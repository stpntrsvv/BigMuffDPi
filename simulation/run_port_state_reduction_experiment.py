"""Сокращает линейные состояния портовой модели сбалансированным усечением."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_frontend_model import nonlinear_terms as frontend_terms
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_experiment import RATE, read_input
from run_port_sparse_linear_experiment import run


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "port_state_reduction"
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_state_reduction"


def finite_gramians(a, b, c, count):
    controllability = np.zeros((len(a), len(a)))
    observability = np.zeros_like(controllability)
    reach = b.copy()
    seen = c.copy()
    for _ in range(count):
        controllability += reach @ reach.T
        observability += seen.T @ seen
        reach = a @ reach
        seen = seen @ a
    return controllability, observability


def factor_psd(matrix):
    value, vector = np.linalg.eigh(0.5 * (matrix + matrix.T))
    limit = max(float(np.max(value)), 1.0) * 1e-14
    value = np.maximum(value, limit)
    return vector @ np.diag(np.sqrt(value))


def balance(a, b, c, horizon):
    p, q = finite_gramians(a, b, c, horizon)
    rp, rq = factor_psd(p), factor_psd(q)
    u, singular, vh = np.linalg.svd(rq.T @ rp)
    root = np.sqrt(np.maximum(singular, singular[0] * 1e-14))
    transform = rp @ vh.T @ np.diag(1.0 / root)
    inverse = np.diag(1.0 / root) @ u.T @ rq.T
    return singular, transform, inverse


def reduced(data, dc_state, rank, kind, method="balanced"):
    result = copy.deepcopy(data)
    a = np.asarray(data["state_transition"])
    if kind == "fast":
        b = np.column_stack((0.1 * data["state_input"], 1e-3 * data["state_port"],
                             -1e-3 * data["state_active"]))
        c = np.vstack((data["active_state"], data["port_state"]))
        bias_rows = (("active_bias", "active_state"), ("port_bias", "port_state"))
        left_rows = ("active_state", "port_state")
    else:
        b = np.column_stack((0.1 * data["state_port"], -1e-3 * data["state_active"]))
        c = np.vstack((data["active_state"], data["output_state"], data["current_state"]))
        bias_rows = (("active_bias", "active_state"), ("output_bias", "output_state"),
                     ("current_bias", "current_state"))
        left_rows = ("active_state", "output_state", "current_state")
    singular, transform, inverse = balance(a, b, c, 4096)
    if method == "modal":
        poles, modal = np.linalg.eig(a)
        modal_inverse = np.linalg.inv(modal)
        modal = np.real_if_close(modal).real
        modal_inverse = np.real_if_close(modal_inverse).real
        modal_b = modal_inverse @ b
        modal_c = c @ modal
        score = (np.linalg.norm(modal_b, axis=1) * np.linalg.norm(modal_c, axis=0)
                 / np.maximum(1.0 - np.abs(poles) ** 2, 1e-9))
        order = np.argsort(score)[::-1]
        t = modal[:, order[:rank]]
        ti = modal_inverse[order[:rank], :]
        singular = score[order]
    else:
        t = transform[:, :rank]
        ti = inverse[:rank, :]
    for bias_name, row_name in bias_rows:
        result[bias_name] = np.asarray(data[bias_name]) + np.asarray(data[row_name]) @ dc_state
    for row_name in left_rows:
        result[row_name] = np.asarray(data[row_name]) @ t
    result["state_bias"] = ti @ (data["state_bias"] + a @ dc_state - dc_state)
    result["state_input"] = ti @ data.get("state_input", np.zeros(len(a)))
    if "state_port" in data:
        result["state_port"] = ti @ data["state_port"]
    result["state_transition"] = ti @ a @ t
    result["state_active"] = ti @ data["state_active"]
    return result, np.zeros(rank), singular


def main():
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR], float(dc_nodes[Q2_COLLECTOR]), True
    )
    fast_reduction = prepare_fast(parameters, 1.0 / (RATE * 4), port_g)
    fast_state = fast_reduction.incidence.T @ dc_nodes[:len(fast_reduction.source)]
    fast = fast_affine(parameters, dc_q[:9], port_g, RATE * 4)
    initial = (fast_state, dc_q[FAST_ACTIVE], slow_state, dc_q[[Q1_FORWARD, Q1_REVERSE]],
               port_i - port_g * dc_nodes[Q2_COLLECTOR], port_g)
    time_s = np.arange(961) / RATE
    source = np.asarray(read_input()(time_s))
    rows = []
    spectra_written = False
    for level_mv in (25, 50, 100):
        signal = source * (level_mv * 1e-3 / np.max(np.abs(source)))
        reference = run(fast, slow, initial, signal, parameters, solver_mode="full_repeat")
        for fast_rank in range(9, 2, -1):
            fast_r, fast_initial, fast_sv = reduced(fast, fast_state, fast_rank, "fast")
            for slow_rank in range(4, 0, -1):
                slow_r, slow_initial, slow_sv = reduced(slow, slow_state, slow_rank, "slow")
                tested_initial = (fast_initial, initial[1], slow_initial, initial[3], initial[4], initial[5])
                if fast_rank == 9 and slow_rank == 4:
                    tested = reference.copy()
                else:
                    tested = run(
                        fast_r, slow_r, tested_initial, signal, parameters,
                        solver_mode="full_repeat",
                    )
                finite = bool(np.all(np.isfinite(tested)))
                error = tested[1:] - reference[1:]
                rms = float(np.sqrt(np.mean(error * error))) if finite else np.inf
                peak = float(np.max(np.abs(error))) if finite else np.inf
                rows.append((level_mv, fast_rank, slow_rank, finite, rms, peak))
                print(level_mv, fast_rank, slow_rank, finite, rms * 1e3, peak * 1e3)
                if not spectra_written:
                    RAW.mkdir(parents=True, exist_ok=True)
                    np.savetxt(RAW / "fast_hankel.csv", fast_sv, delimiter=",")
                    np.savetxt(RAW / "slow_hankel.csv", slow_sv, delimiter=",")
                    spectra_written = True
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level_mv", "fast_rank", "slow_rank", "finite", "rms_v", "peak_v"))
        writer.writerows(rows)
    shared = []
    for fr in range(9, 2, -1):
        for sr in range(4, 0, -1):
            group = [row for row in rows if row[1:3] == (fr, sr)]
            if all(row[3] and row[4] <= 1e-4 and row[5] <= 1e-3 for row in group):
                shared.append((fr, sr, max(row[4] for row in group), max(row[5] for row in group)))
    lines = ["# Сокращение пространства состояний", "",
             "Сбалансированное усечение выполнено на конечном горизонте 4096 внутренних шагов.", "",
             "Критерий: не более 0,1 мВ СКО и 1 мВ пиковой ошибки одновременно при 25, 50 и 100 мВ.", "",
             "| быстрых состояний | медленных состояний | худшее СКО, мВ | худший пик, мВ |",
             "|---:|---:|---:|---:|"]
    for fr, sr, rms, peak in shared:
        lines.append(f"| {fr} | {sr} | {rms*1e3:.6f} | {peak*1e3:.6f} |")
    if not shared:
        lines.append("| — | — | допустимых сокращений нет | — |")
    lines.extend(("", "Сингулярные значения Ханкеля записаны в `simulation/raw/port_state_reduction/`.", ""))
    candidates = []
    for fr, sr in ((8, 4), (7, 4), (9, 3)):
        group = [row for row in rows if row[1:3] == (fr, sr)]
        candidates.append((fr, sr, max(row[4] for row in group), max(row[5] for row in group)))
    lines.extend((
        "## Ближайшие варианты", "",
        "| быстрых состояний | медленных состояний | худшее СКО, мВ | худший пик, мВ |",
        "|---:|---:|---:|---:|",
    ))
    for fr, sr, rms, peak in candidates:
        lines.append(f"| {fr} | {sr} | {rms*1e3:.6f} | {peak*1e3:.6f} |")
    lines.extend((
        "",
        "Удаление одного быстрого состояния даёт 2–4 мВ СКО в зависимости от уровня входа. "
        "При этом сбалансированные координаты превращают диагональную матрицу перехода в плотную, "
        "поэтому уменьшение размерности не означает уменьшения числа умножений на STM32.",
        "",
        "Медленную подсистему сокращать нельзя: переход 4 → 3 даёт около 85–93 мВ СКО.",
        "",
    ))
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
