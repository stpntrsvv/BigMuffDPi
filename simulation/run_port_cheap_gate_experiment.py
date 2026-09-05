"""Подбирает дешёвый признак полного нелинейного шага без пробной невязки."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_experiment import RATE, read_input
from run_port_sparse_linear_experiment import run


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_cheap_gate"
RAW = ROOT / "simulation" / "raw" / "port_cheap_gate"


def main():
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR],
        float(dc_nodes[Q2_COLLECTOR]), True,
    )
    port_offset = port_i - port_g * dc_nodes[Q2_COLLECTOR]
    reduction = prepare_fast(parameters, 1.0 / (RATE * 4), port_g)
    fast_state = reduction.incidence.T @ dc_nodes[:len(reduction.source)]
    initial_state = (
        fast_state, dc_q[FAST_ACTIVE], slow_state,
        dc_q[[Q1_FORWARD, Q1_REVERSE]], port_offset, port_g,
    )
    fast = fast_affine(parameters, dc_q[:9], port_g, RATE * 4)
    time_s = np.arange(961) / RATE
    source = np.asarray(read_input()(time_s))
    rows = []
    for level_mv in (25, 50, 100):
        signal = source * (level_mv * 1e-3 / max(np.max(np.abs(source)), 1e-12))
        reference = run(fast, slow, initial_state, signal, parameters)
        for period in (2, 3, 4, 6, 8, 16):
            for limit in (0.0001, 0.0003, 0.001, 0.003, 0.010):
                stats = {}
                try:
                    tested = run(
                        fast, slow, initial_state, signal, parameters,
                        schedule=(f"quadratic_{limit}_{period}",), stats=stats,
                    )
                    finite = bool(np.all(np.isfinite(tested)))
                    error = tested - reference
                    rms = float(np.sqrt(np.mean(error * error))) if finite else np.inf
                    peak = float(np.max(np.abs(error))) if finite else np.inf
                except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    finite, rms, peak = False, np.inf, np.inf
                total = stats.get("full_steps", 0) + stats.get("predicted_steps", 0)
                predicted = stats.get("predicted_steps", 0) / total if total else 0.0
                rows.append((level_mv, period, limit, finite, predicted, rms, peak))
                print(level_mv, period, limit, finite, predicted, rms * 1e3, peak * 1e3)

    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level_mv", "period", "limit", "finite", "predicted_fraction", "rms_v", "peak_v"))
        writer.writerows(rows)

    acceptable = [row for row in rows if row[3] and row[5] <= 1e-4 and row[6] <= 1e-3]
    shared = []
    for period in (2, 3, 4, 6, 8, 16):
        for limit in (0.0001, 0.0003, 0.001, 0.003, 0.010):
            group = [row for row in rows if row[1] == period and row[2] == limit]
            if len(group) == 3 and all(row[3] and row[5] <= 1e-4 and row[6] <= 1e-3 for row in group):
                shared.append((period, limit))
    lines = [
        "# Дешёвый признак полного шага", "",
        "Признак использует нормированную величину поправки по сохранённому LU-разложению и обязательный полный шаг через заданный период.", "",
        "Критерий отбора отдельной строки: не более 0,1 мВ СКО и 1 мВ пиковой ошибки.", "",
    ]
    if acceptable:
        lines.extend((
            "| уровень, мВ | период | порог поправки | доля прогнозов | СКО, мВ | пик, мВ |",
            "|---:|---:|---:|---:|---:|---:|",
        ))
        for level, period, limit, _, fraction, rms, peak in acceptable:
            lines.append(f"| {level} | {period} | {limit:.3f} | {fraction:.1%} | {rms*1e3:.6f} | {peak*1e3:.6f} |")
    else:
        lines.append("Ни одна проверенная комбинация не прошла критерий на всех режимах.")
    lines.extend(("", f"Общих настроек, прошедших одновременно 25, 50 и 100 мВ: **{len(shared)}**.", ""))
    if not shared:
        lines.append("Квадратичная оценка пригодна только как дешёвый предварительный отказ перед точной проверкой невязки.")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
