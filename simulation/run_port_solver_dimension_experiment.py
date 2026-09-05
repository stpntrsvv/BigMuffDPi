"""Сравнивает блочные приближения решателя быстрого портового ядра."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_frontend_model import nonlinear_terms as frontend_terms
from muff_multirate_model import (
    Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step,
)
from q3_model import Q3Parameters
from run_port_multirate_experiment import DURATION_S, RATE, read_input
from run_port_sparse_linear_experiment import run


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_solver_dimension"
RAW = ROOT / "simulation" / "raw" / "port_solver_dimension"
MODES = ("full_4x4", "3x3_plus_1", "2x2_plus_2x2", "four_scalars")


def main() -> int:
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[
        dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]
    ]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR],
        float(dc_nodes[Q2_COLLECTOR]), True,
    )
    fast = fast_affine(parameters, dc_q[:9], port_g)
    fast_reduction = prepare_fast(parameters, 1.0 / 192_000.0, port_g)
    fast_state = fast_reduction.incidence.T @ dc_nodes[:len(fast_reduction.source)]
    initial = (
        fast_state, dc_q[FAST_ACTIVE], slow_state,
        dc_q[[Q1_FORWARD, Q1_REVERSE]],
        port_i - port_g*dc_nodes[Q2_COLLECTOR], port_g,
    )
    time_s = np.arange(int(round(DURATION_S*RATE)) + 1) / RATE
    input_v = np.asarray(read_input()(time_s))
    rows = []
    for gain in (1.0, 2.0, 4.0, 8.0):
        outputs = {}
        for mode in MODES:
            try:
                outputs[mode] = run(
                    fast, slow, initial, gain*input_v, parameters, mode
                )
            except (FloatingPointError, np.linalg.LinAlgError, OverflowError):
                outputs[mode] = np.full_like(input_v, np.nan)
        reference = outputs["full_4x4"]
        for mode in MODES:
            finite = bool(
                np.all(np.isfinite(outputs[mode]))
                and np.max(np.abs(outputs[mode])) < 100.0
            )
            reference_finite = bool(
                np.all(np.isfinite(reference)) and np.max(np.abs(reference)) < 100.0
            )
            if finite and reference_finite:
                error = outputs[mode][1:] - reference[1:]
                rms = float(np.sqrt(np.mean(error*error)))
                peak = float(np.max(np.abs(error)))
                final = float(error[-1])
            else:
                rms = peak = final = float("nan")
            rows.append((25.0*gain, mode, finite, rms, peak, final))
            print(f"level_mV={25*gain:.0f}", mode, "finite=" + str(finite),
                  f"rms_mV={rms*1e3:.6f}", f"peak_mV={peak*1e3:.6f}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("уровень_мВ", "решатель", "устойчив", "ско_В", "пик_В", "конец_В"))
        writer.writerows(rows)
    labels = {
        "full_4x4": "полный 4×4", "3x3_plus_1": "3×3 + скаляр",
        "2x2_plus_2x2": "2×2 + 2×2", "four_scalars": "четыре скаляра",
    }
    table = "\n".join(
        f"| {level:.0f} | {labels[mode]} | {'да' if finite else 'нет'} | {rms*1e3:.6f} | "
        f"{peak*1e3:.6f} | {final*1e3:.6f} |"
        for level, mode, finite, rms, peak, final in rows
    )
    (EXPERIMENT / "report.md").write_text(
        "# Размерность быстрого решателя\n\n"
        f"Первые {DURATION_S*1e3:.0f} мс гитарного аккорда, 4×/2×. Физические "
        "уравнения и число состояний не менялись; менялся только способ выполнения "
        "одной поправки быстрого нелинейного блока.\n\n"
        "| Уровень, мВ | Решатель | Устойчив | СКО, мВ | Пик, мВ | Конец, мВ |\n"
        "|---:|---|---|---:|---:|---:|\n" + table + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
