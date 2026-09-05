"""Проверяет индивидуальную частоту четырёх нелинейных портов быстрого ядра."""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_experiment import RATE, read_input
from run_port_sparse_linear_experiment import run


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "port_partition_rate"
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_partition_rate"
NAMES = ("Q4 переход", "Q3 переход", "Q3 диоды", "Q2 диоды")


def main():
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR], float(dc_nodes[Q2_COLLECTOR]), True
    )
    port_offset = port_i - port_g * dc_nodes[Q2_COLLECTOR]
    reduction = prepare_fast(parameters, 1.0 / (RATE * 4), port_g)
    initial = (
        reduction.incidence.T @ dc_nodes[:len(reduction.source)], dc_q[FAST_ACTIVE],
        slow_state, dc_q[[Q1_FORWARD, Q1_REVERSE]], port_offset, port_g,
    )
    fast = fast_affine(parameters, dc_q[:9], port_g, RATE * 4)
    time_s = np.arange(961) / RATE
    source = np.asarray(read_input()(time_s))
    rows = []
    configurations = [(1, 1, 1, 1)]
    configurations += [tuple(2 if i == index else 1 for i in range(4)) for index in range(4)]
    configurations += list(product((1, 2), repeat=4))[1:]
    configurations += [tuple(4 if i == index else 1 for i in range(4)) for index in range(4)]
    configurations = list(dict.fromkeys(configurations))
    for level_mv in (25, 50, 100):
        signal = source * (level_mv * 1e-3 / np.max(np.abs(source)))
        reference = run(fast, slow, initial, signal, parameters)
        for periods in configurations:
            stats = {}
            tested = run(
                fast, slow, initial, signal, parameters,
                schedule=("partition_" + "_".join(map(str, periods)),), stats=stats,
            )
            finite = bool(np.all(np.isfinite(tested)))
            error = tested - reference
            rms = float(np.sqrt(np.mean(error * error))) if finite else np.inf
            peak = float(np.max(np.abs(error))) if finite else np.inf
            rows.append((level_mv, *periods, finite, rms, peak))
            print(level_mv, periods, finite, rms * 1e3, peak * 1e3)

    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level_mv", "q4", "q3", "q3_diode", "q2_diode", "finite", "rms_v", "peak_v"))
        writer.writerows(rows)
    lines = ["# Частоты нелинейных портов", "", "Период 1 означает 4×, 2 — 2×, 4 — 1×.", "",
             "| уровень, мВ | Q4 | Q3 | диоды Q3 | диоды Q2 | СКО, мВ | пик, мВ |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for level, q4, q3, d3, d2, finite, rms, peak in rows:
        if finite and rms <= 1e-4 and peak <= 1e-3:
            lines.append(f"| {level} | {q4} | {q3} | {d3} | {d2} | {rms*1e3:.6f} | {peak*1e3:.6f} |")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
