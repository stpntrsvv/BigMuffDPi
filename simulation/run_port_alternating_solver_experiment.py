"""Сравнивает чередование полного шага и прогноза с полноценным ядром 2×."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import (
    Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step,
)
from q3_model import Q3Parameters
from run_port_multirate_experiment import DURATION_S, RATE, read_input
from run_port_sparse_linear_experiment import run


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_alternating_solver"
RAW = ROOT / "simulation" / "raw" / "port_alternating_solver"


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
    port_offset = port_i - port_g*dc_nodes[Q2_COLLECTOR]
    initial_slow = (slow_state, dc_q[[Q1_FORWARD, Q1_REVERSE]])

    def initial(factor):
        reduction = prepare_fast(parameters, 1.0/(RATE*factor), port_g)
        fast_state = reduction.incidence.T @ dc_nodes[:len(reduction.source)]
        return (fast_state, dc_q[FAST_ACTIVE], initial_slow[0], initial_slow[1],
                port_offset, port_g)

    fast4 = fast_affine(parameters, dc_q[:9], port_g, RATE*4)
    fast2 = fast_affine(parameters, dc_q[:9], port_g, RATE*2)
    time_s = np.arange(int(round(DURATION_S*RATE)) + 1) / RATE
    base_input = np.asarray(read_input()(time_s))
    configurations = (
        ("full_4x", fast4, initial(4), 4, None),
        ("full_predict", fast4, initial(4), 4,
         ("full_4x4", "predict", "full_4x4", "predict")),
        ("predict_full", fast4, initial(4), 4,
         ("predict", "full_4x4", "predict", "full_4x4")),
        ("one_full_three_predict", fast4, initial(4), 4,
         ("full_4x4", "predict", "predict", "predict")),
        ("adaptive_001", fast4, initial(4), 4, ("adaptive_0.001",)),
        ("adaptive_010", fast4, initial(4), 4, ("adaptive_0.010",)),
        ("adaptive_050", fast4, initial(4), 4, ("adaptive_0.050",)),
        ("full_2x", fast2, initial(2), 2, None),
    )
    rows = []
    for gain in (1.0, 2.0):
        outputs = {}
        statistics = {}
        for name, fast, seed, factor, schedule in configurations:
            statistics[name] = {}
            outputs[name] = run(
                fast, slow, seed, gain*base_input, parameters,
                "full_4x4", factor, 2, schedule, statistics[name],
            )
        reference = outputs["full_4x"]
        for name, *_ in configurations:
            error = outputs[name][1:] - reference[1:]
            stable = bool(
                np.all(np.isfinite(outputs[name]))
                and np.max(np.abs(outputs[name])) < 100.0
            )
            rms = float(np.sqrt(np.mean(error*error))) if stable else float("nan")
            peak = float(np.max(np.abs(error))) if stable else float("nan")
            total_steps = sum(statistics[name].values())
            predicted_share = (statistics[name]["predicted_steps"] / total_steps
                               if total_steps else 0.0)
            rows.append((25.0*gain, name, stable, rms, peak, float(error[-1]),
                         predicted_share))
            print(f"level_mV={25*gain:.0f}", name, f"stable={stable}",
                  f"rms_mV={rms*1e3:.6f}", f"peak_mV={peak*1e3:.6f}",
                  f"predicted={predicted_share:.3f}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    with (RAW/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("уровень_мВ", "режим", "устойчив", "ско_В", "пик_В", "конец_В", "доля_прогноза"))
        writer.writerows(rows)
    labels = {
        "full_4x": "полный 4×", "full_predict": "полный/прогноз",
        "predict_full": "прогноз/полный",
        "one_full_three_predict": "один полный из четырёх",
        "adaptive_001": "адаптивный 0,001", "adaptive_010": "адаптивный 0,010",
        "adaptive_050": "адаптивный 0,050",
        "full_2x": "полный 2×",
    }
    table = "\n".join(
        f"| {level:.0f} | {labels[name]} | {'да' if stable else 'нет'} | "
        f"{rms*1e3:.6f} | {peak*1e3:.6f} | {share:.3f} |"
        for level, name, stable, rms, peak, _, share in rows
    )
    (EXPERIMENT/"report.md").write_text(
        "# Чередование полного шага и прогноза\n\n"
        "Эталон — полное быстрое ядро 4× с одной поправкой 4×4 на каждом шаге. "
        "Прогноз обновляет линейные состояния и нелинейные координаты, но не решает 4×4.\n\n"
        "| Уровень, мВ | Режим | Устойчив | СКО, мВ | Пик, мВ | Доля прогноза |\n"
        "|---:|---|---|---:|---:|---:|\n" + table + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
