"""Измеряет рабочие диапазоны и чувствительность четырёх портов быстрого ядра."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import active_terms, run

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "fast_port_sensitivity"
EXPERIMENT = ROOT / "simulation" / "experiments" / "fast_port_sensitivity"
RATE, DURATION_S = 48_000, 0.020
LEVELS_MV = (25, 50, 100, 200)
PORTS = ("Q4 B–E", "Q3 B–E", "Q3 диоды", "Q2 диоды")


def rms(value):
    return float(np.sqrt(np.mean(value * value)))


def main():
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR], float(dc_nodes[Q2_COLLECTOR]), True)
    fast = fast_affine(parameters, dc_q[:9], port_g)
    reduction = prepare_fast(parameters, 1.0 / (RATE * 4), port_g)
    fast_state = reduction.incidence.T @ dc_nodes[:len(reduction.source)]
    initial = (fast_state, dc_q[FAST_ACTIVE], slow_state, dc_q[[Q1_FORWARD, Q1_REVERSE]],
               port_i - port_g * dc_nodes[Q2_COLLECTOR], port_g)
    dc_active = dc_q[FAST_ACTIVE].copy()
    dc_current, dc_first = active_terms(dc_active, parameters)
    rows, ranges = [], {name: [np.inf, -np.inf, np.inf, -np.inf] for name in PORTS}
    start = int(round(0.003 * RATE))
    for level_mv in LEVELS_MV:
        time = np.arange(int(round(DURATION_S * RATE)) + 1) / RATE
        signal = np.asarray(source_input(level_mv * 1e-3)(time))

        def observed(q, p):
            current, first = active_terms(q, p)
            for index, name in enumerate(PORTS):
                ranges[name][0] = min(ranges[name][0], float(q[index]))
                ranges[name][1] = max(ranges[name][1], float(q[index]))
                ranges[name][2] = min(ranges[name][2], float(first[index]))
                ranges[name][3] = max(ranges[name][3], float(first[index]))
            return current, first

        reference = run(fast, slow, initial, signal, parameters, "full_repeat", terms_function=observed)
        for index, name in enumerate(PORTS):
            def simplified(q, p, selected=index):
                current, first = active_terms(q, p)
                current[selected] = dc_current[selected] + dc_first[selected] * (q[selected] - dc_active[selected])
                first[selected] = dc_first[selected]
                return current, first
            tested = run(fast, slow, initial, signal, parameters, "full_repeat", terms_function=simplified)
            error = tested[start:] - reference[start:]
            rows.append((name, level_mv, bool(np.all(np.isfinite(tested))),
                         rms(error)*1e3, float(np.max(np.abs(error)))*1e3))

    RAW.mkdir(parents=True, exist_ok=True); EXPERIMENT.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("port", "level_mv", "finite", "rms_mv", "peak_mv")); writer.writerows(rows)
    with (RAW / "ranges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("port", "minimum_v", "maximum_v", "minimum_s", "maximum_s"))
        writer.writerows((name, *ranges[name]) for name in PORTS)
    table = []
    for name in PORTS:
        group = [row for row in rows if row[0] == name]
        qmin, qmax, gmin, gmax = ranges[name]
        table.append(f"| {name} | {qmin:.6f}…{qmax:.6f} | {gmin:.3e}…{gmax:.3e} | "
                     f"{max(r[3] for r in group):.3f} | {max(r[4] for r in group):.3f} |")
    report = """# Чувствительность четырёх портов быстрого ядра

Эталон — две полные поправки на шаге 4×. Каждый кандидат заменяет закон только
одного порта его касательной в рабочей точке; остальные порты и девять состояний
не меняются. Проверена атака аккорда 25–200 мВ при Sustain = Tone = 1.

| Порт | Диапазон напряжения, В | Диапазон производной, См | Худшее СКО, мВ | Худший пик, мВ |
|---|---:|---:|---:|---:|
""" + "\n".join(table) + "\n"
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
