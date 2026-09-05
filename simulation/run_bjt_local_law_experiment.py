"""Проверяет локальный полиномиальный закон активных BJT-портов."""

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
RAW = ROOT / "simulation" / "raw" / "bjt_local_law"
EXPERIMENT = ROOT / "simulation" / "experiments" / "bjt_local_law"
RATE, DURATION_S = 48_000, 0.020
LEVELS_MV = (25, 50, 100, 200)


def main():
    p = Q3Parameters(); nodes, dc_q = operating_point(1.0, 1.0, 0.8, p)
    sr, slow = slow_affine(p)
    ss = sr.incidence.T @ np.r_[nodes[Q2_COLLECTOR], nodes[SLOW_NODES]]
    _, _, _, _, pi, pg = slow_step(sr, ss, dc_q[Q1_NONLINEAR], float(nodes[Q2_COLLECTOR]), True)
    fast = fast_affine(p, dc_q[:9], pg); fr = prepare_fast(p, 1.0/(RATE*4), pg)
    fs = fr.incidence.T @ nodes[:len(fr.source)]
    initial = (fs, dc_q[FAST_ACTIVE], ss, dc_q[[Q1_FORWARD, Q1_REVERSE]],
               pi-pg*nodes[Q2_COLLECTOR], pg)
    dc = dc_q[FAST_ACTIVE]; i0, g0 = active_terms(dc, p)
    scale = p.forward_ideality * p.thermal_voltage_v
    variants = (("Q3 квадратичный", (1,), 2), ("Q3 кубический", (1,), 3),
                ("Q4 квадратичный", (0,), 2), ("Q4+Q3 квадратичные", (0, 1), 2))
    rows, start = [], int(round(0.003*RATE))
    for level in LEVELS_MV:
        time = np.arange(int(round(DURATION_S*RATE))+1)/RATE
        signal = np.asarray(source_input(level*1e-3)(time))
        reference = run(fast, slow, initial, signal, p, "full_repeat")
        for name, selected, order in variants:
            def terms(q, parameters, indices=selected, polynomial_order=order):
                current, first = active_terms(q, parameters)
                for index in indices:
                    d = q[index]-dc[index]; z = d/scale
                    current[index] = i0[index] + g0[index]*d + 0.5*g0[index]*d*z
                    first[index] = g0[index]*(1.0+z)
                    if polynomial_order == 3:
                        current[index] += g0[index]*d*z*z/6.0
                        first[index] += 0.5*g0[index]*z*z
                return current, first
            tested = run(fast, slow, initial, signal, p, "full_repeat", terms_function=terms)
            error = tested[start:]-reference[start:]
            rows.append((name, level, bool(np.all(np.isfinite(tested))),
                         np.sqrt(np.mean(error*error))*1e3, np.max(np.abs(error))*1e3))
    RAW.mkdir(parents=True, exist_ok=True); EXPERIMENT.mkdir(parents=True, exist_ok=True)
    with (RAW/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        w=csv.writer(stream); w.writerow(("variant","level_mv","finite","rms_mv","peak_mv")); w.writerows(rows)
    lines=[]
    for name, _, _ in variants:
        group=[r for r in rows if r[0]==name]
        lines.append(f"| {name} | {max(r[3] for r in group):.6f} | {max(r[4] for r in group):.6f} | {'да' if all(r[2] for r in group) else 'нет'} |")
    report="""# Локальный закон BJT быстрого ядра

Экспоненциальный закон активного перехода заменён разложением около рабочей точки.
Порт, его нелинейная обратная связь, конденсаторы и размер решателя сохранены.
Эталон — `full_repeat`, атака аккорда 25–200 мВ, Sustain = Tone = 1.

| Вариант | Худшее СКО, мВ | Худший пик, мВ | Устойчиво |
|---|---:|---:|---|
"""+"\n".join(lines)+"\n"
    (EXPERIMENT/"report.md").write_text(report, encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__": main()
