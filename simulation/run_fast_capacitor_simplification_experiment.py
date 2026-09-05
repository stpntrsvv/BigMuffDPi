"""Проверяет удаление 470-пФ ёмкостей быстрого ядра."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "fast_capacitor_simplification"
EXPERIMENT = ROOT / "simulation" / "experiments" / "fast_capacitor_simplification"
FIGURES = EXPERIMENT / "figures"
RATE, DURATION_S = 48_000, 0.020
LEVELS_MV = (25, 50, 100, 200)
CAP_NAMES = ("C1", "C10", "C4", "C5", "C12", "C6", "C13", "C11", "C7")
SMALL = (1, 4, 7)


def rms(value):
    return float(np.sqrt(np.mean(value * value)))


def product_count(data, removed):
    keep = np.ones(9, dtype=bool); keep[list(removed)] = False
    arrays = (data["active_state"][:, keep], data["state_transition"][np.ix_(keep, keep)],
              data["state_active"][keep], data["port_state"][keep])
    return sum(int(np.count_nonzero(np.abs(value) >= 1e-7)) for value in arrays)


def main():
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR], float(dc_nodes[Q2_COLLECTOR]), True)
    common = (dc_q[FAST_ACTIVE], slow_state, dc_q[[Q1_FORWARD, Q1_REVERSE]],
              port_i - port_g * dc_nodes[Q2_COLLECTOR], port_g)
    candidates = [()] + [c for n in range(1, 4) for c in itertools.combinations(SMALL, n)]
    models, initials = {}, {}
    for removed in candidates:
        scale = np.ones(9); scale[list(removed)] = 0.0
        models[removed] = fast_affine(parameters, dc_q[:9], port_g, RATE * 4, scale)
        reduction = prepare_fast(parameters, 1.0 / (RATE * 4), port_g, scale)
        state = reduction.incidence.T @ dc_nodes[:len(reduction.source)]
        initials[removed] = (state, *common)

    rows, start = [], int(round(0.003 * RATE))
    for level_mv in LEVELS_MV:
        time = np.arange(int(round(DURATION_S * RATE)) + 1) / RATE
        signal = np.asarray(source_input(level_mv * 1e-3)(time))
        reference = run(models[()], slow, initials[()], signal, parameters, "full_repeat")
        for removed in candidates:
            tested = reference if not removed else run(
                models[removed], slow, initials[removed], signal, parameters, "full_repeat")
            error = tested[start:] - reference[start:]
            finite = bool(np.all(np.isfinite(tested)))
            spectrum = np.fft.rfft(error)
            frequency = np.fft.rfftfreq(len(error), 1.0 / RATE)
            spectrum[frequency < 10_000] = 0.0
            high = np.fft.irfft(spectrum, n=len(error))
            rows.append(("+".join(CAP_NAMES[i] for i in removed) or "полная", level_mv,
                         finite, rms(error)*1e3, np.max(np.abs(error))*1e3,
                         rms(high)*1e3, product_count(models[removed], removed)))

    RAW.mkdir(parents=True, exist_ok=True); EXPERIMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(
            ("removed", "level_mv", "finite", "rms_mv", "peak_mv", "high_rms_mv", "linear_products"))
        writer.writerows(rows)
    worst = {}
    for removed in candidates:
        name = "+".join(CAP_NAMES[i] for i in removed) or "полная"
        group = [r for r in rows if r[0] == name]
        worst[name] = (max(r[3] for r in group), max(r[4] for r in group),
                       max(r[5] for r in group), group[0][6], all(r[2] for r in group))
    table = "\n".join(
        f"| {name} | {9 if name == 'полная' else 9-len(name.split('+'))} | {a:.3f} | {b:.3f} | {c:.3f} | {p} | {'да' if ok else 'нет'} |"
        for name, (a, b, c, p, ok) in worst.items())
    fig, ax = plt.subplots(figsize=(10, 5)); names = list(worst)
    ax.bar(names, [worst[n][0] for n in names]); ax.set_ylabel("Худшая ошибка, мВ СКО")
    ax.tick_params(axis="x", rotation=35); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "error_by_removed_capacitor.png", dpi=160); plt.close(fig)
    report = f"""# Удаление малых ёмкостей быстрого ядра

Проверены C10, C12 и C11 по 470 пФ — межколлекторно-базовые ёмкости Q4, Q3 и Q2.
Все четыре активных нелинейных порта сохранены. Эталон и кандидаты рассчитаны двумя
полными поправками (`full_repeat`) на каждом шаге 4×; медленная часть работает 2×.
Вход — 20 мс атаки ми-мажорного аккорда, 25–200 мВ, Sustain = Tone = 1.
Первые 3 мс исключены из оценки.

| Удалено | Состояний | Худшее СКО, мВ | Худший пик, мВ | Выше 10 кГц, мВ СКО | Линейных произведений | Устойчиво |
|---|---:|---:|---:|---:|---:|---|
{table}

![Ошибка](figures/error_by_removed_capacitor.png)

Число произведений — статическая оценка с порогом `1e-7`, не аппаратный замер.
Удаление ёмкости устраняет накопитель энергии и потому является изменением физики.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
