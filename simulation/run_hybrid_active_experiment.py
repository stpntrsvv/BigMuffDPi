"""Проверяет сокращение гибридной педали с 12 до 6 активных нелинейностей."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete
from muff_hybrid_active import ACTIVE_COUNT, simulate_hybrid_active


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "hybrid_active"
FIGURES = EXPERIMENT / "figures"
FACTOR = 8
DURATION_S = 0.012


def signal(time_s: np.ndarray) -> np.ndarray:
    ramp = np.minimum(1.0, time_s / 1e-3)
    return ramp * (
        0.070 * np.sin(2 * np.pi * 82.41 * time_s)
        + 0.035 * np.sin(2 * np.pi * 164.81 * time_s + 0.3)
        + 0.020 * np.sin(2 * np.pi * 329.63 * time_s + 0.7)
        + 0.010 * np.sin(2 * np.pi * 2637.0 * time_s)
    )


def error_metrics(candidate, reference) -> tuple[float, float, float]:
    error = candidate.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]
    return (
        1e3 * float(np.sqrt(np.mean(error * error))),
        1e3 * float(np.max(np.abs(error))),
        float(np.max(candidate.residual_v)),
    )


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    exact_node_error = 0.0
    exact_q_error = 0.0
    plot_data = None
    for tone in (0.0, 0.5, 1.0):
        reference = simulate_complete(
            1.0, tone, 0.8, FACTOR, signal, DURATION_S, "hybrid", True
        )
        reduced_exact = simulate_hybrid_active(
            1.0, tone, 0.8, FACTOR, signal, DURATION_S, True, 0
        )
        exact_node_error = max(
            exact_node_error,
            float(np.max(np.abs(reduced_exact.node_v - reference.node_v))),
        )
        exact_q_error = max(
            exact_q_error,
            float(np.max(np.abs(reduced_exact.nonlinear_v - reference.nonlinear_v))),
        )
        started = time.perf_counter()
        baseline = simulate_complete(
            1.0, tone, 0.8, FACTOR, signal, DURATION_S,
            "hybrid", False, 1, 1,
        )
        baseline_s = time.perf_counter() - started
        started = time.perf_counter()
        reduced = simulate_hybrid_active(
            1.0, tone, 0.8, FACTOR, signal, DURATION_S, False, 1
        )
        reduced_s = time.perf_counter() - started
        baseline_metrics = error_metrics(baseline, reference)
        reduced_metrics = error_metrics(reduced, reference)
        rows.append((tone, baseline_metrics, reduced_metrics, baseline_s, reduced_s))
        if tone == 1.0:
            plot_data = reference, baseline, reduced

    assert plot_data is not None
    reference, baseline, reduced = plot_data
    time_ms = reference.time_s * 1e3
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_ms, reference.node_v[:, OUTPUT], label="Полное схождение")
    axes[0].plot(time_ms, baseline.node_v[:, OUTPUT], linewidth=0.8, label="12 переменных")
    axes[0].plot(time_ms, reduced.node_v[:, OUTPUT], "--", linewidth=0.8, label="6 переменных")
    axes[0].set_ylabel("Выход, В")
    axes[0].legend()
    axes[1].plot(
        time_ms, 1e3 * (baseline.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]),
        label="12 переменных",
    )
    axes[1].plot(
        time_ms, 1e3 * (reduced.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]),
        label="6 переменных",
    )
    axes[1].set_ylabel("Ошибка, мВ")
    axes[1].set_xlabel("Время, мс")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Гибридная модель: точное исключение аффинных нелинейностей")
    figure.tight_layout()
    figure.savefig(FIGURES / "six_variable_comparison.png", dpi=160)
    plt.close(figure)

    table = "\n".join(
        f"| {tone:.2f} | {base[0]:.3f} | {base[1]:.2f} | {base[2]:.3e} | "
        f"{red[0]:.3f} | {red[1]:.2f} | {red[2]:.3e} | "
        f"{baseline_s:.3f} | {reduced_s:.3f} | {baseline_s/reduced_s:.2f} |"
        for tone, base, red, baseline_s, reduced_s in rows
    )
    report = f"""# Шесть активных нелинейностей

В выбранной гибридной физике из 12 нелинейных напряжений только {ACTIVE_COUNT} остаются
действительно нелинейными: `I_F(Q4)`, `I_F(Q3)`, диоды Q3, диоды Q2, `I_F(Q1)` и `I_R(Q1)`.
Три запертых обратных перехода, две неиспользуемые переменные и линеаризованный
`I_F(Q2)` исключены алгебраически.

При полной сходимости максимальная разность двух форм записи составляет
`{1e9 * exact_node_error:.3f}` нВ по узлам и `{1e9 * exact_q_error:.3f}` нВ по нелинейным
напряжениям. То есть само исключение не меняет решение в пределах численной точности.

| Tone | 12×12: ошибка, мВ СКО | 12×12: пик, мВ | 12×12: невязка, В | 6×6: ошибка, мВ СКО | 6×6: пик, мВ | 6×6: невязка, В | 12×12, с | 6×6, с | Ускорение |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

![Сравнение](figures/six_variable_comparison.png)

Сравнение выполнено с эталоном той же гибридной физики, доведённым до полной сходимости.
Оба быстрых варианта делают одну общую поправку и одну местную поправку Q1. Это проверка
влияния исключения, а не ещё замер тактов STM32.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
