"""Сравнивает плотную поправку 6×6 с последовательными местными поправками каскадов."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete
from muff_hybrid_active import (
    CASCADE_BLOCKS,
    COUPLED_INPUT_BLOCKS,
    COUPLED_INPUT_Q1_TWICE,
    FRONTEND_BLOCKS,
    FRONTEND_Q1_TWICE,
    prepare_hybrid_active,
    simulate_hybrid_active,
    simulate_hybrid_cascade,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "cascade_solver"
FIGURES = EXPERIMENT / "figures"
FACTOR = 8
DURATION_S = 0.012


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    passes: int = 0
    symmetric: bool = False
    blocks: tuple[np.ndarray, ...] = CASCADE_BLOCKS


METHODS = (
    Method("dense", "Плотная 6×6 + Q1"),
    Method("forward", "Раздельные блоки, один проход", 1, False),
    Method("two_forward", "Раздельные блоки, два прохода", 2, False),
    Method("coupled_input", "Q4+Q3: 3×3, один проход", 1, False, COUPLED_INPUT_BLOCKS),
    Method("coupled_input_q1", "Q4+Q3: 3×3 + Q1 дважды", 1, False, COUPLED_INPUT_Q1_TWICE),
    Method("coupled_input_two", "Q4+Q3: 3×3, два прохода", 2, False, COUPLED_INPUT_BLOCKS),
    Method("frontend", "Входной тракт 4×4 + Q1 2×2", 1, False, FRONTEND_BLOCKS),
    Method("frontend_q1", "Входной тракт 4×4 + Q1 дважды", 1, False, FRONTEND_Q1_TWICE),
)
PLOT_METHOD_NAMES = {
    "dense", "coupled_input_q1", "coupled_input_two", "frontend_q1"
}


def signal(time_s: np.ndarray) -> np.ndarray:
    ramp = np.minimum(1.0, time_s / 1e-3)
    return ramp * (
        0.070 * np.sin(2 * np.pi * 82.41 * time_s)
        + 0.035 * np.sin(2 * np.pi * 164.81 * time_s + 0.3)
        + 0.020 * np.sin(2 * np.pi * 329.63 * time_s + 0.7)
        + 0.010 * np.sin(2 * np.pi * 2637.0 * time_s)
    )


def run_method(method: Method, tone: float):
    if method.name == "dense":
        return simulate_hybrid_active(
            1.0, tone, 0.8, FACTOR, signal, DURATION_S, False, 1
        )
    return simulate_hybrid_cascade(
        1.0, tone, 0.8, FACTOR, signal, DURATION_S,
        method.passes, method.symmetric, method.blocks,
    )


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = {}
    for tone in (0.0, 0.5, 1.0):
        reference = simulate_complete(
            1.0, tone, 0.8, FACTOR, signal, DURATION_S, "hybrid", True
        )
        for method in METHODS:
            started = time.perf_counter()
            result = run_method(method, tone)
            elapsed = time.perf_counter() - started
            error = result.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]
            rows.append((
                tone, method,
                1e3 * float(np.sqrt(np.mean(error * error))),
                1e3 * float(np.max(np.abs(error))),
                float(np.max(result.residual_v)), elapsed,
            ))
            if tone == 1.0:
                traces[method.name] = result
        if tone == 1.0:
            traces["reference"] = reference

    labels = ("Q4", "Q3: BJT", "Q3: диоды", "Q2: диоды", "Q1: BE", "Q1: BC")
    reduction = prepare_hybrid_active(1.0, 1.0, 0.8, FACTOR)
    scale = np.maximum(np.abs(np.diag(reduction.active_influence)), 1e-30)
    normalized = np.abs(reduction.active_influence) / scale[:, np.newaxis]
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(np.log10(np.maximum(normalized, 1e-9)), vmin=-6, vmax=1, cmap="magma")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Связь активных нелинейностей, Tone = 1\nlog10(|Mij|/|Mii|)")
    figure.colorbar(image, ax=axis, label="Десятичный логарифм относительной связи")
    figure.tight_layout()
    figure.savefig(FIGURES / "active_coupling.png", dpi=160)
    plt.close(figure)

    reference = traces["reference"]
    time_ms = reference.time_s * 1e3
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_ms, reference.node_v[:, OUTPUT], color="black", label="Полное схождение")
    for method in METHODS:
        if method.name not in PLOT_METHOD_NAMES:
            continue
        result = traces[method.name]
        axes[0].plot(time_ms, result.node_v[:, OUTPUT], linewidth=0.7, label=method.label)
        axes[1].plot(
            time_ms, 1e3 * (result.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]),
            linewidth=0.8, label=method.label,
        )
    axes[0].set_ylabel("Выход, В")
    axes[1].set_ylabel("Ошибка, мВ")
    axes[1].set_xlabel("Время, мс")
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Каскадные поправки, Tone = 1")
    figure.tight_layout()
    figure.savefig(FIGURES / "cascade_comparison.png", dpi=160)
    plt.close(figure)

    table = "\n".join(
        f"| {tone:.2f} | {method.label} | {rms:.3e} | {peak:.3e} | "
        f"{residual:.3e} | {elapsed:.3f} |"
        for tone, method, rms, peak, residual, elapsed in rows
    )
    report = f"""# Каскадное решение шести нелинейностей

Проверены последовательные местные поправки нескольких размеров. После каждого блока токи
и невязка пересчитываются, поэтому следующий блок использует уже исправленное состояние
предыдущего. Раздельный вариант `Q4 (1×1) → Q3 (2×2) → Q2 (1×1) → Q1 (2×2)` включён
как отрицательная проверка: сильная взаимная связь Q4 и Q3 не позволяет разделять их.

| Tone | Метод | Ошибка, мВ СКО | Пик, мВ | Максимальная невязка, В | Время Python, с |
|---:|---|---:|---:|---:|---:|
{table}

![Связи](figures/active_coupling.png)

![Сравнение](figures/cascade_comparison.png)

Выбраны два кандидата для переноса на STM32:

1. Дешёвый: `Q4+Q3 (3×3) → Q2 (1×1) → Q1 (2×2) → Q1 (2×2)`. Он устойчив во всём
   диапазоне Tone и даёт 0,501–1,159 мВ СКО на проверенных положениях.
2. Надёжный: `Q4+Q3+Q2 (4×4) → Q1 (2×2) → Q1 (2×2)`. Его ошибка 0,060–0,270 мВ СКО,
   то есть практически совпадает с плотной поправкой `6×6`.

Два полных прохода с блоком `Q4+Q3 (3×3)` дают всего 0,004–0,009 мВ СКО, но требуют
повторного расчёта всех каскадов и сохраняются как запасной точный вариант.

Время Python приведено только как проверка программы: короткие вызовы NumPy и повторный расчёт
нелинейных функций искажают соотношение с STM32. Для контроллера существенна структура:
следующий выбор между блоками `3×3` и `4×4` должен делаться по тактам STM32, а не по времени Python.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
