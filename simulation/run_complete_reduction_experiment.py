"""Проверяет точное исключение 25 линейных узлов из каждого звукового шага."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete
from muff_complete_reduced import prepare_affine_step, simulate_complete_affine


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "complete_reduction"
FIGURES = EXPERIMENT / "figures"
DURATION_S = 0.012
FACTOR = 8


@dataclass(frozen=True)
class Case:
    sustain: float
    tone: float
    volume: float = 0.8


CASES = (
    Case(0.25, 0.50),
    Case(1.00, 0.00),
    Case(1.00, 0.50),
    Case(1.00, 1.00),
)


def signal(time_s: np.ndarray) -> np.ndarray:
    ramp = np.minimum(1.0, time_s / 1e-3)
    return ramp * (
        0.070 * np.sin(2 * np.pi * 82.41 * time_s)
        + 0.035 * np.sin(2 * np.pi * 164.81 * time_s + 0.3)
        + 0.020 * np.sin(2 * np.pi * 329.63 * time_s + 0.7)
        + 0.010 * np.sin(2 * np.pi * 2637.0 * time_s)
    )


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Case, float, float, float, float, float]] = []
    traces = None
    for case in CASES:
        started = time.perf_counter()
        direct = simulate_complete(
            case.sustain, case.tone, case.volume, FACTOR, signal,
            DURATION_S, "hybrid", False, 1, 1,
        )
        direct_s = time.perf_counter() - started
        started = time.perf_counter()
        affine = simulate_complete_affine(
            case.sustain, case.tone, case.volume, FACTOR, signal,
            DURATION_S, "hybrid", False, 1, 1,
        )
        affine_s = time.perf_counter() - started
        node_error = affine.node_v - direct.node_v
        q_error = affine.nonlinear_v - direct.nonlinear_v
        rows.append((
            case,
            float(np.max(np.abs(node_error))),
            float(np.max(np.abs(q_error))),
            float(np.sqrt(np.mean(node_error[:, OUTPUT] ** 2))),
            direct_s,
            affine_s,
        ))
        if case.tone == 1.0:
            traces = direct, affine

    assert traces is not None
    direct, affine = traces
    time_ms = direct.time_s * 1e3
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_ms, direct.node_v[:, OUTPUT], label="Решение 25×25")
    axes[0].plot(
        time_ms, affine.node_v[:, OUTPUT], "--", linewidth=0.8,
        label="Аффинное сокращение",
    )
    axes[0].set_ylabel("Выход, В")
    axes[0].legend()
    axes[1].plot(time_ms, 1e12 * (affine.node_v[:, OUTPUT] - direct.node_v[:, OUTPUT]))
    axes[1].set_ylabel("Разность, пВ")
    axes[1].set_xlabel("Время, мс")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Точное исключение линейных узлов, Tone = 1")
    figure.tight_layout()
    figure.savefig(FIGURES / "affine_equivalence.png", dpi=160)
    plt.close(figure)

    compact = prepare_affine_step(1.0, 0.5, 0.8, FACTOR)
    coefficient_count = (
        compact.q_state.size + compact.q_input.size + compact.q_bias.size
        + compact.influence.size + compact.state_transition.size
        + compact.state_nonlinear.size + compact.state_input.size
        + compact.state_bias.size + compact.output_state.size
        + compact.output_nonlinear.size + 2
    )
    table = "\n".join(
        f"| {case.sustain:.2f} | {case.tone:.2f} | {1e12 * node:.3f} | "
        f"{1e12 * q:.3f} | {1e12 * output:.3f} | {direct_s:.3f} | "
        f"{affine_s:.3f} | {direct_s / affine_s:.2f} |"
        for case, node, q, output, direct_s, affine_s in rows
    )
    report = f"""# Точное исключение линейных узлов

В исходном шаге дважды решалась линейная система `25×25`: для предиктора
нелинейных напряжений и для восстановления узлов. Обе операции точно заменены
аффинными матричными произведениями. Обращение матрицы выполняется один раз при
подготовке коэффициентов, а не на звуковом шаге.

| Sustain | Tone | Макс. по узлам, пВ | Макс. по нелин. переменным, пВ | Выход, пВ СКО | 25×25, с | Сокращение, с | Ускорение |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

![Совпадение](figures/affine_equivalence.png)

Для минимального шага, который обновляет 13 напряжений состояния, 12 нелинейных
предикторов и один выход, нужно `{coefficient_count}` коэффициента, то есть
`{4 * coefficient_count}` байт в `float32`. Полные 25 узлов на STM32 восстанавливать не нужно.

Это точное алгебраическое сокращение, а не упрощение физики. Следующий шаг — сократить
нелинейную систему `12×12` по уже выбранной физике Q3/Q2/Q1 и перенести полученный шаг в C.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
