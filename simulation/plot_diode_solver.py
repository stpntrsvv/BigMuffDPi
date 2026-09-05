"""Графики точности одной поправки Ньютона и Галлея."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_diode_solver import (
    DiodePair,
    exact_root,
    halley_step,
    newton_step,
    residual,
    simulate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = (
    PROJECT_ROOT / "simulation" / "experiments" / "diode_solver" / "figures"
)


def make_model() -> DiodePair:
    return DiodePair(
        saturation_current_a=2.0e-9,
        ideality=1.9,
        temperature_k=300.15,
        source_resistance_ohm=10_000.0,
    )


def plot_sine_error(model: DiodePair) -> None:
    factors = (4, 8, 16, 32, 64)
    methods = (("Ньютон", "newton", newton_step), ("Галлей", "halley", halley_step))
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for label, method_name, step in methods:
        results = [
            simulate(
                method_name,
                step,
                "sine",
                factor,
                48_000.0,
                0.005,
                1_000.0,
                2.0,
                model,
            )
            for factor in factors
        ]
        axes[0].loglog(
            factors,
            [result.maximum_error_v for result in results],
            marker="o",
            label=label,
        )
        axes[1].loglog(
            factors,
            [result.maximum_residual_a for result in results],
            marker="o",
            label=label,
        )

    axes[0].set_ylabel("Максимальная ошибка, В")
    axes[0].set_title("Одна поправка на отсчёт: синус 1 кГц")
    axes[1].set_ylabel("Максимальная невязка, А")
    axes[1].set_xlabel("Множитель частоты относительно 48 кГц")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "sine_error_and_residual.png", dpi=160)
    plt.close(figure)


def plot_step_failure(model: DiodePair) -> None:
    sample_rate_hz = 48_000.0 * 16.0
    count = 48
    source = np.zeros(count)
    source[8:] = 2.0
    reference = np.array([exact_root(value, model) for value in source])
    traces: dict[str, np.ndarray] = {}

    for label, step in (("Ньютон", newton_step), ("Галлей", halley_step)):
        values = np.zeros(count)
        current = 0.0
        for index, source_v in enumerate(source):
            try:
                current = step(current, float(source_v), model)
                if not np.isfinite(current) or not np.isfinite(residual(current, source_v, model)):
                    current = reference[index]
            except (OverflowError, ZeroDivisionError):
                current = reference[index]
            values[index] = current
        traces[label] = values

    time_us = np.arange(count) / sample_rate_hz * 1.0e6
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.step(time_us, source, where="post", color="black", alpha=0.45, label="Источник")
    axis.step(time_us, reference, where="post", linewidth=2.2, label="Точный корень")
    for label, values in traces.items():
        axis.plot(time_us, values, marker=".", label=f"Одна поправка: {label}")
    axis.set_xlabel("Время, мкс")
    axis.set_ylabel("Напряжение диодной пары, В")
    axis.set_title("Провал продолжения решения на мгновенном перепаде")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "step_failure.png", dpi=160)
    plt.close(figure)


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    model = make_model()
    plot_sine_error(model)
    plot_step_failure(model)
    print(f"Графики: {FIGURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
