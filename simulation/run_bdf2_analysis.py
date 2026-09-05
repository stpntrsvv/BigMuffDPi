"""Сравнивает Эйлер и BDF2 в полной физической модели педали."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muff_complete_model import OUTPUT, simulate_complete
from run_oversampling_analysis import BASE_RATE, align, analyze, lowpass_downsample


EXPERIMENT = Path(__file__).parent / "experiments" / "bdf2_analysis"


def run_tone(
    frequency_hz: float, level_v: float, factor: int, duration_s: float, method: str
) -> np.ndarray:
    cache = EXPERIMENT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (
        f"tone_{frequency_hz:g}hz_{level_v * 1e3:g}mv_"
        f"{factor}x_{method}_{duration_s:g}s.npy"
    )
    if path.exists():
        return np.load(path)
    result = simulate_complete(
        1.0, 1.0, 0.8, factor,
        lambda time_s: level_v * np.sin(2.0 * np.pi * frequency_hz * time_s),
        duration_s, "hybrid", True, integration_method=method,
    )
    output = lowpass_downsample(result.node_v[:, OUTPUT], factor)
    np.save(path, output)
    return output


def spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    magnitude = 20.0 * np.log10(np.maximum(
        np.abs(np.fft.rfft(signal * window)) / np.sum(window), 1e-12
    ))
    return np.fft.rfftfreq(len(signal), 1.0 / BASE_RATE), magnitude


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.06)
    parser.add_argument("--frequencies", default="440,1000,3000,6000")
    parser.add_argument("--levels-mv", default="5,25,100")
    args = parser.parse_args()
    frequencies = tuple(float(value) for value in args.frequencies.split(","))
    levels = tuple(float(value) * 1e-3 for value in args.levels_mv.split(","))
    settle = int(round(min(0.02, args.duration * 0.4) * BASE_RATE))
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []

    for frequency_hz in frequencies:
        for level_v in levels:
            print(f"{frequency_hz:g} Гц, {level_v * 1e3:g} мВ", flush=True)
            reference = run_tone(frequency_hz, level_v, 16, args.duration, "euler")[settle:]
            candidates = {
                "Эйлер 1×": run_tone(frequency_hz, level_v, 1, args.duration, "euler")[settle:],
                "BDF2 1×": run_tone(frequency_hz, level_v, 1, args.duration, "bdf2")[settle:],
                "BDF2 2×": run_tone(frequency_hz, level_v, 2, args.duration, "bdf2")[settle:],
                "Трапеции 1×": run_tone(frequency_hz, level_v, 1, args.duration, "trapezoid")[settle:],
                "Трапеции/Эйлер 1×": run_tone(frequency_hz, level_v, 1, args.duration, "trapezoid_adaptive")[settle:],
            }
            for label, candidate in candidates.items():
                if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > 20.0:
                    rows.append(f"| {frequency_hz:g} | {level_v * 1e3:g} | {label} | СРЫВ | — | — | — |")
                    continue
                metrics = analyze(reference, candidate, 1, frequency_hz)
                rows.append(
                    f"| {frequency_hz:g} | {level_v * 1e3:g} | {label} | "
                    f"{metrics.peak_v:.6f} | {metrics.thd_percent:.3f} | "
                    f"{metrics.relative_error_percent:.4f} | {metrics.magnitude_error_db:.2f} |"
                )

            if abs(level_v - 0.025) < 1e-12:
                figure, axes = plt.subplots(2, 1, figsize=(12, 9))
                signals = {"Эйлер 16×": reference, **candidates}
                for label, signal in signals.items():
                    freq, magnitude = spectrum(signal)
                    axes[0].plot(freq, magnitude, label=label, alpha=0.85)
                for label, candidate in candidates.items():
                    ref, aligned = align(reference, candidate)
                    freq, magnitude = spectrum(aligned - ref)
                    axes[1].plot(freq, magnitude, label=label, alpha=0.85)
                axes[0].set_title(f"Выходной спектр, {frequency_hz:g} Гц, 25 мВ")
                axes[1].set_title("Спектр ошибки относительно Эйлера 16×")
                axes[1].set_xlabel("Частота, Гц")
                for axis in axes:
                    axis.set_xlim(0, 24_000)
                    axis.set_ylim(-130, 10)
                    axis.set_ylabel("Амплитуда, дБВ")
                    axis.grid(True, alpha=0.3)
                    axis.legend()
                figure.tight_layout()
                figure.savefig(EXPERIMENT / f"spectrum_{frequency_hz:g}hz.png", dpi=160)
                plt.close(figure)

    report = f"""# Проверка BDF2

Сравниваются однократный Эйлер, однократный и двукратный BDF2, а также
однократный метод трапеций. Контроль — полная
модель с Эйлером при шестнадцатикратной частоте. Первый шаг BDF2 выполняется
методом Эйлера, затем используются два прошлых напряжения каждого конденсатора.

| Частота, Гц | Вход, мВ (пик) | Способ | Выход, В (пик) | КНИ, % | Среднеквадратичная ошибка, % | Ошибка модуля спектра, дБ |
|---:|---:|:---|---:|---:|---:|---:|
{chr(10).join(rows)}

Чем отрицательнее последний столбец, тем ближе модуль спектра к контрольному
расчёту. Это сравнение оценивает численный способ, но контрольный расчёт Эйлером
16× сам по себе не является точным решением непрерывной схемы.

Чистые трапеции дают высокую точность на слабом сигнале, но теряют устойчивость
в части нелинейных режимов. Ранний повтор опасного шага методом Эйлера возвращает
устойчивость лишь на части точек и обычно ухудшает точность относительно BDF2 1×.
Поэтому смешанный вариант не является общим улучшением BDF2.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
