"""Измеряет выигрыш oversampling полной физической модели Big Muff Pi V3."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muff_complete_model import OUTPUT, simulate_complete


BASE_RATE = 48_000
FACTORS = (1, 2, 4, 8, 16)
EXPERIMENT = Path(__file__).parent / "experiments" / "oversampling_analysis"
FIGURES = EXPERIMENT / "figures"


@dataclass(frozen=True)
class Metrics:
    factor: int
    peak_v: float
    rms_v: float
    thd_percent: float
    relative_error_percent: float
    error_db: float
    high_band_error_db: float
    magnitude_error_db: float


def lowpass_downsample(signal: np.ndarray, factor: int) -> np.ndarray:
    """Windowed-sinc anti-alias filter followed by integer decimation."""
    if factor == 1:
        return signal.copy()
    half = 32 * factor
    index = np.arange(-half, half + 1, dtype=np.float64)
    cutoff = 0.45 / factor
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * index)
    kernel *= np.kaiser(len(kernel), 10.0)
    kernel /= np.sum(kernel)
    filtered = np.convolve(signal, kernel, mode="same")
    return filtered[::factor]


def align(reference: np.ndarray, candidate: np.ndarray, maximum_lag: int = 24) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(reference), len(candidate))
    reference = reference[:count]
    candidate = candidate[:count]
    reference_centered = reference - np.mean(reference)
    candidate_centered = candidate - np.mean(candidate)
    best_lag = max(
        range(-maximum_lag, maximum_lag + 1),
        key=lambda lag: float(np.dot(
            reference_centered[max(0, lag): min(count, count + lag)],
            candidate_centered[max(0, -lag): min(count, count - lag)],
        )),
    )
    if best_lag >= 0:
        return reference[best_lag:], candidate[: count - best_lag]
    return reference[: count + best_lag], candidate[-best_lag:]


def harmonic_amplitudes(signal: np.ndarray, frequency_hz: float) -> np.ndarray:
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    coherent_gain = np.sum(window)
    time_s = np.arange(len(signal), dtype=np.float64) / BASE_RATE
    harmonic_count = int((BASE_RATE / 2) // frequency_hz)
    amplitudes = np.empty(harmonic_count, dtype=np.float64)
    weighted = signal * window
    for harmonic in range(1, harmonic_count + 1):
        phase = np.exp(-2j * np.pi * harmonic * frequency_hz * time_s)
        amplitudes[harmonic - 1] = 2.0 * abs(np.sum(weighted * phase)) / coherent_gain
    return amplitudes


def band_rms(signal: np.ndarray, low_hz: float, high_hz: float) -> float:
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft((signal - np.mean(signal)) * window)
    frequency = np.fft.rfftfreq(len(signal), 1.0 / BASE_RATE)
    selected = (frequency >= low_hz) & (frequency < high_hz)
    return float(np.sqrt(np.sum(np.abs(spectrum[selected]) ** 2)))


def analyze(reference: np.ndarray, candidate: np.ndarray, factor: int, frequency_hz: float) -> Metrics:
    reference, candidate = align(reference, candidate)
    reference -= np.mean(reference)
    candidate -= np.mean(candidate)
    error = candidate - reference
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    error_rms = float(np.sqrt(np.mean(error * error)))
    amplitudes = harmonic_amplitudes(candidate, frequency_hz)
    thd = np.sqrt(np.sum(amplitudes[1:] ** 2)) / max(amplitudes[0], 1e-30)
    high_reference = band_rms(reference, 10_000.0, 24_001.0)
    high_error = band_rms(error, 10_000.0, 24_001.0)
    window = np.hanning(len(reference))
    reference_magnitude = np.abs(np.fft.rfft(reference * window))
    candidate_magnitude = np.abs(np.fft.rfft(candidate * window))
    magnitude_error = float(np.linalg.norm(candidate_magnitude - reference_magnitude))
    magnitude_reference = float(np.linalg.norm(reference_magnitude))
    return Metrics(
        factor=factor,
        peak_v=float(np.max(np.abs(candidate))),
        rms_v=float(np.sqrt(np.mean(candidate * candidate))),
        thd_percent=100.0 * float(thd),
        relative_error_percent=100.0 * error_rms / max(reference_rms, 1e-30),
        error_db=20.0 * np.log10(max(error_rms / max(reference_rms, 1e-30), 1e-15)),
        high_band_error_db=20.0 * np.log10(max(high_error / max(high_reference, 1e-30), 1e-15)),
        magnitude_error_db=20.0 * np.log10(max(magnitude_error / max(magnitude_reference, 1e-30), 1e-15)),
    )


def simulate_tone(frequency_hz: float, level_v: float, factor: int, duration_s: float) -> np.ndarray:
    cache = EXPERIMENT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"tone_{frequency_hz:g}hz_{1e3 * level_v:g}mv_{factor}x_{duration_s:g}s.npy"
    if path.exists():
        return np.load(path)
    result = simulate_complete(
        1.0, 1.0, 0.8, factor,
        lambda time_s: level_v * np.sin(2.0 * np.pi * frequency_hz * time_s),
        duration_s, "hybrid", True,
    )
    output = lowpass_downsample(result.node_v[:, OUTPUT], factor)
    np.save(path, output)
    return output


def plot_spectra(outputs: dict[int, np.ndarray], frequency_hz: float, settle_samples: int) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    for factor, signal in outputs.items():
        body = signal[settle_samples:] - np.mean(signal[settle_samples:])
        window = np.hanning(len(body))
        spectrum = np.fft.rfft(body * window)
        magnitude = 20.0 * np.log10(np.maximum(np.abs(spectrum) / np.sum(window), 1e-12))
        frequencies = np.fft.rfftfreq(len(body), 1.0 / BASE_RATE)
        axes[0].plot(frequencies, magnitude, label=f"{factor}x", alpha=0.85)
    reference = outputs[16][settle_samples:]
    for factor in FACTORS[:-1]:
        ref, candidate = align(reference, outputs[factor][settle_samples:])
        error = candidate - ref
        window = np.hanning(len(error))
        spectrum = np.fft.rfft((error - np.mean(error)) * window)
        magnitude = 20.0 * np.log10(np.maximum(np.abs(spectrum) / np.sum(window), 1e-12))
        frequencies = np.fft.rfftfreq(len(error), 1.0 / BASE_RATE)
        axes[1].plot(frequencies, magnitude, label=f"{factor}x - 16x")
    axes[0].set_title(f"Выходные спектры, синус {frequency_hz:g} Гц")
    axes[0].set_ylabel("Амплитуда, dBV")
    axes[1].set_title("Спектр ошибки относительно 16x")
    axes[1].set_ylabel("Амплитуда, dBV")
    axes[1].set_xlabel("Частота, Гц")
    for axis in axes:
        axis.set_xlim(0, 24_000)
        axis.set_ylim(-130, 10)
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(FIGURES / f"spectrum_{frequency_hz:g}hz.png", dpi=160)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="только 440 Гц, 25 мВ, 40 мс")
    parser.add_argument("--duration", type=float, default=0.08, help="длительность каждого тона, с")
    parser.add_argument("--frequencies", default="", help="частоты через запятую")
    parser.add_argument("--levels-mv", default="", help="амплитуды peak в мВ через запятую")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    FIGURES.mkdir(parents=True, exist_ok=True)
    frequencies = (440.0,) if args.quick else (55.0, 110.0, 440.0, 1000.0, 3000.0, 6000.0)
    levels = (0.025,) if args.quick else (0.005, 0.025, 0.100, 0.200)
    if args.frequencies:
        frequencies = tuple(float(value) for value in args.frequencies.split(","))
    if args.levels_mv:
        levels = tuple(1e-3 * float(value) for value in args.levels_mv.split(","))
    duration_s = 0.04 if args.quick else args.duration
    settle_s = min(0.02, duration_s * 0.4)
    rows: list[str] = []
    for frequency_hz in frequencies:
        for level_v in levels:
            outputs: dict[int, np.ndarray] = {}
            for factor in FACTORS:
                print(f"{frequency_hz:g} Hz, {1e3 * level_v:g} mV, {factor}x", flush=True)
                outputs[factor] = simulate_tone(frequency_hz, level_v, factor, duration_s)
            settle_samples = int(round(settle_s * BASE_RATE))
            reference = outputs[16][settle_samples:]
            for factor in FACTORS:
                candidate = outputs[factor][settle_samples:]
                if not np.all(np.isfinite(candidate)) or float(np.max(np.abs(candidate))) > 20.0:
                    rows.append(
                        f"| {frequency_hz:g} | {1e3 * level_v:g} | {factor}x | "
                        "INVALID | INVALID | INVALID | INVALID | INVALID | INVALID |"
                    )
                    continue
                metrics = analyze(reference, outputs[factor][settle_samples:], factor, frequency_hz)
                rows.append(
                    f"| {frequency_hz:g} | {1e3 * level_v:g} | {factor}x | "
                    f"{metrics.peak_v:.6f} | {metrics.thd_percent:.3f} | "
                    f"{metrics.relative_error_percent:.4f} | {metrics.error_db:.2f} | "
                    f"{metrics.high_band_error_db:.2f} | {metrics.magnitude_error_db:.2f} |"
                )
            if level_v == min(levels, key=lambda value: abs(value - 0.025)):
                plot_spectra(outputs, frequency_hz, settle_samples)
    report = f"""# Влияние oversampling на полную физическую модель

Полная 12-нелинейная Python-модель Big Muff Pi V3 рассчитана с корректным
шагом состояния для 1x/2x/4x/8x/16x. Каждый результат пропущен через общий
оконный sinc low-pass и приведён к 48 кГц; 16x принят за цифровой reference.

THD здесь характеризует создаваемые педалью гармоники, а ошибка относительно
16x показывает одновременно discretization и aliasing. Столбец 10–24 кГц
особенно чувствителен к завёрнутым высокочастотным продуктам.

| Частота, Гц | Вход, мВ peak | Режим | Выход, В peak | THD, % | RMS error, % | Error, dB | Error 10–24 кГц, dB | Magnitude error, dB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Примечание: это сравнение частот дискретизации Python-модели. Отдельные
сравнения с ngspice и финальным host C-ядром будут добавлены следующим этапом,
чтобы не смешивать ошибку редукции с ошибкой oversampling.

## Как читать результат

- На малом уровне высокая ошибка ещё не является aliasing: при почти нулевом
  THD она показывает прежде всего ошибку крупного backward-Euler шага в
  линейных RC-цепях.
- Рост и перераспределение THD с уровнем показывают уже зависимость
  нелинейности от частоты шага.
- `Magnitude error` сравнивает модули FFT и не штрафует модель за небольшую
  разницу фазы; это основная сводная метрика данного опыта.
- `INVALID` означает переход полной узловой модели на нефизическую ветвь
  решения. Такой режим нельзя включать в THD или усреднять с устойчивыми
  результатами.

Пилот показывает, что 1x нельзя заранее объявить эквивалентом 16x. Однако он
ещё не доказывает, что слышимая разница вызвана именно aliasing: сначала нужно
вычесть малосигнальную ошибку дискретизации и отдельно сравнить составное
STM32-ядро с ngspice.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
