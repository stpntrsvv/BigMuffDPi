"""Прогоняет атаку сухого гитарного аккорда через полную и адаптивную модели."""

from __future__ import annotations

import time
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete, simulate_complete_adaptive_q1


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_attack.wav"
EXPERIMENT = ROOT / "simulation" / "experiments" / "guitar_chord"
FIGURES = EXPERIMENT / "figures"
OUTPUT_RATE = 48_000
FACTOR = 8
INTERNAL_RATE = OUTPUT_RATE * FACTOR
DURATION_S = 0.120
INPUT_PEAK_V = 0.100


def read_pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"Ожидается моно PCM16: {path}")
        rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    return rate, np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0


def make_input(source: np.ndarray, rate: int):
    source = source - float(np.mean(source))
    source *= INPUT_PEAK_V / float(np.max(np.abs(source)))

    def input_function(time_s: np.ndarray) -> np.ndarray:
        position = time_s * rate
        return np.interp(position, np.arange(len(source)), source, left=0.0, right=0.0)

    return input_function


def downsample(signal: np.ndarray) -> np.ndarray:
    count = len(signal) // FACTOR
    return signal[:count * FACTOR].reshape(count, FACTOR).mean(axis=1)


def write_pcm16(path: Path, signal: np.ndarray, scale_v: float) -> None:
    pcm = np.int16(np.clip(signal / scale_v, -1.0, 1.0) * 32767.0)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(OUTPUT_RATE)
        stream.writeframes(pcm.astype("<i2", copy=False).tobytes())


def repeated_for_listening(signal: np.ndarray) -> np.ndarray:
    fade_count = min(len(signal) // 4, int(round(0.010 * OUTPUT_RATE)))
    item = signal.copy()
    item[-fade_count:] *= np.linspace(1.0, 0.0, fade_count)
    silence = np.zeros(int(round(0.180 * OUTPUT_RATE)))
    return np.tile(np.concatenate((item, silence)), 8)


def high_band_rms(signal: np.ndarray, lower_hz: float = 10_000.0) -> float:
    spectrum = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(len(signal), 1.0 / INTERNAL_RATE)
    spectrum[frequency < lower_hz] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(signal))
    return float(np.sqrt(np.mean(filtered * filtered)))


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rate, source = read_pcm16_mono(SAMPLE)
    input_function = make_input(source, rate)

    started = time.perf_counter()
    print("Расчёт эталона с полным схождением…", flush=True)
    reference = simulate_complete(
        1.0, 1.0, 1.0, FACTOR, input_function, DURATION_S, "hybrid", True
    )
    reference_seconds = time.perf_counter() - started
    print(f"Эталон готов за {reference_seconds:.1f} с", flush=True)

    started = time.perf_counter()
    adaptive = simulate_complete_adaptive_q1(
        1.0, 1.0, 1.0, FACTOR, input_function, DURATION_S,
        0.10, 0.50, 0.06, 384, 64
    )
    adaptive_seconds = time.perf_counter() - started
    print(f"Адаптивная модель готова за {adaptive_seconds:.1f} с", flush=True)

    reference_output = reference.node_v[:, OUTPUT]
    adaptive_output = adaptive.node_v[:, OUTPUT]
    error = adaptive_output - reference_output
    mix = adaptive.q1_nonlinear_mix
    assert mix is not None

    reference_48 = downsample(reference_output)
    adaptive_48 = downsample(adaptive_output)
    error_48 = downsample(error)
    input_48 = downsample(reference.input_v)
    common_scale = 1.05 * float(max(np.max(np.abs(reference_48)), np.max(np.abs(adaptive_48))))
    input_scale = 1.05 * float(np.max(np.abs(input_48)))
    write_pcm16(EXPERIMENT / "input_100mv_repeated.wav", repeated_for_listening(input_48), input_scale)
    write_pcm16(EXPERIMENT / "reference_repeated.wav", repeated_for_listening(reference_48), common_scale)
    write_pcm16(EXPERIMENT / "adaptive_repeated.wav", repeated_for_listening(adaptive_48), common_scale)
    write_pcm16(EXPERIMENT / "difference_x10_repeated.wav", repeated_for_listening(10.0 * error_48), common_scale)

    time_ms = reference.time_s * 1e3
    figure, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(time_ms, 1e3 * reference.input_v)
    axes[0].set_ylabel("Вход, мВ")
    axes[1].plot(time_ms, reference_output, label="Полное схождение")
    axes[1].plot(time_ms, adaptive_output, linewidth=0.7, alpha=0.85, label="Адаптивный Q1")
    axes[1].set_ylabel("Выход, В")
    axes[1].legend()
    axes[2].plot(time_ms, 1e3 * error)
    axes[2].set_ylabel("Разность, мВ")
    axes[3].plot(time_ms, mix)
    axes[3].set_ylabel("Доля полного Q1")
    axes[3].set_xlabel("Время, мс")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Атака ми-мажорного аккорда, 100 мВ пик, Tone/Sustain/Volume = 1")
    figure.tight_layout()
    figure.savefig(FIGURES / "waveforms_and_q1_mix.png", dpi=160)
    plt.close(figure)

    rms_mv = 1e3 * float(np.sqrt(np.mean(error * error)))
    peak_mv = 1e3 * float(np.max(np.abs(error)))
    high_mv = 1e3 * high_band_rms(error)
    emergency_entries = int(np.count_nonzero((mix[1:] == 0.0) & (mix[:-1] > 0.0)))
    linear_fraction = float(np.mean(mix < 0.01))
    report = f"""# Атака гитарного аккорда

Вход — первые {1e3 * DURATION_S:.0f} мс сухого ми-мажорного аккорда `E2–B2–E3–G♯3–B3–E4`.
Амплитуда приведена к {1e3 * INPUT_PEAK_V:.0f} мВ пик. Частота расчёта 384 кГц, Tone, Sustain и Volume установлены в 1.

| Величина | Значение |
|---|---:|
| Ошибка адаптивной модели, СКО | {rms_mv:.3f} мВ |
| Пиковая ошибка | {peak_mv:.2f} мВ |
| Ошибка выше 10 кГц, СКО | {high_mv:.3f} мВ |
| Время в линейной ветви Q1 | {100 * linear_fraction:.1f} % |
| Аварийные входы в линейную ветвь | {emergency_entries} |
| Время расчёта эталона | {reference_seconds:.1f} с |
| Время расчёта адаптивной модели | {adaptive_seconds:.1f} с |

![Сигналы и смешение Q1](figures/waveforms_and_q1_mix.png)

Для прослушивания отрезок повторён восемь раз с паузой 180 мс. Эталон и адаптивный
вариант записаны с общим масштабом, поэтому сравнивать их можно напрямую:

- [сухой вход 100 мВ](input_100mv_repeated.wav);
- [полное схождение](reference_repeated.wav);
- [адаптивный Q1](adaptive_repeated.wav);
- [разность, усиленная в 10 раз](difference_x10_repeated.wav).

Полный четырёхсекундный сухой аккорд лежит в `simulation/samples/e_major_chord/e_major_dry.wav`.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
