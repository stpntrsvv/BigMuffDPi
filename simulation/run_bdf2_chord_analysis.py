"""Прогоняет атаку гитарного аккорда через Эйлер и BDF2."""

from __future__ import annotations

import time
import wave
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muff_complete_model import OUTPUT, simulate_complete
from run_oversampling_analysis import BASE_RATE, align, lowpass_downsample


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_attack.wav"
EXPERIMENT = ROOT / "simulation" / "experiments" / "bdf2_chord"
DURATION_S = 0.120
INPUT_PEAK_V = 0.025


def read_input() -> tuple[np.ndarray, callable]:
    with wave.open(str(SAMPLE), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Ожидается одноканальный PCM16")
        rate = stream.getframerate()
        source = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64)
    source -= np.mean(source)
    source *= INPUT_PEAK_V / np.max(np.abs(source))

    def input_function(time_s: np.ndarray) -> np.ndarray:
        return np.interp(time_s * rate, np.arange(len(source)), source, left=0.0, right=0.0)

    return source, input_function


def write_wav(path: Path, signal: np.ndarray, scale_v: float) -> None:
    pcm = np.int16(np.clip(signal / scale_v, -1.0, 1.0) * 32767.0)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(BASE_RATE)
        stream.writeframes(pcm.astype("<i2", copy=False).tobytes())


def listening_copy(signal: np.ndarray) -> np.ndarray:
    item = signal.copy()
    fade = min(len(item) // 4, int(0.010 * BASE_RATE))
    item[-fade:] *= np.linspace(1.0, 0.0, fade)
    silence = np.zeros(int(0.180 * BASE_RATE))
    return np.tile(np.concatenate((item, silence)), 8)


def magnitude_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    reference, candidate = align(reference, candidate)
    reference = reference - np.mean(reference)
    candidate = candidate - np.mean(candidate)
    error = candidate - reference
    relative = np.sqrt(np.mean(error * error)) / max(np.sqrt(np.mean(reference * reference)), 1e-30)
    window = np.hanning(len(reference))
    reference_magnitude = np.abs(np.fft.rfft(reference * window))
    candidate_magnitude = np.abs(np.fft.rfft(candidate * window))
    spectral = np.linalg.norm(candidate_magnitude - reference_magnitude) / max(
        np.linalg.norm(reference_magnitude), 1e-30
    )
    return 100.0 * float(relative), 20.0 * np.log10(max(float(spectral), 1e-15))


def spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    magnitude = 20.0 * np.log10(np.maximum(
        np.abs(np.fft.rfft(signal * window)) / np.sum(window), 1e-12
    ))
    return np.fft.rfftfreq(len(signal), 1.0 / BASE_RATE), magnitude


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    _, input_function = read_input()
    modes = (
        ("Эйлер 1×", "euler", 1, "euler_1x"),
        ("BDF2 1×", "bdf2", 1, "bdf2_1x"),
        ("BDF2 2×", "bdf2", 2, "bdf2_2x"),
        ("Трапеции 1×", "trapezoid", 1, "trapezoid_1x"),
        ("Эйлер 16×", "euler", 16, "euler_16x"),
    )
    outputs: dict[str, np.ndarray] = {}
    times: dict[str, float] = {}
    input_48: np.ndarray | None = None
    for label, method, factor, stem in modes:
        print(f"Расчёт: {label}", flush=True)
        started = time.perf_counter()
        result = simulate_complete(
            1.0, 1.0, 0.8, factor, input_function, DURATION_S,
            "hybrid", True, integration_method=method,
        )
        times[label] = time.perf_counter() - started
        output = lowpass_downsample(result.node_v[:, OUTPUT], factor)
        if not np.all(np.isfinite(output)) or np.max(np.abs(output)) > 20.0:
            raise RuntimeError(f"{label}: нефизическое решение")
        outputs[label] = output
        if input_48 is None:
            input_48 = result.input_v

    assert input_48 is not None
    count = min(map(len, outputs.values()))
    outputs = {label: signal[:count] for label, signal in outputs.items()}
    input_48 = input_48[:count]
    reference = outputs["Эйлер 16×"]
    scale = 1.05 * max(float(np.max(np.abs(signal))) for signal in outputs.values())
    input_scale = 1.05 * float(np.max(np.abs(input_48)))
    write_wav(EXPERIMENT / "dry_chord_25mv.wav", listening_copy(input_48), input_scale)
    for _, _, _, stem in modes:
        label = next(item[0] for item in modes if item[3] == stem)
        write_wav(EXPERIMENT / f"{stem}.wav", listening_copy(outputs[label]), scale)

    rows: list[str] = []
    for label, _, _, stem in modes:
        if label == "Эйлер 16×":
            relative, spectral = 0.0, -300.0
        else:
            relative, spectral = magnitude_error(reference, outputs[label])
        rows.append(
            f"| {label} | {np.max(np.abs(outputs[label])):.6f} | {relative:.4f} | "
            f"{spectral:.2f} | {times[label]:.1f} | [{stem}.wav]({stem}.wav) |"
        )

    figure, axes = plt.subplots(3, 1, figsize=(13, 11))
    time_ms = np.arange(count) / BASE_RATE * 1e3
    axes[0].plot(time_ms, 1e3 * input_48)
    axes[0].set_title("Сухая атака аккорда")
    axes[0].set_ylabel("Вход, мВ")
    for label, signal in outputs.items():
        axes[1].plot(time_ms, signal, label=label, alpha=0.82)
        frequency, magnitude = spectrum(signal)
        axes[2].plot(frequency, magnitude, label=label, alpha=0.82)
    axes[1].set_title("Выходные сигналы")
    axes[1].set_ylabel("Выход, В")
    axes[2].set_title("Выходные спектры")
    axes[2].set_xlabel("Частота, Гц")
    axes[2].set_ylabel("Амплитуда, дБВ")
    axes[2].set_xlim(0, 24_000)
    axes[2].set_ylim(-120, 5)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[1].legend(ncol=2)
    axes[2].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(EXPERIMENT / "chord_comparison.png", dpi=160)
    plt.close(figure)

    report = f"""# BDF2 на гитарном аккорде

Первые {DURATION_S * 1e3:.0f} мс сухого ми-мажорного аккорда
`E2–B2–E3–G♯3–B3–E4` приведены к {INPUT_PEAK_V * 1e3:.0f} мВ (пик).
Ручки длительности и тембра установлены в 1, громкость — в 0,8. Контроль — полная модель с
Эйлером при шестнадцатикратной частоте.

| Способ | Выход, В (пик) | Ошибка формы, % | Ошибка модуля спектра, дБ | Время расчёта, с | Прослушивание |
|:---|---:|---:|---:|---:|:---|
{chr(10).join(rows)}

Все обработанные варианты записаны с одним масштабом громкости. Отрезок повторён
восемь раз с паузой 180 мс. Сухой вход записан отдельно:
[сухой аккорд](dry_chord_25mv.wav).

![Сравнение аккорда](chord_comparison.png)
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
