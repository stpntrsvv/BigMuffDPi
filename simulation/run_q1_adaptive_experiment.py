"""Настраивает плавный защитный переход Q1 между полной и линейной моделями."""

from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete, simulate_complete_adaptive_q1


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "q1_adaptive"
EXPERIMENT = ROOT / "simulation" / "experiments" / "q1_adaptive"
FIGURES = EXPERIMENT / "figures"
RATE = 384_000
CONFIGS = tuple((threshold, fade) for threshold in (.10, .15, .25) for fade in (64, 256))


@dataclass(frozen=True)
class Row:
    warning_residual_v: float
    fade_samples: int
    rms_error_mv: float
    peak_error_mv: float
    high_band_error_mv: float
    maximum_error_slew_v_per_sample: float
    linear_fraction: float
    emergency_entries: int


def smooth_level(time_s: np.ndarray) -> np.ndarray:
    level = np.full_like(time_s, .05)
    rising = (time_s >= .008) & (time_s < .012)
    hold = (time_s >= .012) & (time_s < .028)
    falling = (time_s >= .028) & (time_s < .032)
    phase_up = (time_s[rising] - .008) / .004
    phase_down = (time_s[falling] - .028) / .004
    level[rising] = .05 + .15 * (.5 - .5*np.cos(np.pi*phase_up))
    level[hold] = .20
    level[falling] = .20 - .15 * (.5 - .5*np.cos(np.pi*phase_down))
    return level


def signal(time_s: np.ndarray) -> np.ndarray:
    return smooth_level(time_s) * np.sin(2*np.pi*1_000*time_s)


def write_wav(path: Path, signal_v: np.ndarray, scale_v: float) -> None:
    count = len(signal_v) // 8
    downsampled = signal_v[:count*8].reshape(count, 8).mean(axis=1)
    repeated = np.tile(downsampled, 25)
    pcm = np.int16(np.clip(repeated / scale_v, -1, 1) * 32767)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(48_000)
        stream.writeframes(pcm.tobytes())


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    reference = simulate_complete(1, 1, 1, 8, signal, .04, "hybrid", True)
    rows = []; results = {}
    for threshold, fade in CONFIGS:
        print(f"Порог {threshold:.2f} В, переход {fade} шагов")
        result = simulate_complete_adaptive_q1(
            1, 1, 1, 8, signal, .04, threshold, .50, .06, 384, fade
        )
        results[(threshold, fade)] = result
        error = result.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]
        spectrum = np.fft.rfft(error)
        frequency = np.fft.rfftfreq(len(error), 1/RATE)
        spectrum[frequency < 10_000] = 0
        high_band = np.fft.irfft(spectrum, n=len(error))
        mix = result.q1_nonlinear_mix
        entries = int(np.count_nonzero((mix[1:] == 0) & (mix[:-1] > 0)))
        rows.append(Row(
            threshold, fade, float(1e3*np.sqrt(np.mean(error*error))),
            float(1e3*np.max(np.abs(error))),
            float(1e3*np.sqrt(np.mean(high_band*high_band))),
            float(np.max(np.abs(np.diff(error)))), float(np.mean(mix < .01)), entries
        ))

    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(Row.__dataclass_fields__)
        writer.writerows(tuple(vars(row).values()) for row in rows)
    best = min(rows, key=lambda row: (row.high_band_error_mv, row.rms_error_mv))
    result = results[(best.warning_residual_v, best.fade_samples)]
    reference_output = reference.node_v[:, OUTPUT]
    adaptive_output = result.node_v[:, OUTPUT]
    common_scale = 1.05 * float(max(np.max(np.abs(reference_output)), np.max(np.abs(adaptive_output))))
    write_wav(EXPERIMENT / "reference_repeated.wav", reference_output, common_scale)
    write_wav(EXPERIMENT / "adaptive_repeated.wav", adaptive_output, common_scale)
    write_wav(
        EXPERIMENT / "difference_x10_repeated.wav",
        10.0 * (adaptive_output - reference_output), common_scale
    )
    time_ms = result.time_s * 1e3
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(time_ms, 1e3*smooth_level(result.time_s)); axes[0].set_ylabel("Вход, мВ пик")
    axes[1].plot(time_ms, reference.node_v[:,OUTPUT], "--", label="Полное схождение")
    axes[1].plot(time_ms, result.node_v[:,OUTPUT], linewidth=.8, label="Адаптивный Q1")
    axes[1].set_ylabel("Выход, В"); axes[1].legend()
    axes[2].plot(time_ms, result.q1_nonlinear_mix); axes[2].set_ylabel("Доля полной Q1")
    axes[2].set_xlabel("Время, мс")
    for axis in axes: axis.grid(True, alpha=.3)
    figure.suptitle(f"Лучший режим: порог {best.warning_residual_v:.2f} В, переход {best.fade_samples} шагов")
    figure.tight_layout(); figure.savefig(FIGURES / "adaptive_transition.png", dpi=160); plt.close(figure)

    table = "\n".join(f"| {r.warning_residual_v:.2f} | {r.fade_samples} | {r.rms_error_mv:.3f} | {r.peak_error_mv:.2f} | {r.high_band_error_mv:.3f} | {r.maximum_error_slew_v_per_sample:.4f} | {100*r.linear_fraction:.1f} | {r.emergency_entries} |" for r in rows)
    report = f"""# Адаптивный выходной Q1

Уровень синуса 1 кГц плавно меняется `50 → 200 → 50` мВ пик при Tone, Sustain и Volume = 1.
Предварительный уход начинается при огибающей `V_BC = 0,48` В, возврат разрешён ниже 0,30 В.
Невязка служит второй защитой: предупредительные пороги указаны в таблице, а при 0,5 В
выполняется аварийный повтор текущего шага с линейным Q1. Возврат по невязке разрешён ниже 0,06 В.

| Порог, В | Переход, шагов | Ошибка, мВ СКО | Пик, мВ | Ошибка выше 10 кГц, мВ СКО | Макс. изменение ошибки за шаг, В | Линейная ветвь, % | Входы в линейную ветвь |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

![Переход](figures/adaptive_transition.png)

Звуковые файлы повторяют опыт 25 раз для удобства прослушивания; частота 48 кГц,
понижение частоты выполнено усреднением восьми внутренних отсчётов:

- [полное схождение](reference_repeated.wav);
- [адаптивный Q1](adaptive_repeated.wav);
- [разность, усиленная в 10 раз](difference_x10_repeated.wav).

Лучший по добавочной высокочастотной энергии режим: порог `{best.warning_residual_v:.2f}` В и
переход `{best.fade_samples}` шагов. Его ошибка `{best.rms_error_mv:.2f}` мВ СКО, составляющая
выше 10 кГц — `{best.high_band_error_mv:.3f}` мВ СКО. Это численная проверка щелчка; далее
нужны звуковые файлы слепого сравнения и замер тактов на STM32.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
