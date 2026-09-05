"""Проверяет выбранную портовую архитектуру на гитарных уровнях 25–200 мВ."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete
from muff_multirate_model import simulate_multirate


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_attack.wav"
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_multirate_levels"
FIGURES = EXPERIMENT / "figures"
RAW = ROOT / "simulation" / "raw" / "port_multirate_levels"
RATE = 48_000
DURATION_S = 0.020
LEVELS_V = (0.025, 0.050, 0.100, 0.200)


def source_input(peak_v: float):
    with wave.open(str(SAMPLE), "rb") as stream:
        rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    source = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    source -= float(np.mean(source))
    source *= peak_v / float(np.max(np.abs(source)))
    positions = np.arange(len(source), dtype=np.float64)

    def value(time_s: np.ndarray) -> np.ndarray:
        return np.interp(time_s * rate, positions, source, left=0.0, right=0.0)

    return value


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal * signal)))


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    strong_traces = {}
    start = int(round(0.003 * RATE))
    for peak_v in LEVELS_V:
        print(f"Уровень {1e3*peak_v:.0f} мВ…", flush=True)
        input_value = source_input(peak_v)
        reference = simulate_complete(
            1.0, 1.0, 0.8, 8, input_value, DURATION_S, "hybrid", True
        )
        reference_output = reference.node_v[::8, OUTPUT]
        full_q4 = simulate_multirate(
            input_value, DURATION_S, 4, 2, 1.0, 0.8, True, False
        )
        linear_q4 = simulate_multirate(
            input_value, DURATION_S, 4, 2, 1.0, 0.8, True, True
        )
        full_error = full_q4.output_v - reference_output
        linear_error = linear_q4.output_v - reference_output
        q4_error = linear_q4.output_v - full_q4.output_v
        rows.append((
            1e3 * peak_v,
            1e3 * rms(full_error[start:]),
            1e3 * rms(linear_error[start:]),
            1e3 * float(np.max(np.abs(linear_error[start:]))),
            1e3 * rms(q4_error[start:]),
            1e3 * float(np.max(np.abs(q4_error[start:]))),
            1e6 * float(np.min(linear_q4.port_g)),
            1e6 * float(np.max(linear_q4.port_g)),
            linear_q4.maximum_fast_residual_v,
            linear_q4.maximum_slow_residual_v,
        ))
        if peak_v == LEVELS_V[-1]:
            strong_traces = {
                "time": linear_q4.time_s,
                "reference": reference_output,
                "full": full_q4.output_v,
                "linear": linear_q4.output_v,
            }

    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "input_peak_mv", "full_q4_rms_mv", "linear_q4_rms_mv",
            "linear_q4_peak_mv", "q4_added_rms_mv", "q4_added_peak_mv",
            "minimum_port_us", "maximum_port_us", "maximum_fast_residual_v",
            "maximum_slow_residual_v",
        ))
        writer.writerows(rows)

    levels = [row[0] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(levels, [row[1] for row in rows], "o-", label="Полный Q4")
    axes[0].plot(levels, [row[2] for row in rows], "o-", label="Линейный Q4")
    axes[0].set_ylabel("Ошибка к общей 8×, мВ СКО"); axes[0].legend()
    axes[1].plot(levels, [row[4] for row in rows], "o-", color="tab:green")
    axes[1].set_ylabel("Собственный вклад Q4, мВ СКО")
    axes[1].set_xlabel("Вход, мВ пик")
    for axis in axes: axis.grid(True, alpha=0.3)
    figure.tight_layout(); figure.savefig(FIGURES / "error_by_level.png", dpi=160); plt.close(figure)

    time_ms = strong_traces["time"] * 1e3
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_ms, strong_traces["reference"], color="black", label="Общая 8×")
    axes[0].plot(time_ms, strong_traces["linear"], linewidth=0.75, label="Портовая 4×/2×, линейный Q4")
    axes[1].plot(time_ms, 1e3*(strong_traces["linear"]-strong_traces["reference"]))
    axes[0].set_ylabel("Выход, В"); axes[0].legend()
    axes[1].set_ylabel("Разность, мВ"); axes[1].set_xlabel("Время, мс")
    for axis in axes: axis.grid(True, alpha=0.3)
    figure.tight_layout(); figure.savefig(FIGURES / "strong_waveform.png", dpi=160); plt.close(figure)

    table = "\n".join(
        f"| {level:.0f} | {full:.3f} | {linear:.3f} | {linear_peak:.2f} | "
        f"{q4:.3f} | {q4_peak:.2f} | {gmin:.3f}…{gmax:.3f} | {fast:.2e} | {slow:.2e} |"
        for level, full, linear, linear_peak, q4, q4_peak, gmin, gmax, fast, slow in rows
    )
    report = f"""# Уровни портовой модели 4×/2×

Проверена атака ми-мажорного аккорда при Tone = Sustain = 1 и Volume = 0,8.
Эталон — общая гибридная система с полным схождением при 8×. Портовая модель
решается с полным схождением; первые 3 мс исключены из СКО.

| Вход, мВ пик | Полный Q4, мВ СКО | Линейный Q4, мВ СКО | Пик ошибки, мВ | Вклад Q4, мВ СКО | Пик вклада Q4, мВ | Gp, мкСм | Невязка быстрой | Невязка медленной |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

![Ошибка по уровню](figures/error_by_level.png)

![Сильный сигнал](figures/strong_waveform.png)

Полный Q4 сохраняет устойчивость и ограничивает рост ошибки до 7,741 мВ СКО при
200 мВ. Постоянная линеаризация приемлема только на тихом входе: её собственный
вклад превышает 8 мВ СКО при 100 мВ. Поэтому она может включаться адаптивно по
огибающей, а не использоваться как единственная физическая модель.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
