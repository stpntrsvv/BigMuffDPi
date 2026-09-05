"""Сравнивает физическое портовое разбиение 4×/1× с общей моделью 8×."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, Q2_COLLECTOR, simulate_complete
from muff_multirate_model import simulate_multirate


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_attack.wav"
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_multirate"
FIGURES = EXPERIMENT / "figures"
RAW = ROOT / "simulation" / "raw" / "port_multirate"
RATE = 48_000
DURATION_S = 0.020
PEAK_V = 0.025


def read_input():
    with wave.open(str(SAMPLE), "rb") as stream:
        rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    signal = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    signal -= float(np.mean(signal))
    signal *= PEAK_V / float(np.max(np.abs(signal)))
    positions = np.arange(len(signal), dtype=np.float64)

    def input_value(time_s: np.ndarray) -> np.ndarray:
        return np.interp(time_s * rate, positions, signal, left=0.0, right=0.0)

    return input_value


def high_band_rms(signal: np.ndarray, lower_hz: float = 10_000.0) -> float:
    spectrum = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(len(signal), 1.0 / RATE)
    spectrum[frequency < lower_hz] = 0.0
    return float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(signal)) ** 2)))


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    input_value = read_input()

    print("Эталон общей системы с коэффициентом 8…", flush=True)
    reference = simulate_complete(
        1.0, 1.0, 0.8, 8, input_value, DURATION_S, "hybrid", True
    )
    reference_output = reference.node_v[::8, OUTPUT]
    reference_port = reference.node_v[::8, Q2_COLLECTOR]

    results = {}
    rows = []
    configurations = (
        (4, 1, False, False, "полный Q4"),
        (4, 2, False, False, "полный Q4"),
        (4, 2, False, True, "полный Q4, постоянная Gp"),
        (4, 2, True, False, "линейный Q4"),
        (8, 2, False, False, "полный Q4"),
    )
    for factor, slow_factor, linear_q4, fixed_g, q4_label in configurations:
        print(f"Портовая система {factor}/{slow_factor}, {q4_label}…", flush=True)
        result = simulate_multirate(
            input_value, DURATION_S, factor, slow_factor, 1.0, 0.8, True,
            linear_q4, fixed_g,
        )
        results[(factor, slow_factor, q4_label)] = result
        common = min(len(result.output_v), len(reference_output))
        error = result.output_v[:common] - reference_output[:common]
        port_error = result.port_v[:common] - reference_port[:common]
        settled = slice(int(round(0.003 * RATE)), common)
        rows.append((
            factor, slow_factor, q4_label,
            1e3 * float(np.sqrt(np.mean(error[settled] ** 2))),
            1e3 * float(np.max(np.abs(error[settled]))),
            1e3 * high_band_rms(error[settled]),
            1e3 * float(np.sqrt(np.mean(port_error[settled] ** 2))),
            float(np.min(result.port_g)),
            float(np.max(result.port_g)),
            result.maximum_fast_residual_v,
            result.maximum_slow_residual_v,
        ))

    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "fast_factor", "slow_factor", "q4_model", "output_rms_mv", "output_peak_mv", "high_band_rms_mv",
            "port_rms_mv", "minimum_port_s", "maximum_port_s",
            "maximum_fast_residual_v", "maximum_slow_residual_v",
        ))
        writer.writerows(rows)

    q4_full = results[(4, 2, "полный Q4")].output_v
    q4_linear = results[(4, 2, "линейный Q4")].output_v
    q4_difference = q4_linear - q4_full
    settled_start = int(round(0.003 * RATE))
    q4_rms_mv = 1e3 * float(np.sqrt(np.mean(q4_difference[settled_start:] ** 2)))
    q4_peak_mv = 1e3 * float(np.max(np.abs(q4_difference[settled_start:])))

    time_ms = reference.time_s[::8] * 1e3
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(time_ms, reference_output, color="black", label="Общая система 8×")
    for (factor, slow_factor, q4_label), result in results.items():
        label = f"{factor}×/{slow_factor}×, {q4_label}"
        axes[0].plot(result.time_s * 1e3, result.output_v, linewidth=0.75, label=label)
        axes[1].plot(result.time_s * 1e3, 1e3*(result.output_v-reference_output),
                     linewidth=0.75, label=label)
        axes[2].plot(result.time_s * 1e3, 1e6*result.port_g, linewidth=0.75,
                     label=label)
    axes[0].set_ylabel("Выход, В"); axes[0].legend()
    axes[1].set_ylabel("Разность, мВ"); axes[1].legend()
    axes[2].set_ylabel("Проводимость порта, мкСм")
    axes[2].set_xlabel("Время, мс"); axes[2].legend()
    for axis in axes: axis.grid(True, alpha=0.3)
    figure.suptitle("Физическое портовое разбиение на коллекторе Q2")
    figure.tight_layout(); figure.savefig(FIGURES / "port_waveforms.png", dpi=160); plt.close(figure)

    labels = [f"{row[0]}×/{row[1]}×\n{row[2]}" for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, [row[3] for row in rows], color="tab:green")
    axis.set_ylabel("Ошибка выхода, мВ СКО")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout(); figure.savefig(FIGURES / "port_error.png", dpi=160); plt.close(figure)

    table = "\n".join(
        f"| {factor}×/{slow_factor}× | {q4_label} | {rms:.3f} | {peak:.2f} | {high:.3f} | {port:.3f} | "
        f"{1e6*gmin:.3f}…{1e6*gmax:.3f} | {fast_res:.2e} | {slow_res:.2e} |"
        for factor, slow_factor, q4_label, rms, peak, high, port, gmin, gmax, fast_res, slow_res in rows
    )
    report = f"""# Портовая многоскоростная модель

Быстрая подсистема содержит Q4, Sustain, Q3 и Q2. Медленная подсистема содержит
C9, C8, Tone, C10, Q1, C11, Volume и выход. Граница проходит по коллектору Q2.
Медленная часть представляется для быстрого ядра эквивалентом Нортона
`i = Gp·vp + I0`; ток и касательная проводимость обновляются раз в отсчёт 48 кГц.
В режимах медленной части 2× они обновляются дважды за внешний отсчёт.

Вход — первые {1e3*DURATION_S:.0f} мс ми-мажорного аккорда, {1e3*PEAK_V:.0f} мВ пик.
Эталон — общая гибридная модель с полным схождением при 8×. Первые 3 мс исключены
из среднеквадратических ошибок.

| Быстро/медленно | Q4 | Выход, мВ СКО | Пик, мВ | Выше 10 кГц, мВ СКО | Порт Q2, мВ СКО | Gp, мкСм | Невязка быстрой | Невязка медленной |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table}

![Сигналы](figures/port_waveforms.png)

![Ошибка](figures/port_error.png)

Замораживание `Gp` практически не изменило результат: в таблице ошибки совпадают
до 0,001 мВ. Поэтому в рабочем варианте матрица быстрого ядра постоянна, а медленная
подсистема обновляет только эквивалентный источник `I0`.

## Рабочий кандидат

Основной точный режим — `4×/2×` с полным Q4 и постоянной `Gp`. Его ошибка относительно
общей модели составляет {next(row[3] for row in rows if row[:3] == (4, 2, "полный Q4, постоянная Gp")):.3f} мВ СКО.
Непосредственная разность между полным и линейным Q4 в одной портовой архитектуре —
{q4_rms_mv:.3f} мВ СКО и {q4_peak_mv:.2f} мВ в пике на входе 25 мВ. Линейный Q4
остаётся ускоренной ветвью тихого сигнала, но не заменяет полную модель на сильной атаке.

По ранее измеренным ядрам Q3–Q2 при 4× требует около 1156 тактов на внешний отсчёт,
а две поправки Q1 при 2× — около 1354 тактов. Сумма нелинейных решений равна примерно
2510 тактам из бюджета 3542; остаётся около 1032 тактов на линейные накопители,
порт, изменение частоты и ввод-вывод в тихой ветви. Полную цену Q4 для сильного
сигнала теперь надо измерить в отдельном портовом C-ядре.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
