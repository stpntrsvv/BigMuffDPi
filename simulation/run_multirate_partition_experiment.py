"""Проверяет частоту общей модели и разнос нелинейных блоков по частотам."""

from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, operating_point, prepare_complete
from muff_hybrid_active import (
    ACTIVE,
    COUPLED_INPUT_BLOCKS,
    FRONTEND_BLOCKS,
    FRONTEND_Q1_TWICE,
    prepare_hybrid_active,
    simulate_hybrid_cascade,
    step_hybrid_scheduled,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_attack.wav"
EXPERIMENT = ROOT / "simulation" / "experiments" / "multirate_partition"
FIGURES = EXPERIMENT / "figures"
RAW = ROOT / "simulation" / "raw" / "multirate_partition"
BASE_RATE = 48_000
REFERENCE_FACTOR = 8
DURATION_S = 0.040
INPUT_PEAK_V = 0.025

INPUT_BLOCK = COUPLED_INPUT_BLOCKS[0]
Q1_BLOCK = COUPLED_INPUT_BLOCKS[2]
FRONTEND_BLOCK = FRONTEND_BLOCKS[0]


@dataclass(frozen=True)
class Row:
    name: str
    label: str
    internal_factor: int
    q2_period: int
    q1_period: int
    rms_mv: float
    peak_mv: float
    high_band_mv: float
    maximum_residual_v: float
    corrections_per_base_sample: float
    stable: bool


def read_input() -> tuple[int, np.ndarray]:
    with wave.open(str(SAMPLE), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"Ожидается моно PCM16: {SAMPLE}")
        rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    signal = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    signal -= float(np.mean(signal))
    signal *= INPUT_PEAK_V / float(np.max(np.abs(signal)))
    return rate, signal


def input_function(rate: int, source: np.ndarray):
    positions = np.arange(len(source), dtype=np.float64)

    def value(time_s: np.ndarray) -> np.ndarray:
        return np.interp(time_s * rate, positions, source, left=0.0, right=0.0)

    return value


def base_samples(signal: np.ndarray, factor: int) -> np.ndarray:
    """Берёт значения в общих физических моментах времени 48 кГц."""
    return signal[::factor]


def high_band_rms(signal: np.ndarray, lower_hz: float = 10_000.0) -> float:
    spectrum = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(len(signal), 1.0 / BASE_RATE)
    spectrum[frequency < lower_hz] = 0.0
    return float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(signal)) ** 2)))


def scheduled_result(input_value, q2_period: int, q1_period: int):
    factor = REFERENCE_FACTOR
    reduction = prepare_hybrid_active(1.0, 1.0, 0.8, factor)
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, reduction.parameters)
    dynamic = prepare_complete(1.0, 1.0, 0.8, 1.0 / (BASE_RATE * factor), reduction.parameters)
    state = dynamic.capacitor_incidence.T @ dc_nodes
    active_q = dc_q[ACTIVE].copy()
    count = int(round(DURATION_S * BASE_RATE * factor))
    time_s = np.arange(count + 1, dtype=np.float64) / (BASE_RATE * factor)
    input_v = np.asarray(input_value(time_s), dtype=np.float64)
    output = np.empty(count + 1, dtype=np.float64)
    residual = np.zeros(count + 1, dtype=np.float64)
    output[0] = dc_nodes[OUTPUT]
    for index in range(1, count + 1):
        blocks = [FRONTEND_BLOCK if index % q2_period == 0 else INPUT_BLOCK]
        if index % q1_period == 0:
            blocks.extend((Q1_BLOCK, Q1_BLOCK))
        node, state, full_q, residual[index], _ = step_hybrid_scheduled(
            reduction, state, active_q, float(input_v[index]), tuple(blocks)
        )
        active_q = full_q[ACTIVE]
        output[index] = node[OUTPUT]
        if not np.isfinite(output[index]):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return time_s, output, residual


def metrics(name, label, factor, q2_period, q1_period, output, residual, reference):
    common = min(len(output), len(reference))
    error = output[:common] - reference[:common]
    # Первые 5 мс не включаем: там преобладает разница установления рабочих точек.
    settled = error[int(round(0.005 * BASE_RATE)):]
    corrections = factor * (1.0 + 2.0 / q1_period)
    stable = bool(
        np.all(np.isfinite(output))
        and float(np.max(np.abs(output))) < 10.0
        and float(np.max(residual)) < 10.0
    )
    return Row(
        name, label, factor, q2_period, q1_period,
        1e3 * float(np.sqrt(np.mean(settled ** 2))) if stable else float("nan"),
        1e3 * float(np.max(np.abs(settled))) if stable else float("nan"),
        1e3 * high_band_rms(settled) if stable else float("nan"),
        float(np.max(residual)), corrections, stable,
    ), error


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rate, source = read_input()
    input_value = input_function(rate, source)

    print("Общий расчёт с коэффициентом 8…", flush=True)
    reference_result = simulate_hybrid_cascade(
        1.0, 1.0, 0.8, REFERENCE_FACTOR, input_value, DURATION_S,
        1, False, FRONTEND_Q1_TWICE,
    )
    reference = base_samples(reference_result.node_v[:, OUTPUT], REFERENCE_FACTOR)
    rows: list[Row] = []
    errors: dict[str, np.ndarray] = {}

    for factor in (1, 2, 4):
        print(f"Общий расчёт с коэффициентом {factor}…", flush=True)
        result = simulate_hybrid_cascade(
            1.0, 1.0, 0.8, factor, input_value, DURATION_S,
            1, False, FRONTEND_Q1_TWICE,
        )
        row, error = metrics(
            f"global_{factor}", f"Вся схема {factor}×", factor, 1, 1,
            base_samples(result.node_v[:, OUTPUT], factor), result.residual_v, reference,
        )
        rows.append(row); errors[row.name] = error

    schedules = (
        (1, 2, "Q1 4×"),
        (1, 4, "Q1 2×"),
        (1, 8, "Q1 1×"),
        (2, 1, "Полная связь Q2 4×"),
        (4, 1, "Полная связь Q2 2×"),
        (8, 1, "Полная связь Q2 1×"),
    )
    scheduled_outputs = {}
    for q2_period, q1_period, label in schedules:
        print(f"Периоды блоков Q2={q2_period}, Q1={q1_period}…", flush=True)
        _, output, residual = scheduled_result(input_value, q2_period, q1_period)
        output_48 = base_samples(output, REFERENCE_FACTOR)
        name = f"scheduled_q2_{q2_period}_q1_{q1_period}"
        row, error = metrics(
            name, label, REFERENCE_FACTOR, q2_period, q1_period,
            output_48, residual, reference,
        )
        rows.append(row); errors[name] = error; scheduled_outputs[name] = output_48

    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(Row.__dataclass_fields__)
        writer.writerows(tuple(vars(row).values()) for row in rows)

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    time_ms = np.arange(len(reference)) / BASE_RATE * 1e3
    axes[0].plot(time_ms, reference, color="black", linewidth=1.0, label="Общая модель 8×")
    for name in ("global_2", "global_4", "scheduled_q2_2_q1_1", "scheduled_q2_4_q1_1"):
        row = next(item for item in rows if item.name == name)
        if row.stable:
            axes[1].plot(time_ms[:len(errors[name])], 1e3 * errors[name], linewidth=0.75,
                         label=row.label)
    axes[0].set_ylabel("Выход, В")
    axes[1].set_ylabel("Разность, мВ")
    axes[1].set_xlabel("Время, мс")
    axes[0].legend(); axes[1].legend(ncol=2)
    for axis in axes: axis.grid(True, alpha=0.3)
    figure.suptitle("Ми-мажорный аккорд: распределение частот расчёта")
    figure.tight_layout(); figure.savefig(FIGURES / "waveform_error.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    stable_rows = [row for row in rows if row.stable]
    labels = [row.label for row in stable_rows]
    rms = [row.rms_mv for row in stable_rows]
    bars = axis.bar(np.arange(len(stable_rows)), rms, color="tab:blue")
    axis.set_yscale("log")
    axis.set_ylabel("Ошибка выхода, мВ СКО")
    axis.set_xticks(np.arange(len(stable_rows)), labels, rotation=35, ha="right")
    axis.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, rms):
        axis.text(bar.get_x()+bar.get_width()/2, value*1.12, f"{value:.2f}", ha="center", fontsize=8)
    figure.tight_layout(); figure.savefig(FIGURES / "error_by_schedule.png", dpi=160); plt.close(figure)

    def value(number: float, digits: int) -> str:
        return f"{number:.{digits}f}" if np.isfinite(number) else "неустойчив"

    table = "\n".join(
        f"| {row.label} | {value(row.rms_mv, 3)} | {value(row.peak_mv, 2)} | "
        f"{value(row.high_band_mv, 3)} | {row.maximum_residual_v:.3e} | "
        f"{row.corrections_per_base_sample:.1f} |"
        for row in rows
    )
    stable_schedules = [row for row in rows if row.name.startswith("scheduled") and row.stable]
    if stable_schedules:
        best_schedule = min(stable_schedules, key=lambda row: row.rms_mv)
        schedule_conclusion = (
            f"Лучший устойчивый разреженный режим — **{best_schedule.label}**: "
            f"ошибка {best_schedule.rms_mv:.3f} мВ СКО."
        )
    else:
        schedule_conclusion = (
            "Ни один разреженный режим не прошёл критерий устойчивости. "
            "Пропускать неявную поправку внутри общей системы нельзя."
        )
    report = f"""# Разнос блоков по частотам расчёта

Проверка выполнена на первых {1e3*DURATION_S:.0f} мс сухого ми-мажорного аккорда,
{1e3*INPUT_PEAK_V:.0f} мВ пик, Sustain = Tone = 1 и Volume = 0,8. Общая каскадная модель 8×
со связанным блоком Q4+Q3+Q2 служит нулём сравнения. Значения выхода
сопоставляются в одинаковые моменты сетки 48 кГц;
первые 5 мс исключены из СКО.

| Режим | Ошибка, мВ СКО | Пик, мВ | Ошибка выше 10 кГц, мВ СКО | Макс. невязка, В | Местных поправок на отсчёт 48 кГц |
|---|---:|---:|---:|---:|---:|
{table}

![Ошибка вариантов](figures/error_by_schedule.png)

![Форма ошибки](figures/waveform_error.png)

## Что именно проверено

В вариантах с разной частотой блоков все 13 накопителей пока обновляются на 384 кГц.
На каждом внутреннем шаге решается связанный блок Q4+Q3. С указанным периодом он
заменяется полной синхронизацией Q4+Q3+Q2 размером 4×4. Две местные поправки Q1
выполняются только с собственным периодом; между ними используется предсказанное
состояние прошлого малого шага. Поэтому опыт отделяет требуемую частоту нелинейных
решений от будущего разбиения линейных накопителей.

{schedule_conclusion}

Из общих частот устойчивым сокращением относительно 8× оказался режим **4×**.
Он дал {next(row.rms_mv for row in rows if row.name == "global_4"):.3f} мВ СКО
на этом входном уровне. Режимы 1× и 2× сорвались.

## Предварительный бюджет отдельного Q3–Q2

Ранее измеренный связанный Q3–Q2 стоит 280–289 тактов на внутренний шаг.
Поэтому только этот блок потребует 1156 тактов на внешний отсчёт при 4× или
2312 при 8×. Из общего бюджета 3542 такта остаётся соответственно около
2386 или 1230 тактов на Q4, Sustain, темброблок, Q1, преобразователи частоты
и ввод-вывод. Это пока верхнеуровневая оценка: Q4 сильно связан с Q3 и его
нельзя автоматически вынести в медленную часть.

## Такты блочных поправок на STM32

После численного опыта в измеритель добавлены целые рабочие блоки. В отдельной
дешёвой сборке `Q4+Q3 3×3 → Q2 1×1` занял 837 тактов. В надёжной сборке единый
`Q4+Q3+Q2 4×4` занял 917 тактов; разница равна 80 тактам. Одна местная поправка
Q1 заняла 391 такт, две последовательные — 677 тактов в надёжной сборке.

Устойчивый режим «полная связь Q2 4× при общей сетке 8×» чередует дешёвый и
надёжный входные блоки. Его средняя цена входной поправки равна примерно
`(837 + 917)/2 = 877` тактов на внутренний шаг, то есть экономия относительно
постоянного 4×4 составляет около `8·40 = 320` тактов на внешний отсчёт 48 кГц.
Абсолютная цена полного потока после добавления измерительных участков изменилась
из-за размещения кода, поэтому для архитектурного вывода используется устойчивая
разность блоков, совпадающая с прежней разностью полных сборок 83 такта.

## Ограничение результата

Это ещё не окончательная многоскоростная схема: общая матрица состояния сохраняет
обратные связи через разделительные конденсаторы. Следующая реализация должна разорвать
систему только по физическим портам C5 и темброблока, добавить линейное повышение и
понижение частоты и затем снова сравнить весь выход с общей моделью 8×. Нельзя просто
заморозить произвольные строки общей матрицы — это изменит проводимости конденсаторов.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
