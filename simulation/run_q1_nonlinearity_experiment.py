"""Сравнивает способы расчёта выходного Q1 в полной модели педали."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, simulate_complete


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "q1_nonlinearity"
EXPERIMENT = ROOT / "simulation" / "experiments" / "q1_nonlinearity"
FIGURES = EXPERIMENT / "figures"
TONES = (0.0, 0.5, 0.75, 1.0)
AMPLITUDES_MV = (25, 50, 100, 200)
METHODS = (
    ("one", "Одна общая поправка", "hybrid", 0),
    ("q1_local_1", "+1 локальная Q1", "hybrid", 1),
    ("q1_local_2", "+2 локальные Q1", "hybrid", 2),
    ("q1_linear", "Линейный Q1", "hybrid_q1_linear", 0),
)


@dataclass(frozen=True)
class Row:
    tone: float
    amplitude_mv: int
    method: str
    rms_error_mv: float
    peak_error_mv: float
    maximum_residual_v: float


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    traces = {}
    for amplitude_mv in AMPLITUDES_MV:
        amplitude_v = amplitude_mv * 1e-3
        signal = lambda time_s, a=amplitude_v: a * np.sin(2*np.pi*1_000*time_s)
        for tone in TONES:
            print(f"{amplitude_mv} мВ, Tone {tone:.2f}")
            reference = simulate_complete(1, tone, 1, 8, signal, 8e-3, "hybrid", True)
            mask = reference.time_s >= 4e-3
            for key, _, architecture, local in METHODS:
                result = simulate_complete(
                    1, tone, 1, 8, signal, 8e-3, architecture, False, local
                )
                error = result.node_v[mask, OUTPUT] - reference.node_v[mask, OUTPUT]
                rows.append(Row(
                    tone, amplitude_mv, key,
                    float(1e3*np.sqrt(np.mean(error*error))),
                    float(1e3*np.max(np.abs(error))),
                    float(np.max(result.residual_v[mask])),
                ))
                if amplitude_mv == 100 and tone == 1.0:
                    traces[key] = result
            if amplitude_mv == 100 and tone == 1.0:
                traces["reference"] = reference

    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(Row.__dataclass_fields__)
        writer.writerows((r.tone, r.amplitude_mv, r.method, r.rms_error_mv,
                          r.peak_error_mv, r.maximum_residual_v) for r in rows)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for axis, (key, label, _, _) in zip(axes.flat, METHODS):
        grid = np.array([[next(r.rms_error_mv for r in rows if r.method == key and r.tone == tone and r.amplitude_mv == amp)
                          for tone in TONES] for amp in AMPLITUDES_MV])
        image = axis.imshow(grid, origin="lower", aspect="auto", norm="log",
                            extent=(-.125, 1.125, 12.5, 212.5), vmin=.01, vmax=100)
        for row_index, amp in enumerate(AMPLITUDES_MV):
            for column, tone in enumerate(TONES):
                value = grid[row_index, column]
                label_text = "срыв" if value > 1e6 else f"{value:.2f}"
                axis.text(tone, amp, label_text, ha="center", va="center")
        axis.set_title(label); axis.set_xlabel("Tone"); axis.set_ylabel("Вход, мВ пик")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Ошибка выхода, мВ СКО")
    figure.suptitle("Практические варианты Q1 относительно полного схождения")
    figure.savefig(FIGURES / "error_map.png", dpi=160, bbox_inches="tight"); plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    reference = traces["reference"]
    mask = reference.time_s >= 6e-3
    axis.plot(reference.time_s[mask]*1e3, reference.node_v[mask, OUTPUT], "k--", linewidth=2, label="Полное схождение")
    for key, label, _, _ in METHODS:
        axis.plot(traces[key].time_s[mask]*1e3, traces[key].node_v[mask, OUTPUT], linewidth=1, label=label)
    axis.set_title("Tone 1, вход 100 мВ пик"); axis.set_xlabel("Время, мс"); axis.set_ylabel("Выход, В")
    axis.grid(True, alpha=.3); axis.legend(ncol=2)
    figure.tight_layout(); figure.savefig(FIGURES / "tone1_waveform.png", dpi=160); plt.close(figure)

    rows_100 = [r for r in rows if r.amplitude_mv == 100]
    table = "\n".join(
        f"| {r.tone:.2f} | {next(label for key,label,_,_ in METHODS if key == r.method)} | {r.rms_error_mv:.3f} | {r.peak_error_mv:.3f} | {r.maximum_residual_v:.3e} |"
        for r in rows_100
    )
    tone1_table = "\n".join(
        f"| {r.amplitude_mv} | {next(label for key,label,_,_ in METHODS if key == r.method)} | {r.rms_error_mv:.3g} | {r.peak_error_mv:.3g} | {r.maximum_residual_v:.3e} |"
        for r in rows if r.tone == 1.0
    )
    report = f"""# Нелинейность выходного Q1

Сравнены одна общая поправка, одна и две дополнительные локальные поправки двух переменных
Q1 и линеаризация Q1. Эталон — полное схождение полной системы при 8×. Sustain и Volume
максимальны; проверены четыре положения Tone и вход 25, 50, 100 и 200 мВ пик.

Переход база–коллектор Q1 при Tone = 1 и входе 100 мВ достигает +0,457 В: Q1 входит в
насыщение, поэтому поправка только по переходу база–эмиттер была бы физически неверной.

## Вход 100 мВ пик

| Tone | Вариант | Ошибка, мВ СКО | Пиковая ошибка, мВ | Невязка, В |
|---:|---|---:|---:|---:|
{table}

## Tone = 1 по уровням входа

| Вход, мВ пик | Вариант | Ошибка, мВ СКО | Пиковая ошибка, мВ | Невязка, В |
|---:|---|---:|---:|---:|
{tone1_table}

![Карта ошибки](figures/error_map.png)

![Форма сигнала](figures/tone1_waveform.png)

## Вывод

До входа 100 мВ одна локальная двухпеременная поправка Q1 заметно помогает, две снижают ошибку
ещё сильнее. Но при 200 мВ и Tone = 1 все варианты с недосходившимся нелинейным Q1 срываются:
невязка достигает порядка 10^30. Даже фиксированные 2–10 общих поправок не восстанавливают
решение; около 15–20 уже сходятся, что неприемлемо для постоянного звукового пути.

Линеаризация Q1 остаётся ограниченной во всей проверенной сетке и при худшем режиме даёт
15,95 мВ СКО, но теряет физическое насыщение. Практический кандидат теперь адаптивный:
нелинейный Q1 с локальной поправкой в обычном режиме и переход на устойчивый линейный Q1 либо
на редкое полное восстановление при превышении порога невязки. Перед STM32 надо проверить
плавность такого перехода и затем измерить его стоимость.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
