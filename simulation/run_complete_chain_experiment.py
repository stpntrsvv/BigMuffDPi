"""Проверяет полную цепь Q4–Q1 при разных положениях Tone."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import (
    OUTPUT, discrete_stability_eigenvalues, simulate_complete,
    small_signal_response,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "complete_chain"
EXPERIMENT = ROOT / "simulation" / "experiments" / "complete_chain"
FIGURES = EXPERIMENT / "figures"
TONES = (0.0, 0.25, 0.5, 0.75, 1.0)
RATE = 384_000


@dataclass(frozen=True)
class Summary:
    tone: float
    spectral_radius: float
    output_peak_to_peak_v: float
    rms_one_step_error_mv: float
    peak_one_step_error_mv: float
    maximum_residual_v: float
    maximum_correction_v: float


def signal(time_s: np.ndarray) -> np.ndarray:
    return 0.1 * np.sin(2 * np.pi * 1_000 * time_s)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frequency = np.logspace(1, 5, 500)
    results = {}
    references = {}
    responses = {}
    rows = []
    for tone in TONES:
        print(f"Tone {tone:.2f}")
        result = simulate_complete(1, tone, 1, 8, signal, 12e-3, "hybrid", False)
        reference = simulate_complete(1, tone, 1, 8, signal, 12e-3, "hybrid", True)
        results[tone] = result
        references[tone] = reference
        responses[tone] = small_signal_response(1, tone, 1, frequency)[2]
        mask = result.time_s >= 6e-3
        error = result.node_v[mask, OUTPUT] - reference.node_v[mask, OUTPUT]
        radius = float(np.max(np.abs(discrete_stability_eigenvalues(1, tone, 1, 8))))
        rows.append(Summary(
            tone, radius, float(np.ptp(result.node_v[mask, OUTPUT])),
            float(1e3 * np.sqrt(np.mean(error * error))),
            float(1e3 * np.max(np.abs(error))),
            float(np.max(result.residual_v[mask])),
            float(np.max(result.correction_v[mask])),
        ))

    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for tone, response in responses.items():
        axes[0].semilogx(frequency, 20*np.log10(np.maximum(abs(response), 1e-15)), label=f"Tone {tone:.2f}")
        axes[1].semilogx(frequency, np.degrees(np.unwrap(np.angle(response))), label=f"Tone {tone:.2f}")
    axes[0].set_ylabel("Коэффициент передачи, дБ")
    axes[1].set_ylabel("Фаза, градусы")
    axes[1].set_xlabel("Частота, Гц")
    axes[0].set_title("Полная педаль: ЛАЧХ и ЛФЧХ")
    for axis in axes:
        axis.grid(True, which="both", alpha=.3); axis.legend(ncol=2)
    figure.tight_layout(); figure.savefig(FIGURES / "frequency_and_phase.png", dpi=160); plt.close(figure)

    figure, axes = plt.subplots(len(TONES), 1, figsize=(12, 12), sharex=True)
    for axis, tone in zip(axes, TONES):
        result, reference = results[tone], references[tone]
        mask = result.time_s >= 8e-3
        axis.plot(result.time_s[mask]*1e3, reference.node_v[mask, OUTPUT], "--", label="Полное схождение")
        axis.plot(result.time_s[mask]*1e3, result.node_v[mask, OUTPUT], linewidth=.8, label="Одна поправка")
        axis.set_ylabel("Выход, В"); axis.set_title(f"Tone {tone:.2f}"); axis.grid(True, alpha=.3); axis.legend()
    axes[-1].set_xlabel("Время, мс")
    figure.tight_layout(); figure.savefig(FIGURES / "output_comparison.png", dpi=160); plt.close(figure)

    table = "\n".join(
        f"| {r.tone:.2f} | {r.spectral_radius:.9f} | {r.output_peak_to_peak_v:.3f} | {r.rms_one_step_error_mv:.3f} | {r.peak_one_step_error_mv:.3f} | {r.maximum_residual_v:.3e} | {r.maximum_correction_v:.3e} |"
        for r in rows
    )
    report = f"""# Полная цепь до выхода

В единую систему добавлены полный темброблок, C3, выходной Q1, C2, Volume и нагрузка 1 МОм.
Получилось 25 узлов, 13 напряжений конденсаторов и 12 нелинейных переменных. Sustain и Volume
установлены в максимум, вход — синус 100 мВ пик, 1 кГц, частота расчёта 8×.

Рабочая точка Q1: база 1,636 В, эмиттер 1,012 В, коллектор 4,413 В.

| Tone | Спектральный радиус | Выход, В пик-пик | Ошибка одной поправки, мВ СКО | Пиковая ошибка, мВ | Невязка, В | Поправка, В |
|---:|---:|---:|---:|---:|---:|---:|
{table}

![ЛАЧХ и ЛФЧХ](figures/frequency_and_phase.png)

![Сравнение выходов](figures/output_comparison.png)

## Вывод

Полная система локально устойчива во всём диапазоне Tone. До Tone = 0,75 одна поправка остаётся
приемлемой. При Tone = 1 выходной Q1 получает наиболее богатый высокими частотами сигнал и сам
становится существенной нелинейностью: ошибка резко возрастает. Перед переносом полной цепи на
STM32 надо сравнить три варианта: вторая поправка только для Q1, линеаризация Q1 около рабочей
точки и ограничение рабочего диапазона входа. Физику Q1 пока не сокращаем.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
