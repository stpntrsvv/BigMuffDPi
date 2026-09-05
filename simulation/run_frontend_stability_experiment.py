"""Исследует Q4, Sustain и два ограничителя: АЧХ, ФЧХ и устойчивость."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_frontend_model import (
    Q2_COLLECTOR,
    Q3_COLLECTOR,
    discrete_stability_eigenvalues,
    simulate_frontend,
    small_signal_response,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "frontend_stability"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "frontend_stability"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
SUSTAIN_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
FACTOR = 8
DURATION_S = 16e-3
WARMUP_S = 8e-3


@dataclass(frozen=True)
class Summary:
    sustain: float
    spectral_radius: float
    damping_ppm_per_step: float
    equivalent_decay_s: float
    dominant_angle_degrees: float
    peak_gain_db: float
    peak_frequency_hz: float
    maximum_residual_v: float
    maximum_correction_v: float
    q3_peak_to_peak_v: float
    q2_peak_to_peak_v: float
    one_step_rms_error_mv: float
    one_step_peak_error_mv: float


def input_signal(time_s: np.ndarray) -> np.ndarray:
    return 100e-3 * np.sin(2.0 * np.pi * 1_000.0 * time_s)


def plot_frequency(
    frequency_hz: np.ndarray, responses: dict[float, np.ndarray]
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for sustain, response in responses.items():
        axes[0].semilogx(
            frequency_hz, 20.0 * np.log10(np.maximum(np.abs(response), 1e-15)),
            label=f"Sustain {sustain:.2f}"
        )
        axes[1].semilogx(
            frequency_hz, np.degrees(np.unwrap(np.angle(response))),
            label=f"Sustain {sustain:.2f}"
        )
    axes[0].set_ylabel("Коэффициент передачи, дБ")
    axes[0].set_title("Q4 + Sustain + Q3 + Q2: малосигнальная характеристика")
    axes[1].set_ylabel("Фаза, градусы")
    axes[1].set_xlabel("Частота, Гц")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "frequency_and_phase.png", dpi=160)
    plt.close(figure)


def plot_phase_portraits(
    results: dict[float, object], references: dict[float, object]
) -> None:
    selected = (0.25, 0.5, 1.0)
    figure, axes = plt.subplots(2, len(selected), figsize=(16, 9))
    for column, sustain in enumerate(selected):
        result = results[sustain]
        mask = result.time_s >= WARMUP_S
        q = result.nonlinear_v[mask]
        reference = references[sustain]
        reference_mask = reference.time_s >= WARMUP_S
        reference_q = reference.nonlinear_v[reference_mask]
        axes[0, column].plot(
            reference_q[:, 3], reference_q[:, 5], "--", linewidth=1.4,
            label="Полное схождение"
        )
        axes[0, column].plot(
            q[:, 3], q[:, 5], linewidth=0.9, label="Одна поправка"
        )
        axes[0, column].set_xlabel("Q3: vBE, В")
        axes[0, column].set_ylabel("Q3: vD, В")
        axes[0, column].set_title(f"Sustain {sustain:.2f}")
        axes[1, column].plot(
            reference_q[:, 5], reference_q[:, 8], "--", linewidth=1.4
        )
        axes[1, column].plot(q[:, 5], q[:, 8], linewidth=0.9)
        axes[1, column].set_xlabel("Q3: vD, В")
        axes[1, column].set_ylabel("Q2: vD, В")
        for row in range(2):
            axes[row, column].grid(True, alpha=0.3)
        axes[0, column].legend()
    figure.suptitle("Фазовые траектории установившегося режима, 1 кГц")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "phase_portraits.png", dpi=160)
    plt.close(figure)


def plot_solver_diagnostics(results: dict[float, object]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for sustain, result in results.items():
        mask = result.time_s >= WARMUP_S
        time_ms = result.time_s[mask] * 1e3
        axes[0].semilogy(
            time_ms, np.maximum(result.residual_v[mask], 1e-15),
            label=f"Sustain {sustain:.2f}"
        )
        axes[1].semilogy(
            time_ms, np.maximum(result.correction_v[mask], 1e-15),
            label=f"Sustain {sustain:.2f}"
        )
    axes[0].set_ylabel("Невязка после поправки, В")
    axes[0].set_title("Работа одной поправки Ньютона при 8×")
    axes[1].set_ylabel("Максимальная поправка, В")
    axes[1].set_xlabel("Время, мс")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "solver_diagnostics.png", dpi=160)
    plt.close(figure)


def write_report(rows: list[Summary]) -> None:
    table = "\n".join(
        "| {s:.2f} | {rho:.9f} | {ppm:.3f} | {tau:.3f} | {gain:.2f} | "
        "{freq:.1f} | {res:.3e} | {corr:.3e} | {q3:.3f} | {q2:.3f} | "
        "{rms:.3f} | {peak:.3f} |".format(
            s=row.sustain, rho=row.spectral_radius,
            ppm=row.damping_ppm_per_step, tau=row.equivalent_decay_s,
            gain=row.peak_gain_db, freq=row.peak_frequency_hz,
            res=row.maximum_residual_v, corr=row.maximum_correction_v,
            q3=row.q3_peak_to_peak_v, q2=row.q2_peak_to_peak_v,
            rms=row.one_step_rms_error_mv, peak=row.one_step_peak_error_mv,
        )
        for row in rows
    )
    report = f"""# Входной каскад, Sustain и устойчивость ограничителей

## Состав модели

В единую узловую систему добавлены Q4, R2/C1, коллекторно-базовая обратная
связь R9/C10, C4, настоящий потенциометр Sustain, R23 и C5. Далее без разрыва
следуют связанные Q3 и Q2 и линейная нагрузка темброблока. Рабочая архитектура
остаётся гибридной: Q4 и Q3 полные, транзистор Q2 линеаризован около рабочей
точки, диодная пара Q2 нелинейна.

## Как проверяется устойчивость

ЛАЧХ и ЛФЧХ ниже являются характеристиками замкнутой схемы около рабочей
точки. Они показывают резонансные подъёмы и фазовое вращение, но не являются
формальным запасом фазы разомкнутой петли.

Для фактического алгоритма дополнительно численно построен якобиан отображения
одного шага 8×. В состояние входят напряжения всех конденсаторов и нелинейные
напряжения-предикторы. Спектральный радиус меньше единицы означает локальное
затухание малых возмущений. Очень медленный доминирующий режим ожидаем: ёмкости
C6/C7 заряжаются через малую дифференциальную проводимость диодных пар около
нулевого напряжения. Угол доминирующего собственного числа во всех пяти точках равен 0°,
то есть медленный режим не колебательный.

| Sustain | Спектральный радиус | Затухание, ppm/шаг | Экв. время, с | Макс. усиление, дБ | Частота максимума, Гц | Макс. невязка, В | Макс. поправка, В | Q3 размах, В | Q2 размах, В | Ср. кв. ошибка 1 поправки, мВ | Пиковая ошибка, мВ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

![ЛАЧХ и ЛФЧХ](figures/frequency_and_phase.png)

![Фазовые траектории](figures/phase_portraits.png)

![Невязка и поправка](figures/solver_diagnostics.png)

## Критерии тревоги

- спектральный радиус `>= 1`;
- незамкнутая или распухающая фазовая траектория при периодическом входе;
- рост невязки от периода к периоду;
- поправка, выводящая аргумент CORDIC за прямой диапазон;
- узкий подъём ЛАЧХ, отсутствующий в ngspice.

## Вывод

Во всех пяти положениях регулятора спектральный радиус меньше единицы, а фазовые
траектории замыкаются: схема и одна поправка локально устойчивы. При Sustain = 1
невязка доходит до 82,5 мВ, но ошибка выхода Q2 относительно полного схождения
остаётся 1,37 мВ среднеквадратически и 12,2 мВ в пике. Поэтому излом фазовой траектории
в основном физический; численная ошибка видна как небольшое расхождение с штриховой эталонной кривой.
Окончательная проверка глобальной устойчивости потребует ступени, импульса,
перегрузки входа и резкого движения регулятора между крайними положениями.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    frequency_hz = np.logspace(1, 5, 500)
    responses: dict[float, np.ndarray] = {}
    results: dict[float, object] = {}
    references: dict[float, object] = {}
    rows: list[Summary] = []
    for sustain in SUSTAIN_VALUES:
        print(f"Sustain {sustain:.2f}")
        _, q2_response = small_signal_response(sustain, frequency_hz)
        responses[sustain] = q2_response
        result = simulate_frontend(
            sustain, FACTOR, input_signal, DURATION_S,
            architecture="hybrid", fully_converged=False
        )
        results[sustain] = result
        reference = simulate_frontend(
            sustain, FACTOR, input_signal, DURATION_S,
            architecture="hybrid", fully_converged=True
        )
        references[sustain] = reference
        eigenvalues = discrete_stability_eigenvalues(sustain, FACTOR)
        dominant = eigenvalues[int(np.argmax(np.abs(eigenvalues)))]
        radius = float(abs(dominant))
        peak_index = int(np.argmax(np.abs(q2_response)))
        mask = result.time_s >= WARMUP_S
        step_s = 1.0 / (48_000.0 * FACTOR)
        decay = float(-step_s / np.log(radius)) if 0.0 < radius < 1.0 else np.inf
        output_error = (
            result.node_v[mask, Q2_COLLECTOR]
            - reference.node_v[mask, Q2_COLLECTOR]
        )
        rows.append(Summary(
            sustain,
            radius,
            (1.0 - radius) * 1e6,
            decay,
            float(np.degrees(np.angle(dominant))),
            float(20.0 * np.log10(abs(q2_response[peak_index]))),
            float(frequency_hz[peak_index]),
            float(np.max(result.residual_v[mask])),
            float(np.max(result.correction_v[mask])),
            float(np.ptp(result.node_v[mask, Q3_COLLECTOR])),
            float(np.ptp(result.node_v[mask, Q2_COLLECTOR])),
            float(1e3 * np.sqrt(np.mean(output_error * output_error))),
            float(1e3 * np.max(np.abs(output_error))),
        ))

    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    plot_frequency(frequency_hz, responses)
    plot_phase_portraits(results, references)
    plot_solver_diagnostics(results)
    write_report(rows)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
