"""Проверяет разрядность целочисленного ядра Ньютона для Q3."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q3_fixed_point import FixedFormats, simulate_fixed_reduced
from q3_model import OUTPUT
from q3_reduced_model import simulate_reduced


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "q3_fixed_point"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_fixed_point"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
FACTORS = (8, 16, 32)
FORMATS = FixedFormats(voltage=30, jacobian=23, normalized_influence=30)


@dataclass(frozen=True)
class Summary:
    factor: int
    fixed_vs_float_max_v: float
    fixed_vs_full_max_v: float
    float_vs_full_max_v: float
    fixed_residual_max_a: float
    voltage_range_usage: float
    jacobian_range_usage: float


def calculate(factor: int, duration_s: float = 5.0e-3) -> tuple[Summary, object, object]:
    step_s = 1.0 / (48_000.0 * factor)
    full = simulate_reduced("full", duration_s=duration_s, step_s=step_s)
    floating = simulate_reduced("newton", duration_s=duration_s, step_s=step_s)
    fixed, diagnostics = simulate_fixed_reduced(
        duration_s=duration_s, step_s=step_s, formats=FORMATS
    )
    fixed_vs_float = fixed.node_v[:, OUTPUT] - floating.node_v[:, OUTPUT]
    fixed_vs_full = fixed.node_v[:, OUTPUT] - full.node_v[:, OUTPUT]
    float_vs_full = floating.node_v[:, OUTPUT] - full.node_v[:, OUTPUT]
    summary = Summary(
        factor,
        float(np.max(np.abs(fixed_vs_float))),
        float(np.max(np.abs(fixed_vs_full))),
        float(np.max(np.abs(float_vs_full))),
        float(np.max(fixed.residual_a)),
        diagnostics.maximum_voltage_integer / float((1 << 31) - 1),
        diagnostics.maximum_jacobian_integer / float((1 << 31) - 1),
    )
    return summary, floating, fixed


def stress_rows() -> list[tuple[float, float, float]]:
    rows = []
    step_s = 1.0 / (48_000.0 * 16.0)
    for peak_v in (0.05, 0.5, 1.0):
        floating = simulate_reduced(
            "newton", duration_s=2.0e-3, step_s=step_s, input_peak_v=peak_v
        )
        fixed, diagnostics = simulate_fixed_reduced(
            duration_s=2.0e-3,
            step_s=step_s,
            input_peak_v=peak_v,
            formats=FORMATS,
        )
        error = float(
            np.max(np.abs(fixed.node_v[:, OUTPUT] - floating.node_v[:, OUTPUT]))
        )
        rows.append((peak_v, error, diagnostics.maximum_voltage_integer / ((1 << 31) - 1)))
    return rows


def plot_errors(rows: list[Summary]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    factors = [row.factor for row in rows]
    axis.plot(
        factors,
        [row.fixed_vs_float_max_v * 1.0e6 for row in rows],
        "o-",
        label="Фиксированная точка против float",
    )
    axis.plot(
        factors,
        [row.fixed_vs_full_max_v * 1.0e6 for row in rows],
        "o-",
        label="Фиксированная точка против полной сходимости",
    )
    axis.set_xticks(factors)
    axis.set_yscale("log")
    axis.set_xlabel("Повышение частоты")
    axis.set_ylabel("Максимальная ошибка выхода, мкВ")
    axis.set_title("Q3: целочисленная поправка Q2.30")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "error_by_factor.png", dpi=160)
    plt.close(figure)


def write_report(rows: list[Summary], stress: list[tuple[float, float, float]]) -> None:
    factor_table = "\n".join(
        f"| {row.factor}× | {row.fixed_vs_float_max_v * 1e6:.3f} | "
        f"{row.fixed_vs_full_max_v * 1e6:.3f} | {row.float_vs_full_max_v * 1e6:.3f} | "
        f"{row.fixed_residual_max_a:.3e} | {row.voltage_range_usage * 100:.1f}% |"
        for row in rows
    )
    stress_table = "\n".join(
        f"| {peak:.2f} | {error * 1e6:.3f} | {usage * 100:.1f}% |"
        for peak, error, usage in stress
    )
    report = rf"""# Фиксированная точка для ядра Ньютона Q3

## Цель

Проверить, можно ли представить сокращённое ядро Q3 в 32-разрядных целых
числах без неприемлемой ошибки.

## Масштабирование

Один общий Q-формат непригоден: в модели одновременно есть токи порядка
\(10^{{-14}}\) А, напряжения в вольтах и элементы якобиана до 78.

Использованы:

- отклонения нелинейных напряжений от рабочей точки — Q2.30;
- токи сначала умножаются на масштаб своего столбца матрицы, затем хранятся в Q2.30;
- нормированная матрица влияния — Q2.30;
- якобиан — Q9.23;
- решение 3×3 — целочисленный метод Гаусса с 64-разрядными промежуточными результатами.

## Точность

| Частота | Целые против float, мкВ | Целые против полной сходимости, мкВ | Float против полной сходимости, мкВ | Невязка, А | Занято Q2.30 |
|---:|---:|---:|---:|---:|---:|
{factor_table}

![Ошибка по частоте](figures/error_by_factor.png)

### Проверка уровня в 16×

| Амплитуда входа, В | Целые против float, мкВ | Занято Q2.30 |
|---:|---:|---:|
{stress_table}

## Измерение STM32

Решение одной и той же системы 3×3:

- аппаратная плавающая точка с развёрнутой формулой — **72 такта**;
- целочисленный метод Гаусса с 64-разрядными делениями — **1343 такта**.

## Вывод

По точности раздельные форматы работают. Но полная замена `float` на целые числа в
текущей системе 3×3 не даёт ускорения: Cortex-M4 не имеет аппаратного 64-разрядного
деления, а FPU решает эту малую систему очень дёшево. Разумный путь — оставить
решение матрицы в `float`, а целые форматы пробовать для FMAC, CORDIC и хранения состояний.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for factor in FACTORS:
        summary, _, _ = calculate(factor)
        rows.append(summary)
    stress = stress_rows()
    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    plot_errors(rows)
    write_report(rows, stress)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
