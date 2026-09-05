"""Проверяет обновление нелинейных функций Q3 по малому приращению."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q3_incremental_model import simulate_incremental_reduced
from q3_model import OUTPUT
from q3_reduced_model import simulate_reduced


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "q3_incremental"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_incremental"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
FACTORS = (8, 16, 32)
REFRESH_PERIOD = 32
INCREMENTAL_CYCLES = 288
REFRESH_CYCLES = 176


@dataclass(frozen=True)
class Summary:
    factor: int
    incremental_vs_float_max_v: float
    incremental_vs_full_max_v: float
    float_vs_full_max_v: float
    maximum_cache_relative_error: float
    maximum_forward_increment: float
    maximum_diode_increment: float
    fallback_count: int


def calculate_factor(factor: int) -> Summary:
    step_s = 1.0 / (48_000.0 * factor)
    full = simulate_reduced("full", duration_s=5.0e-3, step_s=step_s)
    floating = simulate_reduced("newton", duration_s=5.0e-3, step_s=step_s)
    incremental, diagnostics = simulate_incremental_reduced(
        duration_s=5.0e-3,
        step_s=step_s,
        refresh_period=REFRESH_PERIOD,
    )
    return Summary(
        factor,
        float(np.max(np.abs(incremental.node_v[:, OUTPUT] - floating.node_v[:, OUTPUT]))),
        float(np.max(np.abs(incremental.node_v[:, OUTPUT] - full.node_v[:, OUTPUT]))),
        float(np.max(np.abs(floating.node_v[:, OUTPUT] - full.node_v[:, OUTPUT]))),
        diagnostics.maximum_cache_relative_error,
        diagnostics.maximum_forward_increment,
        diagnostics.maximum_diode_increment,
        diagnostics.fallback_count,
    )


def refresh_sweep() -> list[tuple[int, float, float]]:
    step_s = 1.0 / (48_000.0 * 16.0)
    floating = simulate_reduced("newton", duration_s=5.0e-3, step_s=step_s)
    rows = []
    for period in (1, 8, 16, 32, 64, 128, 256):
        incremental, diagnostics = simulate_incremental_reduced(
            duration_s=5.0e-3,
            step_s=step_s,
            refresh_period=period,
        )
        error = float(
            np.max(np.abs(incremental.node_v[:, OUTPUT] - floating.node_v[:, OUTPUT]))
        )
        rows.append((period, error, diagnostics.maximum_cache_relative_error))
    return rows


def stress_sweep() -> list[tuple[float, int, float, float, int]]:
    rows = []
    for peak_v in (0.05, 0.5, 1.0):
        for factor in FACTORS:
            incremental, diagnostics = simulate_incremental_reduced(
                duration_s=2.0e-3,
                step_s=1.0 / (48_000.0 * factor),
                input_peak_v=peak_v,
                refresh_period=REFRESH_PERIOD,
            )
            del incremental
            rows.append(
                (
                    peak_v,
                    factor,
                    diagnostics.maximum_forward_increment,
                    diagnostics.maximum_diode_increment,
                    diagnostics.fallback_count,
                )
            )
    return rows


def plot_factor_errors(rows: list[Summary]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    factors = [row.factor for row in rows]
    axis.plot(
        factors,
        [row.incremental_vs_float_max_v * 1.0e6 for row in rows],
        "o-",
        label="Малое приращение против обычного Ньютона",
    )
    axis.plot(
        factors,
        [row.incremental_vs_full_max_v * 1.0e6 for row in rows],
        "o-",
        label="Малое приращение против полной сходимости",
    )
    axis.set_xticks(factors)
    axis.set_yscale("log")
    axis.set_xlabel("Повышение частоты")
    axis.set_ylabel("Максимальная ошибка выхода, мкВ")
    axis.set_title("Обновление по малому приращению, пересчёт раз в 32 шага")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "error_by_factor.png", dpi=160)
    plt.close(figure)


def plot_refresh(rows: list[tuple[int, float, float]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    period = [row[0] for row in rows]
    axis.plot(period, [row[1] * 1.0e6 for row in rows], "o-")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(period, labels=[str(value) for value in period])
    axis.set_xlabel("Период полного пересчёта, отсчётов")
    axis.set_ylabel("Ошибка относительно обычного Ньютона, мкВ")
    axis.set_title("Q3 16×: накопление ошибки рекуррентного состояния")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "error_by_refresh_period.png", dpi=160)
    plt.close(figure)


def write_report(
    rows: list[Summary],
    refresh_rows: list[tuple[int, float, float]],
    stress_rows: list[tuple[float, int, float, float, int]],
) -> None:
    factor_table = "\n".join(
        f"| {row.factor}× | {row.incremental_vs_float_max_v * 1e6:.3f} | "
        f"{row.incremental_vs_full_max_v * 1e6:.3f} | "
        f"{row.float_vs_full_max_v * 1e6:.3f} | "
        f"{row.maximum_cache_relative_error:.3e} | "
        f"{row.maximum_forward_increment:.4f} | {row.maximum_diode_increment:.4f} | "
        f"{row.fallback_count} |"
        for row in rows
    )
    refresh_table = "\n".join(
        f"| {period} | {error * 1e6:.3f} | {cache_error:.3e} |"
        for period, error, cache_error in refresh_rows
    )
    stress_table = "\n".join(
        f"| {peak:.2f} | {factor}× | {forward:.4f} | {diode:.4f} | {fallback} |"
        for peak, factor, forward, diode, fallback in stress_rows
    )
    average_cycles = INCREMENTAL_CYCLES + REFRESH_CYCLES / REFRESH_PERIOD
    budget_8 = 170_000_000 / (48_000 * 8)
    spare_8 = budget_8 - average_cycles
    two_stages = 2 * average_cycles
    budget_rows = "\n".join(
        f"| {factor}× | {170_000_000 / (48_000 * factor):.1f} | "
        f"{average_cycles:.1f} | "
        f"{100 * average_cycles / (170_000_000 / (48_000 * factor)):.1f}% |"
        for factor in FACTORS
    )
    report = rf"""# Q3: обновление нелинейностей по малому приращению

## Метод

В состоянии хранятся
экспонента \(E=e^{{v_{{BE}}/V_T}}\) и пара
\(S=\sinh(v_D/nV_T)\), \(C=\cosh(v_D/nV_T)\). После поправки
\(\Delta v\) они обновляются как

\[
E_{{k+1}}=E_k\left(\cosh\Delta z+\sinh\Delta z\right),
\qquad \Delta z=\frac{{\Delta v_{{BE}}}}{{V_T}},
\]

\[
S_{{k+1}}=S_k\cosh\Delta d+C_k\sinh\Delta d,
\qquad
C_{{k+1}}=C_k\cosh\Delta d+S_k\sinh\Delta d.
\]

CORDIC получает малое приращение напрямую, без разложения \(k\ln2+r\),
масштабирования экспоненты и пересборки всего якобиана. Раз в
{REFRESH_PERIOD} шага сохранённое состояние полностью пересчитывается. Если приращение выходит за прямой
диапазон CORDIC, полный пересчёт выполняется немедленно.

## Точность

Синус 1 кГц, 50 мВ, 5 мс. Эмулятор округляет входы и выходы CORDIC до Q1.31,
но не воспроизводит внутреннюю ошибку его итераций.

| Частота | Против обычного Ньютона, мкВ | Против полной сходимости, мкВ | Обычный Ньютон против полной сходимости, мкВ | Ошибка сохранённого состояния | Макс. \(|\Delta BE|\) | Макс. \(|\Delta D|\) | Аварийных пересчётов |
|---:|---:|---:|---:|---:|---:|---:|---:|
{factor_table}

![Ошибка по частоте](figures/error_by_factor.png)

### Период пересчёта при 16×

| Период | Ошибка выхода, мкВ | Ошибка сохранённого состояния |
|---:|---:|---:|
{refresh_table}

![Ошибка по периоду](figures/error_by_refresh_period.png)

### Сильный вход

| Вход, В | Частота | Макс. \(|\Delta BE|\) | Макс. \(|\Delta D|\) | Аварийных пересчётов |
|---:|---:|---:|---:|---:|
{stress_table}

## Изолированный замер тактов STM32G474RE

- поправка с двумя прямыми обновлениями CORDIC — **{INCREMENTAL_CYCLES} такта**;
- полный пересчёт двух нелинейных состояний — **{REFRESH_CYCLES} тактов**;
- средняя цена при периоде {REFRESH_PERIOD} — **{average_cycles:.1f} такта**.

Это расчётная сумма изолированных вызовов, без счётчика периода и выбора
режима. Длительный потоковый опыт дал 371 такт на обычном входе и 392 такта
при автоматическом периоде на сильном входе; см.
[`q3_stream_benchmark/report.md`](../q3_stream_benchmark/report.md).

| Частота | Бюджет, тактов | Цена Q3, тактов | Доля бюджета |
|---:|---:|---:|---:|
{budget_rows}

## Вывод

По изолированной оценке один Q3 помещается в 8× с запасом около {spare_8:.1f} такта. Два одинаковых каскада
потребуют около {two_stages:.1f} такта, поэтому вся педаль в 8× пока не помещается. Следующие
резервы: одно общее полиномиальное обновление вместо двух вызовов CORDIC, упрощение транзистора
или разные частоты для двух каскадов.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [calculate_factor(factor) for factor in FACTORS]
    refresh_rows = refresh_sweep()
    stress_rows = stress_sweep()
    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    plot_factor_errors(rows)
    plot_refresh(refresh_rows)
    write_report(rows, refresh_rows, stress_rows)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
