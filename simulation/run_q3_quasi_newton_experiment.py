"""Проверяет повторное использование обратного якобиана Q3."""

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
from q3_quasi_newton_model import simulate_quasi_reduced
from q3_reduced_model import simulate_reduced


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "q3_quasi_newton"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_quasi_newton"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
FACTORS = (4, 8, 16)
PERIODS = (1, 2, 4, 8, 16, 32)
BOARD_EXACT = (287, 369, 391)
BOARD_RECIPROCAL = (262, 379, 389)
BOARD_FROZEN = (260, np.nan, np.nan)


@dataclass(frozen=True)
class Result:
    factor: int
    mode: str
    inverse_period: int
    versus_incremental_max_v: float
    versus_full_max_v: float
    maximum_residual_a: float
    inverse_refresh_count: int
    nonlinear_fallback_count: int
    maximum_increment: float


def calculate() -> list[Result]:
    rows: list[Result] = []
    for factor in FACTORS:
        step_s = 1.0 / (48_000.0 * factor)
        full = simulate_reduced("full", duration_s=5.0e-3, step_s=step_s)
        incremental, _ = simulate_incremental_reduced(
            duration_s=5.0e-3, step_s=step_s, refresh_period=32
        )
        for period in PERIODS:
            quasi, diagnostics = simulate_quasi_reduced(
                "frozen", period, duration_s=5.0e-3, step_s=step_s
            )
            rows.append(Result(
                factor, "frozen", period,
                float(np.max(np.abs(
                    quasi.node_v[:, OUTPUT] - incremental.node_v[:, OUTPUT]
                ))),
                float(np.max(np.abs(
                    quasi.node_v[:, OUTPUT] - full.node_v[:, OUTPUT]
                ))),
                float(np.max(quasi.residual_a)),
                diagnostics.inverse_refresh_count,
                diagnostics.nonlinear_fallback_count,
                diagnostics.maximum_increment,
            ))
        quasi, diagnostics = simulate_quasi_reduced(
            "reciprocal", 32, duration_s=5.0e-3, step_s=step_s
        )
        rows.append(Result(
            factor, "reciprocal", 32,
            float(np.max(np.abs(
                quasi.node_v[:, OUTPUT] - incremental.node_v[:, OUTPUT]
            ))),
            float(np.max(np.abs(
                quasi.node_v[:, OUTPUT] - full.node_v[:, OUTPUT]
            ))),
            float(np.max(quasi.residual_a)),
            diagnostics.inverse_refresh_count,
            diagnostics.nonlinear_fallback_count,
            diagnostics.maximum_increment,
        ))
    return rows


def stress() -> list[tuple[float, int, str, int, float, int, float]]:
    rows = []
    for peak_v in (0.5, 1.0):
        for factor in (4, 8):
            baseline, _ = simulate_incremental_reduced(
                duration_s=2.0e-3,
                step_s=1.0 / (48_000.0 * factor),
                input_peak_v=peak_v,
                refresh_period=8,
            )
            for mode, period in (("frozen", 4), ("frozen", 8), ("reciprocal", 32)):
                quasi, diagnostics = simulate_quasi_reduced(
                    mode,
                    period,
                    duration_s=2.0e-3,
                    step_s=1.0 / (48_000.0 * factor),
                    input_peak_v=peak_v,
                    nonlinear_refresh_period=8,
                )
                error = float(np.max(np.abs(
                    quasi.node_v[:, OUTPUT] - baseline.node_v[:, OUTPUT]
                )))
                rows.append((
                    peak_v, factor, mode, period, error,
                    diagnostics.nonlinear_fallback_count,
                    diagnostics.maximum_increment,
                ))
    return rows


def plot_errors(rows: list[Result]) -> None:
    figure, axes = plt.subplots(1, len(FACTORS), figsize=(15, 5), sharey=True)
    for axis, factor in zip(axes, FACTORS):
        selected = [row for row in rows if row.factor == factor and row.mode == "frozen"]
        axis.plot(
            [row.inverse_period for row in selected],
            [row.versus_incremental_max_v * 1.0e6 for row in selected],
            "o-",
        )
        reciprocal = next(
            row for row in rows if row.factor == factor and row.mode == "reciprocal"
        )
        axis.axhline(
            reciprocal.versus_incremental_max_v * 1.0e6,
            color="tab:orange", linestyle="--",
            label="Обновление обратного знаменателя",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(PERIODS, labels=[str(value) for value in PERIODS])
        axis.set_title(f"{factor}×")
        axis.set_xlabel("Период пересчёта матрицы")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Ошибка против обычной поправки, мкВ")
    axes[-1].legend()
    figure.suptitle("Q3: повторное использование обратного якобиана")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "error_by_inverse_period.png", dpi=160)
    plt.close(figure)


def plot_board_cycles() -> None:
    labels = ("Изолированная поправка", "Поток 50 мВ", "Поток 1 В")
    positions = np.arange(len(labels))
    width = 0.26
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(positions - width, BOARD_EXACT, width, label="Точный знаменатель")
    axis.bar(positions, BOARD_RECIPROCAL, width, label="Обновляемый знаменатель")
    axis.bar(positions + width, BOARD_FROZEN, width, label="Замороженная матрица")
    axis.axhline(170_000_000 / (48_000 * 8), color="black", linestyle="--", label="Бюджет 8×")
    axis.set_xticks(positions, labels=labels)
    axis.set_ylabel("Тактов")
    axis.set_title("Q3: выигрыш внутри функции и в полном потоке")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "board_cycles.png", dpi=160)
    plt.close(figure)


def write_report(
    rows: list[Result],
    stress_rows: list[tuple[float, int, str, int, float, int, float]],
) -> None:
    mode_title = {
        "frozen": "замороженная матрица",
        "reciprocal": "обратный знаменатель",
    }
    table = "\n".join(
        f"| {row.factor}× | {mode_title[row.mode]} | {row.inverse_period} | "
        f"{row.versus_incremental_max_v * 1e6:.3f} | "
        f"{row.versus_full_max_v * 1e6:.3f} | "
        f"{row.maximum_residual_a:.3e} | {row.inverse_refresh_count} |"
        for row in rows
    )
    stress_table = "\n".join(
        f"| {peak:.1f} | {factor}× | {mode_title[mode]} | {period} | "
        f"{error * 1e6:.3f} | {fallback} | {increment:.4f} |"
        for peak, factor, mode, period, error, fallback, increment in stress_rows
    )
    report = rf"""# Q3: повторное использование обратного якобиана

## Идея

Проверены два продолжения метода малого приращения:

1. `frozen` — обратная матрица поправки сохраняется на несколько отсчётов;
2. `reciprocal` — элементы якобиана считаются каждый шаг, но обратный
   определитель обновляется одной ньютоновской поправкой

\[
r_{{k+1}}=r_k(2-D_{{k+1}}r_k),\qquad r\approx D^{{-1}}.
\]

Оба варианта используют одну поправку состояния на внутренний отсчёт.

## Синус 1 кГц, 50 мВ, 5 мс

| Частота | Режим | Период матрицы | Против обычной поправки, мкВ | Против полной сходимости, мкВ | Макс. невязка, А | Пересчётов матрицы |
|---:|---|---:|---:|---:|---:|---:|
{table}

![Ошибка по периоду](figures/error_by_inverse_period.png)

## Сильный вход

| Вход, В | Частота | Режим | Период | Ошибка, мкВ | Восстановлений нелинейности | Макс. приращение |
|---:|---:|---|---:|---:|---:|---:|
{stress_table}

## Такты STM32G474RE

| Вариант | Изолированная поправка | Поток 50 мВ | Поток 1 В |
|---|---:|---:|---:|
| Точный знаменатель | 287 | 369 | 391 |
| Обновляемый обратный знаменатель | 262 | 379 | 389 |
| Замороженная обратная матрица | 260 | — | — |

Точное восстановление обратного знаменателя стоит 55 тактов, всей обратной
матрицы — 80 тактов. Одиночные результаты всех трёх вариантов побитно совпали.

![Такты на плате](figures/board_cycles.png)

## Вывод

Обновление обратного знаменателя численно удачно: при обычном входе оно добавляет
лишь 5–9 мкВ ошибки. Изолированная поправка ускорилась на 25 тактов. Однако в
полном потоке проверки и восстановление дополнительного состояния съели выигрыш:
обычный режим стал на 10 тактов дороже, а сильный — только на 2 такта дешевле.

Замороженная матрица требует восстановления за 80 тактов. Даже при периоде 4
её расчётная цена около 280 тактов до потокового управления, а ошибка уже
82–364 мкВ в зависимости от частоты. Поэтому ни один вариант пока не заменяет
точный знаменатель в основном пути.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = calculate()
    stress_rows = stress()
    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Result.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    plot_errors(rows)
    plot_board_cycles()
    write_report(rows, stress_rows)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
