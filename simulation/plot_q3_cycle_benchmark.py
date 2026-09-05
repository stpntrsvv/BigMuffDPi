"""Строит отчёт по измерению тактов Q3 на STM32G474RE."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "simulation" / "raw" / "q3_cycle_benchmark" / "board.txt"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_cycle_benchmark"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
INCREMENTAL_REFRESH_PERIOD = 32

QUANTITIES = (
    ("forward_current", "Ток база–эмиттер"),
    ("reverse_current", "Обратный ток база–коллектор"),
    ("diode_current", "Ток диодной пары"),
    ("forward_first", "Производная тока база–эмиттер"),
    ("reverse_first", "Производная обратного тока"),
    ("diode_first", "Производная тока диодов"),
    ("forward_second", "Вторая производная тока база–эмиттер"),
    ("reverse_second", "Вторая производная обратного тока"),
    ("diode_second", "Вторая производная тока диодов"),
)


def load_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name] = int(value)
    required = {
        "core_hz", "repetitions", "overhead_cycles",
        "nonlinear_libm_cycles", "nonlinear_cordic_cycles",
        "form_cordic_cycles", "solve_3x3_cycles",
        "newton_libm_cycles", "newton_cordic_cycles",
        "newton_cordic_optimized_cycles",
        "newton_cordic_incremental_cycles", "incremental_refresh_cycles",
        "halley_libm_cycles", "halley_cordic_cycles",
        "cordic_cosh_sinh_cycles", "cordic_cosh_q31", "cordic_sinh_q31",
    }
    for name, _ in QUANTITIES:
        required.update((f"libm_{name}_bits", f"cordic_{name}_bits"))
    missing = required - values.keys()
    if missing:
        raise ValueError(f"В журнале платы отсутствуют поля: {sorted(missing)}")
    return values


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def plot_cycle_parts(values: dict[str, int]) -> None:
    labels = ("Нелинейности", "Ньютон", "Галлей")
    library = (
        values["nonlinear_libm_cycles"], values["newton_libm_cycles"],
        values["halley_libm_cycles"],
    )
    incremental_average = (
        values["newton_cordic_incremental_cycles"] +
        values["incremental_refresh_cycles"] / INCREMENTAL_REFRESH_PERIOD
    )
    cordic = (
        values["nonlinear_cordic_cycles"], incremental_average,
        values["halley_cordic_cycles"],
    )
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 6))
    left = axis.bar([x - width / 2 for x in positions], library, width, label="Библиотечные функции")
    right = axis.bar([x + width / 2 for x in positions], cordic, width, label="CORDIC с приведением")
    axis.set_yscale("log")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Тактов ядра, логарифмическая шкала")
    axis.set_title("Q3: стоимость до и после переноса на CORDIC")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    axis.bar_label(left, padding=3)
    axis.bar_label(right, padding=3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "cycle_parts.png", dpi=160)
    plt.close(figure)


def plot_budgets(values: dict[str, int]) -> None:
    factors = (8, 16, 32)
    budgets = [values["core_hz"] / (48_000.0 * factor) for factor in factors]
    incremental_average = (
        values["newton_cordic_incremental_cycles"] +
        values["incremental_refresh_cycles"] / INCREMENTAL_REFRESH_PERIOD
    )
    newton = [incremental_average / budget * 100.0 for budget in budgets]
    halley = [values["halley_cordic_cycles"] / budget * 100.0 for budget in budgets]
    positions = list(range(len(factors)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar([x - width / 2 for x in positions], newton, width, label="Ньютон")
    axis.bar([x + width / 2 for x in positions], halley, width, label="Галлей")
    axis.axhline(100.0, color="black", linestyle="--", label="Весь бюджет")
    axis.set_yscale("log")
    axis.set_xticks(positions, labels=[f"{f}×\n{b:.1f} такта" for f, b in zip(factors, budgets)])
    axis.set_ylabel("Доля временного бюджета, %")
    axis.set_title("Один Q3 с CORDIC, без остальной педали")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "budget_usage.png", dpi=160)
    plt.close(figure)


def accuracy_rows(values: dict[str, int]) -> tuple[str, float]:
    rows = []
    maximum_relative = 0.0
    for name, title in QUANTITIES:
        reference = bits_to_float(values[f"libm_{name}_bits"])
        measured = bits_to_float(values[f"cordic_{name}_bits"])
        absolute = abs(measured - reference)
        relative = absolute / abs(reference) if reference else 0.0
        maximum_relative = max(maximum_relative, relative)
        rows.append(f"| {title} | {reference:.8e} | {measured:.8e} | {absolute:.3e} | {relative:.3e} |")
    return "\n".join(rows), maximum_relative


def write_report(values: dict[str, int]) -> None:
    incremental_average = (
        values["newton_cordic_incremental_cycles"] +
        values["incremental_refresh_cycles"] / INCREMENTAL_REFRESH_PERIOD
    )
    factors = (8, 16, 32)
    budget_rows = "\n".join(
        f"| {factor}× | {48_000 * factor} | {budget:.1f} | "
        f"{100 * incremental_average / budget:.1f} | "
        f"{100 * values['halley_cordic_cycles'] / budget:.1f} |"
        for factor in factors
        for budget in [values["core_hz"] / (48_000.0 * factor)]
    )
    rows, maximum_relative = accuracy_rows(values)
    speedup_nonlinear = values["nonlinear_libm_cycles"] / values["nonlinear_cordic_cycles"]
    speedup_newton = values["newton_libm_cycles"] / values["newton_cordic_cycles"]
    speedup_newton_optimized = (
        values["newton_libm_cycles"] / values["newton_cordic_optimized_cycles"]
    )
    speedup_newton_incremental = values["newton_libm_cycles"] / incremental_average
    speedup_halley = values["halley_libm_cycles"] / values["halley_cordic_cycles"]
    budget_8 = values["core_hz"] / (48_000.0 * 8.0)
    newton_8_percent = 100.0 * incremental_average / budget_8
    cordic_cosh = values["cordic_cosh_q31"] / (2.0**30)
    cordic_sinh = values["cordic_sinh_q31"] / (2.0**30)
    report = rf"""# Такты полной поправки Q3 на STM32G474RE

## Цель и условия

Измерена трёхмерная поправка Q3 до упрощения физики. Плата NUCLEO-G474RE,
ядро 170 МГц, GCC 7.2.1, `-O3`, FPv4-SP-D16. Счётчик DWT, по {values['repetitions']}
вызовов, вычтены {values['overhead_cycles']} тактов пустого вызова. Матрица 3×3 решается
развёрнутой формулой. Ввод-вывод, FMAC и остальные каскады не входят.

## Приведение аргумента

Для каждой экспоненты используется

$$x=k\ln 2+r,\qquad k=\operatorname{{round}}(x/\ln 2),\qquad |r|\leq\ln2/2.$$

CORDIC считает `cosh(r)` и `sinh(r)`, затем

$$e^x=2^k(\cosh r+\sinh r),\qquad e^{{-x}}=2^{{-k}}(\cosh r-\sinh r).$$

Так аргументы исходной модели до примерно 24,4 превращаются в остаток
не более 0,347, что заведомо внутри границы 1,118 из [ST AN5325](https://www.st.com/resource/en/application_note/an5325-how-to-use-the-cordic-to-perform-mathematical-functions-on-stm32-mcus-stmicroelectronics.pdf).

## Специализация Ньютона

В горячем пути убраны указатель на функцию, массивы, циклы, деления на постоянные
масштабы и вторые производные. Если обратная экспонента после масштабирования всё равно
округляется до нуля в `float`, вызов CORDIC пропускается. При нулевой производной обратного
перехода система 3×3 точно распадается на систему 2×2 и одно обратное вычисление. В измеренной
точке три новых значения \(q\) побитно совпали с общим вариантом.

## Обновление по малому приращению

Между отсчётами хранятся \(E=e^x\), \(S=\sinh y\) и \(C=\cosh y\).
После ньютоновской поправки CORDIC получает непосредственно \(\Delta x\) и
\(\Delta y\); состояние обновляется по формулам сложения гиперболических функций.
Проверка диапазона выполняется в каждом вызове. Раз в {INCREMENTAL_REFRESH_PERIOD}
отсчёта состояние полностью восстанавливается по абсолютным аргументам.

## Измеренные такты

| Часть | Библиотечные функции | CORDIC | Ускорение |
|---|---:|---:|---:|
| Нелинейные токи и производные | {values['nonlinear_libm_cycles']} | {values['nonlinear_cordic_cycles']} | {speedup_nonlinear:.2f}× |
| Одна поправка Ньютона | {values['newton_libm_cycles']} | {values['newton_cordic_cycles']} | {speedup_newton:.2f}× |
| Специализированная поправка Ньютона | {values['newton_libm_cycles']} | {values['newton_cordic_optimized_cycles']} | {speedup_newton_optimized:.2f}× |
| Малое приращение с учётом восстановления | {values['newton_libm_cycles']} | {incremental_average:.1f} | {speedup_newton_incremental:.2f}× |
| Одна поправка Галлея | {values['halley_libm_cycles']} | {values['halley_cordic_cycles']} | {speedup_halley:.2f}× |

Отдельно: нелинейности вместе со сборкой невязки и матрицы —
{values['form_cordic_cycles']} тактов; решение 3×3 — {values['solve_3x3_cycles']} такта; один вызов CORDIC
`cosh+sinh` — {values['cordic_cosh_sinh_cycles']} такта.

![Состав стоимости](figures/cycle_parts.png)

## Точность в рабочей точке

| Величина | `libm` | CORDIC | Абс. ошибка | Отн. ошибка |
|---|---:|---:|---:|---:|
{rows}

Максимальная относительная ошибка ненулевых величин: {maximum_relative:.3e}.
Для аргумента 0,75 CORDIC вернул `cosh={cordic_cosh:.9f}` и
`sinh={cordic_sinh:.9f}`.

## Сравнение с бюджетом

| Повышение | Частота, Гц | Бюджет, тактов | Ньютон, % | Галлей, % |
|---:|---:|---:|---:|---:|
{budget_rows}

![Использование бюджета](figures/budget_usage.png)

## Вывод

Приведение аргумента решает задачу диапазона и даёт около шести значащих
цифр совпадения с `libm`. Малое приращение вместе с проверкой диапазона и
амортизированным восстановлением занимает {newton_8_percent:.1f}% бюджета 8×.
Один Q3 помещается, но два таких каскада уже требуют около
{2.0 * incremental_average:.1f} такта. Следующий крупный резерв — одно общее
приближение вместо двух вызовов CORDIC либо разные частоты расчёта каскадов.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    values = load_values()
    plot_cycle_parts(values)
    plot_budgets(values)
    write_report(values)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
