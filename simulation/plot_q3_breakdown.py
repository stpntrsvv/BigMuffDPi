"""Строит отчёт разделения физики и обслуживающей арифметики Q3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "simulation" / "raw" / "q3_breakdown" / "board.txt"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_breakdown"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"

COMPONENTS = (
    ("nonlinear", "Токи и производные"),
    ("residual", "Невязка схемы"),
    ("newton", "Поправка Ньютона"),
    ("range_check", "Проверки диапазона"),
    ("cordic", "Два вызова CORDIC"),
    ("state_update", "Обновление состояния"),
    ("unattributed", "Организация функции"),
)


def load_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name] = int(value)
    for name, _ in COMPONENTS:
        key = f"breakdown_{name}_cycles"
        if key not in values:
            raise ValueError(f"Нет поля {key}")
    return values


def plot_components(values: dict[str, int]) -> None:
    labels = [title for _, title in COMPONENTS]
    cycles = [values[f"breakdown_{name}_cycles"] for name, _ in COMPONENTS]
    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(labels, cycles)
    bars[-1].set_color("tab:red")
    axis.set_ylabel("Тактов")
    axis.set_title("Q3: состав одной поправки малого приращения")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.3)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "components.png", dpi=160)
    plt.close(figure)


def plot_groups(values: dict[str, int]) -> None:
    physics = sum(values[f"breakdown_{name}_cycles"] for name in (
        "nonlinear", "residual", "cordic", "state_update"
    ))
    numerical = sum(values[f"breakdown_{name}_cycles"] for name in (
        "newton", "range_check"
    ))
    implementation = values["breakdown_unattributed_cycles"]
    labels = ("Физическая модель", "Численный метод", "Организация функции")
    cycles = (physics, numerical, implementation)
    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(labels, cycles, color=("tab:green", "tab:blue", "tab:red"))
    axis.set_ylabel("Тактов")
    axis.set_title("Укрупнённое разделение стоимости Q3")
    axis.grid(True, axis="y", alpha=0.3)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "groups.png", dpi=160)
    plt.close(figure)


def write_report(values: dict[str, int]) -> None:
    total = values["newton_cordic_incremental_cycles"]
    measured = values["breakdown_measured_sum_cycles"]
    remainder = values["breakdown_unattributed_cycles"]
    rows = "\n".join(
        f"| {title} | {values[f'breakdown_{name}_cycles']} | "
        f"{100 * values[f'breakdown_{name}_cycles'] / total:.1f}% |"
        for name, title in COMPONENTS
    )
    physics = sum(values[f"breakdown_{name}_cycles"] for name in (
        "nonlinear", "residual", "cordic", "state_update"
    ))
    numerical = sum(values[f"breakdown_{name}_cycles"] for name in (
        "newton", "range_check"
    ))
    report = f"""# Q3: разделение физики и арифметики

## Метод

Внутри одной поправки расставлены метки счётчика DWT. Цена пары меток —
{values['breakdown_marker_cycles']} такт — вычтена из каждого участка. Выполнено
{values['repetitions']} повторений; второй запуск дал те же значения.

Полная изолированная поправка измерена отдельно и занимает **{total} тактов**.
Размеченные участки суммарно дали {measured} такта. Остаток {remainder} тактов
выделен отдельной строкой.

| Участок | Тактов | Доля полной функции |
|---|---:|---:|
{rows}

![Подробные части](figures/components.png)

## Укрупнённое разделение

- физические токи, топология, CORDIC и нелинейное состояние — {physics} тактов;
- поправка Ньютона и проверки — {numerical} тактов;
- организация скомпилированной функции — {remainder} тактов.

![Укрупнённые группы](figures/groups.png)

## Что входит в остаток

Остаток нельзя трактовать как одну инструкцию. Размеченный цикл позволяет
компилятору удерживать часть констант между повторениями, тогда как настоящая
функция вызывается заново. В её машинном коде видны сохранение двенадцати
регистров FPU, загрузки состояния и многочисленных констант, возврат регистров,
ветвления и более тяжёлое распределение временных значений.

Поэтому {remainder} тактов — это верхнеуровневая цена формы горячей функции и
движения данных, а не точная стоимость только памяти. Для следующего шага надо
сравнить нынешнюю функцию с вариантом, встроенным непосредственно в цикл
обработки, где константы и часть состояния смогут оставаться в регистрах.

## Вывод

Физику преждевременно резать. Непосредственные физические вычисления занимают
около {100 * physics / total:.1f}% полной функции, а Ньютон с проверками —
{100 * numerical / total:.1f}%. Наибольшая отдельная категория — организация
скомпилированного ядра, {100 * remainder / total:.1f}%. Следующий опыт должен
проверять встраивание и уплотнение данных, не меняя уравнения схемы.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    values = load_values()
    plot_components(values)
    plot_groups(values)
    write_report(values)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
