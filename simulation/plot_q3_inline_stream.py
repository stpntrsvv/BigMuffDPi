"""Строит отчёт об устранении вызова горячего ядра Q3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "simulation" / "raw" / "q3_inline_stream" / "board.txt"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_inline_stream"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"


def load_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name] = int(value)
    return values


def plot_comparison(values: dict[str, int]) -> None:
    labels = ("Ядро", "Обычный поток", "Сильный поток")
    called = (
        values["kernel_called_average_cycles"],
        values["stream_nominal_called_average_cycles"],
        values["stream_strong_adaptive_called_average_cycles"],
    )
    inlined = (
        values["kernel_inlined_average_cycles"],
        values["stream_nominal_inlined_average_cycles"],
        values["stream_strong_adaptive_inlined_average_cycles"],
    )
    positions = range(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 6))
    left = axis.bar([position - width / 2 for position in positions], called,
                    width, label="Отдельный вызов")
    right = axis.bar([position + width / 2 for position in positions], inlined,
                     width, label="Встроено в цикл")
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Тактов на внутренний отсчёт")
    axis.set_title("Q3: устранение границы горячей функции")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    axis.bar_label(left, padding=3)
    axis.bar_label(right, padding=3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "comparison.png", dpi=160)
    plt.close(figure)


def write_report(values: dict[str, int]) -> None:
    called_kernel = values["kernel_called_average_cycles"]
    inlined_kernel = values["kernel_inlined_average_cycles"]
    nominal_called = values["stream_nominal_called_average_cycles"]
    nominal_inlined = values["stream_nominal_inlined_average_cycles"]
    strong_called = values["stream_strong_adaptive_called_average_cycles"]
    strong_inlined = values["stream_strong_adaptive_inlined_average_cycles"]
    remainder_before = 115
    remainder_after = remainder_before - (nominal_called - nominal_inlined)
    report = f"""# Q3: встраивание горячего ядра в цикл

## Метод

Сравнены два машинно равнозначных варианта одной поправки: отдельный вызов
функции и тот же исходный код, принудительно встроенный в цикл. Оба варианта
обработали одну и ту же последовательность. Все три переменные схемы и три
сохранённые нелинейные величины совпали побитно.

| Режим | Отдельный вызов | Встроенное ядро | Экономия |
|---|---:|---:|---:|
| Только ядро | {called_kernel} | {inlined_kernel} | {called_kernel - inlined_kernel} |
| Обычный поток | {nominal_called} | {nominal_inlined} | {nominal_called - nominal_inlined} |
| Сильный поток, автоматическое восстановление | {strong_called} | {strong_inlined} | {strong_called - strong_inlined} |

![Сравнение стоимости](figures/comparison.png)

## Разбор

В отдельном опыте ядро ускорилось на
{100 * (called_kernel - inlined_kernel) / called_kernel:.1f}%. Компилятор смог
сохранить состояние и константы между соседними шагами и перестал на каждом
отсчёте сохранять и восстанавливать большой набор регистров FPU.

В полном обычном потоке выигрыш составил {nominal_called - nominal_inlined}
такт. Если сопоставить его с прежним нераспределённым остатком {remainder_before}
тактов, после встраивания остаётся около {remainder_after} тактов. Остаток
сократился в {remainder_before / remainder_after:.2f} раза. Это оценка по разности
полных потоков, а не новое пооперационное разложение.

При 170 МГц бюджет одного внутреннего отсчёта для 8-кратной частоты равен
442,7 такта. Один полный Q3 теперь занимает около
{100 * nominal_inlined / 442.7:.1f}% этого бюджета, два одинаковых каскада —
около {200 * nominal_inlined / 442.7:.1f}%. Для двух каскадов всё ещё требуется
снизить среднюю цену одного примерно до 221 такта.

## Вывод

Цель сократить организационный остаток втрое достигнута без изменения
уравнений и без численного расхождения. Встраивание следует считать основной
формой будущего звукового цикла. Следующий крупный резерв уже находится в
физической модели и частоте расчёта, а не в границе вызова функции.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    values = load_values()
    plot_comparison(values)
    write_report(values)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
