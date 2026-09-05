"""Строит отчёт о компиляторе и блочном измерении Q3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "q3_asm_compiler" / "board.txt"
OUT = ROOT / "simulation" / "experiments" / "q3_asm_compiler"
FIGURES = OUT / "figures"


def load_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name] = int(value)
    return values


def plot(values: dict[str, int]) -> None:
    labels = ("GCC 7\nстрого", "GCC 12\nстрого", "GCC 12\nбыстро")
    nominal = (
        values["gcc7_strict_nominal_cycles"],
        values["gcc12_strict_nominal_cycles"],
        values["gcc12_fast_nominal_cycles"],
    )
    strong = (
        values["gcc7_strict_strong_cycles"],
        values["gcc12_strict_strong_cycles"],
        values["gcc12_fast_strong_cycles"],
    )
    positions = range(3)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 6))
    left = axis.bar([x - width / 2 for x in positions], nominal, width,
                    label="Обычный сигнал")
    right = axis.bar([x + width / 2 for x in positions], strong, width,
                     label="Сильный сигнал")
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Тактов на внутренний отсчёт")
    axis.set_title("Q3: компилятор и порядок арифметики")
    axis.set_ylim(0, 220)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    axis.bar_label(left, padding=3)
    axis.bar_label(right, padding=3)
    figure.tight_layout()
    figure.savefig(FIGURES / "comparison.png", dpi=160)
    plt.close(figure)


def report(values: dict[str, int]) -> None:
    strict_nominal = values["gcc12_strict_nominal_cycles"]
    strict_strong = values["gcc12_strict_strong_cycles"]
    fast_nominal = values["gcc12_fast_nominal_cycles"]
    fast_strong = values["gcc12_fast_strong_cycles"]
    text = f"""# Q3: проверка машинного кода и компилятора

## Правильная граница измерения

Чтение DWT вокруг каждой встроенной поправки оказалось недостаточной границей
для нового компилятора: чистая арифметика могла переноситься через метку. После
замены меток невстраиваемыми функциями цена одного шага выросла с ошибочных 65
до 143 тактов. Такой способ, в свою очередь, мешает удерживать состояние между
отсчётами.

Итоговый замер охватывает целый блок из {values['block_samples']} шагов одной
парой жёстких меток. Это соответствует будущему обработчику звукового блока и
не создаёт границу вызова на каждом отсчёте. Конечные состояния блочного и
поотсчётного вариантов совпали побитно.

| Сборка | Обычный сигнал | Сильный сигнал |
|---|---:|---:|
| GCC 7.2.1, строгая арифметика | {values['gcc7_strict_nominal_cycles']} | {values['gcc7_strict_strong_cycles']} |
| GCC 12.3.1, строгая арифметика | {strict_nominal} | {strict_strong} |
| GCC 12.3.1, `fast-math` | {fast_nominal} | {fast_strong} |

![Сравнение](figures/comparison.png)

## Что показал машинный код

Главный выигрыш создаёт не отдельная команда, а отсутствие границы между
отсчётами: состояние и часть констант остаются в регистрах FPU. GCC 12 в строгом
режиме экономит ещё 5–7 тактов относительно GCC 7. Ручное удаление трёх записей
и трёх чтений линейной части оказалось регрессией из-за давления на регистры:
ядро подорожало со 129 до 143 тактов, сильный поток — с 302 до 311.

Разрешение переупорядочивать суммы и произведения даёт ещё
{strict_nominal - fast_nominal}–{strict_strong - fast_strong} тактов. При этом
конечное состояние изменилось на 1–15 единиц младшего разряда `float`, поэтому
этот режим пока остаётся исследовательской верхней границей, а не основным.

Два строгих Q3 требуют {2 * strict_nominal}–{2 * strict_strong} тактов из
бюджета 442,7 такта при 8×. На остальную педаль остаётся примерно
{442.7 - 2 * strict_strong:.1f}–{442.7 - 2 * strict_nominal:.1f} такта.

## Вывод

Ручной ассемблер имеет смысл только для локального получения оставшихся
примерно 11–18 тактов: длинных цепочек умножений и сложений, деления Ньютона и
обслуживания двух обращений к CORDIC. Переписывать всё ядро на ассемблере сейчас
невыгодно. Основной вариант — GCC 12, строгая арифметика и блочная обработка;
агрессивную арифметику сначала надо сравнить с эталоном на длинных сигналах.
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = load_values()
    plot(values)
    report(values)
    print(f"Отчёт: {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
