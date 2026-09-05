"""Строит отчёт длительного потокового опыта Q3 на STM32G474RE."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "simulation" / "raw" / "q3_stream_benchmark" / "board.txt"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_stream_benchmark"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"

MODES = (
    ("nominal", "50 мВ, период 32"),
    ("strong_r1", "1 В, период 1"),
    ("strong_r8", "1 В, период 8"),
    ("strong_r16", "1 В, период 16"),
    ("strong_r32", "1 В, период 32"),
    ("strong_adaptive", "1 В, автоматический"),
)


def load_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name] = int(value)
    for mode, _ in MODES:
        for suffix in (
            "steps", "fallback_count", "large_step_count",
            "periodic_refresh_count", "average_cycles", "maximum_cycles",
            "maximum_cache_error_ppb", "nonfinite_count",
        ):
            name = f"stream_{mode}_{suffix}"
            if name not in values:
                raise ValueError(f"В журнале платы отсутствует поле {name}")
    return values


def plot_cycles(values: dict[str, int]) -> None:
    labels = [title for _, title in MODES]
    average = [values[f"stream_{mode}_average_cycles"] for mode, _ in MODES]
    maximum = [values[f"stream_{mode}_maximum_cycles"] for mode, _ in MODES]
    budget_8 = values["core_hz"] / (48_000 * 8)
    positions = list(range(len(MODES)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar([x - width / 2 for x in positions], average, width, label="Среднее")
    axis.bar([x + width / 2 for x in positions], maximum, width, label="Наибольшее")
    axis.axhline(budget_8, color="black", linestyle="--", label="Бюджет 8×")
    axis.set_xticks(positions, labels=labels, rotation=18, ha="right")
    axis.set_ylabel("Тактов на внутренний отсчёт")
    axis.set_title("Потоковая стоимость Q3 на STM32G474RE")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "cycle_cost.png", dpi=160)
    plt.close(figure)


def plot_error(values: dict[str, int]) -> None:
    modes = MODES[1:]
    labels = [title.replace("1 В, ", "") for _, title in modes]
    error_ppm = [
        values[f"stream_{mode}_maximum_cache_error_ppb"] / 1_000.0
        for mode, _ in modes
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(labels, error_ppm)
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_ylabel("Наибольшая относительная ошибка состояния, млн⁻¹")
    axis.set_title("Сильный вход 1 В: влияние периода восстановления")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "state_error.png", dpi=160)
    plt.close(figure)


def write_report(values: dict[str, int]) -> None:
    budget_8 = values["core_hz"] / (48_000 * 8)
    budget_16 = values["core_hz"] / (48_000 * 16)
    rows = []
    for mode, title in MODES:
        prefix = f"stream_{mode}"
        error = values[f"{prefix}_maximum_cache_error_ppb"] / 1.0e9
        rows.append(
            f"| {title} | {values[f'{prefix}_average_cycles']} | "
            f"{values[f'{prefix}_maximum_cycles']} | {error:.6g} | "
            f"{values[f'{prefix}_periodic_refresh_count']} | "
            f"{values[f'{prefix}_fallback_count']} | "
            f"{values[f'{prefix}_nonfinite_count']} |"
        )
    table = "\n".join(rows)
    nominal_cycles = values["stream_nominal_average_cycles"]
    adaptive_cycles = values["stream_strong_adaptive_average_cycles"]
    report = f"""# Q3: длительный поток на STM32G474RE

## Постановка

Из полной модели сформированы периодические последовательности линейной части
Q3 при 16×, синусе 1 кГц и входах 50 мВ и 1 В. Перед записью последовательности
схема прошла 30 периодов установления. Каждый период из 768 внутренних отсчётов
повторён на плате 1000 раз: по **768 000 поправок**, что соответствует одной
секунде внутреннего потока.

В измеряемый участок входят поправка Ньютона, проверка диапазона, выбор периода
и полное восстановление состояния. Контрольный абсолютный пересчёт для оценки
ошибки выполняется после измеряемого участка.

## Результаты

| Режим | Среднее, тактов | Наибольшее, тактов | Ошибка состояния | Восстановлений | Выходов за диапазон | Нечисловых результатов |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Стоимость потока](figures/cycle_cost.png)

![Ошибка сохранённого состояния](figures/state_error.png)

## Автоматический период

Если модуль безразмерной поправки транзистора или диодов превышает 0,25,
на следующие 32 шага включается восстановление раз в 8 шагов. В остальное
время используется период 32. На сильном входе это дало 36 000 восстановлений
вместо 96 000 у постоянного периода 8, при ошибке состояния
{values['stream_strong_adaptive_maximum_cache_error_ppb'] / 1e9:.6g}.

## Временной бюджет

- бюджет 8×: {budget_8:.1f} такта;
- бюджет 16×: {budget_16:.1f} такта;
- обычный вход: {nominal_cycles} такт, {100 * nominal_cycles / budget_8:.1f}% бюджета 8×;
- сильный вход с автоматическим периодом: {adaptive_cycles} такта,
  {100 * adaptive_cycles / budget_8:.1f}% бюджета 8×.

Среднее одного Q3 помещается в 8×, но не в 16×. Отдельные шаги восстановления
дороже бюджета одного внутреннего отсчёта; при обработке звукового блока это
допустимо только за счёт более дешёвых соседних шагов. Два каскада по-прежнему
не помещаются в общий бюджет 8×.

## Вывод

Поток подтвердил устойчивость малого приращения на обычном входе и показал,
что одной проверки диапазона CORDIC недостаточно для сильного сигнала.
Постоянный период 32 при 1 В непригоден. Автоматическое переключение удерживает
ошибку около {values['stream_strong_adaptive_maximum_cache_error_ppb'] / 1e3:.1f}
миллионных долей без нечисловых результатов и выходов за диапазон.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    values = load_values()
    plot_cycles(values)
    plot_error(values)
    write_report(values)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
