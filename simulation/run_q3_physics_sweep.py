"""Сравнивает физику Q3 и частоту расчёта с высокочастотным эталоном."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q3_model import OUTPUT
from q3_physics_model import Physics, simulate_physics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "q3_physics_sweep"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_physics_sweep"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
FACTORS = (4, 8, 16, 32)
REFERENCE_FACTOR = 64
DURATION_S = 12.0e-3
WARMUP_S = 4.0e-3
PHYSICS: tuple[Physics, ...] = ("full", "bc_saturated", "bjt_linear")
NAMES = {
    "full": "Полный BJT",
    "bc_saturated": "Запертый B–C",
    "bjt_linear": "Линейный BJT",
}
CORE_HZ = 170_000_000.0
FULL_CYCLES = (183, 191)
SCALAR_CYCLES = (97, 98)


@dataclass(frozen=True)
class Summary:
    signal: str
    physics: str
    factor: int
    maximum_error_v: float
    rms_error_v: float
    relative_error_db: float
    high_band_error_db: float
    model_rms_error_v: float
    maximum_residual_v: float


def nominal_input(time_s: np.ndarray) -> np.ndarray:
    return 50.0e-3 * np.sin(2.0 * np.pi * 1_000.0 * time_s)


def strong_input(time_s: np.ndarray) -> np.ndarray:
    raw = (
        np.sin(2.0 * np.pi * 997.0 * time_s)
        + 0.55 * np.sin(2.0 * np.pi * 3_217.0 * time_s + 0.37)
    )
    return raw * (1.0 / 1.55)


SIGNALS = (("50 мВ, 1 кГц", nominal_input), ("1 В, два тона", strong_input))


def _summary(
    signal: str,
    physics: Physics,
    factor: int,
    reference_output: np.ndarray,
    full_same_factor_output: np.ndarray,
    candidate,
) -> Summary:
    output = candidate.node_v[::factor, OUTPUT]
    start = int(round(WARMUP_S * 48_000.0))
    reference = reference_output[start:]
    output = output[start:]
    full_same_factor = full_same_factor_output[start:]
    error = output - reference
    reference_ac = reference - np.mean(reference)
    error_ac = error - np.mean(error)
    rms_reference = float(np.sqrt(np.mean(reference_ac * reference_ac)))
    rms_error = float(np.sqrt(np.mean(error_ac * error_ac)))
    relative_error_db = 20.0 * np.log10(max(rms_error, 1.0e-18) / rms_reference)

    window = np.hanning(len(error_ac))
    error_spectrum = np.fft.rfft(error_ac * window)
    reference_spectrum = np.fft.rfft(reference_ac * window)
    frequency_hz = np.fft.rfftfreq(len(error_ac), 1.0 / 48_000.0)
    high = frequency_hz >= 8_000.0
    high_error = float(np.linalg.norm(error_spectrum[high]))
    reference_energy = float(np.linalg.norm(reference_spectrum))
    high_band_error_db = 20.0 * np.log10(
        max(high_error, 1.0e-18) / reference_energy
    )
    model_error = output - full_same_factor
    return Summary(
        signal,
        physics,
        factor,
        float(np.max(np.abs(error))),
        rms_error,
        float(relative_error_db),
        float(high_band_error_db),
        float(np.sqrt(np.mean(model_error * model_error))),
        candidate.maximum_reduced_residual_v,
    )


def _plot(rows: list[Summary], field: str, ylabel: str, filename: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for axis, (signal, _) in zip(axes, SIGNALS, strict=True):
        for physics in PHYSICS:
            selected = [
                row for row in rows
                if row.signal == signal and row.physics == physics
            ]
            values = [getattr(row, field) for row in selected]
            axis.plot(FACTORS, values, "o-", label=NAMES[physics])
        axis.set_title(signal)
        axis.set_xticks(FACTORS)
        axis.set_xlabel("Повышение частоты")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend()
    figure.suptitle("Q3: частота расчёта и физическая модель")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / filename, dpi=160)
    plt.close(figure)


def _plot_budget() -> None:
    budget = CORE_HZ / (48_000.0 * 8.0)
    names = ("Два полных", "Полный + скалярный", "Два скалярных")
    nominal = (2 * FULL_CYCLES[0], FULL_CYCLES[0] + SCALAR_CYCLES[0], 2 * SCALAR_CYCLES[0])
    strong = (2 * FULL_CYCLES[1], FULL_CYCLES[1] + SCALAR_CYCLES[1], 2 * SCALAR_CYCLES[1])
    x = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.4), layout="constrained")
    bars_a = axis.bar(x - width / 2, nominal, width, label="Обычный сигнал")
    bars_b = axis.bar(x + width / 2, strong, width, label="Сильный сигнал")
    axis.axhline(budget, color="black", linestyle="--", label="Весь бюджет 8×")
    axis.bar_label(bars_a, padding=3)
    axis.bar_label(bars_b, padding=3)
    axis.set_xticks(x, names)
    axis.set_ylabel("Тактов на внутренний шаг")
    axis.set_title("Цена двух каскадов ограничения при 8×")
    axis.set_ylim(0, 480)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(FIGURE_ROOT / "cycle_budget.png", dpi=160)
    plt.close(figure)


def _write_report(rows: list[Summary]) -> None:
    table = "\n".join(
        "| {signal} | {model} | {factor}× | {maximum:.1f} | {rms:.1f} | "
        "{relative:.1f} | {high:.1f} | {model_error:.1f} |".format(
            signal=row.signal,
            model=NAMES[row.physics],
            factor=row.factor,
            maximum=row.maximum_error_v * 1.0e6,
            rms=row.rms_error_v * 1.0e6,
            relative=row.relative_error_db,
            high=row.high_band_error_db,
            model_error=row.model_rms_error_v * 1.0e6,
        )
        for row in rows
    )
    report = f"""# Q3: выбор физики и частоты расчёта

## Постановка

Эталон — полная модель Эберса—Молла с полной сходимостью Ньютона при 64×.
Кандидаты работают с одной поправкой на внутренний отсчёт при 4×, 8×, 16× и
32×. Выходы сравниваются в общих точках сетки 48 кГц после 4 мс установления.

Проверены три уровня физики:

- **полный BJT** — обе экспоненты транзистора и диодная пара;
- **запертый B–C** — обратный переход заменён током `-I_S` с нулевой
  производной; это соответствует уже используемому быстрому ядру STM32;
- **линейный BJT** — оба перехода транзистора линеаризованы около рабочей
  точки, нелинейной остаётся только диодная пара. После переноса линейной части
  в постоянную матрицу система сводится к одному скалярному уравнению.

Отрицательная относительная ошибка показывает, на сколько децибел ошибка ниже
полезного выходного сигнала. Последний столбец считает только ошибку в полосе
8–24 кГц и лучше обнаруживает спектральные продукты наложения.

| Сигнал | Физика | Частота | Макс. ошибка, мкВ | СКО, мкВ | Относительная, дБ | Ошибка 8–24 кГц, дБ | Вклад упрощения, СКО мкВ |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

![Относительная ошибка](figures/relative_error.png)

![Высокочастотная ошибка](figures/high_band_error.png)

## Измерение на STM32G474RE

Скалярное диодное ядро реализовано с одной поправкой Ньютона, одним обращением
к CORDIC и тем же рекуррентным обновлением гиперболических функций. Блочный
замер 16 периодов по 768 шагов дал **97 тактов** на обычном и **98 тактов** на
сильном сигнале. Полный текущий Q3 требует 183/191 такт. Ускорение составляет
1,89–1,95 раза; аварийных полных пересчётов не было.
Отдельная имитация арифметики `float32` и округления CORDIC дала конечные слова
`3189244338` и `1038296515`; они побитно совпали с двумя результатами платы.

| Состав двух каскадов при 8× | Обычный сигнал | Сильный сигнал | Остаток от 442,7 такта |
|---|---:|---:|---:|
| Два полных | 366 | 382 | 60,7–76,7 |
| Полный + скалярный | 280 | 289 | 153,7–162,7 |
| Два скалярных | 194 | 196 | 246,7–248,7 |

![Бюджет двух каскадов](figures/cycle_budget.png)

## Выбор

Рабочая исходная точка — **8×**. При 4× сильный двухтональный сигнал имеет лишь
около -30,5 дБ относительной точности и слишком большую невязку после одной
поправки. Переход 8× → 16× улучшает точность примерно на 7 дБ, но два даже
скалярных каскада съедают 194–196 тактов из всего бюджета 221,4 такта и не
оставляют места остальной педали.

На первом проходе разумно оставить **первый каскад полным, второй сделать
скалярным**. Такой вариант сохраняет одну транзисторную нелинейность и оставляет
154–163 такта на входной каскад, темброблок, выход, повышение/понижение частоты
и передачу звука. После сборки всей цепи его надо напрямую сравнить с двумя
полными каскадами; если транзисторная нелинейность второго каскада слышимо не
влияет, можно перейти к двум скалярным и получить ещё около 90 тактов запаса.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[Summary] = []
    for signal_name, input_function in SIGNALS:
        print(f"Эталон: {signal_name}")
        reference = simulate_physics(
            "full", REFERENCE_FACTOR, input_function, DURATION_S, fully_converged=True
        )
        reference_output = reference.node_v[::REFERENCE_FACTOR, OUTPUT]
        full_same_factor: dict[int, np.ndarray] = {}
        for factor in FACTORS:
            baseline = simulate_physics(
                "full", factor, input_function, DURATION_S, fully_converged=False
            )
            full_same_factor[factor] = baseline.node_v[::factor, OUTPUT]
        for physics in PHYSICS:
            for factor in FACTORS:
                print(f"  {NAMES[physics]}, {factor}x")
                candidate = simulate_physics(
                    physics, factor, input_function, DURATION_S, fully_converged=False
                )
                rows.append(
                    _summary(
                        signal_name,
                        physics,
                        factor,
                        reference_output,
                        full_same_factor[factor],
                        candidate,
                    )
                )

    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    _plot(rows, "relative_error_db", "Ошибка относительно сигнала, дБ", "relative_error.png")
    _plot(rows, "high_band_error_db", "Ошибка 8–24 кГц, дБ", "high_band_error.png")
    _plot_budget()
    _write_report(rows)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
