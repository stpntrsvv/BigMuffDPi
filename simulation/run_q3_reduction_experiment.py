"""Проверка сокращения Q3 и одной поправки Ньютона/Галлея."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q3_model import COLLECTOR, DIODE_NODE, OUTPUT, Q3Parameters, simulate_transient
from q3_reduced_model import prepare_reduction, simulate_reduced


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "q3_reduction"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "q3_reduction"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
FACTORS = (8, 16, 32)
METHODS = ("newton", "halley")
METHOD_RUSSIAN = {"newton": "Ньютон", "halley": "Галлей"}


@dataclass(frozen=True)
class Summary:
    method: str
    factor: int
    sample_rate_hz: float
    maximum_output_error_v: float
    rms_output_error_v: float
    maximum_diode_error_v: float
    rms_diode_error_v: float
    maximum_physical_residual_a: float
    nonlinear_evaluations_per_step: int
    linear_solves_3x3_per_step: int


def summarize(method: str, factor: int, reference, approximation) -> Summary:
    output_error = approximation.node_v[:, OUTPUT] - reference.node_v[:, OUTPUT]
    reference_diode = (
        reference.node_v[:, DIODE_NODE] - reference.node_v[:, COLLECTOR]
    )
    approximation_diode = (
        approximation.node_v[:, DIODE_NODE] - approximation.node_v[:, COLLECTOR]
    )
    diode_error = approximation_diode - reference_diode
    return Summary(
        method,
        factor,
        48_000.0 * factor,
        float(np.max(np.abs(output_error))),
        float(np.sqrt(np.mean(output_error * output_error))),
        float(np.max(np.abs(diode_error))),
        float(np.sqrt(np.mean(diode_error * diode_error))),
        float(np.max(approximation.residual_a)),
        1,
        1 if method == "newton" else 2,
    )


def write_summary(rows: list[Summary]) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_trace(reference, newton, halley) -> None:
    table = np.column_stack(
        (
            reference.time_s,
            reference.input_v,
            reference.node_v[:, OUTPUT],
            newton.node_v[:, OUTPUT],
            halley.node_v[:, OUTPUT],
            reference.node_v[:, DIODE_NODE] - reference.node_v[:, COLLECTOR],
            newton.node_v[:, DIODE_NODE] - newton.node_v[:, COLLECTOR],
            halley.node_v[:, DIODE_NODE] - halley.node_v[:, COLLECTOR],
            newton.residual_a,
            halley.residual_a,
        )
    )
    np.savetxt(
        RAW_ROOT / "trace_16x.txt",
        table,
        header=(
            "time_s input_v output_reference_v output_newton_v output_halley_v "
            "diode_reference_v diode_newton_v diode_halley_v "
            "newton_residual_a halley_residual_a"
        ),
        comments="",
    )


def plot_accuracy(rows: list[Summary]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for method in METHODS:
        selected = [row for row in rows if row.method == method]
        factors = [row.factor for row in selected]
        axes[0].plot(
            factors,
            [row.maximum_output_error_v * 1.0e6 for row in selected],
            "o-",
            label=METHOD_RUSSIAN[method],
        )
        axes[1].plot(
            factors,
            [row.maximum_diode_error_v * 1.0e6 for row in selected],
            "o-",
            label=METHOD_RUSSIAN[method],
        )
    axes[0].set_ylabel("Максимальная ошибка выхода, мкВ")
    axes[0].set_title("Q3: одна численная поправка на внутренний отсчёт")
    axes[1].set_ylabel("Максимальная ошибка напряжения диодов, мкВ")
    axes[1].set_xlabel("Повышение частоты относительно 48 кГц")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xticks(FACTORS, labels=[str(value) for value in FACTORS])
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "accuracy_by_factor.png", dpi=160)
    plt.close(figure)


def plot_residual(rows: list[Summary]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for method in METHODS:
        selected = [row for row in rows if row.method == method]
        axis.plot(
            [row.factor for row in selected],
            [row.maximum_physical_residual_a for row in selected],
            "o-",
            label=METHOD_RUSSIAN[method],
        )
    axis.set_xticks(FACTORS, labels=[str(value) for value in FACTORS])
    axis.set_yscale("log")
    axis.set_xlabel("Повышение частоты относительно 48 кГц")
    axis.set_ylabel("Максимальная физическая невязка, А")
    axis.set_title("Невязка после одной поправки")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "residual_by_factor.png", dpi=160)
    plt.close(figure)


def plot_trace(reference, newton, halley) -> None:
    time_ms = reference.time_s * 1000.0
    mask = time_ms >= time_ms[-1] - 2.0
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for result, label, style in (
        (reference, "Полная сходимость", "-"),
        (newton, "Один шаг Ньютона", "--"),
        (halley, "Один шаг Галлея", ":"),
    ):
        axes[0].plot(
            time_ms[mask], result.node_v[mask, OUTPUT], style, label=label
        )
    axes[0].set_ylabel("Выход, В")
    axes[0].set_title("Q3 при 16×: установившийся сигнал")
    axes[0].legend()

    axes[1].semilogy(
        time_ms[mask],
        np.maximum(
            np.abs(newton.node_v[mask, OUTPUT] - reference.node_v[mask, OUTPUT]),
            1.0e-12,
        ),
        label="Ньютон",
    )
    axes[1].semilogy(
        time_ms[mask],
        np.maximum(
            np.abs(halley.node_v[mask, OUTPUT] - reference.node_v[mask, OUTPUT]),
            1.0e-12,
        ),
        label="Галлей",
    )
    axes[1].set_ylabel("Ошибка выхода, В")
    axes[1].legend()

    axes[2].semilogy(
        time_ms[mask], np.maximum(newton.residual_a[mask], 1.0e-18), label="Ньютон"
    )
    axes[2].semilogy(
        time_ms[mask], np.maximum(halley.residual_a[mask], 1.0e-18), label="Галлей"
    )
    axes[2].set_ylabel("Невязка, А")
    axes[2].set_xlabel("Время, мс")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "trace_16x.png", dpi=160)
    plt.close(figure)


def write_report(
    rows: list[Summary],
    equivalence_error_v: float,
    reduced_residual_a: float,
    matrix_rank: int,
    matrix_determinant: float,
) -> None:
    table = "\n".join(
        "| {method} | {factor}× | {out:.3f} | {rms:.3f} | {diode:.3f} | "
        "{residual:.3e} | {solves} |".format(
            method=METHOD_RUSSIAN[row.method],
            factor=row.factor,
            out=row.maximum_output_error_v * 1.0e6,
            rms=row.rms_output_error_v * 1.0e6,
            diode=row.maximum_diode_error_v * 1.0e6,
            residual=row.maximum_physical_residual_a,
            solves=row.linear_solves_3x3_per_step,
        )
        for row in rows
    )
    newton_32 = next(
        row for row in rows if row.method == "newton" and row.factor == 32
    )
    halley_16 = next(
        row for row in rows if row.method == "halley" and row.factor == 16
    )
    advantage = newton_32.maximum_output_error_v / halley_16.maximum_output_error_v
    report = rf"""# Сокращение Q3 и одна численная поправка

## Цель

Без приближений исключить линейные узлы каскада Q3, определить размер
оставшейся нелинейной системы и сравнить одну поправку Ньютона и Галлея при
8×, 16× и 32× относительно 48 кГц.

## Сокращённая система

Полные узловые уравнения представлены как

\[
\mathbf L\mathbf v-\mathbf r+\mathbf P\boldsymbol\varphi(\mathbf Q\mathbf v)=0.
\]

После исключения шести узлов остаются три нелинейных напряжения
\(\mathbf q=(v_{{BE}},v_{{BC}},v_D)^T\):

\[
\mathbf g(\mathbf q)=\mathbf q-\mathbf a+
\mathbf H\boldsymbol\varphi(\mathbf q)=0,
\quad
\mathbf H=\mathbf Q\mathbf L^{{-1}}\mathbf P.
\]

При 16× матрица \(\mathbf H\) имеет ранг **{matrix_rank}** и определитель
**{matrix_determinant:.6e}**. Следовательно, линейной зависимости, позволяющей
точно превратить систему в одно или два уравнения, нет. Дальнейшее сокращение
потребует уже физического приближения либо вложенного нелинейного решения.

Сокращённая модель с полной сходимостью сравнена с исходным шестимерным
решателем на 1 мс при шаге 0,5 мкс. Максимальная разница всех узлов —
**{equivalence_error_v:.3e} В**, максимальная физическая невязка сокращённого
решения — **{reduced_residual_a:.3e} А**. Разница ограничена прежде всего
более мягким порогом остановки исходного решателя.

## Одна поправка на отсчёт

Сигнал — синус 1 кГц с амплитудой 50 мВ, длительность 5 мс. Эталон для каждой
частоты — та же сокращённая модель с полной сходимостью. Поэтому таблица
измеряет ошибку неполного нелинейного решения, а не ошибку дискретизации по
времени.

| Метод | Частота | Макс. ошибка выхода, мкВ | СКО выхода, мкВ | Макс. ошибка диодов, мкВ | Макс. невязка, А | Решений 3×3 |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Точность по частоте](figures/accuracy_by_factor.png)

![Невязка по частоте](figures/residual_by_factor.png)

![Сигнал и ошибки при 16×](figures/trace_16x.png)

## Вывод

При 16× один многомерный шаг Галлея даёт максимальную ошибку выхода
**{halley_16.maximum_output_error_v * 1.0e6:.3f} мкВ**, а один шаг Ньютона при
32× — **{newton_32.maximum_output_error_v * 1.0e6:.3f} мкВ**. В этом опыте
вариант `16× + Галлей` точнее примерно в **{advantage:.1f} раза**.

Но это ещё не означает меньшую стоимость на STM32. Многомерной формуле Галлея
нужны одна оценка нелинейностей и два решения системы 3×3; Ньютону — та же
оценка и одно решение 3×3. Преимущество дешёвой второй производной полностью
реализовалось бы в скалярном уравнении, но точное сокращение Q3 скалярного
уравнения не дало.

## Следующий шаг

Проверить последовательные физические упрощения транзистора: сначала убрать
обратный ток перехода база–коллектор, затем сравнить упрощённую модель с полной
по сигналу, спектру и переходам. Параллельно нужно измерить на STM32 стоимость
одного и двух решений системы 3×3; без измерения тактов выбирать Ньютон или
Галлей рано.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    original = simulate_transient(duration_s=1.0e-3, step_s=0.5e-6)
    reduced = simulate_reduced("full", duration_s=1.0e-3, step_s=0.5e-6)
    equivalence_error_v = float(np.max(np.abs(original.node_v - reduced.node_v)))
    reduced_residual_a = float(np.max(reduced.residual_a))

    rows: list[Summary] = []
    traces: dict[tuple[int, str], object] = {}
    for factor in FACTORS:
        step_s = 1.0 / (48_000.0 * factor)
        reference = simulate_reduced("full", step_s=step_s)
        traces[(factor, "full")] = reference
        for method in METHODS:
            approximation = simulate_reduced(method, step_s=step_s)
            traces[(factor, method)] = approximation
            rows.append(summarize(method, factor, reference, approximation))

    reduction = prepare_reduction(Q3Parameters(), 1.0 / (48_000.0 * 16.0))
    matrix_rank = int(np.linalg.matrix_rank(reduction.influence_matrix))
    matrix_determinant = float(np.linalg.det(reduction.influence_matrix))

    write_summary(rows)
    save_trace(traces[(16, "full")], traces[(16, "newton")], traces[(16, "halley")])
    plot_accuracy(rows)
    plot_residual(rows)
    plot_trace(traces[(16, "full")], traces[(16, "newton")], traces[(16, "halley")])
    write_report(
        rows,
        equivalence_error_v,
        reduced_residual_a,
        matrix_rank,
        matrix_determinant,
    )
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
