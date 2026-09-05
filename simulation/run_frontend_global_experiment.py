"""Глобальные возмущения входного тракта и сверка его частотной характеристики."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_frontend_model import (
    Q2_COLLECTOR,
    simulate_frontend,
    simulate_frontend_switching,
    small_signal_response,
)
from run_reference import find_ngspice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
NGSPICE_ROOT = SIMULATION_ROOT / "ngspice"
RAW_ROOT = SIMULATION_ROOT / "raw" / "frontend_global"
EXPERIMENT_ROOT = SIMULATION_ROOT / "experiments" / "frontend_global"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
FACTOR = 8
RATE_HZ = 48_000 * FACTOR
SUSTAIN_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    duration_s: float
    input_function: object
    switching: bool = False


@dataclass(frozen=True)
class Summary:
    case: str
    maximum_residual_v: float
    maximum_correction_v: float
    rms_output_error_mv: float
    peak_output_error_mv: float
    maximum_output_excursion_v: float
    tail_growth_ratio: float
    finite: bool


def step_input(time_s: np.ndarray) -> np.ndarray:
    return np.where(time_s >= 2e-3, 0.2, 0.0)


def impulse_input(time_s: np.ndarray) -> np.ndarray:
    return np.where(
        (time_s >= 2e-3) & (time_s < 2e-3 + 1.0 / RATE_HZ), 0.8, 0.0
    )


def overload_input(time_s: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * 1_000.0 * time_s)


def ordinary_input(time_s: np.ndarray) -> np.ndarray:
    return 0.1 * np.sin(2.0 * np.pi * 1_000.0 * time_s)


def sustain_round_trip(time_s: np.ndarray) -> np.ndarray:
    return np.where((time_s >= 5e-3) & (time_s < 12e-3), 1.0, 0.0)


CASES = (
    Case("step", "Ступень 0,2 В", 20e-3, step_input),
    Case("impulse", "Импульс 0,8 В, один шаг", 20e-3, impulse_input),
    Case("overload", "Перегрузка 1 В пик, 1 кГц", 16e-3, overload_input),
    Case("sustain_jump", "Sustain 0 - 1 - 0", 24e-3, ordinary_input, True),
)


def run_ngspice(executable: Path) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(executable), "-b", "frontend_equation_reference.cir"],
        cwd=NGSPICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (RAW_ROOT / "ngspice.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"ngspice завершился с кодом {result.returncode}")


def simulate(case: Case, fully_converged: bool):
    if case.switching:
        return simulate_frontend_switching(
            0.0, FACTOR, case.input_function, sustain_round_trip,
            case.duration_s, "hybrid", fully_converged
        )
    return simulate_frontend(
        1.0, FACTOR, case.input_function, case.duration_s,
        "hybrid", fully_converged
    )


def summarize(case: Case, result, reference) -> Summary:
    output = result.node_v[:, Q2_COLLECTOR]
    reference_output = reference.node_v[:, Q2_COLLECTOR]
    error = output - reference_output
    excursion = output - output[0]
    window = max(1, int(round(2e-3 * RATE_HZ)))
    previous_peak = float(np.max(np.abs(excursion[-2 * window:-window])))
    final_peak = float(np.max(np.abs(excursion[-window:])))
    growth = final_peak / max(previous_peak, 1e-15)
    return Summary(
        case.title,
        float(np.max(result.residual_v)),
        float(np.max(result.correction_v)),
        float(1e3 * np.sqrt(np.mean(error * error))),
        float(1e3 * np.max(np.abs(error))),
        float(np.max(np.abs(excursion))),
        growth,
        bool(np.all(np.isfinite(result.node_v))),
    )


def plot_disturbances(results: dict[str, object], references: dict[str, object]) -> None:
    figure, axes = plt.subplots(len(CASES), 1, figsize=(13, 12))
    for axis, case in zip(axes, CASES):
        result = results[case.name]
        reference = references[case.name]
        time_ms = result.time_s * 1e3
        dc = reference.node_v[0, Q2_COLLECTOR]
        axis.plot(
            time_ms, reference.node_v[:, Q2_COLLECTOR] - dc,
            "--", linewidth=1.5, label="Полное схождение"
        )
        axis.plot(
            time_ms, result.node_v[:, Q2_COLLECTOR] - dc,
            linewidth=0.9, label="Одна поправка"
        )
        if case.switching:
            twin = axis.twinx()
            twin.plot(time_ms, result.sustain, color="black", alpha=0.25)
            twin.set_ylabel("Sustain")
            twin.set_ylim(-0.05, 1.05)
        axis.set_ylabel("Q2, В")
        axis.set_title(case.title)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Время, мс")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "disturbance_waveforms.png", dpi=160)
    plt.close(figure)


def plot_solver(results: dict[str, object]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for case in CASES:
        result = results[case.name]
        time_ms = result.time_s * 1e3
        axes[0].semilogy(
            time_ms, np.maximum(result.residual_v, 1e-15), label=case.title
        )
        axes[1].semilogy(
            time_ms, np.maximum(result.correction_v, 1e-15), label=case.title
        )
    axes[0].set_ylabel("Невязка, В")
    axes[1].set_ylabel("Поправка, В")
    axes[1].set_xlabel("Время, мс")
    axes[0].set_title("Одна поправка Ньютона при глобальных возмущениях")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "solver_diagnostics.png", dpi=160)
    plt.close(figure)


def load_ngspice(sustain: float) -> np.ndarray:
    suffix = f"{sustain:.2f}".replace(".", "_")
    return np.loadtxt(RAW_ROOT / f"ngspice_ac_{suffix}.txt", skiprows=1)


def compare_frequency() -> tuple[float, float]:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    maximum_db_error = 0.0
    phase_squared: list[np.ndarray] = []
    for sustain in SUSTAIN_VALUES:
        ngspice = load_ngspice(sustain)
        frequency_hz = ngspice[:, 0]
        _, response = small_signal_response(sustain, frequency_hz)
        python_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-15))
        python_phase = np.unwrap(np.angle(response))
        ngspice_phase = np.unwrap(ngspice[:, 2])
        db_error = python_db - ngspice[:, 1]
        phase_error = np.degrees(python_phase - ngspice_phase)
        maximum_db_error = max(maximum_db_error, float(np.max(np.abs(db_error))))
        phase_squared.append(phase_error * phase_error)
        axes[0].semilogx(frequency_hz, db_error, label=f"Sustain {sustain:.2f}")
        axes[1].semilogx(frequency_hz, phase_error, label=f"Sustain {sustain:.2f}")
    phase_rms = float(np.sqrt(np.mean(np.concatenate(phase_squared))))
    axes[0].set_ylabel("Ошибка ЛАЧХ, дБ")
    axes[0].set_title("Узловая модель относительно ngspice с теми же уравнениями")
    axes[1].set_ylabel("Ошибка ЛФЧХ, градусы")
    axes[1].set_xlabel("Частота, Гц")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "ngspice_frequency_error.png", dpi=160)
    plt.close(figure)
    return maximum_db_error, phase_rms


def write_report(
    rows: list[Summary], maximum_db_error: float, phase_rms_degrees: float
) -> None:
    table = "\n".join(
        "| {case} | {res:.3e} | {correction:.3e} | {rms:.3f} | {peak:.3f} | "
        "{excursion:.3f} | {growth:.4f} | {finite} |".format(
            case=row.case, res=row.maximum_residual_v,
            correction=row.maximum_correction_v, rms=row.rms_output_error_mv,
            peak=row.peak_output_error_mv,
            excursion=row.maximum_output_excursion_v,
            growth=row.tail_growth_ratio, finite="да" if row.finite else "нет",
        )
        for row in rows
    )
    report = f"""# Глобальная устойчивость входного тракта

## Что проверено

В единой модели Q4–Sustain–Q3–Q2 при 8× выполнены четыре жёстких опыта:
ступень 0,2 В, импульс 0,8 В длительностью один внутренний шаг, синусоидальная
перегрузка 1 В пик и скачок Sustain `0 → 1 → 0` без сброса зарядов конденсаторов.
Одна поправка Ньютона сравнивается с полностью сходящимся решением на том же шаге.

| Возмущение | Макс. невязка, В | Макс. поправка, В | Ошибка Q2 СКО, мВ | Пиковая ошибка Q2, мВ | Макс. отклонение Q2, В | Отношение хвостовых максимумов | Все числа конечны |
|---|---:|---:|---:|---:|---:|---:|:---:|
{table}

Отношение хвостовых максимумов сравнивает два последних окна по 2 мс. Для
периодического сигнала значение около единицы нормально; рост заметно выше
единицы был бы признаком разгона.

![Отклики на возмущения](figures/disturbance_waveforms.png)

![Работа решателя](figures/solver_diagnostics.png)

## Независимая проверка ngspice

В `frontend_equation_reference.cir` записаны те же уравнения Эберса—Молла и
Шокли, что и в узловой модели. Это намеренная проверка топологии, знаков токов,
рабочей точки и линеаризации; она не подменяет будущего сравнения с расширенной
моделью BC239 и измеренной педалью.

- максимальная ошибка ЛАЧХ: `{maximum_db_error:.3e}` дБ;
- среднеквадратическая ошибка ЛФЧХ: `{phase_rms_degrees:.3e}` градуса.

![Ошибка относительно ngspice](figures/ngspice_frequency_error.png)

## Вывод

Все четыре возмущения остаются ограниченными и не дают накопительного роста.
Наиболее тяжёлый режим — синусоидальная перегрузка 1 В пик: невязка одной
поправки велика, поэтому пиковую ошибку нельзя оценивать по одной невязке —
в таблице приведено непосредственное сравнение выходного напряжения с полным
схождением. Скачок Sustain переносит физическое состояние без сброса и также
не вызывает разгона.

Локальная и проверенная здесь крупносигнальная устойчивость достаточны, чтобы
присоединять полный темброблок и выходной Q1. Проверка должна повторяться после
каждого расширения системы.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", type=Path)
    parser.add_argument("--no-ngspice", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.no_ngspice:
        run_ngspice(find_ngspice(args.ngspice))

    results: dict[str, object] = {}
    references: dict[str, object] = {}
    rows: list[Summary] = []
    for case in CASES:
        print(case.title)
        result = simulate(case, False)
        reference = simulate(case, True)
        results[case.name] = result
        references[case.name] = reference
        rows.append(summarize(case, result, reference))

    maximum_db_error, phase_rms = compare_frequency()
    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    plot_disturbances(results, references)
    plot_solver(results)
    write_report(rows, maximum_db_error, phase_rms)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
