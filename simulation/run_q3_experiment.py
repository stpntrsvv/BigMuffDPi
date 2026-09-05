"""Сопоставление полной узловой модели Q3 на Python с ngspice."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q3_model import NODE_NAMES, operating_point, simulate_transient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
NGSPICE_ROOT = SIMULATION_ROOT / "ngspice"
RAW_ROOT = SIMULATION_ROOT / "raw" / "q3_reference"
EXPERIMENT_ROOT = SIMULATION_ROOT / "experiments" / "q3_reference"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"


def find_ngspice(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("NGSPICE_EXE"):
        candidates.append(Path(os.environ["NGSPICE_EXE"]))
    for name in ("ngspice_con", "ngspice"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))
    candidates.append(
        PROJECT_ROOT.parent
        / ".tools"
        / "ngspice-47"
        / "Spice64"
        / "bin"
        / "ngspice_con.exe"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("ngspice не найден; задайте --ngspice или NGSPICE_EXE")


def run_ngspice(executable: Path) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(executable), "-b", "q3_reference.cir"],
        cwd=NGSPICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = result.stdout + result.stderr
    (RAW_ROOT / "ngspice.log").write_text(log, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"ngspice завершился с кодом {result.returncode}; "
            f"смотрите {RAW_ROOT / 'ngspice.log'}"
        )


def load_table(path: Path, columns: int) -> np.ndarray:
    table = np.loadtxt(path, skiprows=1)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] != columns:
        raise ValueError(f"{path}: ожидалось {columns} столбцов, есть {table.shape[1]}")
    return table


def load_ngspice_operating_point() -> dict[str, float]:
    text = (RAW_ROOT / "operating_point.txt").read_text(encoding="utf-8")
    pattern = re.compile(r"v\(([^)]+)\)\s*=\s*([-+0-9.eE]+)")
    values = {match.group(1): float(match.group(2)) for match in pattern.finditer(text)}
    missing = set(NODE_NAMES) - values.keys()
    if missing:
        raise ValueError(f"В рабочей точке ngspice нет узлов: {sorted(missing)}")
    return values


def save_python_trace(result) -> None:
    table = np.column_stack(
        (
            result.time_s,
            result.input_v,
            result.node_v,
            result.iterations,
            result.residual_a,
        )
    )
    header = "time_s input_v " + " ".join(NODE_NAMES) + " iterations residual_a"
    np.savetxt(RAW_ROOT / "transient_python.txt", table, header=header, comments="")


def interpolate_python(result, time_s: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(time_s, result.time_s, result.node_v[:, column]) for column in range(6)]
    )


def plot_frequency_response() -> None:
    data = load_table(RAW_ROOT / "ac.txt", 5)
    frequency_hz = data[:, 0]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].semilogx(frequency_hz, data[:, 1], label="Вход")
    axes[0].semilogx(frequency_hz, data[:, 2], label="Коллектор Q3")
    axes[0].semilogx(frequency_hz, data[:, 3], label="После C13 и нагрузки")
    axes[0].set_ylabel("Амплитуда, дБ относительно 1 В")
    axes[0].set_title("Изолированный каскад Q3: малосигнальная характеристика")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].semilogx(frequency_hz, np.degrees(np.unwrap(data[:, 4])))
    axes[1].set_xlabel("Частота, Гц")
    axes[1].set_ylabel("Фаза выхода, градусы")
    axes[1].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "frequency_response.png", dpi=160)
    plt.close(figure)


def plot_waveform_comparison(
    ngspice: np.ndarray, python_at_ngspice: np.ndarray
) -> None:
    time_ms = ngspice[:, 0] * 1000.0
    mask = time_ms >= time_ms.max() - 2.0
    ng_nodes = ngspice[:, 2:]
    quantities = (
        (1, "База", "В"),
        (2, "Коллектор", "В"),
        (None, "Напряжение диодной пары", "В"),
        (5, "Выход после C13", "В"),
    )
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for axis, (column, title, unit) in zip(axes, quantities):
        if column is None:
            ng_values = ng_nodes[:, 4] - ng_nodes[:, 2]
            py_values = python_at_ngspice[:, 4] - python_at_ngspice[:, 2]
        else:
            ng_values = ng_nodes[:, column]
            py_values = python_at_ngspice[:, column]
        axis.plot(time_ms[mask], ng_values[mask], label="ngspice", linewidth=1.8)
        axis.plot(
            time_ms[mask],
            py_values[mask],
            "--",
            label="Узловой решатель Python",
            linewidth=1.1,
        )
        axis.set_ylabel(f"{title}\n{unit}")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Q3: сопоставление временных сигналов")
    axes[-1].set_xlabel("Время, мс")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "waveform_comparison.png", dpi=160)
    plt.close(figure)


def plot_errors_and_solver(
    ngspice: np.ndarray, python_at_ngspice: np.ndarray, python_result
) -> None:
    time_ms = ngspice[:, 0] * 1000.0
    error = np.abs(python_at_ngspice - ngspice[:, 2:])
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for column, name in enumerate(NODE_NAMES):
        axes[0].semilogy(time_ms, np.maximum(error[:, column], 1.0e-15), label=name)
    axes[0].set_xlabel("Время, мс")
    axes[0].set_ylabel("Абсолютная ошибка, В")
    axes[0].set_title("Разница Python и ngspice")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(ncol=3)

    mask = python_result.time_s >= 5.0e-3
    axes[1].plot(
        python_result.time_s[mask] * 1000.0,
        python_result.iterations[mask],
        label="Итерации Ньютона",
    )
    residual_axis = axes[1].twinx()
    residual_axis.semilogy(
        python_result.time_s[mask] * 1000.0,
        np.maximum(python_result.residual_a[mask], 1.0e-18),
        color="tab:red",
        alpha=0.75,
        label="Невязка",
    )
    axes[1].set_xlabel("Время, мс")
    axes[1].set_ylabel("Число итераций")
    residual_axis.set_ylabel("Невязка, А")
    axes[1].grid(True, alpha=0.3)
    lines = axes[1].get_lines() + residual_axis.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines])
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "errors_and_solver.png", dpi=160)
    plt.close(figure)


def write_report(
    ngspice: np.ndarray,
    python_at_ngspice: np.ndarray,
    python_result,
    ngspice_op: dict[str, float],
) -> None:
    error = python_at_ngspice - ngspice[:, 2:]
    maximum_error = np.max(np.abs(error), axis=0)
    rms_error = np.sqrt(np.mean(error * error, axis=0))
    python_op = operating_point().voltage_v
    op_error = python_op - np.array([ngspice_op[name] for name in NODE_NAMES])
    active = python_result.time_s >= 5.0e-3

    rows_op = "\n".join(
        f"| `{name}` | {ngspice_op[name]:.9f} | {python_op[index]:.9f} | {op_error[index]:.3e} |"
        for index, name in enumerate(NODE_NAMES)
    )
    rows_transient = "\n".join(
        f"| `{name}` | {maximum_error[index]:.3e} | {rms_error[index]:.3e} |"
        for index, name in enumerate(NODE_NAMES)
    )
    report = f"""# Полная узловая модель каскада Q3

## Цель

Проверить независимый узловой решатель Python сравнением с ngspice. Оба
решателя используют одинаковые явно записанные уравнения Эберса—Молла и
встречно-параллельной диодной пары, но имеют независимые реализации матрицы,
якобиана и интегрирования.

## Состав каскада

- C5 = 100 нФ и R19 = 10 кОм на входе;
- R20 = 100 кОм;
- R18 = 10 кОм;
- R21 = 150 Ом;
- R17 = 470 кОм и C12 = 470 пФ в обратной связи;
- C6 = 1 мкФ и точная гиперболическая диодная пара;
- C13 = 100 нФ и линейный эквивалент нагрузки 110 кОм.

Вход переходного расчёта: синус 1 кГц, амплитуда 50 мВ. Шаг решателя Python —
0,5 мкс, метод — обратный Эйлер, на каждом шаге выполняется полная сходимость
Ньютона с аналитическим якобианом и поиском длины шага.

## Рабочая точка

| Узел | ngspice, В | Python, В | Разница, В |
|---|---:|---:|---:|
{rows_op}

## Переходный расчёт

| Узел | Максимальная ошибка, В | СКО ошибки, В |
|---|---:|---:|
{rows_transient}

Максимальное число итераций Ньютона после 5 мс:
**{int(np.max(python_result.iterations[active]))}**.

Максимальная невязка Python после 5 мс:
**{float(np.max(python_result.residual_a[active])):.3e} А**.

![Сопоставление сигналов](figures/waveform_comparison.png)

![Ошибки и работа решателя](figures/errors_and_solver.png)

## Частотная характеристика ngspice

![Частотная характеристика](figures/frequency_response.png)

## Ограничения опыта

- нагрузка следующего транзистора заменена линейными R12 + R16 = 110 кОм;
- паразитные ёмкости, сопротивления выводов и эффект Эрли не включены;
- ngspice применяет внутренний адаптивный шаг Gear первого порядка, Python —
  постоянный шаг обратного Эйлера; поэтому временные решения не обязаны
  совпадать до машинной точности;
- параметры полупроводников пока исследовательские, а не измеренные.

## Следующий шаг

Линейные узлы уже исключены: результат и сравнение одной поправки Ньютона и
Галлея приведены в [следующем опыте](../q3_reduction/report.md).
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", type=Path)
    parser.add_argument("--no-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    arguments = parse_arguments()
    if not arguments.no_run:
        run_ngspice(find_ngspice(arguments.ngspice))

    ngspice_transient = load_table(RAW_ROOT / "transient_ngspice.txt", 8)
    python_result = simulate_transient()
    save_python_trace(python_result)
    python_at_ngspice = interpolate_python(python_result, ngspice_transient[:, 0])
    ngspice_op = load_ngspice_operating_point()

    plot_frequency_response()
    plot_waveform_comparison(ngspice_transient, python_at_ngspice)
    plot_errors_and_solver(ngspice_transient, python_at_ngspice, python_result)
    write_report(
        ngspice_transient,
        python_at_ngspice,
        python_result,
        ngspice_op,
    )
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
