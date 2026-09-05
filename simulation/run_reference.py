"""Запуск эталонной схемы ngspice и построение основных графиков."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
NGSPICE_ROOT = SIMULATION_ROOT / "ngspice"
RAW_ROOT = SIMULATION_ROOT / "raw" / "ngspice_reference"
EXPERIMENT_ROOT = SIMULATION_ROOT / "experiments" / "reference_baseline"
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
    raise FileNotFoundError(
        "ngspice не найден. Укажите --ngspice или переменную NGSPICE_EXE."
    )


def run_ngspice(executable: Path) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(executable), "-b", "reference.cir"],
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
            f"ngspice завершился с кодом {result.returncode}. "
            f"Смотрите {RAW_ROOT / 'ngspice.log'}"
        )


def load_table(path: Path, expected_columns: int) -> np.ndarray:
    table = np.loadtxt(path, skiprows=1)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] != expected_columns:
        raise ValueError(
            f"В {path} ожидалось {expected_columns} столбцов, "
            f"получено {table.shape[1]}"
        )
    return table


def plot_frequency_response() -> None:
    data = load_table(RAW_ROOT / "ac.txt", expected_columns=8)
    frequency_hz = data[:, 0]
    labels = (
        "Вход",
        "После Q4",
        "После Q3",
        "После Q2",
        "После темброблока",
        "Выход",
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for column, label in enumerate(labels, start=1):
        axes[0].semilogx(frequency_hz, data[:, column], label=label)
    axes[0].set_ylabel("Амплитуда, дБ относительно 1 В")
    axes[0].set_title("Big Muff Pi V3: малосигнальная частотная характеристика")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(ncol=2)

    phase_degrees = np.degrees(np.unwrap(data[:, 7]))
    axes[1].semilogx(frequency_hz, phase_degrees, color="tab:purple")
    axes[1].set_xlabel("Частота, Гц")
    axes[1].set_ylabel("Фаза выхода, градусы")
    axes[1].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "frequency_response.png", dpi=160)
    plt.close(figure)


def plot_transient() -> None:
    data = load_table(RAW_ROOT / "transient.txt", expected_columns=7)
    time_ms = data[:, 0] * 1000.0
    names = (
        "Вход",
        "После Q4",
        "После Q3",
        "После Q2",
        "После темброблока",
        "Выход",
    )
    colors = ("black", "tab:orange", "tab:red", "tab:blue", "tab:green", "tab:pink")

    figure, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    for index, (axis, name, color) in enumerate(zip(axes, names, colors), start=1):
        signal = data[:, index]
        centered = signal - np.mean(signal)
        axis.plot(time_ms, centered, color=color, linewidth=1.0)
        axis.set_ylabel(f"{name}\nВ")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Время, мс")
    axes[0].set_title("Big Muff Pi V3: сигналы каскадов, вход 200 мВ пик-пик, 1 кГц")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "stage_waveforms.png", dpi=160)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", type=Path)
    parser.add_argument("--no-run", action="store_true", help="строить из имеющихся данных")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.no_run:
        run_ngspice(find_ngspice(args.ngspice))
    plot_frequency_response()
    plot_transient()
    print(f"Графики: {FIGURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
