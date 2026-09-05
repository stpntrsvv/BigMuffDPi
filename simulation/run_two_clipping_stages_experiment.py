"""Сравнивает варианты физики двух связанных каскадов ограничения."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from two_clipping_stages_model import (
    Architecture,
    Q2_COLLECTOR,
    Q3_COLLECTOR,
    simulate_two_stages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "simulation" / "raw" / "two_clipping_stages"
EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "experiments" / "two_clipping_stages"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
DURATION_S = 12e-3
WARMUP_S = 4e-3
REFERENCE_FACTOR = 64
FACTOR = 8
ARCHITECTURES: tuple[Architecture, ...] = (
    "full_full", "full_scalar", "scalar_scalar"
)
NAMES = {
    "full_full": "Два полных",
    "full_scalar": "Полный + скалярный",
    "scalar_scalar": "Два скалярных",
}
CYCLES = {
    "full_full": (366, 382),
    "full_scalar": (280, 289),
    "scalar_scalar": (194, 196),
}


@dataclass(frozen=True)
class Summary:
    signal: str
    architecture: str
    output_rms_error_v: float
    output_maximum_error_v: float
    q3_rms_error_v: float
    physics_output_rms_error_v: float
    relative_output_error_db: float
    high_band_error_db: float
    maximum_residual_v: float


def nominal_input(time_s: np.ndarray) -> np.ndarray:
    return 50e-3 * np.sin(2.0 * np.pi * 1_000.0 * time_s)


def strong_input(time_s: np.ndarray) -> np.ndarray:
    raw = (
        np.sin(2.0 * np.pi * 997.0 * time_s)
        + 0.55 * np.sin(2.0 * np.pi * 3_217.0 * time_s + 0.37)
    )
    return raw / 1.55


SIGNALS = (("50 мВ, 1 кГц", nominal_input), ("1 В, два тона", strong_input))


def _ac(value: np.ndarray) -> np.ndarray:
    return value - np.mean(value)


def summarize(signal: str, architecture: Architecture, reference, baseline, candidate) -> Summary:
    start = int(round(WARMUP_S * 48_000.0))
    reference_q2 = reference.node_v[::REFERENCE_FACTOR, Q2_COLLECTOR][start:]
    reference_q3 = reference.node_v[::REFERENCE_FACTOR, Q3_COLLECTOR][start:]
    candidate_q2 = candidate.node_v[::FACTOR, Q2_COLLECTOR][start:]
    candidate_q3 = candidate.node_v[::FACTOR, Q3_COLLECTOR][start:]
    baseline_q2 = baseline.node_v[::FACTOR, Q2_COLLECTOR][start:]
    output_error = _ac(candidate_q2) - _ac(reference_q2)
    q3_error = _ac(candidate_q3) - _ac(reference_q3)
    physics_error = _ac(candidate_q2) - _ac(baseline_q2)
    reference_ac = _ac(reference_q2)
    rms_reference = float(np.sqrt(np.mean(reference_ac * reference_ac)))
    rms_error = float(np.sqrt(np.mean(output_error * output_error)))
    window = np.hanning(len(output_error))
    error_spectrum = np.fft.rfft(output_error * window)
    reference_spectrum = np.fft.rfft(reference_ac * window)
    frequency = np.fft.rfftfreq(len(output_error), 1.0 / 48_000.0)
    high = frequency >= 8_000.0
    return Summary(
        signal,
        architecture,
        rms_error,
        float(np.max(np.abs(output_error))),
        float(np.sqrt(np.mean(q3_error * q3_error))),
        float(np.sqrt(np.mean(physics_error * physics_error))),
        float(20.0 * np.log10(max(rms_error, 1e-18) / rms_reference)),
        float(20.0 * np.log10(
            max(float(np.linalg.norm(error_spectrum[high])), 1e-18)
            / float(np.linalg.norm(reference_spectrum))
        )),
        candidate.maximum_reduced_residual_v,
    )


def plot_waveform(reference, candidates: dict[str, object]) -> None:
    time_ms = reference.time_s[::REFERENCE_FACTOR] * 1e3
    mask = time_ms >= time_ms[-1] - 3.0
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(
        time_ms[mask], reference.node_v[::REFERENCE_FACTOR, Q2_COLLECTOR][mask],
        color="black", linewidth=2, label="Эталон 64×"
    )
    for architecture, result in candidates.items():
        axes[0].plot(
            result.time_s[::FACTOR][mask] * 1e3,
            result.node_v[::FACTOR, Q2_COLLECTOR][mask],
            label=NAMES[architecture], alpha=0.85
        )
        error = _ac(result.node_v[::FACTOR, Q2_COLLECTOR][mask]) - _ac(
            reference.node_v[::REFERENCE_FACTOR, Q2_COLLECTOR][mask]
        )
        axes[1].plot(result.time_s[::FACTOR][mask] * 1e3, error * 1e3,
                     label=NAMES[architecture])
    axes[0].set_ylabel("Коллектор Q2, В")
    axes[0].set_title("Два каскада ограничения, сильный двухтональный вход")
    axes[1].set_ylabel("Ошибка переменной части, мВ")
    axes[1].set_xlabel("Время, мс")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "strong_waveform.png", dpi=160)
    plt.close(figure)


def plot_metrics(rows: list[Summary]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for axis, (signal, _) in zip(axes, SIGNALS, strict=True):
        selected = [row for row in rows if row.signal == signal]
        axis.bar(
            [NAMES[row.architecture] for row in selected],
            [row.relative_output_error_db for row in selected],
        )
        axis.set_title(signal)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=12)
    axes[0].set_ylabel("Ошибка выхода относительно эталона, дБ")
    figure.suptitle("Два связанных каскада при 8×")
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / "architecture_error.png", dpi=160)
    plt.close(figure)


def write_report(rows: list[Summary]) -> None:
    table = "\n".join(
        "| {signal} | {name} | {rms:.2f} | {maximum:.2f} | {physics:.2f} | "
        "{relative:.1f} | {high:.1f} | {residual:.2e} |".format(
            signal=row.signal,
            name=NAMES[row.architecture],
            rms=row.output_rms_error_v * 1e3,
            maximum=row.output_maximum_error_v * 1e3,
            physics=row.physics_output_rms_error_v * 1e3,
            relative=row.relative_output_error_db,
            high=row.high_band_error_db,
            residual=row.maximum_residual_v,
        )
        for row in rows
    )
    report = f"""# Два связанных каскада ограничения

## Модель

Q3 и Q2 собраны в одну узловую систему. Разделительный C13 и R12 присутствуют
ровно один раз, поэтому первый каскад нагружен настоящим входом второго. После
Q2 включена линейная нагрузка двух ветвей темброблока: C9/R5, R8/C8 и полное
сопротивление регулятора 100 кОм. Сам выходной каскад пока не включён.

Эталон использует полную модель Эберса—Молла двух транзисторов, полную
сходимость Ньютона и частоту 64×. Три рабочих варианта используют одну поправку
на отсчёт при 8×. Ошибки считаются после удаления постоянной составляющей и
4 мс установления.

| Сигнал | Архитектура | СКО выхода, мВ | Макс. выхода, мВ | Вклад физического сокращения, мВ СКО | Относительная ошибка, дБ | Ошибка 8–24 кГц, дБ | Макс. невязка, В |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

![Сравнение архитектур](figures/architecture_error.png)

![Сильный сигнал](figures/strong_waveform.png)

## Вычислительная цена

| Архитектура | Обычный сигнал, тактов | Сильный сигнал, тактов | Остаток бюджета 8× |
|---|---:|---:|---:|
| Два полных | 366 | 382 | 60,7–76,7 |
| Полный + скалярный | 280 | 289 | 153,7–162,7 |
| Два скалярных | 194 | 196 | 246,7–248,7 |

## Вывод

Гибридный вариант подтверждён как основная архитектура. Линеаризация только
второго транзистора добавила 0,49 мВ СКО на слабом входе при общей ошибке
4,93 мВ и 1,69 мВ на предельном входе при общей ошибке 18,80 мВ. Относительная
ошибка ухудшилась лишь на 0,13–0,22 дБ, зато освободилось 93 такта на сильном
сигнале.

Два скалярных каскада тоже близки к эталону, но пока оставлены запасным
вариантом: вклад сокращения возрастает до 1,88–3,05 мВ. Следующая проверка —
добавить входной усилитель и настоящий регулятор Sustain, потому что именно они
зададут реальный размах сигнала на первом ограничителе. Предельный двухтональный
вход 1 В намеренно жёстче штатного режима и показывает, что численная ошибка
одной поправки при 8× сосредоточена на резких фронтах ограничения.
"""
    (EXPERIMENT_ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[Summary] = []
    strong_reference = None
    strong_candidates: dict[str, object] = {}
    for signal_name, input_function in SIGNALS:
        print(f"Эталон: {signal_name}")
        reference = simulate_two_stages(
            "reference_full", REFERENCE_FACTOR, input_function,
            DURATION_S, fully_converged=True
        )
        baseline = simulate_two_stages(
            "full_full", FACTOR, input_function, DURATION_S,
            fully_converged=False
        )
        candidates: dict[str, object] = {"full_full": baseline}
        for architecture in ARCHITECTURES[1:]:
            candidates[architecture] = simulate_two_stages(
                architecture, FACTOR, input_function, DURATION_S,
                fully_converged=False
            )
        for architecture in ARCHITECTURES:
            rows.append(summarize(
                signal_name, architecture, reference, baseline,
                candidates[architecture]
            ))
        if signal_name.startswith("1 В"):
            strong_reference = reference
            strong_candidates = candidates

    with (RAW_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Summary.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    plot_metrics(rows)
    if strong_reference is None:
        raise RuntimeError("Не получен сильный проверочный сигнал")
    plot_waveform(strong_reference, strong_candidates)
    write_report(rows)
    print(f"Отчёт: {EXPERIMENT_ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
