"""Проверяет точную замену координат состояния полной педали."""

from __future__ import annotations

import csv
from dataclasses import fields, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, operating_point, prepare_complete
from muff_hybrid_active import (
    ACTIVE,
    HybridActiveStep,
    modal_state_transform,
    prepare_hybrid_active,
    step_hybrid_cascade,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "modal_state"
EXPERIMENT = ROOT / "simulation" / "experiments" / "modal_state"
FIGURES = EXPERIMENT / "figures"
FACTOR = 8
RATE = 48_000 * FACTOR
COUNT = 768


def fixture_input() -> np.ndarray:
    time_s = np.arange(COUNT, dtype=np.float64) / RATE
    envelope = np.minimum(1.0, time_s / 1e-3)
    return envelope * (
        0.070 * np.sin(2 * np.pi * 82.41 * time_s)
        + 0.035 * np.sin(2 * np.pi * 164.81 * time_s + 0.3)
        + 0.020 * np.sin(2 * np.pi * 329.63 * time_s + 0.7)
        + 0.010 * np.sin(2 * np.pi * 2637.0 * time_s)
    )


def quantize(reduction: HybridActiveStep) -> HybridActiveStep:
    values = {}
    for field in fields(reduction):
        value = getattr(reduction, field.name)
        values[field.name] = value.astype(np.float32) if isinstance(value, np.ndarray) else value
    return replace(reduction, **values)


def run_pair(
    physical: HybridActiveStep,
    modal: HybridActiveStep,
    to_modal: np.ndarray,
    to_physical: np.ndarray,
    dc_state: np.ndarray,
    dc_active: np.ndarray,
    input_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_p = dc_state.copy()
    state_m = to_modal @ dc_state
    active_p = dc_active.copy()
    active_m = dc_active.copy()
    output_p = np.empty(len(input_v))
    output_m = np.empty(len(input_v))
    state_error = np.empty(len(input_v))
    blocks = (
        np.array([0, 1, 2], dtype=np.intp),
        np.array([3], dtype=np.intp),
        np.array([4, 5], dtype=np.intp),
    )
    for index, sample in enumerate(input_v):
        node_p, state_p, full_q_p, _, _ = step_hybrid_cascade(
            physical, state_p, active_p, float(sample), blocks=blocks,
        )
        node_m, state_m, full_q_m, _, _ = step_hybrid_cascade(
            modal, state_m, active_m, float(sample), blocks=blocks,
        )
        active_p = full_q_p[ACTIVE]
        active_m = full_q_m[ACTIVE]
        output_p[index] = node_p[OUTPUT]
        output_m[index] = node_m[OUTPUT]
        state_error[index] = np.max(np.abs(state_p - to_physical @ state_m))
    return output_p, output_m, state_error


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    reduction = prepare_hybrid_active(1.0, 1.0, 0.8, FACTOR)
    modal_transform = modal_state_transform(reduction)
    modal = modal_transform.reduction
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, reduction.parameters)
    dynamic = prepare_complete(1.0, 1.0, 0.8, 1.0 / RATE, reduction.parameters)
    dc_state = dynamic.capacitor_incidence.T @ dc_nodes
    input_v = fixture_input()

    original, transformed, state_error = run_pair(
        reduction, modal, modal_transform.to_modal, modal_transform.to_physical,
        dc_state, dc_q[ACTIVE], input_v
    )
    quantized_transform = modal_state_transform(quantize(reduction))
    original32, transformed32, state_error32 = run_pair(
        quantize(reduction), quantize(quantized_transform.reduction),
        quantized_transform.to_modal.astype(np.float32),
        quantized_transform.to_physical.astype(np.float32), dc_state.astype(np.float32),
        dc_q[ACTIVE].astype(np.float32), input_v,
    )

    rows = [
        ("float64", np.sqrt(np.mean((transformed-original)**2)), np.max(np.abs(transformed-original)), np.max(state_error)),
        ("коэффициенты float32", np.sqrt(np.mean((transformed32-original32)**2)), np.max(np.abs(transformed32-original32)), np.max(state_error32)),
    ]
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("режим", "ско_выхода_В", "пик_выхода_В", "пик_состояния_В"))
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].imshow(np.abs(reduction.state_transition), aspect="auto", cmap="magma")
    axes[0].set_title("Исходная матрица перехода 13×13")
    axes[1].imshow(np.abs(modal.state_transition), aspect="auto", cmap="magma")
    axes[1].set_title("Та же система в координатах режимов")
    for axis in axes:
        axis.set_xlabel("Столбец"); axis.set_ylabel("Строка")
    figure.tight_layout(); figure.savefig(FIGURES / "transition_matrices.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.semilogy(np.maximum(np.abs(transformed-original), 1e-18), label="float64")
    axis.semilogy(np.maximum(np.abs(transformed32-original32), 1e-18), label="коэффициенты float32")
    axis.set_xlabel("Шаг"); axis.set_ylabel("Ошибка выхода, В")
    axis.set_title("Ошибка, внесённая заменой координат")
    axis.grid(True, which="both", alpha=.3); axis.legend()
    figure.tight_layout(); figure.savefig(FIGURES / "output_error.png", dpi=160); plt.close(figure)

    nonzero_before = int(np.count_nonzero(np.abs(reduction.state_transition) > 1e-12))
    nonzero_after = int(np.count_nonzero(np.abs(modal.state_transition) > 1e-12))
    report = f"""# Диагональные координаты линейного состояния

Напряжения 13 конденсаторов заменены 13 независимыми режимами той же линейной системы. Физика и нелинейные уравнения не меняются. Матрица перехода преобразована как `V⁻¹ A V` и стала диагональной.

- число обусловленности преобразования: {np.linalg.cond(modal_transform.to_physical):.3f};
- наибольшая мнимая часть: {max(np.max(np.abs(np.imag(np.linalg.eigvals(reduction.state_transition)))), 0.0):.3e};
- ненулевые коэффициенты перехода: {nonzero_before} → {nonzero_after};
- умножения для перехода состояния: 169 → 13.

| Точность коэффициентов | СКО выхода, В | Пиковая ошибка выхода, В | Пиковая ошибка состояния, В |
|---|---:|---:|---:|
| float64 | {rows[0][1]:.3e} | {rows[0][2]:.3e} | {rows[0][3]:.3e} |
| float32 | {rows[1][1]:.3e} | {rows[1][2]:.3e} | {rows[1][3]:.3e} |

![Матрицы перехода](figures/transition_matrices.png)

![Ошибка выхода](figures/output_error.png)

## Вывод

Это точная алгебраическая замена координат. После проверки округления `float32` диагональный переход можно перенести в измеритель STM32: ожидаемое сокращение — 156 умножений с накоплением на каждый внутренний отсчёт, за вычетом нескольких дополнительных операций в формуле выхода.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    print(f"Переход: {nonzero_before} -> {nonzero_after}; cond={np.linalg.cond(modal_transform.to_physical):.3f}")
    print(f"Пик ошибки float64: {rows[0][2]:.3e} В; float32: {rows[1][2]:.3e} В")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
