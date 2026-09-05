"""Квазиньютоновские варианты Q3 с повторным использованием обратного якобиана."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from q3_incremental_model import (
    NonlinearCache,
    cached_currents,
    rebuild_cache,
    update_cache,
)
from q3_model import (
    Q3Parameters,
    TransientResult,
    capacitor_voltage,
    operating_point,
    residual_and_jacobian,
)
from q3_reduced_model import (
    Reduction,
    prepare_reduction,
    reconstruct_nodes,
    right_hand_side,
)


Mode = Literal["frozen", "reciprocal"]


@dataclass
class InverseState:
    m00: np.float32
    m02: np.float32
    m10: np.float32
    m12: np.float32
    m20: np.float32
    m22: np.float32
    reciprocal_determinant: np.float32


@dataclass(frozen=True)
class QuasiDiagnostics:
    inverse_refresh_count: int
    nonlinear_fallback_count: int
    maximum_increment: float


def inverse_state(first: np.ndarray, influence: np.ndarray) -> InverseState:
    h = np.asarray(influence, dtype=np.float32)
    first0 = np.float32(first[0])
    first2 = np.float32(first[2])
    a = np.float32(1.0) + h[0, 0] * first0
    c = h[0, 2] * first2
    d = h[1, 0] * first0
    f = h[1, 2] * first2
    g = h[2, 0] * first0
    i = np.float32(1.0) + h[2, 2] * first2
    reciprocal = np.float32(1.0) / np.float32(a * i - c * g)
    m00 = np.float32(i * reciprocal)
    m02 = np.float32(-c * reciprocal)
    m20 = np.float32(-g * reciprocal)
    m22 = np.float32(a * reciprocal)
    m10 = np.float32(-d * m00 - f * m20)
    m12 = np.float32(-d * m02 - f * m22)
    return InverseState(m00, m02, m10, m12, m20, m22, reciprocal)


def inverse_state_with_reciprocal_update(
    previous: InverseState, first: np.ndarray, influence: np.ndarray
) -> InverseState:
    h = np.asarray(influence, dtype=np.float32)
    first0 = np.float32(first[0])
    first2 = np.float32(first[2])
    a = np.float32(1.0) + h[0, 0] * first0
    c = h[0, 2] * first2
    d = h[1, 0] * first0
    f = h[1, 2] * first2
    g = h[2, 0] * first0
    i = np.float32(1.0) + h[2, 2] * first2
    determinant = np.float32(a * i - c * g)
    reciprocal = np.float32(
        previous.reciprocal_determinant
        * np.float32(2.0 - determinant * previous.reciprocal_determinant)
    )
    if not np.isfinite(reciprocal) or reciprocal <= 0.0:
        reciprocal = np.float32(1.0) / determinant
    m00 = np.float32(i * reciprocal)
    m02 = np.float32(-c * reciprocal)
    m20 = np.float32(-g * reciprocal)
    m22 = np.float32(a * reciprocal)
    m10 = np.float32(-d * m00 - f * m20)
    m12 = np.float32(-d * m02 - f * m22)
    return InverseState(m00, m02, m10, m12, m20, m22, reciprocal)


def apply_inverse(inverse: InverseState, right: np.ndarray) -> np.ndarray:
    right0 = np.float32(right[0])
    right1 = np.float32(right[1])
    right2 = np.float32(right[2])
    return np.array(
        [
            inverse.m00 * right0 + inverse.m02 * right2,
            right1 + inverse.m10 * right0 + inverse.m12 * right2,
            inverse.m20 * right0 + inverse.m22 * right2,
        ],
        dtype=np.float32,
    )


def simulate_quasi_reduced(
    mode: Mode,
    inverse_refresh_period: int,
    duration_s: float = 5.0e-3,
    step_s: float = 1.0 / (48_000.0 * 8.0),
    input_peak_v: float = 50.0e-3,
    input_frequency_hz: float = 1_000.0,
    nonlinear_refresh_period: int = 32,
    parameters: Q3Parameters | None = None,
) -> tuple[TransientResult, QuasiDiagnostics]:
    if inverse_refresh_period <= 0:
        raise ValueError("Период пересчёта обратной матрицы должен быть положительным")
    parameters = parameters or Q3Parameters()
    reduction = prepare_reduction(parameters, step_s)
    dc = operating_point(parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = input_peak_v * np.sin(2.0 * np.pi * input_frequency_hz * time_s)
    node_v = np.empty((count + 1, 6), dtype=np.float64)
    node_v[0] = dc.voltage_v
    iterations = np.ones(count + 1, dtype=np.int32)
    residual_a = np.zeros(count + 1, dtype=np.float64)
    q_v = reduction.voltage_matrix @ node_v[0]
    cache: NonlinearCache = rebuild_cache(q_v, parameters)
    _, first = cached_currents(cache, parameters)
    inverse = inverse_state(first, reduction.influence_matrix)
    previous_capacitor_v = capacitor_voltage(node_v[0], input_v[0])
    refresh_count = 1
    fallback_count = 0
    maximum_increment = 0.0
    force_refresh = False

    for index in range(1, count + 1):
        rhs = right_hand_side(reduction, previous_capacitor_v, float(input_v[index]))
        linear_q_v = reduction.voltage_matrix @ np.linalg.solve(
            reduction.linear_matrix, rhs
        )
        current, first = cached_currents(cache, parameters)
        q32 = np.asarray(q_v, dtype=np.float32)
        linear32 = np.asarray(linear_q_v, dtype=np.float32)
        influence32 = np.asarray(reduction.influence_matrix, dtype=np.float32)
        residual = q32 - linear32 + influence32 @ current

        periodic = (index - 1) % inverse_refresh_period == 0
        if force_refresh or periodic:
            inverse = inverse_state(first, influence32)
            refresh_count += 1
            force_refresh = False
        elif mode == "reciprocal":
            inverse = inverse_state_with_reciprocal_update(
                inverse, first, influence32
            )
        elif mode != "frozen":
            raise ValueError(f"Неизвестный режим {mode}")

        correction = apply_inverse(inverse, -residual)
        new_q_v = (q32 + correction).astype(np.float32)
        cache, increment, fallback = update_cache(
            cache, q32.astype(np.float64), new_q_v.astype(np.float64), parameters
        )
        if nonlinear_refresh_period > 0 and index % nonlinear_refresh_period == 0:
            cache = rebuild_cache(new_q_v, parameters)
        maximum_increment = max(maximum_increment, float(np.max(np.abs(increment))))
        fallback_count += int(fallback)
        if fallback or np.max(np.abs(increment)) > 0.25:
            force_refresh = True
        q_v = new_q_v.astype(np.float64)
        node_v[index] = reconstruct_nodes(q_v, rhs, reduction)
        physical_residual, _ = residual_and_jacobian(
            node_v[index], parameters, previous_capacitor_v,
            float(input_v[index]), step_s,
        )
        residual_a[index] = float(np.max(np.abs(physical_residual)))
        previous_capacitor_v = capacitor_voltage(node_v[index], input_v[index])
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")

    diagnostics = QuasiDiagnostics(
        refresh_count, fallback_count, maximum_increment
    )
    return TransientResult(time_s, input_v, node_v, iterations, residual_a), diagnostics
