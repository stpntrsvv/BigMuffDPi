"""Варианты физики Q3 для выбора частоты расчёта и размера системы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from q3_model import Q3Parameters, capacitor_voltage, operating_point
from q3_reduced_model import (
    Reduction,
    nonlinear_currents,
    prepare_reduction,
    right_hand_side,
)


Physics = Literal["full", "bc_saturated", "bjt_linear"]
InputFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PhysicsResult:
    time_s: np.ndarray
    input_v: np.ndarray
    node_v: np.ndarray
    maximum_reduced_residual_v: float


def linearized_bjt_reduction(
    reduction: Reduction, dc_q_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Возвращает A^-1, постоянное смещение и влияние диодного тока."""
    dc_currents, dc_first, _ = nonlinear_currents(dc_q_v, reduction.parameters)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:, 0] += reduction.influence_matrix[:, 0] * dc_first[0]
    matrix[:, 1] += reduction.influence_matrix[:, 1] * dc_first[1]
    inverse = np.linalg.inv(matrix)
    transistor_offset = reduction.influence_matrix[:, :2] @ (
        dc_currents[:2] - dc_first[:2] * dc_q_v[:2]
    )
    diode_influence = float((inverse @ reduction.influence_matrix[:, 2])[2])
    return inverse, transistor_offset, diode_influence


def _variant_currents(
    q_v: np.ndarray,
    reduction: Reduction,
    physics: Physics,
    dc_q_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    currents, first, _ = nonlinear_currents(q_v, reduction.parameters)
    if physics == "full":
        return currents, first
    if physics == "bc_saturated":
        currents[1] = -reduction.parameters.reverse_saturation_current_a
        first[1] = 0.0
        return currents, first
    if physics == "bjt_linear":
        dc_currents, dc_first, _ = nonlinear_currents(dc_q_v, reduction.parameters)
        currents[:2] = dc_currents[:2] + dc_first[:2] * (q_v[:2] - dc_q_v[:2])
        first[:2] = dc_first[:2]
        return currents, first
    raise ValueError(f"Неизвестный вариант физики: {physics}")


def _residual_and_jacobian(
    q_v: np.ndarray,
    linear_q_v: np.ndarray,
    reduction: Reduction,
    physics: Physics,
    dc_q_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    currents, first = _variant_currents(q_v, reduction, physics, dc_q_v)
    residual = q_v - linear_q_v + reduction.influence_matrix @ currents
    jacobian = np.eye(3) + reduction.influence_matrix * first[np.newaxis, :]
    return residual, jacobian


def _solve_full(
    initial_q_v: np.ndarray,
    linear_q_v: np.ndarray,
    reduction: Reduction,
    physics: Physics,
    dc_q_v: np.ndarray,
) -> np.ndarray:
    q_v = initial_q_v.copy()
    for _ in range(30):
        residual, jacobian = _residual_and_jacobian(
            q_v, linear_q_v, reduction, physics, dc_q_v
        )
        if float(np.max(np.abs(residual))) <= 1.0e-12:
            return q_v
        correction = np.linalg.solve(jacobian, -residual)
        damping = 1.0
        norm = float(np.max(np.abs(residual)))
        while damping >= 1.0 / 1024.0:
            candidate = q_v + damping * correction
            candidate_residual, _ = _residual_and_jacobian(
                candidate, linear_q_v, reduction, physics, dc_q_v
            )
            if float(np.max(np.abs(candidate_residual))) <= norm:
                q_v = candidate
                break
            damping *= 0.5
        else:
            q_v += correction
    raise RuntimeError("Вариант Q3 не сошёлся")


def _reconstruct_nodes(
    q_v: np.ndarray,
    rhs: np.ndarray,
    reduction: Reduction,
    physics: Physics,
    dc_q_v: np.ndarray,
) -> np.ndarray:
    currents, _ = _variant_currents(q_v, reduction, physics, dc_q_v)
    return np.linalg.solve(
        reduction.linear_matrix,
        rhs - reduction.injection_matrix @ currents,
    )


def simulate_physics(
    physics: Physics,
    factor: int,
    input_function: InputFunction,
    duration_s: float = 12.0e-3,
    fully_converged: bool = False,
    parameters: Q3Parameters | None = None,
) -> PhysicsResult:
    """Считает вариант Q3 на factor × 48 кГц."""
    parameters = parameters or Q3Parameters()
    step_s = 1.0 / (48_000.0 * factor)
    reduction = prepare_reduction(parameters, step_s)
    dc = operating_point(parameters)
    dc_q_v = reduction.voltage_matrix @ dc.voltage_v
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    if input_v.shape != time_s.shape:
        raise ValueError("Функция входа вернула массив неправильного размера")

    node_v = np.empty((count + 1, 6), dtype=np.float64)
    node_v[0] = dc.voltage_v
    q_v = dc_q_v.copy()
    previous_capacitor_v = capacitor_voltage(node_v[0], float(input_v[0]))
    maximum_residual = 0.0

    for index in range(1, count + 1):
        rhs = right_hand_side(reduction, previous_capacitor_v, float(input_v[index]))
        linear_nodes = np.linalg.solve(reduction.linear_matrix, rhs)
        linear_q_v = reduction.voltage_matrix @ linear_nodes
        if fully_converged:
            q_v = _solve_full(
                q_v, linear_q_v, reduction, physics, dc_q_v
            )
        else:
            residual, jacobian = _residual_and_jacobian(
                q_v, linear_q_v, reduction, physics, dc_q_v
            )
            q_v = q_v + np.linalg.solve(jacobian, -residual)
        node_v[index] = _reconstruct_nodes(
            q_v, rhs, reduction, physics, dc_q_v
        )
        reduced_residual, _ = _residual_and_jacobian(
            q_v, linear_q_v, reduction, physics, dc_q_v
        )
        maximum_residual = max(
            maximum_residual, float(np.max(np.abs(reduced_residual)))
        )
        previous_capacitor_v = capacitor_voltage(
            node_v[index], float(input_v[index])
        )
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")

    return PhysicsResult(time_s, input_v, node_v, maximum_residual)
