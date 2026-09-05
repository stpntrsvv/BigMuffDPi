"""Точное сокращение узловой модели Q3 до трёх нелинейных напряжений."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from q3_model import (
    BASE,
    COLLECTOR,
    DIODE_NODE,
    DRIVE,
    EMITTER,
    OUTPUT,
    Q3Parameters,
    TransientResult,
    capacitor_voltage,
    linear_system,
    operating_point,
    residual_and_jacobian,
)


Method = Literal["full", "newton", "halley"]


@dataclass(frozen=True)
class Reduction:
    parameters: Q3Parameters
    step_s: float
    linear_matrix: np.ndarray
    source: np.ndarray
    injection_matrix: np.ndarray
    voltage_matrix: np.ndarray
    influence_matrix: np.ndarray
    capacitor_conductance: np.ndarray


def prepare_reduction(parameters: Q3Parameters, step_s: float) -> Reduction:
    """Готовит постоянные матрицы для выбранного временного шага."""
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("Временной шаг должен быть положительным")

    linear_matrix, source = linear_system(parameters)
    capacitor_conductance = np.array(
        [parameters.c5_f, parameters.c12_f, parameters.c6_f, parameters.c13_f],
        dtype=np.float64,
    ) / step_s

    def stamp(first: int, second: int, conductance: float) -> None:
        linear_matrix[first, first] += conductance
        linear_matrix[second, second] += conductance
        linear_matrix[first, second] -= conductance
        linear_matrix[second, first] -= conductance

    linear_matrix[DRIVE, DRIVE] += capacitor_conductance[0]
    stamp(COLLECTOR, BASE, capacitor_conductance[1])
    stamp(BASE, DIODE_NODE, capacitor_conductance[2])
    stamp(COLLECTOR, OUTPUT, capacitor_conductance[3])

    # Столбцы соответствуют I_F(v_BE), I_R(v_BC), I_D(v_D).
    injection_matrix = np.zeros((6, 3), dtype=np.float64)
    injection_matrix[:, 0] = (
        0.0,
        1.0 - parameters.alpha_forward,
        parameters.alpha_forward,
        -1.0,
        0.0,
        0.0,
    )
    injection_matrix[:, 1] = (
        0.0,
        1.0 - parameters.alpha_reverse,
        -1.0,
        parameters.alpha_reverse,
        0.0,
        0.0,
    )
    injection_matrix[:, 2] = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0)

    # q = (v_BE, v_BC, v_D), причём v_D = v(diode_node)-v(collector).
    voltage_matrix = np.zeros((3, 6), dtype=np.float64)
    voltage_matrix[0, BASE] = 1.0
    voltage_matrix[0, EMITTER] = -1.0
    voltage_matrix[1, BASE] = 1.0
    voltage_matrix[1, COLLECTOR] = -1.0
    voltage_matrix[2, DIODE_NODE] = 1.0
    voltage_matrix[2, COLLECTOR] = -1.0

    influence_matrix = voltage_matrix @ np.linalg.solve(
        linear_matrix, injection_matrix
    )
    return Reduction(
        parameters,
        step_s,
        linear_matrix,
        source,
        injection_matrix,
        voltage_matrix,
        influence_matrix,
        capacitor_conductance,
    )


def right_hand_side(
    reduction: Reduction, previous_capacitor_v: np.ndarray, input_v: float
) -> np.ndarray:
    """Формирует правую часть линейной системы с историческими токами."""
    g = reduction.capacitor_conductance
    rhs = reduction.source.copy()
    rhs[DRIVE] += g[0] * (input_v + previous_capacitor_v[0])
    rhs[COLLECTOR] += g[1] * previous_capacitor_v[1]
    rhs[BASE] -= g[1] * previous_capacitor_v[1]
    rhs[BASE] += g[2] * previous_capacitor_v[2]
    rhs[DIODE_NODE] -= g[2] * previous_capacitor_v[2]
    rhs[COLLECTOR] += g[3] * previous_capacitor_v[3]
    rhs[OUTPUT] -= g[3] * previous_capacitor_v[3]
    return rhs


def nonlinear_currents(
    q_v: np.ndarray, parameters: Q3Parameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает токи, первые и вторые производные по трём напряжениям."""
    scales = np.array(
        [
            parameters.forward_ideality * parameters.thermal_voltage_v,
            parameters.reverse_ideality * parameters.thermal_voltage_v,
            parameters.diode_ideality * parameters.thermal_voltage_v,
        ],
        dtype=np.float64,
    )
    saturation = np.array(
        [
            parameters.forward_saturation_current_a,
            parameters.reverse_saturation_current_a,
            2.0 * parameters.diode_saturation_current_a,
        ],
        dtype=np.float64,
    )
    argument = np.clip(np.asarray(q_v, dtype=np.float64) / scales, -80.0, 80.0)
    currents = saturation * np.expm1(argument)
    currents[2] = saturation[2] * np.sinh(argument[2])
    first = saturation * np.exp(argument) / scales
    first[2] = saturation[2] * np.cosh(argument[2]) / scales[2]
    second = saturation * np.exp(argument) / (scales * scales)
    second[2] = saturation[2] * np.sinh(argument[2]) / (scales[2] ** 2)
    return currents, first, second


def reduced_residual_and_derivatives(
    q_v: np.ndarray, linear_q_v: np.ndarray, reduction: Reduction
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    currents, first, second = nonlinear_currents(q_v, reduction.parameters)
    residual_v = q_v - linear_q_v + reduction.influence_matrix @ currents
    jacobian = np.eye(3) + reduction.influence_matrix * first[np.newaxis, :]
    return residual_v, jacobian, second


def reconstruct_nodes(
    q_v: np.ndarray, rhs: np.ndarray, reduction: Reduction
) -> np.ndarray:
    currents, _, _ = nonlinear_currents(q_v, reduction.parameters)
    return np.linalg.solve(
        reduction.linear_matrix,
        rhs - reduction.injection_matrix @ currents,
    )


def full_step(
    initial_q_v: np.ndarray,
    linear_q_v: np.ndarray,
    reduction: Reduction,
    tolerance_v: float = 1.0e-12,
    maximum_iterations: int = 30,
) -> tuple[np.ndarray, int]:
    """Доводит сокращённую систему Ньютоном с поиском длины шага."""
    q_v = np.array(initial_q_v, dtype=np.float64, copy=True)
    for iteration in range(maximum_iterations + 1):
        residual_v, jacobian, _ = reduced_residual_and_derivatives(
            q_v, linear_q_v, reduction
        )
        norm = float(np.max(np.abs(residual_v)))
        if norm <= tolerance_v:
            return q_v, iteration
        correction = np.linalg.solve(jacobian, -residual_v)
        damping = 1.0
        candidate = q_v + correction
        candidate_residual, _, _ = reduced_residual_and_derivatives(
            candidate, linear_q_v, reduction
        )
        while (
            float(np.max(np.abs(candidate_residual))) > norm
            and damping > 1.0 / 1024.0
        ):
            damping *= 0.5
            candidate = q_v + damping * correction
            candidate_residual, _, _ = reduced_residual_and_derivatives(
                candidate, linear_q_v, reduction
            )
        q_v = candidate
    raise RuntimeError("Сокращённый решатель Q3 не сошёлся")


def one_correction(
    q_v: np.ndarray, linear_q_v: np.ndarray, reduction: Reduction, method: Method
) -> np.ndarray:
    """Выполняет одну поправку Ньютона или многомерного метода Галлея."""
    residual_v, jacobian, second = reduced_residual_and_derivatives(
        q_v, linear_q_v, reduction
    )
    newton_correction = np.linalg.solve(jacobian, -residual_v)
    if method == "newton":
        return q_v + newton_correction
    if method != "halley":
        raise ValueError(f"Неизвестный метод: {method}")

    # Многомерное продолжение формулы Галлея:
    # [J + 1/2 F''(delta_N, ·)] delta_H = -F.
    directional_hessian = reduction.influence_matrix * (
        second * newton_correction
    )[np.newaxis, :]
    halley_matrix = jacobian + 0.5 * directional_hessian
    return q_v + np.linalg.solve(halley_matrix, -residual_v)


def simulate_reduced(
    method: Method,
    duration_s: float = 5.0e-3,
    step_s: float = 1.0 / (48_000.0 * 16.0),
    input_peak_v: float = 50.0e-3,
    input_frequency_hz: float = 1_000.0,
    parameters: Q3Parameters | None = None,
) -> TransientResult:
    """Считает Q3 полной сходимостью либо одной поправкой на отсчёт."""
    parameters = parameters or Q3Parameters()
    reduction = prepare_reduction(parameters, step_s)
    dc = operating_point(parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = input_peak_v * np.sin(2.0 * np.pi * input_frequency_hz * time_s)
    node_v = np.empty((count + 1, 6), dtype=np.float64)
    iterations = np.zeros(count + 1, dtype=np.int32)
    residual_a = np.zeros(count + 1, dtype=np.float64)
    node_v[0] = dc.voltage_v
    q_v = reduction.voltage_matrix @ node_v[0]
    previous_capacitor_v = capacitor_voltage(node_v[0], input_v[0])

    for index in range(1, count + 1):
        rhs = right_hand_side(reduction, previous_capacitor_v, float(input_v[index]))
        linear_q_v = reduction.voltage_matrix @ np.linalg.solve(
            reduction.linear_matrix, rhs
        )
        if method == "full":
            q_v, iterations[index] = full_step(q_v, linear_q_v, reduction)
        else:
            q_v = one_correction(q_v, linear_q_v, reduction, method)
            iterations[index] = 1
        node_v[index] = reconstruct_nodes(q_v, rhs, reduction)
        physical_residual, _ = residual_and_jacobian(
            node_v[index],
            parameters,
            previous_capacitor_v,
            float(input_v[index]),
            step_s,
        )
        residual_a[index] = float(np.max(np.abs(physical_residual)))
        previous_capacitor_v = capacitor_voltage(node_v[index], input_v[index])

        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")

    return TransientResult(time_s, input_v, node_v, iterations, residual_a)
