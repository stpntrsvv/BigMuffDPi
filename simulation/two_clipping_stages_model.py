"""Единая узловая модель двух связанных каскадов ограничения Big Muff Pi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from q3_model import Q3Parameters


Q3_DRIVE, Q3_BASE, Q3_COLLECTOR, Q3_EMITTER, Q3_DIODE = range(5)
Q2_DRIVE, Q2_BASE, Q2_COLLECTOR, Q2_EMITTER, Q2_DIODE = range(5, 10)
TONE_HIGH, TONE_LOW = 10, 11
NODE_COUNT = 12
STAGE_COUNT = 2
NONLINEAR_COUNT = 6

Architecture = Literal[
    "reference_full", "full_full", "full_scalar", "scalar_scalar"
]
InputFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TwoStageResult:
    time_s: np.ndarray
    input_v: np.ndarray
    node_v: np.ndarray
    maximum_reduced_residual_v: float


@dataclass(frozen=True)
class Reduction:
    parameters: Q3Parameters
    linear_matrix: np.ndarray
    source: np.ndarray
    injection: np.ndarray
    voltage: np.ndarray
    influence: np.ndarray
    capacitor_incidence: np.ndarray
    capacitor_conductance: np.ndarray


def _branch(matrix: np.ndarray, first: int, second: int, conductance: float) -> None:
    matrix[first, first] += conductance
    matrix[second, second] += conductance
    matrix[first, second] -= conductance
    matrix[second, first] -= conductance


def _resistive_system(parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    source = np.zeros(NODE_COUNT, dtype=np.float64)
    for base, collector, emitter, drive in (
        (Q3_BASE, Q3_COLLECTOR, Q3_EMITTER, Q3_DRIVE),
        (Q2_BASE, Q2_COLLECTOR, Q2_EMITTER, Q2_DRIVE),
    ):
        _branch(matrix, drive, base, 1.0 / 10_000.0)
        matrix[base, base] += 1.0 / 100_000.0
        matrix[collector, collector] += 1.0 / 10_000.0
        source[collector] += parameters.supply_v / 10_000.0
        matrix[emitter, emitter] += 1.0 / 150.0
        _branch(matrix, collector, base, 1.0 / 470_000.0)
        # Численно безопасный путь, совпадающий с эталоном ngspice.
        diode = Q3_DIODE if collector == Q3_COLLECTOR else Q2_DIODE
        _branch(matrix, collector, diode, 1.0e-12)

    # Нагрузка второго коллектора настоящими двумя ветвями темброблока.
    _branch(matrix, Q2_COLLECTOR, TONE_LOW, 1.0 / 39_000.0)
    matrix[TONE_HIGH, TONE_HIGH] += 1.0 / 22_000.0
    _branch(matrix, TONE_HIGH, TONE_LOW, 1.0 / 100_000.0)
    return matrix, source


def _nonlinear_matrices(parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    injection = np.zeros((NODE_COUNT, NONLINEAR_COUNT), dtype=np.float64)
    voltage = np.zeros((NONLINEAR_COUNT, NODE_COUNT), dtype=np.float64)
    for stage, (base, collector, emitter, diode) in enumerate(
        (
            (Q3_BASE, Q3_COLLECTOR, Q3_EMITTER, Q3_DIODE),
            (Q2_BASE, Q2_COLLECTOR, Q2_EMITTER, Q2_DIODE),
        )
    ):
        first = 3 * stage
        injection[:, first] = (
            np.eye(NODE_COUNT)[base] * (1.0 - parameters.alpha_forward)
            + np.eye(NODE_COUNT)[collector] * parameters.alpha_forward
            - np.eye(NODE_COUNT)[emitter]
        )
        injection[:, first + 1] = (
            np.eye(NODE_COUNT)[base] * (1.0 - parameters.alpha_reverse)
            - np.eye(NODE_COUNT)[collector]
            + np.eye(NODE_COUNT)[emitter] * parameters.alpha_reverse
        )
        injection[collector, first + 2] = -1.0
        injection[diode, first + 2] = 1.0
        voltage[first, base] = 1.0
        voltage[first, emitter] = -1.0
        voltage[first + 1, base] = 1.0
        voltage[first + 1, collector] = -1.0
        voltage[first + 2, diode] = 1.0
        voltage[first + 2, collector] = -1.0
    return injection, voltage


def prepare_reduction(parameters: Q3Parameters, step_s: float | None) -> Reduction:
    matrix, source = _resistive_system(parameters)
    injection, voltage = _nonlinear_matrices(parameters)
    incidence = np.zeros((NODE_COUNT, 8), dtype=np.float64)
    incidence[Q3_DRIVE, 0] = 1.0  # C5, второй вывод — известный вход.
    for column, (first, second) in enumerate(
        (
            (Q3_COLLECTOR, Q3_BASE),
            (Q3_BASE, Q3_DIODE),
            (Q3_COLLECTOR, Q2_DRIVE),
            (Q2_COLLECTOR, Q2_BASE),
            (Q2_BASE, Q2_DIODE),
            (Q2_COLLECTOR, TONE_HIGH),
        ),
        start=1,
    ):
        incidence[first, column] = 1.0
        incidence[second, column] = -1.0
    incidence[TONE_LOW, 7] = 1.0  # C8 на землю.
    capacitance = np.array(
        [100e-9, 470e-12, 1e-6, 100e-9, 470e-12, 1e-6, 4e-9, 10e-9]
    )
    conductance = np.zeros(8) if step_s is None else capacitance / step_s
    if step_s is not None:
        matrix += (incidence * conductance[np.newaxis, :]) @ incidence.T
    influence = voltage @ np.linalg.solve(matrix, injection)
    return Reduction(
        parameters, matrix, source, injection, voltage, influence,
        incidence, conductance
    )


def _full_currents(q_v: np.ndarray, parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    current = np.empty(NONLINEAR_COUNT, dtype=np.float64)
    first = np.empty(NONLINEAR_COUNT, dtype=np.float64)
    scales = (
        parameters.forward_ideality * parameters.thermal_voltage_v,
        parameters.reverse_ideality * parameters.thermal_voltage_v,
        parameters.diode_ideality * parameters.thermal_voltage_v,
    )
    for stage in range(STAGE_COUNT):
        offset = 3 * stage
        forward = float(np.clip(q_v[offset] / scales[0], -80.0, 80.0))
        reverse = float(np.clip(q_v[offset + 1] / scales[1], -80.0, 80.0))
        diode = float(np.clip(q_v[offset + 2] / scales[2], -80.0, 80.0))
        current[offset] = parameters.forward_saturation_current_a * np.expm1(forward)
        current[offset + 1] = parameters.reverse_saturation_current_a * np.expm1(reverse)
        current[offset + 2] = 2.0 * parameters.diode_saturation_current_a * np.sinh(diode)
        first[offset] = parameters.forward_saturation_current_a * np.exp(forward) / scales[0]
        first[offset + 1] = parameters.reverse_saturation_current_a * np.exp(reverse) / scales[1]
        first[offset + 2] = 2.0 * parameters.diode_saturation_current_a * np.cosh(diode) / scales[2]
    return current, first


def _variant_currents(
    q_v: np.ndarray,
    parameters: Q3Parameters,
    architecture: Architecture,
    dc_q_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current, first = _full_currents(q_v, parameters)
    dc_current, dc_first = _full_currents(dc_q_v, parameters)
    linear_stages = {
        "reference_full": (),
        "full_full": (),
        "full_scalar": (1,),
        "scalar_scalar": (0, 1),
    }[architecture]
    for stage in range(STAGE_COUNT):
        offset = 3 * stage
        if architecture != "reference_full":
            # Обратный переход во всех рабочих вариантах глубоко заперт.
            current[offset + 1] = -parameters.reverse_saturation_current_a
            first[offset + 1] = 0.0
        if stage in linear_stages:
            current[offset] = dc_current[offset] + dc_first[offset] * (
                q_v[offset] - dc_q_v[offset]
            )
            first[offset] = dc_first[offset]
    return current, first


def _residual_jacobian(
    q_v: np.ndarray,
    linear_q_v: np.ndarray,
    reduction: Reduction,
    architecture: Architecture,
    dc_q_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current, first = _variant_currents(
        q_v, reduction.parameters, architecture, dc_q_v
    )
    residual = q_v - linear_q_v + reduction.influence @ current
    jacobian = np.eye(NONLINEAR_COUNT) + reduction.influence * first[np.newaxis, :]
    return residual, jacobian


def _solve(
    q_v: np.ndarray,
    linear_q_v: np.ndarray,
    reduction: Reduction,
    architecture: Architecture,
    dc_q_v: np.ndarray,
    fully_converged: bool,
) -> tuple[np.ndarray, float]:
    iterations = 30 if fully_converged else 1
    q_v = q_v.copy()
    for _ in range(iterations):
        residual, jacobian = _residual_jacobian(
            q_v, linear_q_v, reduction, architecture, dc_q_v
        )
        norm = float(np.max(np.abs(residual)))
        if fully_converged and norm <= 1.0e-12:
            break
        correction = np.linalg.solve(jacobian, -residual)
        if not fully_converged:
            q_v += correction
            break
        damping = 1.0
        while damping >= 1.0 / 1024.0:
            candidate = q_v + damping * correction
            candidate_residual, _ = _residual_jacobian(
                candidate, linear_q_v, reduction, architecture, dc_q_v
            )
            if float(np.max(np.abs(candidate_residual))) <= norm:
                q_v = candidate
                break
            damping *= 0.5
    residual, _ = _residual_jacobian(
        q_v, linear_q_v, reduction, architecture, dc_q_v
    )
    return q_v, float(np.max(np.abs(residual)))


def _dc_state(parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    matrix, source = _resistive_system(parameters)
    injection, voltage = _nonlinear_matrices(parameters)
    node_v = np.zeros(NODE_COUNT, dtype=np.float64)
    node_v[[Q3_DRIVE, Q3_BASE, Q2_DRIVE, Q2_BASE]] = 0.70
    node_v[[Q3_COLLECTOR, Q3_DIODE]] = 4.5
    node_v[[Q2_COLLECTOR, Q2_DIODE]] = 4.8
    node_v[[Q3_EMITTER, Q2_EMITTER]] = 0.065
    node_v[TONE_LOW] = 3.65
    node_v[TONE_HIGH] = 0.65

    for _ in range(80):
        q_v = voltage @ node_v
        current, first = _full_currents(q_v, parameters)
        residual = matrix @ node_v - source + injection @ current
        norm = float(np.max(np.abs(residual)))
        if norm <= 1.0e-12:
            return node_v, q_v
        jacobian = matrix + (injection * first[np.newaxis, :]) @ voltage
        correction = np.linalg.solve(jacobian, -residual)
        damping = 1.0
        while damping >= 1.0 / 4096.0:
            candidate = node_v + damping * correction
            candidate_q = voltage @ candidate
            candidate_current, _ = _full_currents(candidate_q, parameters)
            candidate_residual = (
                matrix @ candidate - source + injection @ candidate_current
            )
            if float(np.max(np.abs(candidate_residual))) < norm:
                node_v = candidate
                break
            damping *= 0.5
        else:
            raise RuntimeError("Рабочая точка двух каскадов не сошлась")
    raise RuntimeError("Рабочая точка двух каскадов не сошлась за 80 итераций")


def simulate_two_stages(
    architecture: Architecture,
    factor: int,
    input_function: InputFunction,
    duration_s: float = 12e-3,
    fully_converged: bool = False,
    parameters: Q3Parameters | None = None,
) -> TwoStageResult:
    parameters = parameters or Q3Parameters()
    step_s = 1.0 / (48_000.0 * factor)
    reduction = prepare_reduction(parameters, step_s)
    dc_nodes, dc_q = _dc_state(parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    node_v = np.empty((count + 1, NODE_COUNT), dtype=np.float64)
    node_v[0] = dc_nodes
    q_v = dc_q.copy()
    known = np.zeros(8)
    known[0] = input_v[0]
    previous_capacitor_v = reduction.capacitor_incidence.T @ node_v[0] - known
    maximum_residual = 0.0

    for index in range(1, count + 1):
        known[0] = input_v[index]
        rhs = reduction.source + reduction.capacitor_incidence @ (
            reduction.capacitor_conductance * (known + previous_capacitor_v)
        )
        linear_nodes = np.linalg.solve(reduction.linear_matrix, rhs)
        linear_q = reduction.voltage @ linear_nodes
        q_v, residual = _solve(
            q_v, linear_q, reduction, architecture, dc_q, fully_converged
        )
        maximum_residual = max(maximum_residual, residual)
        current, _ = _variant_currents(q_v, parameters, architecture, dc_q)
        node_v[index] = np.linalg.solve(
            reduction.linear_matrix, rhs - reduction.injection @ current
        )
        previous_capacitor_v = reduction.capacitor_incidence.T @ node_v[index] - known
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return TwoStageResult(time_s, input_v, node_v, maximum_residual)
