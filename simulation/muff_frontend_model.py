"""Единая модель Q4, Sustain, Q3, Q2 и нагрузки темброблока."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from q3_model import Q3Parameters


(
    C1_LEFT, Q4_BASE, Q4_COLLECTOR, Q4_EMITTER,
    SUSTAIN_TOP, SUSTAIN_WIPER, SUSTAIN_BOTTOM,
    Q3_DRIVE, Q3_BASE, Q3_COLLECTOR, Q3_EMITTER, Q3_DIODE,
    Q2_DRIVE, Q2_BASE, Q2_COLLECTOR, Q2_EMITTER, Q2_DIODE,
    TONE_HIGH, TONE_LOW,
) = range(19)
NODE_COUNT = 19
STAGES = (
    (Q4_BASE, Q4_COLLECTOR, Q4_EMITTER, None),
    (Q3_BASE, Q3_COLLECTOR, Q3_EMITTER, Q3_DIODE),
    (Q2_BASE, Q2_COLLECTOR, Q2_EMITTER, Q2_DIODE),
)
NONLINEAR_COUNT = 9
Architecture = Literal["reference_full", "hybrid"]
InputFunction = Callable[[np.ndarray], np.ndarray]
SustainFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FrontendReduction:
    parameters: Q3Parameters
    sustain: float
    step_s: float | None
    linear_matrix: np.ndarray
    source: np.ndarray
    input_vector: np.ndarray
    injection: np.ndarray
    voltage: np.ndarray
    influence: np.ndarray | None
    capacitor_incidence: np.ndarray
    capacitance: np.ndarray
    capacitor_conductance: np.ndarray


@dataclass(frozen=True)
class FrontendResult:
    time_s: np.ndarray
    input_v: np.ndarray
    node_v: np.ndarray
    nonlinear_v: np.ndarray
    residual_v: np.ndarray
    correction_v: np.ndarray
    sustain: np.ndarray | None = None


def _branch(matrix: np.ndarray, first: int, second: int, conductance: float) -> None:
    matrix[first, first] += conductance
    matrix[second, second] += conductance
    matrix[first, second] -= conductance
    matrix[second, first] -= conductance


def _linear_system(
    parameters: Q3Parameters, sustain: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= sustain <= 1.0:
        raise ValueError("Sustain должен находиться в диапазоне 0…1")
    matrix = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    source = np.zeros(NODE_COUNT, dtype=np.float64)
    input_vector = np.zeros(NODE_COUNT, dtype=np.float64)

    input_conductance = 1.0 / 39_000.0
    matrix[C1_LEFT, C1_LEFT] += input_conductance
    input_vector[C1_LEFT] = input_conductance

    matrix[Q4_BASE, Q4_BASE] += 1.0 / 47_000.0
    matrix[Q4_COLLECTOR, Q4_COLLECTOR] += 1.0 / 10_000.0
    source[Q4_COLLECTOR] += parameters.supply_v / 10_000.0
    matrix[Q4_EMITTER, Q4_EMITTER] += 1.0 / 100.0
    _branch(matrix, Q4_COLLECTOR, Q4_BASE, 1.0 / 470_000.0)

    top_resistance = max(1.0, 100_000.0 * (1.0 - sustain))
    bottom_resistance = max(1.0, 100_000.0 * sustain)
    _branch(matrix, SUSTAIN_TOP, SUSTAIN_WIPER, 1.0 / top_resistance)
    _branch(matrix, SUSTAIN_WIPER, SUSTAIN_BOTTOM, 1.0 / bottom_resistance)
    matrix[SUSTAIN_BOTTOM, SUSTAIN_BOTTOM] += 1.0 / 1_000.0

    for drive, base, collector, emitter, diode in (
        (Q3_DRIVE, Q3_BASE, Q3_COLLECTOR, Q3_EMITTER, Q3_DIODE),
        (Q2_DRIVE, Q2_BASE, Q2_COLLECTOR, Q2_EMITTER, Q2_DIODE),
    ):
        _branch(matrix, drive, base, 1.0 / 10_000.0)
        matrix[base, base] += 1.0 / 100_000.0
        matrix[collector, collector] += 1.0 / 10_000.0
        source[collector] += parameters.supply_v / 10_000.0
        matrix[emitter, emitter] += 1.0 / 150.0
        _branch(matrix, collector, base, 1.0 / 470_000.0)
        _branch(matrix, collector, diode, 1.0e-12)

    _branch(matrix, Q2_COLLECTOR, TONE_LOW, 1.0 / 39_000.0)
    matrix[TONE_HIGH, TONE_HIGH] += 1.0 / 22_000.0
    _branch(matrix, TONE_HIGH, TONE_LOW, 1.0 / 100_000.0)
    return matrix, source, input_vector


def _nonlinear_matrices(parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    injection = np.zeros((NODE_COUNT, NONLINEAR_COUNT), dtype=np.float64)
    voltage = np.zeros((NONLINEAR_COUNT, NODE_COUNT), dtype=np.float64)
    identity = np.eye(NODE_COUNT)
    for stage, (base, collector, emitter, diode) in enumerate(STAGES):
        offset = 3 * stage
        injection[:, offset] = (
            identity[base] * (1.0 - parameters.alpha_forward)
            + identity[collector] * parameters.alpha_forward
            - identity[emitter]
        )
        injection[:, offset + 1] = (
            identity[base] * (1.0 - parameters.alpha_reverse)
            - identity[collector]
            + identity[emitter] * parameters.alpha_reverse
        )
        voltage[offset, base] = 1.0
        voltage[offset, emitter] = -1.0
        voltage[offset + 1, base] = 1.0
        voltage[offset + 1, collector] = -1.0
        if diode is not None:
            injection[collector, offset + 2] = -1.0
            injection[diode, offset + 2] = 1.0
            voltage[offset + 2, diode] = 1.0
            voltage[offset + 2, collector] = -1.0
    return injection, voltage


def _capacitors() -> tuple[np.ndarray, np.ndarray]:
    pairs = (
        (C1_LEFT, Q4_BASE),
        (Q4_COLLECTOR, Q4_BASE),
        (Q4_COLLECTOR, SUSTAIN_TOP),
        (SUSTAIN_WIPER, Q3_DRIVE),
        (Q3_COLLECTOR, Q3_BASE),
        (Q3_BASE, Q3_DIODE),
        (Q3_COLLECTOR, Q2_DRIVE),
        (Q2_COLLECTOR, Q2_BASE),
        (Q2_BASE, Q2_DIODE),
        (Q2_COLLECTOR, TONE_HIGH),
        (TONE_LOW, None),
    )
    capacitance = np.array(
        [1e-6, 470e-12, 1e-6, 100e-9, 470e-12, 1e-6,
         100e-9, 470e-12, 1e-6, 4e-9, 10e-9],
        dtype=np.float64,
    )
    incidence = np.zeros((NODE_COUNT, len(pairs)), dtype=np.float64)
    for column, (first, second) in enumerate(pairs):
        incidence[first, column] = 1.0
        if second is not None:
            incidence[second, column] = -1.0
    return incidence, capacitance


def prepare_frontend(
    sustain: float, step_s: float | None, parameters: Q3Parameters | None = None
) -> FrontendReduction:
    parameters = parameters or Q3Parameters()
    matrix, source, input_vector = _linear_system(parameters, sustain)
    injection, voltage = _nonlinear_matrices(parameters)
    incidence, capacitance = _capacitors()
    conductance = np.zeros_like(capacitance)
    influence = None
    if step_s is not None:
        conductance = capacitance / step_s
        matrix += (incidence * conductance[np.newaxis, :]) @ incidence.T
        influence = voltage @ np.linalg.solve(matrix, injection)
    return FrontendReduction(
        parameters, sustain, step_s, matrix, source, input_vector,
        injection, voltage, influence, incidence, capacitance, conductance
    )


def nonlinear_terms(
    q_v: np.ndarray,
    parameters: Q3Parameters,
    architecture: Architecture,
    dc_q_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.zeros(NONLINEAR_COUNT, dtype=np.float64)
    first = np.zeros(NONLINEAR_COUNT, dtype=np.float64)
    dc_current = np.zeros(NONLINEAR_COUNT, dtype=np.float64)
    dc_first = np.zeros(NONLINEAR_COUNT, dtype=np.float64)
    forward_scale = parameters.forward_ideality * parameters.thermal_voltage_v
    reverse_scale = parameters.reverse_ideality * parameters.thermal_voltage_v
    diode_scale = parameters.diode_ideality * parameters.thermal_voltage_v
    for stage, (_, _, _, diode) in enumerate(STAGES):
        offset = 3 * stage
        for values, derivatives, q in (
            (current, first, q_v), (dc_current, dc_first, dc_q_v)
        ):
            forward = float(np.clip(q[offset] / forward_scale, -80.0, 80.0))
            reverse = float(np.clip(q[offset + 1] / reverse_scale, -80.0, 80.0))
            values[offset] = parameters.forward_saturation_current_a * np.expm1(forward)
            values[offset + 1] = parameters.reverse_saturation_current_a * np.expm1(reverse)
            derivatives[offset] = parameters.forward_saturation_current_a * np.exp(forward) / forward_scale
            derivatives[offset + 1] = parameters.reverse_saturation_current_a * np.exp(reverse) / reverse_scale
            if diode is not None:
                argument = float(np.clip(q[offset + 2] / diode_scale, -80.0, 80.0))
                values[offset + 2] = 2.0 * parameters.diode_saturation_current_a * np.sinh(argument)
                derivatives[offset + 2] = 2.0 * parameters.diode_saturation_current_a * np.cosh(argument) / diode_scale

        if architecture == "hybrid":
            current[offset + 1] = -parameters.reverse_saturation_current_a
            first[offset + 1] = 0.0
            if stage == 2:
                current[offset] = dc_current[offset] + dc_first[offset] * (
                    q_v[offset] - dc_q_v[offset]
                )
                first[offset] = dc_first[offset]
    return current, first


def operating_point(
    sustain: float, parameters: Q3Parameters | None = None
) -> tuple[np.ndarray, np.ndarray]:
    reduction = prepare_frontend(sustain, None, parameters)
    node_v = np.zeros(NODE_COUNT, dtype=np.float64)
    node_v[[Q4_BASE, Q3_BASE, Q3_DRIVE, Q2_BASE, Q2_DRIVE]] = 0.70
    node_v[[Q4_COLLECTOR]] = 7.0
    node_v[[Q3_COLLECTOR, Q3_DIODE]] = 4.5
    node_v[[Q2_COLLECTOR, Q2_DIODE]] = 4.5
    node_v[[Q4_EMITTER, Q3_EMITTER, Q2_EMITTER]] = 0.06
    node_v[TONE_LOW] = 3.5
    node_v[TONE_HIGH] = 0.6
    dc_q = reduction.voltage @ node_v
    for _ in range(100):
        q_v = reduction.voltage @ node_v
        current, first = nonlinear_terms(
            q_v, reduction.parameters, "reference_full", dc_q
        )
        residual = (
            reduction.linear_matrix @ node_v - reduction.source
            + reduction.injection @ current
        )
        norm = float(np.max(np.abs(residual)))
        if norm <= 1e-12:
            return node_v, q_v
        jacobian = reduction.linear_matrix + (
            reduction.injection * first[np.newaxis, :]
        ) @ reduction.voltage
        correction = np.linalg.solve(jacobian, -residual)
        damping = 1.0
        while damping >= 1.0 / 4096.0:
            candidate = node_v + damping * correction
            candidate_q = reduction.voltage @ candidate
            candidate_current, _ = nonlinear_terms(
                candidate_q, reduction.parameters, "reference_full", dc_q
            )
            candidate_residual = (
                reduction.linear_matrix @ candidate - reduction.source
                + reduction.injection @ candidate_current
            )
            if float(np.max(np.abs(candidate_residual))) < norm:
                node_v = candidate
                break
            damping *= 0.5
        else:
            raise RuntimeError("Рабочая точка входного тракта не сошлась")
    raise RuntimeError("Рабочая точка входного тракта не сошлась за 100 итераций")


def _step(
    reduction: FrontendReduction,
    previous_capacitor_v: np.ndarray,
    q_v: np.ndarray,
    dc_q_v: np.ndarray,
    input_v: float,
    architecture: Architecture,
    fully_converged: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if reduction.influence is None:
        raise ValueError("Для временного шага нужна динамическая матрица")
    rhs = (
        reduction.source + reduction.input_vector * input_v
        + reduction.capacitor_incidence @ (
            reduction.capacitor_conductance * previous_capacitor_v
        )
    )
    linear_nodes = np.linalg.solve(reduction.linear_matrix, rhs)
    linear_q = reduction.voltage @ linear_nodes
    maximum_iterations = 30 if fully_converged else 1
    correction_norm = 0.0
    for _ in range(maximum_iterations):
        current, first = nonlinear_terms(
            q_v, reduction.parameters, architecture, dc_q_v
        )
        residual = q_v - linear_q + reduction.influence @ current
        norm = float(np.max(np.abs(residual)))
        if fully_converged and norm <= 1e-12:
            break
        jacobian = np.eye(NONLINEAR_COUNT) + (
            reduction.influence * first[np.newaxis, :]
        )
        correction = np.linalg.solve(jacobian, -residual)
        correction_norm = max(correction_norm, float(np.max(np.abs(correction))))
        q_v = q_v + correction
    current, _ = nonlinear_terms(q_v, reduction.parameters, architecture, dc_q_v)
    node_v = np.linalg.solve(
        reduction.linear_matrix, rhs - reduction.injection @ current
    )
    capacitor_v = reduction.capacitor_incidence.T @ node_v
    current, _ = nonlinear_terms(q_v, reduction.parameters, architecture, dc_q_v)
    final_residual = q_v - linear_q + reduction.influence @ current
    return node_v, capacitor_v, q_v, float(np.max(np.abs(final_residual))), correction_norm


def simulate_frontend(
    sustain: float,
    factor: int,
    input_function: InputFunction,
    duration_s: float = 12e-3,
    architecture: Architecture = "hybrid",
    fully_converged: bool = False,
) -> FrontendResult:
    step_s = 1.0 / (48_000.0 * factor)
    reduction = prepare_frontend(sustain, step_s)
    dc_nodes, dc_q = operating_point(sustain, reduction.parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1) * step_s
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    node_v = np.empty((count + 1, NODE_COUNT))
    nonlinear_v = np.empty((count + 1, NONLINEAR_COUNT))
    residual_v = np.zeros(count + 1)
    correction_v = np.zeros(count + 1)
    node_v[0] = dc_nodes
    nonlinear_v[0] = dc_q
    capacitor_v = reduction.capacitor_incidence.T @ dc_nodes
    q_v = dc_q.copy()
    for index in range(1, count + 1):
        node_v[index], capacitor_v, q_v, residual_v[index], correction_v[index] = _step(
            reduction, capacitor_v, q_v, dc_q, float(input_v[index]),
            architecture, fully_converged
        )
        nonlinear_v[index] = q_v
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return FrontendResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v
    )


def simulate_frontend_switching(
    initial_sustain: float,
    factor: int,
    input_function: InputFunction,
    sustain_function: SustainFunction,
    duration_s: float,
    architecture: Architecture = "hybrid",
    fully_converged: bool = False,
) -> FrontendResult:
    """Считает тракт при скачках Sustain, сохраняя заряды и предикторы."""
    step_s = 1.0 / (48_000.0 * factor)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1) * step_s
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    sustain = np.asarray(sustain_function(time_s), dtype=np.float64)
    if sustain.shape != time_s.shape:
        raise ValueError("Функция Sustain должна вернуть по значению на каждый шаг")
    if np.any((sustain < 0.0) | (sustain > 1.0)):
        raise ValueError("Sustain должен находиться в диапазоне 0…1")

    parameters = Q3Parameters()
    reductions: dict[float, FrontendReduction] = {}

    def reduction_at(value: float) -> FrontendReduction:
        key = float(value)
        if key not in reductions:
            reductions[key] = prepare_frontend(key, step_s, parameters)
        return reductions[key]

    dc_nodes, dc_q = operating_point(initial_sustain, parameters)
    initial_reduction = reduction_at(initial_sustain)
    capacitor_v = initial_reduction.capacitor_incidence.T @ dc_nodes
    q_v = dc_q.copy()
    node_v = np.empty((count + 1, NODE_COUNT))
    nonlinear_v = np.empty((count + 1, NONLINEAR_COUNT))
    residual_v = np.zeros(count + 1)
    correction_v = np.zeros(count + 1)
    node_v[0] = dc_nodes
    nonlinear_v[0] = q_v
    for index in range(1, count + 1):
        reduction = reduction_at(float(sustain[index]))
        node_v[index], capacitor_v, q_v, residual_v[index], correction_v[index] = _step(
            reduction, capacitor_v, q_v, dc_q, float(input_v[index]),
            architecture, fully_converged
        )
        nonlinear_v[index] = q_v
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return FrontendResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v, sustain
    )


def small_signal_response(
    sustain: float, frequency_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reduction = prepare_frontend(sustain, None)
    dc_nodes, dc_q = operating_point(sustain, reduction.parameters)
    current, first = nonlinear_terms(
        dc_q, reduction.parameters, "reference_full", dc_q
    )
    del current
    dc_jacobian = reduction.linear_matrix + (
        reduction.injection * first[np.newaxis, :]
    ) @ reduction.voltage
    capacitance_matrix = (
        reduction.capacitor_incidence * reduction.capacitance[np.newaxis, :]
    ) @ reduction.capacitor_incidence.T
    q4 = np.empty(len(frequency_hz), dtype=np.complex128)
    q2 = np.empty(len(frequency_hz), dtype=np.complex128)
    for index, frequency in enumerate(frequency_hz):
        matrix = dc_jacobian + 2j * np.pi * frequency * capacitance_matrix
        response = np.linalg.solve(matrix, reduction.input_vector.astype(complex))
        q4[index] = response[Q4_COLLECTOR]
        q2[index] = response[Q2_COLLECTOR]
    return q4, q2


def discrete_stability_eigenvalues(
    sustain: float, factor: int = 8
) -> np.ndarray:
    reduction = prepare_frontend(sustain, 1.0 / (48_000.0 * factor))
    dc_nodes, dc_q = operating_point(sustain, reduction.parameters)
    dc_caps = reduction.capacitor_incidence.T @ dc_nodes
    state = np.concatenate((dc_caps, dc_q))

    def mapping(value: np.ndarray) -> np.ndarray:
        caps = value[: len(dc_caps)]
        q = value[len(dc_caps):]
        _, new_caps, new_q, _, _ = _step(
            reduction, caps, q, dc_q, 0.0, "hybrid", False
        )
        return np.concatenate((new_caps, new_q))

    base = mapping(state)
    jacobian = np.empty((len(state), len(state)))
    epsilon = 1e-6
    for column in range(len(state)):
        perturbed = state.copy()
        perturbed[column] += epsilon
        jacobian[:, column] = (mapping(perturbed) - base) / epsilon
    return np.linalg.eigvals(jacobian)


def discrete_stability_radius(sustain: float, factor: int = 8) -> float:
    eigenvalues = discrete_stability_eigenvalues(sustain, factor)
    return float(np.max(np.abs(eigenvalues)))
