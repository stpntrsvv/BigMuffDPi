"""Полная узловая модель от входного Q4 до выхода после Volume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from muff_frontend_model import (
    NODE_COUNT as FRONTEND_NODE_COUNT,
    NONLINEAR_COUNT as FRONTEND_NONLINEAR_COUNT,
    Q2_COLLECTOR,
    TONE_HIGH,
    TONE_LOW,
    nonlinear_terms as frontend_nonlinear_terms,
    operating_point as frontend_operating_point,
    prepare_frontend,
)
from q3_model import Q3Parameters


TONE_WIPER = 19
Q1_BASE = 20
Q1_COLLECTOR = 21
Q1_EMITTER = 22
VOLUME_TOP = 23
OUTPUT = 24
NODE_COUNT = 25
Q1_FORWARD = 9
Q1_REVERSE = 10
Q1_UNUSED = 11
NONLINEAR_COUNT = 12
Architecture = Literal["reference_full", "hybrid", "hybrid_q1_linear"]
IntegrationMethod = Literal["euler", "bdf2", "trapezoid", "trapezoid_adaptive"]
InputFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CompleteReduction:
    parameters: Q3Parameters
    sustain: float
    tone: float
    volume: float
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
class CompleteResult:
    time_s: np.ndarray
    input_v: np.ndarray
    node_v: np.ndarray
    nonlinear_v: np.ndarray
    residual_v: np.ndarray
    correction_v: np.ndarray
    q1_nonlinear_mix: np.ndarray | None = None
    integration_fallback_count: int = 0


def _branch(matrix: np.ndarray, first: int, second: int, conductance: float) -> None:
    matrix[first, first] += conductance
    matrix[second, second] += conductance
    matrix[first, second] -= conductance
    matrix[second, first] -= conductance


def _linear_system(
    parameters: Q3Parameters, sustain: float, tone: float, volume: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= tone <= 1.0 or not 0.0 <= volume <= 1.0:
        raise ValueError("Tone и Volume должны находиться в диапазоне 0…1")
    frontend = prepare_frontend(sustain, None, parameters)
    matrix = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    matrix[:FRONTEND_NODE_COUNT, :FRONTEND_NODE_COUNT] = frontend.linear_matrix
    source = np.zeros(NODE_COUNT, dtype=np.float64)
    source[:FRONTEND_NODE_COUNT] = frontend.source
    input_vector = np.zeros(NODE_COUNT, dtype=np.float64)
    input_vector[:FRONTEND_NODE_COUNT] = frontend.input_vector

    # В модели входного тракта потенциометр был только нагрузкой 100 кОм.
    direct_tone_conductance = 1.0 / 100_000.0
    matrix[TONE_HIGH, TONE_HIGH] -= direct_tone_conductance
    matrix[TONE_LOW, TONE_LOW] -= direct_tone_conductance
    matrix[TONE_HIGH, TONE_LOW] += direct_tone_conductance
    matrix[TONE_LOW, TONE_HIGH] += direct_tone_conductance

    tone_top = max(1.0, 100_000.0 * (1.0 - tone))
    tone_bottom = max(1.0, 100_000.0 * tone)
    _branch(matrix, TONE_HIGH, TONE_WIPER, 1.0 / tone_top)
    _branch(matrix, TONE_WIPER, TONE_LOW, 1.0 / tone_bottom)

    matrix[Q1_BASE, Q1_BASE] += 1.0 / 430_000.0 + 1.0 / 100_000.0
    source[Q1_BASE] += parameters.supply_v / 430_000.0
    matrix[Q1_COLLECTOR, Q1_COLLECTOR] += 1.0 / 15_000.0
    source[Q1_COLLECTOR] += parameters.supply_v / 15_000.0
    matrix[Q1_EMITTER, Q1_EMITTER] += 1.0 / 3_300.0

    volume_top = max(1.0, 100_000.0 * (1.0 - volume))
    volume_bottom = max(1.0, 100_000.0 * volume)
    _branch(matrix, VOLUME_TOP, OUTPUT, 1.0 / volume_top)
    matrix[OUTPUT, OUTPUT] += 1.0 / volume_bottom + 1.0 / 1_000_000.0
    return matrix, source, input_vector


def _nonlinear_matrices(
    parameters: Q3Parameters, sustain: float
) -> tuple[np.ndarray, np.ndarray]:
    frontend = prepare_frontend(sustain, None, parameters)
    injection = np.zeros((NODE_COUNT, NONLINEAR_COUNT), dtype=np.float64)
    voltage = np.zeros((NONLINEAR_COUNT, NODE_COUNT), dtype=np.float64)
    injection[:FRONTEND_NODE_COUNT, :FRONTEND_NONLINEAR_COUNT] = frontend.injection
    voltage[:FRONTEND_NONLINEAR_COUNT, :FRONTEND_NODE_COUNT] = frontend.voltage
    identity = np.eye(NODE_COUNT)
    injection[:, Q1_FORWARD] = (
        identity[Q1_BASE] * (1.0 - parameters.alpha_forward)
        + identity[Q1_COLLECTOR] * parameters.alpha_forward
        - identity[Q1_EMITTER]
    )
    injection[:, Q1_REVERSE] = (
        identity[Q1_BASE] * (1.0 - parameters.alpha_reverse)
        - identity[Q1_COLLECTOR]
        + identity[Q1_EMITTER] * parameters.alpha_reverse
    )
    voltage[Q1_FORWARD, Q1_BASE] = 1.0
    voltage[Q1_FORWARD, Q1_EMITTER] = -1.0
    voltage[Q1_REVERSE, Q1_BASE] = 1.0
    voltage[Q1_REVERSE, Q1_COLLECTOR] = -1.0
    return injection, voltage


def _capacitors(sustain: float, parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    frontend = prepare_frontend(sustain, None, parameters)
    columns = frontend.capacitor_incidence.shape[1]
    incidence = np.zeros((NODE_COUNT, columns + 2), dtype=np.float64)
    incidence[:FRONTEND_NODE_COUNT, :columns] = frontend.capacitor_incidence
    incidence[TONE_WIPER, columns] = 1.0
    incidence[Q1_BASE, columns] = -1.0
    incidence[Q1_COLLECTOR, columns + 1] = 1.0
    incidence[VOLUME_TOP, columns + 1] = -1.0
    capacitance = np.concatenate((frontend.capacitance, [100e-9, 100e-9]))
    return incidence, capacitance


def prepare_complete(
    sustain: float,
    tone: float,
    volume: float,
    step_s: float | None,
    parameters: Q3Parameters | None = None,
    integration_scale: float = 1.0,
) -> CompleteReduction:
    parameters = parameters or Q3Parameters()
    matrix, source, input_vector = _linear_system(parameters, sustain, tone, volume)
    injection, voltage = _nonlinear_matrices(parameters, sustain)
    incidence, capacitance = _capacitors(sustain, parameters)
    conductance = np.zeros_like(capacitance)
    influence = None
    if step_s is not None:
        conductance = integration_scale * capacitance / step_s
        matrix += (incidence * conductance[np.newaxis, :]) @ incidence.T
        influence = voltage @ np.linalg.solve(matrix, injection)
    return CompleteReduction(
        parameters, sustain, tone, volume, step_s, matrix, source, input_vector,
        injection, voltage, influence, incidence, capacitance, conductance
    )


def nonlinear_terms(
    q_v: np.ndarray,
    parameters: Q3Parameters,
    architecture: Architecture,
    dc_q_v: np.ndarray,
    q1_nonlinear_mix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.zeros(NONLINEAR_COUNT, dtype=np.float64)
    first = np.zeros(NONLINEAR_COUNT, dtype=np.float64)
    frontend_architecture = (
        "hybrid" if architecture.startswith("hybrid") else "reference_full"
    )
    current[:FRONTEND_NONLINEAR_COUNT], first[:FRONTEND_NONLINEAR_COUNT] = (
        frontend_nonlinear_terms(
            q_v[:FRONTEND_NONLINEAR_COUNT], parameters, frontend_architecture,
            dc_q_v[:FRONTEND_NONLINEAR_COUNT]
        )
    )
    forward_scale = parameters.forward_ideality * parameters.thermal_voltage_v
    reverse_scale = parameters.reverse_ideality * parameters.thermal_voltage_v
    forward = float(np.clip(q_v[Q1_FORWARD] / forward_scale, -80.0, 80.0))
    reverse = float(np.clip(q_v[Q1_REVERSE] / reverse_scale, -80.0, 80.0))
    current[Q1_FORWARD] = parameters.forward_saturation_current_a * np.expm1(forward)
    current[Q1_REVERSE] = parameters.reverse_saturation_current_a * np.expm1(reverse)
    first[Q1_FORWARD] = (
        parameters.forward_saturation_current_a * np.exp(forward) / forward_scale
    )
    first[Q1_REVERSE] = (
        parameters.reverse_saturation_current_a * np.exp(reverse) / reverse_scale
    )
    if architecture == "hybrid_q1_linear" or q1_nonlinear_mix < 1.0:
        dc_forward = float(np.clip(
            dc_q_v[Q1_FORWARD] / forward_scale, -80.0, 80.0
        ))
        dc_reverse = float(np.clip(
            dc_q_v[Q1_REVERSE] / reverse_scale, -80.0, 80.0
        ))
        dc_forward_current = (
            parameters.forward_saturation_current_a * np.expm1(dc_forward)
        )
        dc_reverse_current = (
            parameters.reverse_saturation_current_a * np.expm1(dc_reverse)
        )
        dc_forward_first = (
            parameters.forward_saturation_current_a * np.exp(dc_forward)
            / forward_scale
        )
        dc_reverse_first = (
            parameters.reverse_saturation_current_a * np.exp(dc_reverse)
            / reverse_scale
        )
        linear_forward_current = dc_forward_current + dc_forward_first * (
            q_v[Q1_FORWARD] - dc_q_v[Q1_FORWARD]
        )
        linear_reverse_current = dc_reverse_current + dc_reverse_first * (
            q_v[Q1_REVERSE] - dc_q_v[Q1_REVERSE]
        )
        mix = 0.0 if architecture == "hybrid_q1_linear" else q1_nonlinear_mix
        current[Q1_FORWARD] = (
            mix * current[Q1_FORWARD] + (1.0 - mix) * linear_forward_current
        )
        current[Q1_REVERSE] = (
            mix * current[Q1_REVERSE] + (1.0 - mix) * linear_reverse_current
        )
        first[Q1_FORWARD] = (
            mix * first[Q1_FORWARD] + (1.0 - mix) * dc_forward_first
        )
        first[Q1_REVERSE] = (
            mix * first[Q1_REVERSE] + (1.0 - mix) * dc_reverse_first
        )
    return current, first


def operating_point(
    sustain: float, tone: float, volume: float,
    parameters: Q3Parameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reduction = prepare_complete(sustain, tone, volume, None, parameters)
    node_v = np.zeros(NODE_COUNT, dtype=np.float64)
    node_v[Q1_BASE] = 1.5
    node_v[Q1_EMITTER] = 0.8
    node_v[Q1_COLLECTOR] = 4.5
    frontend_dc, _ = frontend_operating_point(sustain, reduction.parameters)
    node_v[:FRONTEND_NODE_COUNT] = frontend_dc
    dc_q = reduction.voltage @ node_v
    for _ in range(120):
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
            raise RuntimeError("Рабочая точка полной педали не сошлась")
    raise RuntimeError("Рабочая точка полной педали не сошлась за 120 итераций")


def _step(
    reduction: CompleteReduction,
    previous_capacitor_v: np.ndarray,
    q_v: np.ndarray,
    dc_q_v: np.ndarray,
    input_v: float,
    architecture: Architecture,
    fully_converged: bool,
    q1_local_corrections: int = 0,
    fixed_corrections: int = 1,
    q1_nonlinear_mix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if reduction.influence is None:
        raise ValueError("Для шага нужна динамическая матрица")
    rhs = (
        reduction.source + reduction.input_vector * input_v
        + reduction.capacitor_incidence @ (
            reduction.capacitor_conductance * previous_capacitor_v
        )
    )
    linear_nodes = np.linalg.solve(reduction.linear_matrix, rhs)
    linear_q = reduction.voltage @ linear_nodes
    correction_norm = 0.0
    for _ in range(30 if fully_converged else fixed_corrections):
        current, first = nonlinear_terms(
            q_v, reduction.parameters, architecture, dc_q_v, q1_nonlinear_mix
        )
        residual = q_v - linear_q + reduction.influence @ current
        if fully_converged and float(np.max(np.abs(residual))) <= 1e-12:
            break
        jacobian = np.eye(NONLINEAR_COUNT) + reduction.influence * first[np.newaxis, :]
        correction = np.linalg.solve(jacobian, -residual)
        correction_norm = max(correction_norm, float(np.max(np.abs(correction))))
        q_v = q_v + correction
    q1_indices = np.array([Q1_FORWARD, Q1_REVERSE])
    for _ in range(q1_local_corrections):
        current, first = nonlinear_terms(
            q_v, reduction.parameters, architecture, dc_q_v, q1_nonlinear_mix
        )
        residual = q_v - linear_q + reduction.influence @ current
        jacobian = np.eye(NONLINEAR_COUNT) + (
            reduction.influence * first[np.newaxis, :]
        )
        local_jacobian = jacobian[np.ix_(q1_indices, q1_indices)]
        local_correction = np.linalg.solve(
            local_jacobian, -residual[q1_indices]
        )
        correction_norm = max(
            correction_norm, float(np.max(np.abs(local_correction)))
        )
        q_v[q1_indices] += local_correction
    current, _ = nonlinear_terms(
        q_v, reduction.parameters, architecture, dc_q_v, q1_nonlinear_mix
    )
    node_v = np.linalg.solve(
        reduction.linear_matrix, rhs - reduction.injection @ current
    )
    capacitor_v = reduction.capacitor_incidence.T @ node_v
    final_current, _ = nonlinear_terms(
        q_v, reduction.parameters, architecture, dc_q_v, q1_nonlinear_mix
    )
    final_residual = q_v - linear_q + reduction.influence @ final_current
    return (
        node_v, capacitor_v, q_v,
        float(np.max(np.abs(final_residual))), correction_norm
    )


def simulate_complete(
    sustain: float,
    tone: float,
    volume: float,
    factor: int,
    input_function: InputFunction,
    duration_s: float = 12e-3,
    architecture: Architecture = "hybrid",
    fully_converged: bool = False,
    q1_local_corrections: int = 0,
    fixed_corrections: int = 1,
    integration_method: IntegrationMethod = "euler",
) -> CompleteResult:
    if integration_method not in ("euler", "bdf2", "trapezoid", "trapezoid_adaptive"):
        raise ValueError(f"Неизвестный способ интегрирования: {integration_method}")
    step_s = 1.0 / (48_000.0 * factor)
    euler_reduction = prepare_complete(sustain, tone, volume, step_s)
    scale = {
        "euler": 1.0, "bdf2": 1.5,
        "trapezoid": 2.0, "trapezoid_adaptive": 2.0,
    }[integration_method]
    reduction = prepare_complete(
        sustain, tone, volume, step_s, integration_scale=scale
    )
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
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
    older_capacitor_v = capacitor_v.copy()
    capacitor_current = np.zeros_like(capacitor_v)
    q_v = dc_q.copy()
    integration_fallback_count = 0
    for index in range(1, count + 1):
        used_euler_fallback = False
        previous_capacitor_v = capacitor_v
        if integration_method == "bdf2" and index > 1:
            history_v = (4.0 * previous_capacitor_v - older_capacitor_v) / 3.0
            step_reduction = reduction
        elif integration_method in ("trapezoid", "trapezoid_adaptive"):
            history_v = previous_capacitor_v + (
                capacitor_current / reduction.capacitor_conductance
            )
            step_reduction = reduction
        else:
            history_v = previous_capacitor_v
            step_reduction = euler_reduction
        candidate = _step(
            step_reduction, history_v, q_v, dc_q, float(input_v[index]),
            architecture, fully_converged, q1_local_corrections,
            fixed_corrections
        )
        if integration_method == "trapezoid_adaptive" and (
            not np.all(np.isfinite(candidate[0]))
            or float(np.max(np.abs(candidate[0]))) > 12.0
            or candidate[3] > 1e-7
            or candidate[4] > 0.5
        ):
            candidate = _step(
                euler_reduction, previous_capacitor_v, q_v.copy(), dc_q,
                float(input_v[index]), architecture, fully_converged,
                q1_local_corrections, fixed_corrections
            )
            integration_fallback_count += 1
            used_euler_fallback = True
        node_v[index], capacitor_v, q_v, residual_v[index], correction_v[index] = candidate
        older_capacitor_v = previous_capacitor_v
        if integration_method in ("trapezoid", "trapezoid_adaptive"):
            if used_euler_fallback:
                capacitor_current = (
                    euler_reduction.capacitor_conductance
                    * (capacitor_v - previous_capacitor_v)
                )
            else:
                capacitor_current = (
                    reduction.capacitor_conductance
                    * (capacitor_v - previous_capacitor_v)
                    - capacitor_current
                )
        nonlinear_v[index] = q_v
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return CompleteResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v,
        integration_fallback_count=integration_fallback_count
    )


def simulate_complete_generalized_alpha(
    sustain: float,
    tone: float,
    volume: float,
    factor: int,
    input_function: InputFunction,
    rho_infinity: float,
    duration_s: float = 12e-3,
    architecture: Architecture = "hybrid",
    fully_converged: bool = False,
) -> CompleteResult:
    """Обобщённый α-метод второго порядка для системы первого порядка."""
    if not 0.0 <= rho_infinity <= 1.0:
        raise ValueError("rho_infinity должен находиться в диапазоне 0…1")
    step_s = 1.0 / (48_000.0 * factor)
    alpha_m = 0.5 * (3.0 - rho_infinity) / (1.0 + rho_infinity)
    alpha_f = 1.0 / (1.0 + rho_infinity)
    gamma = 0.5 + alpha_m - alpha_f
    conductance_scale = alpha_m / (gamma * alpha_f)
    reduction = prepare_complete(
        sustain, tone, volume, step_s,
        integration_scale=conductance_scale,
    )
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
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
    capacitor_derivative = np.zeros_like(capacitor_v)
    q_v = dc_q.copy()
    derivative_history_scale = (
        alpha_f * step_s * (alpha_m - gamma) / alpha_m
    )
    for index in range(1, count + 1):
        previous_nodes = node_v[index - 1]
        previous_caps = capacitor_v
        previous_q = q_v
        history_v = previous_caps + derivative_history_scale * capacitor_derivative
        stage_input = (
            (1.0 - alpha_f) * input_v[index - 1]
            + alpha_f * input_v[index]
        )
        stage_nodes, stage_caps, stage_q, residual, correction = _step(
            reduction, history_v, previous_q.copy(), dc_q, float(stage_input),
            architecture, fully_converged,
        )
        node_v[index] = previous_nodes + (stage_nodes - previous_nodes) / alpha_f
        capacitor_v = previous_caps + (stage_caps - previous_caps) / alpha_f
        q_v = previous_q + (stage_q - previous_q) / alpha_f
        capacitor_derivative = (
            (capacitor_v - previous_caps) / step_s
            - (1.0 - gamma) * capacitor_derivative
        ) / gamma
        nonlinear_v[index] = q_v
        residual_v[index] = residual
        correction_v[index] = correction
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(
                f"Нечисловой результат обобщённого α-метода на шаге {index}"
            )
    return CompleteResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v
    )


def simulate_complete_adaptive_q1(
    sustain: float,
    tone: float,
    volume: float,
    factor: int,
    input_function: InputFunction,
    duration_s: float,
    enter_linear_residual_v: float = 0.15,
    emergency_residual_v: float = 0.50,
    leave_linear_residual_v: float = 0.10,
    hold_samples: int = 384,
    fade_samples: int = 128,
    saturation_warning_v: float = 0.48,
    saturation_leave_v: float = 0.30,
) -> CompleteResult:
    """Плавно уводит Q1 в линейную ветвь при опасной невязке."""
    step_s = 1.0 / (48_000.0 * factor)
    reduction = prepare_complete(sustain, tone, volume, step_s)
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1) * step_s
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    node_v = np.empty((count + 1, NODE_COUNT))
    nonlinear_v = np.empty((count + 1, NONLINEAR_COUNT))
    residual_v = np.zeros(count + 1)
    correction_v = np.zeros(count + 1)
    mix_v = np.ones(count + 1)
    node_v[0] = dc_nodes
    nonlinear_v[0] = dc_q
    capacitor_v = reduction.capacitor_incidence.T @ dc_nodes
    q_v = dc_q.copy()
    mix = 1.0
    hold = 0
    envelope = 0.0
    saturation_envelope = 0.0
    envelope_decay = float(np.exp(-step_s / 2e-3))
    fade_step = 1.0 / max(1, fade_samples)
    for index in range(1, count + 1):
        previous_caps = capacitor_v
        previous_q = q_v
        saturation_envelope = max(
            saturation_envelope * envelope_decay, float(previous_q[Q1_REVERSE])
        )
        if hold > 0:
            hold -= 1
        elif (
            envelope < leave_linear_residual_v
            and saturation_envelope < saturation_leave_v
        ):
            mix = min(1.0, mix + fade_step)
        if saturation_envelope > saturation_warning_v:
            mix = max(0.0, mix - fade_step)
        candidate = _step(
            reduction, previous_caps, previous_q.copy(), dc_q,
            float(input_v[index]), "hybrid", False, 1, 1, mix
        )
        unsafe = (
            not np.all(np.isfinite(candidate[0]))
            or candidate[3] > emergency_residual_v
        )
        if not unsafe and candidate[3] > enter_linear_residual_v:
            mix = max(0.0, mix - fade_step)
            candidate = _step(
                reduction, previous_caps, previous_q.copy(), dc_q,
                float(input_v[index]), "hybrid", False, 1, 1, mix
            )
            unsafe = (
                not np.all(np.isfinite(candidate[0]))
                or candidate[3] > emergency_residual_v
            )
        if unsafe:
            mix = 0.0
            hold = hold_samples
            candidate = _step(
                reduction, previous_caps, previous_q.copy(), dc_q,
                float(input_v[index]), "hybrid", False, 0, 1, 0.0
            )
        node_v[index], capacitor_v, q_v, residual_v[index], correction_v[index] = candidate
        nonlinear_v[index] = q_v
        mix_v[index] = mix
        envelope = max(envelope * envelope_decay, residual_v[index])
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return CompleteResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v, mix_v
    )


def small_signal_response(
    sustain: float, tone: float, volume: float, frequency_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduction = prepare_complete(sustain, tone, volume, None)
    _, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
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
    tone_wiper = np.empty(len(frequency_hz), dtype=np.complex128)
    q1_collector = np.empty_like(tone_wiper)
    output = np.empty_like(tone_wiper)
    for index, frequency in enumerate(frequency_hz):
        matrix = dc_jacobian + 2j * np.pi * frequency * capacitance_matrix
        response = np.linalg.solve(matrix, reduction.input_vector.astype(complex))
        tone_wiper[index] = response[TONE_WIPER]
        q1_collector[index] = response[Q1_COLLECTOR]
        output[index] = response[OUTPUT]
    return tone_wiper, q1_collector, output


def discrete_stability_eigenvalues(
    sustain: float, tone: float, volume: float, factor: int = 8
) -> np.ndarray:
    reduction = prepare_complete(
        sustain, tone, volume, 1.0 / (48_000.0 * factor)
    )
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
    dc_caps = reduction.capacitor_incidence.T @ dc_nodes
    state = np.concatenate((dc_caps, dc_q))

    def mapping(value: np.ndarray) -> np.ndarray:
        caps = value[:len(dc_caps)]
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
