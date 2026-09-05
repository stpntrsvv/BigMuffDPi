"""Портовая многоскоростная модель: быстрый входной тракт и медленный Tone–Q1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muff_complete_model import (
    OUTPUT,
    Q1_FORWARD,
    Q1_REVERSE,
    Q1_BASE,
    Q1_COLLECTOR,
    Q1_EMITTER,
    TONE_WIPER,
    VOLUME_TOP,
    operating_point,
    prepare_complete,
)
from muff_frontend_model import (
    NONLINEAR_COUNT as FRONTEND_NONLINEAR_COUNT,
    Q2_COLLECTOR,
    TONE_HIGH,
    TONE_LOW,
    nonlinear_terms as frontend_nonlinear_terms,
    prepare_frontend,
)
from q3_model import Q3Parameters


FAST_NODE_COUNT = TONE_HIGH
SLOW_NODES = np.array(
    [TONE_HIGH, TONE_LOW, TONE_WIPER, Q1_BASE, Q1_COLLECTOR,
     Q1_EMITTER, VOLUME_TOP, OUTPUT],
    dtype=np.intp,
)
SLOW_CAPACITORS = np.array([9, 10, 11, 12], dtype=np.intp)
Q1_NONLINEAR = np.array([Q1_FORWARD, Q1_REVERSE], dtype=np.intp)


@dataclass(frozen=True)
class FastReduction:
    parameters: Q3Parameters
    matrix: np.ndarray
    source: np.ndarray
    input_vector: np.ndarray
    injection: np.ndarray
    voltage: np.ndarray
    influence: np.ndarray
    incidence: np.ndarray
    conductance: np.ndarray


@dataclass(frozen=True)
class SlowReduction:
    parameters: Q3Parameters
    matrix: np.ndarray
    source: np.ndarray
    port_column: np.ndarray
    injection: np.ndarray
    voltage: np.ndarray
    influence: np.ndarray
    incidence: np.ndarray
    conductance: np.ndarray
    resistor_conductance: float


@dataclass(frozen=True)
class MultirateResult:
    time_s: np.ndarray
    input_v: np.ndarray
    output_v: np.ndarray
    port_v: np.ndarray
    port_i: np.ndarray
    port_g: np.ndarray
    maximum_fast_residual_v: float
    maximum_slow_residual_v: float


def prepare_fast(
    parameters: Q3Parameters, step_s: float, port_conductance: float,
    capacitance_scale: np.ndarray | None = None, sustain: float = 1.0,
) -> FastReduction:
    frontend = prepare_frontend(sustain, None, parameters)
    matrix = frontend.linear_matrix[:FAST_NODE_COUNT, :FAST_NODE_COUNT].copy()
    # Удаляем прежнюю резистивную нагрузку темброблока: её заменяет порт Нортона.
    matrix[Q2_COLLECTOR, Q2_COLLECTOR] -= 1.0 / 39_000.0
    matrix[Q2_COLLECTOR, Q2_COLLECTOR] += port_conductance
    incidence = frontend.capacitor_incidence[:FAST_NODE_COUNT, :9].copy()
    capacitance = frontend.capacitance[:9].copy()
    if capacitance_scale is not None:
        scale = np.asarray(capacitance_scale, dtype=np.float64)
        if scale.shape != (9,):
            raise ValueError("Для быстрого ядра нужны девять масштабов ёмкости")
        capacitance *= scale
    conductance = capacitance / step_s
    matrix += (incidence * conductance[np.newaxis, :]) @ incidence.T
    injection = frontend.injection[:FAST_NODE_COUNT].copy()
    voltage = frontend.voltage[:, :FAST_NODE_COUNT].copy()
    influence = voltage @ np.linalg.solve(matrix, injection)
    return FastReduction(
        parameters, matrix, frontend.source[:FAST_NODE_COUNT].copy(),
        frontend.input_vector[:FAST_NODE_COUNT].copy(), injection, voltage,
        influence, incidence, conductance,
    )


def prepare_slow(
    parameters: Q3Parameters, tone: float, volume: float, step_s: float
) -> SlowReduction:
    complete = prepare_complete(1.0, tone, volume, step_s, parameters)
    matrix = complete.linear_matrix[np.ix_(SLOW_NODES, SLOW_NODES)]
    port_column = complete.linear_matrix[SLOW_NODES, Q2_COLLECTOR]
    injection = complete.injection[np.ix_(SLOW_NODES, Q1_NONLINEAR)]
    voltage = complete.voltage[np.ix_(Q1_NONLINEAR, SLOW_NODES)]
    influence = voltage @ np.linalg.solve(matrix, injection)
    # Первый ряд — порт Q2, остальные — внутренние узлы медленной подсистемы.
    port_and_slow = np.concatenate(([Q2_COLLECTOR], SLOW_NODES))
    incidence = complete.capacitor_incidence[np.ix_(port_and_slow, SLOW_CAPACITORS)]
    conductance = complete.capacitor_conductance[SLOW_CAPACITORS]
    return SlowReduction(
        parameters, matrix, complete.source[SLOW_NODES].copy(), port_column.copy(),
        injection, voltage, influence, incidence, conductance, 1.0 / 39_000.0,
    )


def _q1_terms(q_v: np.ndarray, parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    scale_f = parameters.forward_ideality * parameters.thermal_voltage_v
    scale_r = parameters.reverse_ideality * parameters.thermal_voltage_v
    arguments = np.array([
        np.clip(q_v[0] / scale_f, -80.0, 80.0),
        np.clip(q_v[1] / scale_r, -80.0, 80.0),
    ])
    saturation = np.array([
        parameters.forward_saturation_current_a,
        parameters.reverse_saturation_current_a,
    ])
    scales = np.array([scale_f, scale_r])
    current = saturation * np.expm1(arguments)
    first = saturation * np.exp(arguments) / scales
    return current, first


def _reduced_newton(
    q_v: np.ndarray,
    linear_q: np.ndarray,
    influence: np.ndarray,
    terms,
    fully_converged: bool,
) -> tuple[np.ndarray, float]:
    q_v = q_v.copy()
    for _ in range(30 if fully_converged else 1):
        current, first = terms(q_v)
        residual = q_v - linear_q + influence @ current
        norm = float(np.max(np.abs(residual)))
        if fully_converged and norm <= 1.0e-12:
            return q_v, norm
        correction = np.linalg.solve(
            np.eye(len(q_v)) + influence * first[np.newaxis, :], -residual
        )
        if not fully_converged:
            q_v += correction
            break
        damping = 1.0
        while damping >= 1.0 / 4096.0:
            candidate = q_v + damping * correction
            candidate_current, _ = terms(candidate)
            candidate_residual = candidate - linear_q + influence @ candidate_current
            if float(np.max(np.abs(candidate_residual))) < norm:
                q_v = candidate
                break
            damping *= 0.5
        else:
            break
    current, _ = terms(q_v)
    residual = q_v - linear_q + influence @ current
    return q_v, float(np.max(np.abs(residual)))


def fast_step(
    reduction: FastReduction,
    capacitor_v: np.ndarray,
    q_v: np.ndarray,
    dc_q_v: np.ndarray,
    input_v: float,
    port_offset_a: float,
    fully_converged: bool,
    linear_q4: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rhs = (
        reduction.source + reduction.input_vector * input_v
        + reduction.incidence @ (reduction.conductance * capacitor_v)
    )
    rhs[Q2_COLLECTOR] -= port_offset_a
    linear_nodes = np.linalg.solve(reduction.matrix, rhs)
    linear_q = reduction.voltage @ linear_nodes

    def terms(value):
        current, first = frontend_nonlinear_terms(
            value, reduction.parameters, "hybrid", dc_q_v
        )
        if linear_q4:
            dc_current, dc_first = frontend_nonlinear_terms(
                dc_q_v, reduction.parameters, "hybrid", dc_q_v
            )
            current[0] = dc_current[0] + dc_first[0] * (value[0] - dc_q_v[0])
            first[0] = dc_first[0]
        return current, first

    q_v, residual = _reduced_newton(
        q_v, linear_q, reduction.influence, terms, fully_converged
    )
    current, _ = terms(q_v)
    nodes = np.linalg.solve(
        reduction.matrix, rhs - reduction.injection @ current
    )
    return nodes, reduction.incidence.T @ nodes, q_v, residual


def slow_step(
    reduction: SlowReduction,
    capacitor_v: np.ndarray,
    q_v: np.ndarray,
    port_v: float,
    fully_converged: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    history_full = reduction.incidence @ (reduction.conductance * capacitor_v)
    rhs = reduction.source + history_full[1:] - reduction.port_column * port_v
    linear_nodes = np.linalg.solve(reduction.matrix, rhs)
    linear_q = reduction.voltage @ linear_nodes

    def terms(value):
        return _q1_terms(value, reduction.parameters)

    q_v, residual = _reduced_newton(
        q_v, linear_q, reduction.influence, terms, fully_converged
    )
    current, first = terms(q_v)
    nodes = np.linalg.solve(
        reduction.matrix, rhs - reduction.injection @ current
    )
    # Ток из быстрого порта в R5 и C9.
    high = nodes[0]
    low = nodes[1]
    port_current = (
        reduction.resistor_conductance * (port_v - low)
        + reduction.conductance[0] * (port_v - high - capacitor_v[0])
    )
    nodal_jacobian = reduction.matrix + (
        reduction.injection * first[np.newaxis, :]
    ) @ reduction.voltage
    node_sensitivity = np.linalg.solve(nodal_jacobian, -reduction.port_column)
    port_conductance = (
        reduction.resistor_conductance * (1.0 - node_sensitivity[1])
        + reduction.conductance[0] * (1.0 - node_sensitivity[0])
    )
    all_nodes = np.concatenate(([port_v], nodes))
    next_capacitor_v = reduction.incidence.T @ all_nodes
    return (
        nodes, next_capacitor_v, q_v, residual,
        float(port_current), float(port_conductance),
    )


def simulate_multirate(
    input_function,
    duration_s: float,
    factor: int = 4,
    slow_factor: int = 1,
    tone: float = 1.0,
    volume: float = 0.8,
    fully_converged: bool = True,
    linear_q4: bool = False,
    fixed_port_conductance: bool = False,
) -> MultirateResult:
    if factor % slow_factor != 0:
        raise ValueError("Частота быстрого ядра должна быть кратна частоте медленного")
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, tone, volume, parameters)
    slow = prepare_slow(parameters, tone, volume, 1.0 / (48_000.0 * slow_factor))
    slow_nodes = dc_nodes[SLOW_NODES].copy()
    slow_caps = slow.incidence.T @ np.concatenate(
        ([dc_nodes[Q2_COLLECTOR]], slow_nodes)
    )
    slow_q = dc_q[Q1_NONLINEAR].copy()
    # На постоянном токе C9 разомкнут, ток порта задаёт R5.
    port_v = float(dc_nodes[Q2_COLLECTOR])
    _, _, _, _, port_i, port_g = slow_step(
        slow, slow_caps, slow_q, port_v, True
    )
    port_offset = port_i - port_g * port_v
    load_conductance = port_g

    fast = prepare_fast(parameters, 1.0 / (48_000.0 * factor), port_g)
    fast_nodes = dc_nodes[:FAST_NODE_COUNT].copy()
    fast_caps = fast.incidence.T @ fast_nodes
    fast_q = dc_q[:FRONTEND_NONLINEAR_COUNT].copy()

    count = int(round(duration_s * 48_000.0))
    time_s = np.arange(count + 1, dtype=np.float64) / 48_000.0
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    output = np.empty(count + 1, dtype=np.float64)
    port_voltage = np.empty(count + 1, dtype=np.float64)
    port_current = np.empty(count + 1, dtype=np.float64)
    port_conductance = np.empty(count + 1, dtype=np.float64)
    output[0] = dc_nodes[OUTPUT]
    port_voltage[0] = port_v
    port_current[0] = port_i
    port_conductance[0] = port_g
    maximum_fast_residual = 0.0
    maximum_slow_residual = 0.0

    for sample in range(1, count + 1):
        # Касательная проводимость меняется раз в внешний шаг. Матрица обновляется
        # здесь явно; на контроллере это заменяется формулой Шермана—Моррисона.
        previous_input = input_v[sample - 1]
        next_input = input_v[sample]
        fast_per_slow = factor // slow_factor
        for slow_index in range(slow_factor):
            fast = prepare_fast(
                parameters, 1.0 / (48_000.0 * factor), load_conductance
            )
            for local_fast in range(1, fast_per_slow + 1):
                completed_fast = slow_index * fast_per_slow + local_fast
                fraction = completed_fast / factor
                interpolated = previous_input + fraction * (next_input - previous_input)
                fast_nodes, fast_caps, fast_q, residual = fast_step(
                    fast, fast_caps, fast_q, dc_q[:FRONTEND_NONLINEAR_COUNT],
                    float(interpolated), port_offset, fully_converged, linear_q4,
                )
                maximum_fast_residual = max(maximum_fast_residual, residual)
            port_v = float(fast_nodes[Q2_COLLECTOR])
            slow_nodes, slow_caps, slow_q, residual, port_i, port_g = slow_step(
                slow, slow_caps, slow_q, port_v, fully_converged
            )
            maximum_slow_residual = max(maximum_slow_residual, residual)
            if not fixed_port_conductance:
                load_conductance = port_g
            port_offset = port_i - load_conductance * port_v
        output[sample] = slow_nodes[-1]
        port_voltage[sample] = port_v
        port_current[sample] = port_i
        port_conductance[sample] = port_g
        if not np.all(np.isfinite((output[sample], port_v, port_i, port_g))):
            raise FloatingPointError(f"Нечисловой результат на внешнем шаге {sample}")

    return MultirateResult(
        time_s, input_v, output, port_voltage, port_current, port_conductance,
        maximum_fast_residual, maximum_slow_residual,
    )
