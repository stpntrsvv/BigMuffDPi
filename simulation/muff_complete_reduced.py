"""Точно исключает линейные узлы из динамического шага полной модели."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muff_complete_model import (
    Architecture,
    CompleteResult,
    InputFunction,
    NODE_COUNT,
    NONLINEAR_COUNT,
    OUTPUT,
    Q1_FORWARD,
    Q1_REVERSE,
    nonlinear_terms,
    operating_point,
    prepare_complete,
)


@dataclass(frozen=True)
class CompleteAffineStep:
    """Аффинное описание линейной части одного шага назадного Эйлера."""

    parameters: object
    node_bias: np.ndarray
    node_input: np.ndarray
    node_state: np.ndarray
    node_nonlinear: np.ndarray
    q_bias: np.ndarray
    q_input: np.ndarray
    q_state: np.ndarray
    influence: np.ndarray
    state_bias: np.ndarray
    state_input: np.ndarray
    state_transition: np.ndarray
    state_nonlinear: np.ndarray
    output_bias: float
    output_input: float
    output_state: np.ndarray
    output_nonlinear: np.ndarray

    @property
    def state_count(self) -> int:
        return len(self.state_bias)


def prepare_affine_step(
    sustain: float, tone: float, volume: float, factor: int
) -> CompleteAffineStep:
    step_s = 1.0 / (48_000.0 * factor)
    reduction = prepare_complete(sustain, tone, volume, step_s)
    history = reduction.capacitor_incidence * (
        reduction.capacitor_conductance[np.newaxis, :]
    )
    right = np.column_stack((
        reduction.source,
        reduction.input_vector,
        history,
        reduction.injection,
    ))
    solved = np.linalg.solve(reduction.linear_matrix, right)
    state_count = history.shape[1]
    node_bias = solved[:, 0]
    node_input = solved[:, 1]
    node_state = solved[:, 2:2 + state_count]
    node_nonlinear = solved[:, 2 + state_count:]

    q_bias = reduction.voltage @ node_bias
    q_input = reduction.voltage @ node_input
    q_state = reduction.voltage @ node_state
    influence = reduction.voltage @ node_nonlinear
    incidence_t = reduction.capacitor_incidence.T
    state_bias = incidence_t @ node_bias
    state_input = incidence_t @ node_input
    state_transition = incidence_t @ node_state
    state_nonlinear = incidence_t @ node_nonlinear
    return CompleteAffineStep(
        reduction.parameters,
        node_bias, node_input, node_state, node_nonlinear,
        q_bias, q_input, q_state, influence,
        state_bias, state_input, state_transition, state_nonlinear,
        float(node_bias[OUTPUT]), float(node_input[OUTPUT]),
        node_state[OUTPUT].copy(), node_nonlinear[OUTPUT].copy(),
    )


def step_affine(
    reduction: CompleteAffineStep,
    previous_state_v: np.ndarray,
    q_v: np.ndarray,
    dc_q_v: np.ndarray,
    input_v: float,
    architecture: Architecture = "hybrid",
    fully_converged: bool = False,
    q1_local_corrections: int = 0,
    fixed_corrections: int = 1,
    q1_nonlinear_mix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    linear_q = (
        reduction.q_bias + reduction.q_input * input_v
        + reduction.q_state @ previous_state_v
    )
    correction_norm = 0.0
    for _ in range(30 if fully_converged else fixed_corrections):
        current, first = nonlinear_terms(
            q_v, reduction.parameters, architecture, dc_q_v, q1_nonlinear_mix
        )
        residual = q_v - linear_q + reduction.influence @ current
        if fully_converged and float(np.max(np.abs(residual))) <= 1e-12:
            break
        jacobian = np.eye(NONLINEAR_COUNT) + (
            reduction.influence * first[np.newaxis, :]
        )
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
    state_v = (
        reduction.state_bias + reduction.state_input * input_v
        + reduction.state_transition @ previous_state_v
        - reduction.state_nonlinear @ current
    )
    node_v = (
        reduction.node_bias + reduction.node_input * input_v
        + reduction.node_state @ previous_state_v
        - reduction.node_nonlinear @ current
    )
    final_current, _ = nonlinear_terms(
        q_v, reduction.parameters, architecture, dc_q_v, q1_nonlinear_mix
    )
    final_residual = q_v - linear_q + reduction.influence @ final_current
    return (
        node_v, state_v, q_v,
        float(np.max(np.abs(final_residual))), correction_norm,
    )


def simulate_complete_affine(
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
) -> CompleteResult:
    reduction = prepare_affine_step(sustain, tone, volume, factor)
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
    state_v = prepare_complete(
        sustain, tone, volume, 1.0 / (48_000.0 * factor)
    ).capacitor_incidence.T @ dc_nodes
    count = int(round(duration_s * 48_000.0 * factor))
    time_s = np.arange(count + 1) / (48_000.0 * factor)
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    node_v = np.empty((count + 1, NODE_COUNT))
    nonlinear_v = np.empty((count + 1, NONLINEAR_COUNT))
    residual_v = np.zeros(count + 1)
    correction_v = np.zeros(count + 1)
    node_v[0] = dc_nodes
    nonlinear_v[0] = dc_q
    q_v = dc_q.copy()
    for index in range(1, count + 1):
        node_v[index], state_v, q_v, residual_v[index], correction_v[index] = (
            step_affine(
                reduction, state_v, q_v, dc_q, float(input_v[index]),
                architecture, fully_converged, q1_local_corrections,
                fixed_corrections,
            )
        )
        nonlinear_v[index] = q_v
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return CompleteResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v
    )
