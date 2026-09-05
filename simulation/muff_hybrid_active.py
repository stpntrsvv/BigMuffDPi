"""Сокращает гибридную модель полной педали до шести активных нелинейностей."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from muff_complete_model import (
    CompleteResult,
    InputFunction,
    NODE_COUNT,
    NONLINEAR_COUNT,
    operating_point,
    prepare_complete,
    nonlinear_terms,
)
from muff_complete_reduced import prepare_affine_step


# Q4 I_F; Q3 I_F и диоды; Q2 диоды; Q1 I_F и I_R.
ACTIVE = np.array([0, 3, 5, 8, 9, 10], dtype=np.intp)
Q1_ACTIVE = np.array([4, 5], dtype=np.intp)
ACTIVE_COUNT = len(ACTIVE)
CASCADE_BLOCKS = (
    np.array([0], dtype=np.intp),
    np.array([1, 2], dtype=np.intp),
    np.array([3], dtype=np.intp),
    np.array([4, 5], dtype=np.intp),
)
COUPLED_INPUT_BLOCKS = (
    np.array([0, 1, 2], dtype=np.intp),
    np.array([3], dtype=np.intp),
    np.array([4, 5], dtype=np.intp),
)
FRONTEND_BLOCKS = (
    np.array([0, 1, 2, 3], dtype=np.intp),
    np.array([4, 5], dtype=np.intp),
)
COUPLED_INPUT_Q1_TWICE = COUPLED_INPUT_BLOCKS + (COUPLED_INPUT_BLOCKS[-1],)
FRONTEND_Q1_TWICE = FRONTEND_BLOCKS + (FRONTEND_BLOCKS[-1],)


@dataclass(frozen=True)
class HybridActiveStep:
    parameters: object
    dc_q: np.ndarray
    active_bias: np.ndarray
    active_input: np.ndarray
    active_state: np.ndarray
    active_influence: np.ndarray
    full_q_bias: np.ndarray
    full_q_input: np.ndarray
    full_q_state: np.ndarray
    full_q_active: np.ndarray
    node_bias: np.ndarray
    node_input: np.ndarray
    node_state: np.ndarray
    node_active: np.ndarray
    state_bias: np.ndarray
    state_input: np.ndarray
    state_transition: np.ndarray
    state_active: np.ndarray


@dataclass(frozen=True)
class ModalStateTransform:
    """Замена напряжений конденсаторов независимыми линейными режимами."""

    reduction: HybridActiveStep
    to_physical: np.ndarray
    to_modal: np.ndarray
    poles: np.ndarray

    def encode(self, physical_state: np.ndarray) -> np.ndarray:
        return self.to_modal @ physical_state

    def decode(self, modal_state: np.ndarray) -> np.ndarray:
        return self.to_physical @ modal_state


def modal_state_transform(reduction: HybridActiveStep) -> ModalStateTransform:
    """Точно диагонализует линейный переход состояния.

    Для ``state = V @ modal_state`` матрица перехода становится диагональной.
    Остальные связи пересчитываются так, чтобы узловые напряжения и нелинейные
    переменные не изменились.
    """
    poles, to_physical = np.linalg.eig(reduction.state_transition)
    imaginary = max(
        float(np.max(np.abs(np.imag(poles)))),
        float(np.max(np.abs(np.imag(to_physical)))),
    )
    if imaginary > 1e-10:
        raise ValueError(
            f"Матрица состояния имеет существенно комплексные режимы: {imaginary:.3e}"
        )
    poles = np.real(poles)
    to_physical = np.real(to_physical)
    order = np.argsort(poles)
    poles = poles[order]
    to_physical = to_physical[:, order]
    # Закрепляем знаки столбцов, чтобы создаваемый заголовок был воспроизводим.
    for column in range(to_physical.shape[1]):
        pivot = int(np.argmax(np.abs(to_physical[:, column])))
        if to_physical[pivot, column] < 0.0:
            to_physical[:, column] *= -1.0
    to_modal = np.linalg.inv(to_physical)
    transformed = replace(
        reduction,
        active_state=reduction.active_state @ to_physical,
        full_q_state=reduction.full_q_state @ to_physical,
        node_state=reduction.node_state @ to_physical,
        state_bias=to_modal @ reduction.state_bias,
        state_input=to_modal @ reduction.state_input,
        state_transition=np.diag(poles),
        state_active=to_modal @ reduction.state_active,
    )
    return ModalStateTransform(transformed, to_physical, to_modal, poles)


def _active_terms(
    active_q: np.ndarray, reduction: HybridActiveStep
) -> tuple[np.ndarray, np.ndarray]:
    q = reduction.dc_q.copy()
    q[ACTIVE] = active_q
    current, first = nonlinear_terms(
        q, reduction.parameters, "hybrid", reduction.dc_q
    )
    return current[ACTIVE], first[ACTIVE]


def prepare_hybrid_active(
    sustain: float, tone: float, volume: float, factor: int
) -> HybridActiveStep:
    affine = prepare_affine_step(sustain, tone, volume, factor)
    _, dc_q = operating_point(sustain, tone, volume, affine.parameters)
    dc_current, dc_first = nonlinear_terms(
        dc_q, affine.parameters, "hybrid", dc_q
    )

    slope = np.zeros(NONLINEAR_COUNT)
    offset = np.zeros(NONLINEAR_COUNT)
    passive = np.ones(NONLINEAR_COUNT, dtype=bool)
    passive[ACTIVE] = False
    slope[passive] = dc_first[passive]
    offset[passive] = dc_current[passive] - slope[passive] * dc_q[passive]

    diagonal = np.diag(slope)
    elimination = np.linalg.inv(np.eye(NONLINEAR_COUNT) + affine.influence @ diagonal)
    active_columns = np.eye(NONLINEAR_COUNT)[:, ACTIVE]
    full_q_active = elimination @ affine.influence[:, ACTIVE]
    full_q_bias = elimination @ (affine.q_bias - affine.influence @ offset)
    full_q_input = elimination @ affine.q_input
    full_q_state = elimination @ affine.q_state

    current_active = active_columns - diagonal @ full_q_active
    current_bias = offset + diagonal @ full_q_bias
    current_input = diagonal @ full_q_input
    current_state = diagonal @ full_q_state

    node_bias = affine.node_bias - affine.node_nonlinear @ current_bias
    node_input = affine.node_input - affine.node_nonlinear @ current_input
    node_state = affine.node_state - affine.node_nonlinear @ current_state
    node_active = affine.node_nonlinear @ current_active
    state_bias = affine.state_bias - affine.state_nonlinear @ current_bias
    state_input = affine.state_input - affine.state_nonlinear @ current_input
    state_transition = affine.state_transition - affine.state_nonlinear @ current_state
    state_active = affine.state_nonlinear @ current_active

    return HybridActiveStep(
        affine.parameters, dc_q,
        full_q_bias[ACTIVE], full_q_input[ACTIVE], full_q_state[ACTIVE],
        full_q_active[np.ix_(ACTIVE, np.arange(ACTIVE_COUNT))],
        full_q_bias, full_q_input, full_q_state, full_q_active,
        node_bias, node_input, node_state, node_active,
        state_bias, state_input, state_transition, state_active,
    )


def step_hybrid_active(
    reduction: HybridActiveStep,
    previous_state: np.ndarray,
    active_q: np.ndarray,
    input_v: float,
    fully_converged: bool = False,
    q1_local_corrections: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    linear_active = (
        reduction.active_bias + reduction.active_input * input_v
        + reduction.active_state @ previous_state
    )
    correction_norm = 0.0
    for _ in range(30 if fully_converged else 1):
        current, first = _active_terms(active_q, reduction)
        residual = active_q - linear_active + reduction.active_influence @ current
        if fully_converged and float(np.max(np.abs(residual))) <= 1e-12:
            break
        jacobian = np.eye(ACTIVE_COUNT) + (
            reduction.active_influence * first[np.newaxis, :]
        )
        correction = np.linalg.solve(jacobian, -residual)
        correction_norm = max(correction_norm, float(np.max(np.abs(correction))))
        active_q = active_q + correction

    for _ in range(0 if fully_converged else q1_local_corrections):
        current, first = _active_terms(active_q, reduction)
        residual = active_q - linear_active + reduction.active_influence @ current
        jacobian = np.eye(ACTIVE_COUNT) + (
            reduction.active_influence * first[np.newaxis, :]
        )
        local = jacobian[np.ix_(Q1_ACTIVE, Q1_ACTIVE)]
        correction = np.linalg.solve(local, -residual[Q1_ACTIVE])
        correction_norm = max(correction_norm, float(np.max(np.abs(correction))))
        active_q[Q1_ACTIVE] += correction

    current, _ = _active_terms(active_q, reduction)
    state = (
        reduction.state_bias + reduction.state_input * input_v
        + reduction.state_transition @ previous_state
        - reduction.state_active @ current
    )
    node = (
        reduction.node_bias + reduction.node_input * input_v
        + reduction.node_state @ previous_state
        - reduction.node_active @ current
    )
    full_q = (
        reduction.full_q_bias + reduction.full_q_input * input_v
        + reduction.full_q_state @ previous_state
        - reduction.full_q_active @ current
    )
    final_current, _ = _active_terms(active_q, reduction)
    residual = active_q - linear_active + reduction.active_influence @ final_current
    return node, state, full_q, float(np.max(np.abs(residual))), correction_norm


def step_hybrid_cascade(
    reduction: HybridActiveStep,
    previous_state: np.ndarray,
    active_q: np.ndarray,
    input_v: float,
    passes: int = 1,
    symmetric: bool = False,
    blocks: tuple[np.ndarray, ...] = CASCADE_BLOCKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Выполняет последовательные местные поправки Q4, Q3, Q2 и Q1."""
    if passes < 1:
        raise ValueError("Число каскадных проходов должно быть положительным")
    linear_active = (
        reduction.active_bias + reduction.active_input * input_v
        + reduction.active_state @ previous_state
    )
    correction_norm = 0.0
    orders = []
    for _ in range(passes):
        orders.append(blocks)
        if symmetric:
            orders.append(tuple(reversed(blocks)))
    for blocks in orders:
        for block in blocks:
            current, first = _active_terms(active_q, reduction)
            residual = active_q - linear_active + reduction.active_influence @ current
            jacobian = np.eye(ACTIVE_COUNT) + (
                reduction.active_influence * first[np.newaxis, :]
            )
            local = jacobian[np.ix_(block, block)]
            correction = np.linalg.solve(local, -residual[block])
            correction_norm = max(
                correction_norm, float(np.max(np.abs(correction)))
            )
            active_q[block] += correction

    current, _ = _active_terms(active_q, reduction)
    state = (
        reduction.state_bias + reduction.state_input * input_v
        + reduction.state_transition @ previous_state
        - reduction.state_active @ current
    )
    node = (
        reduction.node_bias + reduction.node_input * input_v
        + reduction.node_state @ previous_state
        - reduction.node_active @ current
    )
    full_q = (
        reduction.full_q_bias + reduction.full_q_input * input_v
        + reduction.full_q_state @ previous_state
        - reduction.full_q_active @ current
    )
    final_current, _ = _active_terms(active_q, reduction)
    residual = active_q - linear_active + reduction.active_influence @ final_current
    return node, state, full_q, float(np.max(np.abs(residual))), correction_norm


def step_hybrid_scheduled(
    reduction: HybridActiveStep,
    previous_state: np.ndarray,
    active_q: np.ndarray,
    input_v: float,
    blocks: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Выполняет только назначенные на текущий внутренний отсчёт блоки.

    Накопители линейной части при этом по-прежнему обновляются на каждом
    внутреннем отсчёте. Функция нужна для раздельной проверки требуемой частоты
    нелинейных поправок; она ещё не является окончательным многоскоростным
    разбиением всей схемы.
    """
    linear_active = (
        reduction.active_bias + reduction.active_input * input_v
        + reduction.active_state @ previous_state
    )
    correction_norm = 0.0
    for block in blocks:
        current, first = _active_terms(active_q, reduction)
        residual = active_q - linear_active + reduction.active_influence @ current
        jacobian = np.eye(ACTIVE_COUNT) + (
            reduction.active_influence * first[np.newaxis, :]
        )
        local = jacobian[np.ix_(block, block)]
        correction = np.linalg.solve(local, -residual[block])
        correction_norm = max(
            correction_norm, float(np.max(np.abs(correction)))
        )
        active_q[block] += correction

    current, _ = _active_terms(active_q, reduction)
    state = (
        reduction.state_bias + reduction.state_input * input_v
        + reduction.state_transition @ previous_state
        - reduction.state_active @ current
    )
    node = (
        reduction.node_bias + reduction.node_input * input_v
        + reduction.node_state @ previous_state
        - reduction.node_active @ current
    )
    full_q = (
        reduction.full_q_bias + reduction.full_q_input * input_v
        + reduction.full_q_state @ previous_state
        - reduction.full_q_active @ current
    )
    final_current, _ = _active_terms(active_q, reduction)
    residual = active_q - linear_active + reduction.active_influence @ final_current
    return node, state, full_q, float(np.max(np.abs(residual))), correction_norm


def simulate_hybrid_active(
    sustain: float,
    tone: float,
    volume: float,
    factor: int,
    input_function: InputFunction,
    duration_s: float = 12e-3,
    fully_converged: bool = False,
    q1_local_corrections: int = 1,
) -> CompleteResult:
    reduction = prepare_hybrid_active(sustain, tone, volume, factor)
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
    dynamic = prepare_complete(
        sustain, tone, volume, 1.0 / (48_000.0 * factor), reduction.parameters
    )
    state = dynamic.capacitor_incidence.T @ dc_nodes
    active_q = dc_q[ACTIVE].copy()
    count = int(round(duration_s * 48_000.0 * factor))
    time_s = np.arange(count + 1) / (48_000.0 * factor)
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    node_v = np.empty((count + 1, NODE_COUNT))
    nonlinear_v = np.empty((count + 1, NONLINEAR_COUNT))
    residual_v = np.zeros(count + 1)
    correction_v = np.zeros(count + 1)
    node_v[0] = dc_nodes
    nonlinear_v[0] = dc_q
    for index in range(1, count + 1):
        node_v[index], state, full_q, residual_v[index], correction_v[index] = (
            step_hybrid_active(
                reduction, state, active_q, float(input_v[index]),
                fully_converged, q1_local_corrections,
            )
        )
        active_q = full_q[ACTIVE]
        nonlinear_v[index] = full_q
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return CompleteResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v
    )


def simulate_hybrid_cascade(
    sustain: float,
    tone: float,
    volume: float,
    factor: int,
    input_function: InputFunction,
    duration_s: float = 12e-3,
    passes: int = 1,
    symmetric: bool = False,
    blocks: tuple[np.ndarray, ...] = CASCADE_BLOCKS,
) -> CompleteResult:
    reduction = prepare_hybrid_active(sustain, tone, volume, factor)
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
    dynamic = prepare_complete(
        sustain, tone, volume, 1.0 / (48_000.0 * factor), reduction.parameters
    )
    state = dynamic.capacitor_incidence.T @ dc_nodes
    active_q = dc_q[ACTIVE].copy()
    count = int(round(duration_s * 48_000.0 * factor))
    time_s = np.arange(count + 1) / (48_000.0 * factor)
    input_v = np.asarray(input_function(time_s), dtype=np.float64)
    node_v = np.empty((count + 1, NODE_COUNT))
    nonlinear_v = np.empty((count + 1, NONLINEAR_COUNT))
    residual_v = np.zeros(count + 1)
    correction_v = np.zeros(count + 1)
    node_v[0] = dc_nodes
    nonlinear_v[0] = dc_q
    for index in range(1, count + 1):
        node_v[index], state, full_q, residual_v[index], correction_v[index] = (
            step_hybrid_cascade(
                reduction, state, active_q, float(input_v[index]),
                passes, symmetric, blocks,
            )
        )
        active_q = full_q[ACTIVE]
        nonlinear_v[index] = full_q
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")
    return CompleteResult(
        time_s, input_v, node_v, nonlinear_v, residual_v, correction_v
    )
