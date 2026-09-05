"""Проверяет опорные элементы фиксированного исключения 4×4 полной педали."""

from __future__ import annotations

import numpy as np

from muff_complete_model import operating_point, prepare_complete
from muff_hybrid_active import ACTIVE, modal_state_transform, prepare_hybrid_active
from run_full_cached_nonlinearity_experiment import Cache, RATE, fixture_input


def fixed_pivots(matrix: np.ndarray) -> np.ndarray:
    work = matrix.copy()
    pivots = [work[0, 0]]
    work[1:, 1:] -= np.outer(work[1:, 0] / work[0, 0], work[0, 1:])
    pivots.append(work[1, 1])
    work[2:, 2:] -= np.outer(work[2:, 1] / work[1, 1], work[1, 2:])
    pivots.append(work[2, 2])
    work[3, 3] -= work[3, 2] * work[2, 3] / work[2, 2]
    pivots.append(work[3, 3])
    return np.asarray(pivots)


def fixed_solve(matrix: np.ndarray, right: np.ndarray) -> np.ndarray:
    work = matrix.copy(); value = right.copy()
    for pivot in range(3):
        factors = work[pivot+1:, pivot] / work[pivot, pivot]
        work[pivot+1:, pivot+1:] -= np.outer(factors, work[pivot, pivot+1:])
        value[pivot+1:] -= factors * value[pivot]
    result = np.empty(4)
    for row in range(3, -1, -1):
        result[row] = (value[row] - work[row, row+1:] @ result[row+1:]) / work[row, row]
    return result


def check(gain: float) -> tuple[float, float, float]:
    physical = prepare_hybrid_active(1.0, 1.0, 0.8, 8)
    transform = modal_state_transform(physical); reduction = transform.reduction
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, reduction.parameters)
    dynamic = prepare_complete(1.0, 1.0, 0.8, 1.0/RATE, reduction.parameters)
    state = transform.encode(dynamic.capacitor_incidence.T @ dc_nodes)
    q = dc_q[ACTIVE].copy(); cache = Cache(q)
    minimum_pivot = np.inf; maximum_condition = 0.0; maximum_difference = 0.0
    q1 = np.array([4, 5])
    for sample in gain * fixture_input():
        linear = reduction.active_bias + reduction.active_input*sample + reduction.active_state@state
        for block in (np.arange(4), q1, q1):
            current, first = cache.terms()
            residual = q-linear+reduction.active_influence@current
            jacobian = np.eye(6)+reduction.active_influence*first[np.newaxis, :]
            local = jacobian[np.ix_(block, block)]
            if len(block) == 4:
                pivots = fixed_pivots(local)
                minimum_pivot = min(minimum_pivot, float(np.min(np.abs(pivots))))
                maximum_condition = max(maximum_condition, float(np.linalg.cond(local)))
                fixed = fixed_solve(local, residual[block])
                reference = np.linalg.solve(local, residual[block])
                maximum_difference = max(maximum_difference, float(np.max(np.abs(fixed-reference))))
            old_q=q.copy(); q[block]-=np.linalg.solve(local,residual[block]); cache.update(q,old_q,block)
        current,_=cache.terms()
        state = reduction.state_bias+reduction.state_input*sample+reduction.state_transition@state-reduction.state_active@current
        projected=linear-reduction.active_influence@current; cache.update(projected,q,np.arange(6)); q=projected
    return minimum_pivot, maximum_condition, maximum_difference


def main() -> int:
    for gain in (1.0, 5.0):
        pivot, condition, difference = check(gain)
        print(f"input x{gain:g}: min_pivot={pivot:.6e}, max_condition={condition:.3f}, difference={difference:.3e}")
        if pivot < 1e-5 or not np.isfinite(condition):
            return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
