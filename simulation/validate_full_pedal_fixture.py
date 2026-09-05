"""Печатает эталонный итог потока, встроенного в измеритель полного шага."""

from __future__ import annotations

import numpy as np

from muff_complete_model import OUTPUT, operating_point, prepare_complete
from muff_hybrid_active import (
    ACTIVE,
    COUPLED_INPUT_Q1_TWICE,
    FRONTEND_Q1_TWICE,
    prepare_hybrid_active,
    step_hybrid_cascade,
)


def main() -> int:
    factor = 8
    rate = 48_000 * factor
    count = 768
    reduction = prepare_hybrid_active(1.0, 1.0, 0.8, factor)
    nodes, nonlinear = operating_point(1.0, 1.0, 0.8, reduction.parameters)
    dynamic = prepare_complete(1.0, 1.0, 0.8, 1.0 / rate, reduction.parameters)
    initial_state = dynamic.capacitor_incidence.T @ nodes
    time_s = np.arange(count) / rate
    stream = np.minimum(1.0, time_s / 1e-3) * (
        0.070 * np.sin(2 * np.pi * 82.41 * time_s)
        + 0.035 * np.sin(2 * np.pi * 164.81 * time_s + 0.3)
        + 0.020 * np.sin(2 * np.pi * 329.63 * time_s + 0.7)
        + 0.010 * np.sin(2 * np.pi * 2637.0 * time_s)
    )

    methods = (
        ("cheap", COUPLED_INPUT_Q1_TWICE),
        ("robust", FRONTEND_Q1_TWICE),
    )
    for name, blocks in methods:
        state = initial_state.copy()
        active = nonlinear[ACTIVE].copy()
        output = 0.0
        for input_v in stream:
            node, state, full_q, _, _ = step_hybrid_cascade(
                reduction, state, active, float(input_v), 1, False, blocks
            )
            active = full_q[ACTIVE]
            output = float(node[OUTPUT])
        print(f"{name}_reference_output_v={output:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
