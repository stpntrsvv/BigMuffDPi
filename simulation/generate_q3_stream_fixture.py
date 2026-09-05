"""Готовит периодическую последовательность линейной части Q3 для проверки на плате."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from q3_model import Q3Parameters, capacitor_voltage, operating_point
from q3_reduced_model import (
    full_step,
    prepare_reduction,
    reconstruct_nodes,
    right_hand_side,
)
from q3_physics_model import linearized_bjt_reduction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "src" / "q3_stream_fixture.h"
FACTOR = 16
BASE_RATE_HZ = 48_000
SIGNAL_HZ = 1_000
WARMUP_PERIODS = 30
PEAKS_V = (("nominal", 0.05), ("strong", 1.0))


def make_fixture(peak_v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    parameters = Q3Parameters()
    step_s = 1.0 / (BASE_RATE_HZ * FACTOR)
    reduction = prepare_reduction(parameters, step_s)
    period_samples = int(round(BASE_RATE_HZ * FACTOR / SIGNAL_HZ))
    warmup_samples = WARMUP_PERIODS * period_samples
    dc = operating_point(parameters)
    inverse, transistor_offset, diode_influence = linearized_bjt_reduction(
        reduction, reduction.voltage_matrix @ dc.voltage_v
    )
    node_v = dc.voltage_v.copy()
    q_v = reduction.voltage_matrix @ node_v
    previous_input = 0.0
    previous_capacitor_v = capacitor_voltage(node_v, previous_input)
    linear_trace: list[np.ndarray] = []
    seed_q: np.ndarray | None = None

    for index in range(1, warmup_samples + period_samples + 1):
        input_v = peak_v * np.sin(2.0 * np.pi * SIGNAL_HZ * index * step_s)
        rhs = right_hand_side(reduction, previous_capacitor_v, float(input_v))
        linear_q_v = reduction.voltage_matrix @ np.linalg.solve(
            reduction.linear_matrix, rhs
        )
        if index == warmup_samples + 1:
            seed_q = q_v.copy()
        if index > warmup_samples:
            linear_trace.append(linear_q_v)
        q_v, _ = full_step(q_v, linear_q_v, reduction)
        node_v = reconstruct_nodes(q_v, rhs, reduction)
        previous_capacitor_v = capacitor_voltage(node_v, float(input_v))
        previous_input = float(input_v)

    del previous_input
    if seed_q is None:
        raise RuntimeError("Не удалось получить начальное состояние")
    closure_error = float(np.max(np.abs(q_v - seed_q)))
    trace = np.asarray(linear_trace)
    scalar_trace = (
        inverse @ (trace - transistor_offset[np.newaxis, :]).T
    )[2]
    return seed_q, trace, scalar_trace, closure_error, diode_influence


def float_literal(value: float) -> str:
    return f"{np.float32(value):.9e}F"


def array_initializer(values: np.ndarray, indent: str = "    ") -> str:
    lines = []
    for row in values:
        lines.append(indent + "{" + ", ".join(float_literal(x) for x in row) + "},")
    return "\n".join(lines)


def main() -> int:
    fixtures = []
    for name, peak_v in PEAKS_V:
        seed_q, trace, scalar_trace, closure_error, diode_influence = make_fixture(peak_v)
        fixtures.append(
            (name, peak_v, seed_q, trace, scalar_trace, closure_error, diode_influence)
        )

    sample_count = fixtures[0][3].shape[0]
    if any(item[3].shape != (sample_count, 3) for item in fixtures):
        raise RuntimeError("Размеры проверочных последовательностей различаются")

    blocks = []
    for name, peak_v, seed_q, trace, scalar_trace, closure_error, _ in fixtures:
        blocks.append(
            f"/* Вход {peak_v:.2f} В; ошибка замыкания периода "
            f"{closure_error:.3e} В. */\n"
            f"static const float q3_stream_{name}_seed_q[3] = {{"
            + ", ".join(float_literal(x) for x in seed_q)
            + "};\n"
            f"static const float q3_stream_{name}_linear_q"
            f"[Q3_STREAM_SAMPLE_COUNT][3] = {{\n"
            + array_initializer(trace)
            + "\n};\n"
            f"static const float q3_stream_{name}_scalar_linear_q"
            f"[Q3_STREAM_SAMPLE_COUNT] = {{\n    "
            + ",\n    ".join(float_literal(x) for x in scalar_trace)
            + "\n};"
        )

    header = (
        "/* Создано simulation/generate_q3_stream_fixture.py; не править вручную. */\n"
        "#ifndef Q3_STREAM_FIXTURE_H\n"
        "#define Q3_STREAM_FIXTURE_H\n\n"
        f"#define Q3_STREAM_SAMPLE_COUNT {sample_count}U\n"
        f"#define Q3_STREAM_FACTOR {FACTOR}U\n"
        f"#define Q3_STREAM_SIGNAL_HZ {SIGNAL_HZ}U\n\n"
        f"#define Q3_SCALAR_DIODE_INFLUENCE "
        f"{float_literal(fixtures[0][6])}\n\n"
        + "\n\n".join(blocks)
        + "\n\n#endif\n"
    )
    OUTPUT_PATH.write_text(header, encoding="utf-8", newline="\n")
    print(f"Создано: {OUTPUT_PATH}")
    for name, _, _, _, _, closure_error, _ in fixtures:
        print(f"{name}: ошибка замыкания периода {closure_error:.3e} В")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
