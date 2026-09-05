"""Создаёт коэффициенты и входной поток для замера полного шага на STM32."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from muff_complete_model import OUTPUT as OUTPUT_NODE, operating_point, prepare_complete
from muff_hybrid_active import ACTIVE, modal_state_transform, prepare_hybrid_active


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "include" / "full_pedal_fixture.h"
FACTOR = 8
RATE = 48_000 * FACTOR
COUNT = 768


def c_float(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "F"


def array_1d(name: str, values: np.ndarray) -> str:
    body = ",\n    ".join(c_float(value) for value in np.asarray(values).ravel())
    return f"static const float {name}[{len(values)}] = {{\n    {body}\n}};\n"


def array_2d(name: str, values: np.ndarray) -> str:
    values = np.asarray(values)
    rows = []
    for row in values:
        rows.append("    {" + ", ".join(c_float(value) for value in row) + "}")
    return (
        f"static const float {name}[{values.shape[0]}][{values.shape[1]}] = {{\n"
        + ",\n".join(rows) + "\n};\n"
    )


def main() -> int:
    sustain, tone, volume = 1.0, 1.0, 0.8
    physical_reduction = prepare_hybrid_active(sustain, tone, volume, FACTOR)
    transform = modal_state_transform(physical_reduction)
    reduction = transform.reduction
    dc_nodes, dc_q = operating_point(sustain, tone, volume, reduction.parameters)
    dynamic = prepare_complete(sustain, tone, volume, 1.0 / RATE, reduction.parameters)
    dc_state = transform.encode(dynamic.capacitor_incidence.T @ dc_nodes)
    time_s = np.arange(COUNT, dtype=np.float64) / RATE
    envelope = np.minimum(1.0, time_s / 1e-3)
    input_v = envelope * (
        0.070 * np.sin(2 * np.pi * 82.41 * time_s)
        + 0.035 * np.sin(2 * np.pi * 164.81 * time_s + 0.3)
        + 0.020 * np.sin(2 * np.pi * 329.63 * time_s + 0.7)
        + 0.010 * np.sin(2 * np.pi * 2637.0 * time_s)
    )

    parts = [
        "#ifndef FULL_PEDAL_FIXTURE_H\n#define FULL_PEDAL_FIXTURE_H\n\n",
        "#define FULL_PEDAL_STATE_COUNT 13U\n",
        "#define FULL_PEDAL_ACTIVE_COUNT 6U\n",
        f"#define FULL_PEDAL_STREAM_COUNT {COUNT}U\n\n",
        array_1d("full_pedal_active_bias", reduction.active_bias),
        array_1d("full_pedal_active_input", reduction.active_input),
        array_2d("full_pedal_active_state", reduction.active_state),
        array_2d("full_pedal_active_influence", reduction.active_influence),
        array_1d("full_pedal_state_bias", reduction.state_bias),
        array_1d("full_pedal_state_input", reduction.state_input),
        array_1d("full_pedal_state_pole", np.diag(reduction.state_transition)),
        array_2d("full_pedal_state_active", reduction.state_active),
        array_1d("full_pedal_output_state", reduction.node_state[OUTPUT_NODE]),
        array_1d("full_pedal_output_active", reduction.node_active[OUTPUT_NODE]),
        f"static const float full_pedal_output_bias = {c_float(reduction.node_bias[OUTPUT_NODE])};\n",
        f"static const float full_pedal_output_input = {c_float(reduction.node_input[OUTPUT_NODE])};\n",
        array_1d("full_pedal_dc_state", dc_state),
        array_1d("full_pedal_dc_active", dc_q[ACTIVE]),
        array_1d("full_pedal_stream_input", input_v),
        "\n#endif\n",
    ]
    OUTPUT_PATH.write_text("".join(parts), encoding="ascii")
    print(f"Создано: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
