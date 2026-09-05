"""Создаёт сокращённые коэффициенты выбранной портовой схемы 4×/2× для STM32."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_frontend_model import nonlinear_terms as frontend_terms
from muff_multirate_model import (
    Q1_NONLINEAR,
    Q2_COLLECTOR,
    SLOW_NODES,
    _q1_terms,
    prepare_fast,
    prepare_slow,
    slow_step,
)
from q3_model import Q3Parameters


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "include" / "port_multirate_fixture.h"
FAST_ACTIVE = np.array([0, 3, 5, 8], dtype=np.intp)
FAST_RATE = 192_000
SLOW_RATE = 96_000
COUNT = 256
FAST_LINEAR_THRESHOLD = 1.0e-7


def c_float(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "F"


def array_1d(name: str, values: np.ndarray) -> str:
    values = np.asarray(values).ravel()
    body = ",\n    ".join(c_float(value) for value in values)
    return f"static const float {name}[{len(values)}] = {{\n    {body}\n}};\n"


def array_2d(name: str, values: np.ndarray) -> str:
    values = np.asarray(values)
    body = ",\n".join(
        "    {" + ", ".join(c_float(value) for value in row) + "}"
        for row in values
    )
    return f"static const float {name}[{values.shape[0]}][{values.shape[1]}] = {{\n{body}\n}};\n"


def generated_fast_linear_functions(
    fast: dict[str, np.ndarray], dc_state: np.ndarray,
    dc_current: np.ndarray,
) -> str:
    """Разворачивает и разреживает быстрые матрицы, сохраняя рабочую точку."""
    threshold = FAST_LINEAR_THRESHOLD

    def kept_terms(coefficients: np.ndarray, variable: str, sign: float = 1.0):
        return [
            f"{c_float(sign * coefficient)} * {variable}[{index}]"
            for index, coefficient in enumerate(coefficients)
            if abs(coefficient) >= threshold
        ]

    lines = [
        "\n#if defined(PORT_MULTIRATE_GENERATED_LINEAR)\n",
        f"#define PORT_FAST_LINEAR_THRESHOLD {c_float(threshold)}\n",
        "__attribute__((always_inline)) static inline void\n",
        "port_fast_predict_generated(const float state[9], float input, ",
        "float port_offset, float q[4])\n{\n",
    ]
    for row in range(4):
        coefficients = fast["active_state"][row]
        dropped = np.where(np.abs(coefficients) < threshold, coefficients, 0.0)
        corrected_bias = fast["active_bias"][row] + dropped @ dc_state
        terms = [
            c_float(corrected_bias),
            f"{c_float(fast['active_input'][row])} * input",
            f"{c_float(fast['active_port'][row])} * port_offset",
        ] + kept_terms(coefficients, "state")
        lines.append(f"    q[{row}] = " + " + ".join(terms) + ";\n")
    lines.extend(("}\n\n", "__attribute__((always_inline)) static inline void\n",
                  "port_fast_state_generated(const float state[9], float input, ",
                  "float port_offset, const float current[4], float next[9])\n{\n"))
    for row in range(9):
        transition = fast["state_transition"][row]
        active = fast["state_active"][row]
        dropped_transition = np.where(
            np.abs(transition) < threshold, transition, 0.0)
        dropped_active = np.where(np.abs(active) < threshold, active, 0.0)
        corrected_bias = (fast["state_bias"][row]
                          + dropped_transition @ dc_state
                          - dropped_active @ dc_current)
        terms = [
            c_float(corrected_bias),
            f"{c_float(fast['state_input'][row])} * input",
            f"{c_float(fast['state_port'][row])} * port_offset",
        ] + kept_terms(transition, "state")
        terms += kept_terms(active, "current", -1.0)
        lines.append(f"    next[{row}] = " + " + ".join(terms) + ";\n")
    lines.extend(("}\n\n", "__attribute__((always_inline)) static inline float\n",
                  "port_fast_project_generated(const float state[9], float input, ",
                  "float port_offset, const float current[4], ",
                  "const float linear_q[4], float q[4])\n{\n"))
    port_state = fast["port_state"]
    port_active = fast["port_active"]
    dropped_port_state = np.where(
        np.abs(port_state) < threshold, port_state, 0.0)
    dropped_port_active = np.where(
        np.abs(port_active) < threshold, port_active, 0.0)
    corrected_port_bias = (fast["port_bias"] + dropped_port_state @ dc_state
                           - dropped_port_active @ dc_current)
    port_terms = [c_float(corrected_port_bias),
                  f"{c_float(fast['port_input'])} * input",
                  f"{c_float(fast['port_port'])} * port_offset"]
    port_terms += kept_terms(port_state, "state")
    port_terms += kept_terms(port_active, "current", -1.0)
    lines.append("    const float port_voltage = " + " + ".join(port_terms) + ";\n")
    for row in range(4):
        terms = [f"linear_q[{row}]"] + [
            f"- port_fast_active_influence[{row}][{column}] * current[{column}]"
            for column in range(4)
        ]
        lines.append(f"    q[{row}] = " + " + ".join(terms) + ";\n")
    lines.extend(("    return port_voltage;\n}\n", "#endif\n"))
    return "".join(lines)


def fast_affine(
    parameters, dc_q, port_g, fast_rate=FAST_RATE, capacitance_scale=None,
    sustain=1.0,
):
    reduction = prepare_fast(
        parameters, 1.0 / fast_rate, port_g, capacitance_scale, sustain,
    )
    inverse = np.linalg.inv(reduction.matrix)
    node_bias0 = inverse @ reduction.source
    node_input0 = inverse @ reduction.input_vector
    node_state0 = inverse @ (reduction.incidence * reduction.conductance[np.newaxis, :])
    port_source = np.zeros(len(reduction.source)); port_source[Q2_COLLECTOR] = -1.0
    node_port0 = inverse @ port_source
    node_nonlinear = inverse @ reduction.injection
    q_bias0 = reduction.voltage @ node_bias0
    q_input0 = reduction.voltage @ node_input0
    q_state0 = reduction.voltage @ node_state0
    q_port0 = reduction.voltage @ node_port0

    dc_current, dc_first = frontend_terms(dc_q, parameters, "hybrid", dc_q)
    passive = np.ones(len(dc_q), dtype=bool); passive[FAST_ACTIVE] = False
    slope = np.zeros(len(dc_q)); offset = np.zeros(len(dc_q))
    slope[passive] = dc_first[passive]
    offset[passive] = dc_current[passive] - slope[passive] * dc_q[passive]
    diagonal = np.diag(slope)
    elimination = np.linalg.inv(np.eye(len(dc_q)) + reduction.influence @ diagonal)
    active_columns = np.eye(len(dc_q))[:, FAST_ACTIVE]
    full_q_active = elimination @ reduction.influence[:, FAST_ACTIVE]
    q_bias = elimination @ (q_bias0 - reduction.influence @ offset)
    q_input = elimination @ q_input0
    q_state = elimination @ q_state0
    q_port = elimination @ q_port0
    current_active = active_columns - diagonal @ full_q_active
    current_bias = offset + slope * q_bias
    current_input = slope * q_input
    current_state = slope[:, np.newaxis] * q_state
    current_port = slope * q_port
    node_bias = node_bias0 - node_nonlinear @ current_bias
    node_input = node_input0 - node_nonlinear @ current_input
    node_state = node_state0 - node_nonlinear @ current_state
    node_port = node_port0 - node_nonlinear @ current_port
    node_active = node_nonlinear @ current_active
    return {
        "active_bias": q_bias[FAST_ACTIVE],
        "active_input": q_input[FAST_ACTIVE],
        "active_state": q_state[FAST_ACTIVE],
        "active_port": q_port[FAST_ACTIVE],
        "active_influence": full_q_active[np.ix_(FAST_ACTIVE, np.arange(4))],
        "state_bias": reduction.incidence.T @ node_bias,
        "state_input": reduction.incidence.T @ node_input,
        "state_transition": reduction.incidence.T @ node_state,
        "state_port": reduction.incidence.T @ node_port,
        "state_active": reduction.incidence.T @ node_active,
        "port_bias": node_bias[Q2_COLLECTOR],
        "port_input": node_input[Q2_COLLECTOR],
        "port_state": node_state[Q2_COLLECTOR],
        "port_port": node_port[Q2_COLLECTOR],
        "port_active": node_active[Q2_COLLECTOR],
    }


def slow_affine(parameters, tone=1.0, volume=0.8, slow_rate=SLOW_RATE):
    reduction = prepare_slow(parameters, tone, volume, 1.0 / slow_rate)
    inverse = np.linalg.inv(reduction.matrix)
    node_bias = inverse @ reduction.source
    node_state = inverse @ (
        reduction.incidence[1:] * reduction.conductance[np.newaxis, :]
    )
    node_port = inverse @ (-reduction.port_column)
    node_active = inverse @ reduction.injection
    q_bias = reduction.voltage @ node_bias
    q_state = reduction.voltage @ node_state
    q_port = reduction.voltage @ node_port
    output_row = len(SLOW_NODES) - 1
    high_row, low_row = 0, 1
    g_c9 = reduction.conductance[0]
    g_r5 = reduction.resistor_conductance
    direct_state = np.zeros(4); direct_state[0] = -g_c9
    port_node_row = -(g_c9 * node_state[high_row] + g_r5 * node_state[low_row])
    port_current_state = direct_state + port_node_row
    port_current_port = (
        g_c9 + g_r5 - g_c9 * node_port[high_row] - g_r5 * node_port[low_row]
    )
    port_current_bias = -g_c9 * node_bias[high_row] - g_r5 * node_bias[low_row]
    port_current_active = (
        g_c9 * node_active[high_row] + g_r5 * node_active[low_row]
    )
    return reduction, {
        "active_bias": q_bias,
        "active_port": q_port,
        "active_state": q_state,
        "active_influence": reduction.influence,
        "state_bias": reduction.incidence.T @ np.concatenate(([0.0], node_bias)),
        "state_port": reduction.incidence.T @ np.concatenate(([1.0], node_port)),
        "state_transition": reduction.incidence.T @ np.vstack((np.zeros(4), node_state)),
        "state_active": reduction.incidence.T @ np.vstack((np.zeros(2), node_active)),
        "output_bias": node_bias[output_row],
        "output_port": node_port[output_row],
        "output_state": node_state[output_row],
        "output_active": node_active[output_row],
        "current_bias": port_current_bias,
        "current_port": port_current_port,
        "current_state": port_current_state,
        "current_active": port_current_active,
    }


def main() -> int:
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_nodes = dc_nodes[SLOW_NODES]
    slow_caps = slow_reduction.incidence.T @ np.concatenate(
        ([dc_nodes[Q2_COLLECTOR]], slow_nodes)
    )
    slow_q = dc_q[Q1_NONLINEAR]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_caps, slow_q, float(dc_nodes[Q2_COLLECTOR]), True
    )
    fast = fast_affine(parameters, dc_q[:9], port_g)
    port_offset = port_i - port_g * float(dc_nodes[Q2_COLLECTOR])
    frontend = prepare_fast(parameters, 1.0 / FAST_RATE, port_g)
    fast_state = frontend.incidence.T @ dc_nodes[:len(frontend.source)]
    fast_current, fast_first = frontend_terms(
        dc_q[:9], parameters, "hybrid", dc_q[:9]
    )
    fast_active_current = fast_current[FAST_ACTIVE]
    fast_linear_q = (
        fast["active_bias"] + fast["active_state"] @ fast_state
        + fast["active_port"] * port_offset
    )
    fast_residual = (
        dc_q[FAST_ACTIVE] - fast_linear_q
        + fast["active_influence"] @ fast_active_current
    )
    fast_next_state = (
        fast["state_bias"] + fast["state_transition"] @ fast_state
        + fast["state_port"] * port_offset
        - fast["state_active"] @ fast_active_current
    )
    fast_port = (
        fast["port_bias"] + fast["port_state"] @ fast_state
        + fast["port_port"] * port_offset
        - fast["port_active"] @ fast_active_current
    )
    slow_current, _ = _q1_terms(slow_q, parameters)
    slow_linear_q = (
        slow["active_bias"] + slow["active_port"] * dc_nodes[Q2_COLLECTOR]
        + slow["active_state"] @ slow_caps
    )
    slow_residual = slow_q - slow_linear_q + slow["active_influence"] @ slow_current
    slow_next_state = (
        slow["state_bias"] + slow["state_port"] * dc_nodes[Q2_COLLECTOR]
        + slow["state_transition"] @ slow_caps
        - slow["state_active"] @ slow_current
    )
    slow_output = (
        slow["output_bias"] + slow["output_port"] * dc_nodes[Q2_COLLECTOR]
        + slow["output_state"] @ slow_caps
        - slow["output_active"] @ slow_current
    )
    reconstructed_port_i = (
        slow["current_bias"] + slow["current_port"] * dc_nodes[Q2_COLLECTOR]
        + slow["current_state"] @ slow_caps
        + slow["current_active"] @ slow_current
    )
    checks = {
        "fast_residual": float(np.max(np.abs(fast_residual))),
        "fast_state": float(np.max(np.abs(fast_next_state - fast_state))),
        "fast_port": abs(float(fast_port - dc_nodes[Q2_COLLECTOR])),
        "slow_residual": float(np.max(np.abs(slow_residual))),
        "slow_state": float(np.max(np.abs(slow_next_state - slow_caps))),
        "slow_output": abs(float(slow_output - dc_nodes[-1])),
        "slow_port_current": abs(float(reconstructed_port_i - port_i)),
    }
    if max(checks.values()) > 1.0e-8:
        raise RuntimeError(f"Ошибка сокращения портовой модели: {checks}")
    time_s = np.arange(COUNT, dtype=np.float64) / 48_000.0
    envelope = np.minimum(1.0, time_s / 1e-3)
    stream = envelope * (
        0.0130*np.sin(2*np.pi*82.41*time_s)
        + 0.0065*np.sin(2*np.pi*164.81*time_s + 0.3)
        + 0.0037*np.sin(2*np.pi*329.63*time_s + 0.7)
        + 0.0018*np.sin(2*np.pi*2637.0*time_s)
    )

    parts = [
        "#ifndef PORT_MULTIRATE_FIXTURE_H\n#define PORT_MULTIRATE_FIXTURE_H\n\n",
        "#define PORT_FAST_STATE_COUNT 9U\n#define PORT_FAST_ACTIVE_COUNT 4U\n",
        "#define PORT_SLOW_STATE_COUNT 4U\n#define PORT_SLOW_ACTIVE_COUNT 2U\n",
        f"#define PORT_STREAM_COUNT {COUNT}U\n\n",
    ]
    for prefix, values in (("port_fast", fast), ("port_slow", slow)):
        for name, value in values.items():
            full_name = f"{prefix}_{name}"
            array = np.asarray(value)
            if array.ndim == 0:
                parts.append(f"static const float {full_name} = {c_float(array)};\n")
            elif array.ndim == 1:
                parts.append(array_1d(full_name, array))
            else:
                parts.append(array_2d(full_name, array))
    parts.extend((
        array_1d("port_fast_dc_state", fast_state),
        array_1d("port_fast_dc_active", dc_q[FAST_ACTIVE]),
        array_1d("port_fast_dc_current", fast_current[FAST_ACTIVE]),
        array_1d("port_fast_dc_first", fast_first[FAST_ACTIVE]),
        array_1d("port_slow_dc_state", slow_caps),
        array_1d("port_slow_dc_active", dc_q[[Q1_FORWARD, Q1_REVERSE]]),
        f"static const float port_dc_offset = {c_float(port_offset)};\n",
        f"static const float port_fixed_conductance = {c_float(port_g)};\n",
        array_1d("port_stream_input", stream),
        generated_fast_linear_functions(fast, fast_state, fast_active_current),
        "\n#endif\n",
    ))
    OUTPUT.write_text("".join(parts), encoding="ascii")
    print(f"Создано: {OUTPUT}")
    print(f"Коэффициентов fast: {sum(np.asarray(v).size for v in fast.values())}")
    print(f"Коэффициентов slow: {sum(np.asarray(v).size for v in slow.values())}")
    print("Максимальная ошибка проверки: " + format(max(checks.values()), ".3e"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
