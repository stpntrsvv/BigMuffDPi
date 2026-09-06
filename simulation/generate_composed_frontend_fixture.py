"""Создаёт коэффициенты составного ядра для Эйлера или BDF2."""
import argparse
from pathlib import Path
import numpy as np

from generate_port_multirate_fixture import array_1d, array_2d, slow_affine
from muff_complete_model import operating_point, Q2_COLLECTOR
from muff_composed_frontend_model import prepare_blocks
from muff_frontend_model import nonlinear_terms
from muff_multirate_model import Q1_NONLINEAR, SLOW_NODES, slow_step
from q3_model import Q3Parameters

ROOT = Path(__file__).resolve().parents[1]
SPARSE_THRESHOLD = 1.0e-7


def sparse(values):
    values = np.asarray(values).copy()
    values[np.abs(values) < SPARSE_THRESHOLD] = 0.0
    return values


def c_array_1d(name, values):
    return array_1d(name, sparse(values))


def c_array_2d(name, values):
    return array_2d(name, sparse(values))


def reduce_block(block, active, source_extra=None):
    matrix = block.matrix.copy()
    source = block.source.copy()
    if source_extra is not None:
        source += source_extra
    inverse = np.linalg.inv(matrix)
    node_bias = inverse @ source
    node_input = inverse @ block.input
    node_state = inverse @ (block.incidence * block.conductance[np.newaxis, :])
    node_active = inverse @ block.injection[:, active]
    voltage = block.voltage[active]
    return {
        "q_bias": voltage @ node_bias,
        "q_input": voltage @ node_input,
        "q_state": voltage @ node_state,
        "influence": voltage @ node_active,
        "state_bias": block.incidence.T @ node_bias,
        "state_input": block.incidence.T @ node_input,
        "state_transition": block.incidence.T @ node_state,
        "state_active": block.incidence.T @ node_active,
        "node_bias": node_bias,
        "node_input": node_input,
        "node_state": node_state,
        "node_active": node_active,
    }


def emit(prefix, data):
    parts = []
    for name in ("q_bias", "q_input", "q_state", "influence",
                 "state_bias", "state_input", "state_transition", "state_active"):
        value = data[name]
        parts.append(c_array_1d(prefix + name, value) if value.ndim == 1
                     else c_array_2d(prefix + name, value))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf2", action="store_true")
    parser.add_argument("--alpha02", action="store_true")
    args = parser.parse_args()
    if args.bdf2 and args.alpha02:
        parser.error("Выберите только один способ интегрирования")
    output = ROOT / "include" / (
        "composed_frontend_fixture_alpha02.h" if args.alpha02
        else "composed_frontend_fixture_bdf2.h" if args.bdf2
        else "composed_frontend_fixture.h"
    )
    # Проводимость BDF2 равна 3C/(2h), то есть коэффициенты совпадают
    # с коэффициентами Эйлера при условной частоте 3Fs/2.
    # Для rho=0,2: alpha_m=7/6, alpha_f=gamma=5/6,
    # alpha_m/(gamma*alpha_f)=42/25=1,68.
    coefficient_rate = 80_640 if args.alpha02 else 72_000 if args.bdf2 else 48_000
    p = Q3Parameters()
    nodes, dc_q = operating_point(1.0, 1.0, 0.8, p)
    slow_reduction, slow = slow_affine(p, 1.0, 0.8, slow_rate=coefficient_rate)
    slow_dc_state = slow_reduction.incidence.T @ np.concatenate(
        ([nodes[Q2_COLLECTOR]], nodes[SLOW_NODES]))
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_dc_state, dc_q[Q1_NONLINEAR],
        float(nodes[Q2_COLLECTOR]), True)
    port_offset = port_i - port_g * float(nodes[Q2_COLLECTOR])
    first, second = prepare_blocks(p, 1.0, 1.0 / coefficient_rate, port_g, dc_q)
    dc_current, dc_first = nonlinear_terms(dc_q[:9], p, "hybrid", dc_q[:9])

    # Fold both reverse leakage currents into the first affine source.
    first_extra = -(first.injection[:, 1] * dc_current[1]
                    + first.injection[:, 4] * dc_current[4])
    a = reduce_block(first, np.array([0, 3, 5]), first_extra)
    # Boundary input is Vbase/R12, in addition to the audio input map.
    boundary = np.zeros(len(first.source)); boundary[12] = 1.0 / 10_000.0
    inv1 = np.linalg.inv(first.matrix)
    a["q_boundary"] = first.voltage[[0, 3, 5]] @ inv1 @ boundary
    a["state_boundary"] = first.incidence.T @ inv1 @ boundary
    a["drive_bias"] = np.array([a["node_bias"][12]])
    a["drive_input"] = np.array([a["node_input"][12]])
    a["drive_boundary"] = np.array([(inv1 @ boundary)[12]])
    a["drive_state"] = a["node_state"][[12]][:]
    a["drive_active"] = a["node_active"][[12]][:]

    # Q2 forward law is exactly affine in the hybrid model: absorb it into A.
    vf = second.voltage[0]
    jf = second.injection[:, 0]
    slope = dc_first[6]
    offset = dc_current[6] - slope * dc_q[6]
    second.matrix[:] += np.outer(jf, slope * vf)
    second.source[:] -= jf * offset + second.injection[:, 1] * dc_current[7]
    b = reduce_block(second, np.array([2]))
    inv2 = np.linalg.inv(second.matrix)
    # Tone Norton offset has a negative RHS sign at local collector index 1.
    port = np.zeros(4); port[1] = -1.0
    b["q_port"] = second.voltage[[2]] @ inv2 @ port
    b["state_port"] = second.incidence.T @ inv2 @ port
    for name, index in (("base", 0), ("collector", 1)):
        b[name + "_bias"] = np.array([b["node_bias"][index]])
        b[name + "_input"] = np.array([b["node_input"][index]])
        b[name + "_port"] = np.array([(inv2 @ port)[index]])
        b[name + "_state"] = b["node_state"][[index]][:]
        b[name + "_active"] = b["node_active"][[index]][:]

    guard = (
        "COMPOSED_FRONTEND_FIXTURE_ALPHA02_H" if args.alpha02
        else "COMPOSED_FRONTEND_FIXTURE_BDF2_H" if args.bdf2
        else "COMPOSED_FRONTEND_FIXTURE_H"
    )
    parts = [f"#ifndef {guard}\n#define {guard}\n",
             "#define COMPOSED_STREAM_COUNT 256U\n", emit("composed_a_", a)]
    for name in ("q_boundary", "state_boundary", "drive_bias", "drive_input",
                 "drive_boundary", "drive_state", "drive_active"):
        parts.append(c_array_1d("composed_a_" + name, a[name]))
    parts.append(emit("composed_b_", b))
    for name in ("q_port", "state_port", "base_bias", "base_input", "base_port",
                 "base_state", "base_active", "collector_bias", "collector_input",
                 "collector_port", "collector_state", "collector_active"):
        parts.append(c_array_1d("composed_b_" + name, b[name]))
    for name, value in slow.items():
        value = np.asarray(value)
        parts.append(c_array_1d("composed_slow_" + name, value) if value.ndim <= 1
                     else c_array_2d("composed_slow_" + name, value))
    parts += [c_array_1d("composed_a_dc_state", first.incidence.T @ nodes[:13]),
              c_array_1d("composed_b_dc_state", second.incidence.T @ nodes[13:17]),
              c_array_1d("composed_a_dc_q", dc_q[[0, 3, 5]]),
              c_array_1d("composed_b_dc_q", dc_q[[8]]),
              c_array_1d("composed_slow_dc_state", slow_dc_state),
              c_array_1d("composed_slow_dc_q", dc_q[Q1_NONLINEAR]),
              c_array_1d("composed_port", np.array([port_offset, port_g])),
              "#endif\n"]
    text = "\n".join(parts)
    output.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
