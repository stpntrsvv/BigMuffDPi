"""Проверяет накопленную ошибку разреженной линейной части портовой модели."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_frontend_model import nonlinear_terms as frontend_terms
from muff_multirate_model import (
    Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, _q1_terms, prepare_fast, slow_step,
)
from q3_model import Q3Parameters
from run_port_multirate_experiment import DURATION_S, RATE, read_input


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_sparse_linear"
RAW = ROOT / "simulation" / "raw" / "port_sparse_linear"


def sparse_fast(dense, dc_state, dc_current, threshold):
    result = copy.deepcopy(dense)
    for matrix_name, bias_name, equilibrium, sign in (
        ("active_state", "active_bias", dc_state, 1.0),
        ("state_transition", "state_bias", dc_state, 1.0),
        ("state_active", "state_bias", dc_current, -1.0),
        ("port_state", "port_bias", dc_state, 1.0),
        ("port_active", "port_bias", dc_current, -1.0),
    ):
        matrix = np.asarray(result[matrix_name])
        dropped = np.where(np.abs(matrix) < threshold, matrix, 0.0)
        correction = dropped @ equilibrium
        result[bias_name] = np.asarray(result[bias_name]) + sign * correction
        result[matrix_name] = np.where(np.abs(matrix) < threshold, 0.0, matrix)
    return result


def active_terms(q, parameters):
    full_q = np.zeros(9)
    full_q[FAST_ACTIVE] = q
    current, first = frontend_terms(full_q, parameters, "full", full_q)
    return current[FAST_ACTIVE], first[FAST_ACTIVE]


def correct_fast(q, linear_q, fast, parameters, mode, terms=active_terms, scale=1.0):
    current, first = terms(q, parameters)
    residual = q - linear_q + fast["active_influence"] @ current
    jacobian = np.eye(4) + fast["active_influence"] * first[np.newaxis, :]
    if mode == "full_4x4":
        q -= scale * np.linalg.solve(jacobian, residual)
        return
    blocks = {
        "3x3_plus_1": ((0, 3), (3, 4)),
        "2x2_plus_2x2": ((0, 2), (2, 4)),
        "four_scalars": ((0, 1), (1, 2), (2, 3), (3, 4)),
    }[mode]
    for begin, end in blocks:
        current, first = terms(q, parameters)
        residual = q - linear_q + fast["active_influence"] @ current
        jacobian = np.eye(4) + fast["active_influence"] * first[np.newaxis, :]
        q[begin:end] -= np.linalg.solve(
            jacobian[begin:end, begin:end], residual[begin:end]
        )


def run(
    fast, slow, initial, input_v, parameters, solver_mode="full_4x4",
    fast_factor=4, slow_factor=2, schedule=None, stats=None, terms_function=None,
):
    fast_state, fast_q, slow_state, slow_q, port_offset, port_g = (
        value.copy() if isinstance(value, np.ndarray) else value for value in initial
    )
    terms = terms_function or active_terms
    initial_current, initial_first = terms(fast_q, parameters)
    held_current = initial_current.copy()
    saved_inverse = np.linalg.inv(
        np.eye(4) + fast["active_influence"] * initial_first[np.newaxis, :]
    )
    steps_since_full = 0
    output = np.empty(len(input_v)); output[0] = 0.0
    if stats is not None:
        stats.update(full_steps=0, predicted_steps=0)
    for sample in range(1, len(input_v)):
        delta_input = input_v[sample] - input_v[sample - 1]
        for substep in range(1, fast_factor + 1):
            input_now = input_v[sample - 1] + substep / fast_factor * delta_input
            linear_q = (fast["active_bias"] + fast["active_input"] * input_now
                        + fast["active_port"] * port_offset
                        + fast["active_state"] @ fast_state)
            if "active_history" in fast:
                linear_q -= fast["active_history"] @ held_current
            mode = solver_mode if schedule is None else schedule[(substep - 1) % len(schedule)]
            if mode.startswith("partition_"):
                periods = np.asarray(tuple(int(value) for value in mode.split("_")[1:]))
                due = np.flatnonzero(((sample - 1) * fast_factor + substep) % periods == 0)
                exact_current, exact_first = terms(fast_q, parameters)
                held_current[due] = exact_current[due]
                residual = fast_q - linear_q + fast["active_influence"] @ held_current
                jacobian = np.eye(4) + fast["active_influence"] * exact_first[np.newaxis, :]
                if len(due):
                    fast_q[due] -= np.linalg.solve(jacobian[np.ix_(due, due)], residual[due])
                    exact_current, _ = terms(fast_q, parameters)
                    held_current[due] = exact_current[due]
                current = held_current.copy()
                if stats is not None:
                    stats["full_steps"] += int(len(due) == 4)
                    stats["predicted_steps"] += int(len(due) != 4)
            elif (mode == "predict" or mode.startswith("adaptive_")
                    or mode.startswith("gated_") or mode.startswith("quadratic_")):
                predicted_current, predicted_first = terms(fast_q, parameters)
                predicted_residual = (
                    fast_q - linear_q
                    + fast["active_influence"] @ predicted_current
                )
                predicted_correction = saved_inverse @ predicted_residual
                accept_prediction = mode == "predict"
                if mode.startswith("adaptive_"):
                    limit = float(mode.split("_", 1)[1])
                    scales = np.array((0.0258649, 0.0258649, 0.04914331, 0.04914331))
                    candidate_q = fast_q - predicted_correction
                    candidate_current, _ = terms(candidate_q, parameters)
                    candidate_residual = (
                        candidate_q - linear_q
                        + fast["active_influence"] @ candidate_current
                    )
                    accept_prediction = bool(
                        np.max(np.abs(candidate_residual) / scales) <= limit
                    )
                elif mode.startswith("gated_"):
                    _, limit_text, period_text = mode.split("_")
                    limit = float(limit_text)
                    period = int(period_text)
                    scales = np.array((0.0258649, 0.0258649, 0.04914331, 0.04914331))
                    correction_size = np.max(np.abs(predicted_correction) / scales)
                    accept_prediction = bool(
                        correction_size <= limit and steps_since_full < period - 1
                    )
                elif mode.startswith("quadratic_"):
                    _, limit_text, period_text = mode.split("_")
                    limit = float(limit_text)
                    period = int(period_text)
                    scales = np.array((0.0258649, 0.0258649, 0.04914331, 0.04914331))
                    second = np.empty(4)
                    second[:2] = predicted_first[:2] / 0.0258649
                    second[2:] = predicted_current[2:] / (0.04914331 ** 2)
                    estimated_residual = 0.5 * fast["active_influence"] @ (
                        second * predicted_correction * predicted_correction
                    )
                    estimate = np.max(np.abs(estimated_residual) / scales)
                    accept_prediction = bool(
                        estimate <= limit and steps_since_full < period - 1
                    )
                if accept_prediction:
                    fast_q -= predicted_correction
                    steps_since_full += 1
                    if stats is not None:
                        stats["predicted_steps"] += 1
                else:
                    full_current, full_first = terms(fast_q, parameters)
                    full_residual = (
                        fast_q - linear_q + fast["active_influence"] @ full_current
                    )
                    full_jacobian = (
                        np.eye(4) + fast["active_influence"] * full_first[np.newaxis, :]
                    )
                    saved_inverse = np.linalg.inv(full_jacobian)
                    fast_q -= saved_inverse @ full_residual
                    steps_since_full = 0
                    if stats is not None:
                        stats["full_steps"] += 1
            elif schedule is not None and mode == "full_4x4":
                full_current, full_first = terms(fast_q, parameters)
                full_residual = fast_q - linear_q + fast["active_influence"] @ full_current
                full_jacobian = (
                    np.eye(4) + fast["active_influence"] * full_first[np.newaxis, :]
                )
                saved_inverse = np.linalg.inv(full_jacobian)
                fast_q -= saved_inverse @ full_residual
                steps_since_full = 0
                if stats is not None:
                    stats["full_steps"] += 1
            else:
                if mode in ("full_repeat", "full_triple", "full_quad"):
                    count = {"full_repeat": 2, "full_triple": 3, "full_quad": 4}[mode]
                    for _ in range(count):
                        correct_fast(fast_q, linear_q, fast, parameters, "full_4x4", terms)
                elif mode in ("damped_half", "damped_quarter"):
                    scales = ((0.5, 1.0) if mode == "damped_half"
                              else (0.25, 0.75, 1.0))
                    for scale in scales:
                        correct_fast(fast_q, linear_q, fast, parameters,
                                     "full_4x4", terms, scale)
                else:
                    correct_fast(fast_q, linear_q, fast, parameters, mode, terms)
                if stats is not None:
                    stats["full_steps"] += 1
            if not mode.startswith("partition_"):
                current, _ = terms(fast_q, parameters)
            next_state = (fast["state_bias"] + fast["state_input"] * input_now
                          + fast["state_port"] * port_offset
                          + fast["state_transition"] @ fast_state
                          - fast["state_active"] @ current)
            if "state_active_history" in fast:
                next_state -= fast["state_active_history"] @ held_current
            port_v = (fast["port_bias"] + fast["port_input"] * input_now
                      + fast["port_port"] * port_offset
                      + fast["port_state"] @ fast_state
                      - fast["port_active"] @ current)
            if "port_active_history" in fast:
                port_v -= fast["port_active_history"] @ held_current
            fast_q = linear_q - fast["active_influence"] @ current
            fast_state = next_state
            if "active_history" in fast:
                held_current = current.copy()
            if substep % (fast_factor // slow_factor) == 0:
                linear_slow_q = (slow["active_bias"] + slow["active_port"] * port_v
                                 + slow["active_state"] @ slow_state)
                for _ in range(2):
                    slow_current, slow_first = _q1_terms(slow_q, parameters)
                    residual = (slow_q - linear_slow_q
                                + slow["active_influence"] @ slow_current)
                    jacobian = (np.eye(2) + slow["active_influence"]
                                * slow_first[np.newaxis, :])
                    slow_q -= np.linalg.solve(jacobian, residual)
                slow_current, _ = _q1_terms(slow_q, parameters)
                next_slow = (slow["state_bias"] + slow["state_port"] * port_v
                             + slow["state_transition"] @ slow_state
                             - slow["state_active"] @ slow_current)
                output[sample] = (slow["output_bias"] + slow["output_port"] * port_v
                                  + slow["output_state"] @ slow_state
                                  - slow["output_active"] @ slow_current)
                port_current = (slow["current_bias"] + slow["current_port"] * port_v
                                + slow["current_state"] @ slow_state
                                + slow["current_active"] @ slow_current)
                slow_q = linear_slow_q - slow["active_influence"] @ slow_current
                slow_state = next_slow
                port_offset = port_current - port_g * port_v
    return output


def main() -> int:
    parameters = Q3Parameters()
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, parameters)
    slow_reduction, slow = slow_affine(parameters)
    slow_state = slow_reduction.incidence.T @ np.r_[
        dc_nodes[Q2_COLLECTOR], dc_nodes[SLOW_NODES]
    ]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, dc_q[Q1_NONLINEAR],
        float(dc_nodes[Q2_COLLECTOR]), True,
    )
    dense = fast_affine(parameters, dc_q[:9], port_g)
    fast_reduction = prepare_fast(parameters, 1.0 / 192_000.0, port_g)
    fast_state = fast_reduction.incidence.T @ dc_nodes[:len(fast_reduction.source)]
    full_current, _ = frontend_terms(dc_q[:9], parameters, "hybrid", dc_q[:9])
    dc_current = full_current[FAST_ACTIVE]
    initial = (fast_state, dc_q[FAST_ACTIVE], slow_state,
               dc_q[[Q1_FORWARD, Q1_REVERSE]], port_i - port_g*dc_nodes[Q2_COLLECTOR],
               port_g)
    input_function = read_input()
    time_s = np.arange(int(round(DURATION_S*RATE)) + 1) / RATE
    input_v = np.asarray(input_function(time_s))
    dense_output = run(dense, slow, initial, input_v, parameters)
    rows = []
    for threshold in (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        sparse = sparse_fast(dense, fast_state, dc_current, threshold)
        sparse_output = run(sparse, slow, initial, input_v, parameters)
        error = sparse_output[1:] - dense_output[1:]
        rms = float(np.sqrt(np.mean(error*error)))
        peak = float(np.max(np.abs(error)))
        final = float(error[-1])
        removed = sum(np.count_nonzero(np.abs(np.asarray(dense[name])) < threshold)
                      for name in ("active_state", "state_transition", "state_active",
                                   "port_state", "port_active"))
        rows.append((threshold, removed, rms, peak, final))
        print(f"threshold={threshold:.0e} removed={removed} rms_mV={rms*1e3:.6f} "
              f"peak_mV={peak*1e3:.6f} final_mV={final*1e3:.6f}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("порог", "удалено", "ско_В", "пик_В", "конец_В"))
        writer.writerows(rows)
    table = "\n".join(
        f"| `{threshold:.0e}` | {removed} | {rms*1e3:.6f} | {peak*1e3:.6f} | {final*1e3:.6f} |"
        for threshold, removed, rms, peak, final in rows
    )
    (EXPERIMENT / "report.md").write_text(
        "# Разрежение быстрой линейной части\n\n"
        f"Первые {DURATION_S*1e3:.0f} мс гитарного аккорда, плотная однопоправочная модель — эталон.\n\n"
        "| Порог | Удалено | СКО, мВ | Пик, мВ | Конец, мВ |\n|---|---:|---:|---:|---:|\n"
        + table + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
