"""Полная узловая модель изолированного каскада ограничения Q3.

Модель совпадает с q3_reference.cir:

- четыре явных конденсатора C5, C12, C6 и C13;
- безынерционная модель Эберса—Молла;
- точная встречно-параллельная диодная пара Шокли;
- обратный метод Эйлера;
- полная сходимость Ньютона с аналитическим якобианом и поиском длины шага.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DRIVE = 0
BASE = 1
COLLECTOR = 2
EMITTER = 3
DIODE_NODE = 4
OUTPUT = 5
NODE_NAMES = ("drive", "base", "collector", "emitter", "diode_node", "output")


@dataclass(frozen=True)
class Q3Parameters:
    supply_v: float = 9.0
    r19_ohm: float = 10_000.0
    r20_ohm: float = 100_000.0
    r18_ohm: float = 10_000.0
    r21_ohm: float = 150.0
    r17_ohm: float = 470_000.0
    load_ohm: float = 110_000.0
    c5_f: float = 100.0e-9
    c12_f: float = 470.0e-12
    c6_f: float = 1.0e-6
    c13_f: float = 100.0e-9
    thermal_voltage_v: float = 25.8649e-3
    diode_saturation_current_a: float = 2.0e-9
    diode_ideality: float = 1.9
    forward_saturation_current_a: float = 10.025e-15
    reverse_saturation_current_a: float = 12.0e-15
    forward_ideality: float = 1.0
    reverse_ideality: float = 1.0
    alpha_forward: float = 0.9975062344
    alpha_reverse: float = 0.8333333333


@dataclass(frozen=True)
class NewtonResult:
    voltage_v: np.ndarray
    iterations: int
    residual_a: float


@dataclass(frozen=True)
class TransientResult:
    time_s: np.ndarray
    input_v: np.ndarray
    node_v: np.ndarray
    iterations: np.ndarray
    residual_a: np.ndarray


def _stamp_branch(matrix: np.ndarray, first: int, second: int, conductance: float) -> None:
    matrix[first, first] += conductance
    matrix[second, second] += conductance
    matrix[first, second] -= conductance
    matrix[second, first] -= conductance


def linear_system(parameters: Q3Parameters) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((6, 6), dtype=np.float64)
    source = np.zeros(6, dtype=np.float64)

    _stamp_branch(matrix, DRIVE, BASE, 1.0 / parameters.r19_ohm)
    matrix[BASE, BASE] += 1.0 / parameters.r20_ohm
    matrix[COLLECTOR, COLLECTOR] += 1.0 / parameters.r18_ohm
    source[COLLECTOR] += parameters.supply_v / parameters.r18_ohm
    matrix[EMITTER, EMITTER] += 1.0 / parameters.r21_ohm
    _stamp_branch(matrix, COLLECTOR, BASE, 1.0 / parameters.r17_ohm)
    matrix[OUTPUT, OUTPUT] += 1.0 / parameters.load_ohm
    return matrix, source


def _limited_expm1(argument: float) -> float:
    return float(np.expm1(np.clip(argument, -80.0, 80.0)))


def nonlinear_residual_and_jacobian(
    voltage_v: np.ndarray, parameters: Q3Parameters
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros(6, dtype=np.float64)
    jacobian = np.zeros((6, 6), dtype=np.float64)

    v_be = voltage_v[BASE] - voltage_v[EMITTER]
    v_bc = voltage_v[BASE] - voltage_v[COLLECTOR]
    forward_scale = parameters.forward_ideality * parameters.thermal_voltage_v
    reverse_scale = parameters.reverse_ideality * parameters.thermal_voltage_v
    forward_argument = v_be / forward_scale
    reverse_argument = v_bc / reverse_scale
    forward_current = parameters.forward_saturation_current_a * _limited_expm1(
        forward_argument
    )
    reverse_current = parameters.reverse_saturation_current_a * _limited_expm1(
        reverse_argument
    )
    forward_conductance = (
        parameters.forward_saturation_current_a
        * float(np.exp(np.clip(forward_argument, -80.0, 80.0)))
        / forward_scale
    )
    reverse_conductance = (
        parameters.reverse_saturation_current_a
        * float(np.exp(np.clip(reverse_argument, -80.0, 80.0)))
        / reverse_scale
    )

    base_current = (
        (1.0 - parameters.alpha_forward) * forward_current
        + (1.0 - parameters.alpha_reverse) * reverse_current
    )
    collector_current = parameters.alpha_forward * forward_current - reverse_current
    emitter_current = -forward_current + parameters.alpha_reverse * reverse_current
    residual[BASE] += base_current
    residual[COLLECTOR] += collector_current
    residual[EMITTER] += emitter_current

    base_forward = (1.0 - parameters.alpha_forward) * forward_conductance
    base_reverse = (1.0 - parameters.alpha_reverse) * reverse_conductance
    jacobian[BASE, BASE] += base_forward + base_reverse
    jacobian[BASE, EMITTER] -= base_forward
    jacobian[BASE, COLLECTOR] -= base_reverse

    jacobian[COLLECTOR, BASE] += (
        parameters.alpha_forward * forward_conductance - reverse_conductance
    )
    jacobian[COLLECTOR, EMITTER] -= parameters.alpha_forward * forward_conductance
    jacobian[COLLECTOR, COLLECTOR] += reverse_conductance

    jacobian[EMITTER, BASE] += (
        -forward_conductance + parameters.alpha_reverse * reverse_conductance
    )
    jacobian[EMITTER, EMITTER] += forward_conductance
    jacobian[EMITTER, COLLECTOR] -= parameters.alpha_reverse * reverse_conductance

    diode_voltage = voltage_v[DIODE_NODE] - voltage_v[COLLECTOR]
    diode_scale = parameters.diode_ideality * parameters.thermal_voltage_v
    diode_argument = float(np.clip(diode_voltage / diode_scale, -80.0, 80.0))
    diode_current = (
        2.0 * parameters.diode_saturation_current_a * float(np.sinh(diode_argument))
    )
    diode_conductance = (
        2.0
        * parameters.diode_saturation_current_a
        * float(np.cosh(diode_argument))
        / diode_scale
    )
    residual[DIODE_NODE] += diode_current
    residual[COLLECTOR] -= diode_current
    jacobian[DIODE_NODE, DIODE_NODE] += diode_conductance
    jacobian[DIODE_NODE, COLLECTOR] -= diode_conductance
    jacobian[COLLECTOR, DIODE_NODE] -= diode_conductance
    jacobian[COLLECTOR, COLLECTOR] += diode_conductance
    return residual, jacobian


def capacitor_voltage(node_v: np.ndarray, input_v: float) -> np.ndarray:
    return np.array(
        [
            node_v[DRIVE] - input_v,
            node_v[COLLECTOR] - node_v[BASE],
            node_v[BASE] - node_v[DIODE_NODE],
            node_v[COLLECTOR] - node_v[OUTPUT],
        ],
        dtype=np.float64,
    )


def residual_and_jacobian(
    node_v: np.ndarray,
    parameters: Q3Parameters,
    previous_capacitor_v: np.ndarray | None,
    input_v: float,
    step_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix, source = linear_system(parameters)
    residual = matrix @ node_v - source
    jacobian = matrix.copy()

    if previous_capacitor_v is not None and step_s is not None:
        conductances = np.array(
            [parameters.c5_f, parameters.c12_f, parameters.c6_f, parameters.c13_f]
        ) / step_s
        current_capacitor_v = capacitor_voltage(node_v, input_v)
        currents = conductances * (current_capacitor_v - previous_capacitor_v)

        residual[DRIVE] += currents[0]
        jacobian[DRIVE, DRIVE] += conductances[0]

        residual[COLLECTOR] += currents[1]
        residual[BASE] -= currents[1]
        _stamp_branch(jacobian, COLLECTOR, BASE, conductances[1])

        residual[BASE] += currents[2]
        residual[DIODE_NODE] -= currents[2]
        _stamp_branch(jacobian, BASE, DIODE_NODE, conductances[2])

        residual[COLLECTOR] += currents[3]
        residual[OUTPUT] -= currents[3]
        _stamp_branch(jacobian, COLLECTOR, OUTPUT, conductances[3])

    nonlinear_residual, nonlinear_jacobian = nonlinear_residual_and_jacobian(
        node_v, parameters
    )
    return residual + nonlinear_residual, jacobian + nonlinear_jacobian


def solve_newton(
    initial_v: np.ndarray,
    parameters: Q3Parameters,
    previous_capacitor_v: np.ndarray | None = None,
    input_v: float = 0.0,
    step_s: float | None = None,
    maximum_iterations: int = 40,
    residual_tolerance_a: float = 1.0e-11,
    voltage_tolerance_v: float = 1.0e-10,
) -> NewtonResult:
    voltage_v = np.array(initial_v, dtype=np.float64, copy=True)

    for iteration in range(1, maximum_iterations + 1):
        residual, jacobian = residual_and_jacobian(
            voltage_v, parameters, previous_capacitor_v, input_v, step_s
        )
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= residual_tolerance_a:
            return NewtonResult(voltage_v, iteration - 1, residual_norm)

        correction = np.linalg.solve(jacobian, -residual)
        damping = 1.0
        accepted_v = voltage_v + correction
        accepted_residual, _ = residual_and_jacobian(
            accepted_v, parameters, previous_capacitor_v, input_v, step_s
        )
        accepted_norm = float(np.max(np.abs(accepted_residual)))

        while accepted_norm > residual_norm and damping > 1.0 / 1024.0:
            damping *= 0.5
            accepted_v = voltage_v + damping * correction
            accepted_residual, _ = residual_and_jacobian(
                accepted_v, parameters, previous_capacitor_v, input_v, step_s
            )
            accepted_norm = float(np.max(np.abs(accepted_residual)))

        voltage_v = accepted_v
        if (
            float(np.max(np.abs(damping * correction))) <= voltage_tolerance_v
            and accepted_norm <= residual_tolerance_a
        ):
            return NewtonResult(voltage_v, iteration, accepted_norm)

    raise RuntimeError(
        f"Ньютон не сошёлся за {maximum_iterations} итераций; "
        f"невязка {accepted_norm:.6e} А"
    )


def operating_point(parameters: Q3Parameters | None = None) -> NewtonResult:
    parameters = parameters or Q3Parameters()
    initial_v = np.array([0.70, 0.70, 4.5, 0.066, 4.5, 0.0], dtype=np.float64)
    return solve_newton(initial_v, parameters)


def simulate_transient(
    duration_s: float = 10.0e-3,
    step_s: float = 0.5e-6,
    input_peak_v: float = 50.0e-3,
    input_frequency_hz: float = 1_000.0,
    parameters: Q3Parameters | None = None,
) -> TransientResult:
    parameters = parameters or Q3Parameters()
    dc = operating_point(parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = input_peak_v * np.sin(2.0 * np.pi * input_frequency_hz * time_s)
    node_v = np.empty((count + 1, 6), dtype=np.float64)
    iterations = np.zeros(count + 1, dtype=np.int32)
    residual_a = np.zeros(count + 1, dtype=np.float64)
    node_v[0] = dc.voltage_v
    residual_a[0] = dc.residual_a
    previous_capacitor_v = capacitor_voltage(node_v[0], input_v[0])

    for index in range(1, count + 1):
        result = solve_newton(
            node_v[index - 1],
            parameters,
            previous_capacitor_v=previous_capacitor_v,
            input_v=float(input_v[index]),
            step_s=step_s,
        )
        node_v[index] = result.voltage_v
        iterations[index] = result.iterations
        residual_a[index] = result.residual_a
        previous_capacitor_v = capacitor_voltage(node_v[index], input_v[index])

    return TransientResult(time_s, input_v, node_v, iterations, residual_a)


def simulate_input(input_v: np.ndarray, step_s: float,
                   parameters: Q3Parameters | None = None) -> TransientResult:
    """Считает полный каскад Q3 для произвольного массива входных отсчётов."""
    parameters = parameters or Q3Parameters()
    input_v = np.asarray(input_v, dtype=np.float64)
    dc = operating_point(parameters)
    time_s = np.arange(len(input_v), dtype=np.float64) * step_s
    node_v = np.empty((len(input_v), 6), dtype=np.float64)
    iterations = np.zeros(len(input_v), dtype=np.int32)
    residual_a = np.zeros(len(input_v), dtype=np.float64)
    node_v[0] = dc.voltage_v
    previous = capacitor_voltage(node_v[0], float(input_v[0]))
    for index in range(1, len(input_v)):
        result = solve_newton(node_v[index-1], parameters, previous,
                              float(input_v[index]), step_s)
        node_v[index] = result.voltage_v
        iterations[index] = result.iterations
        residual_a[index] = result.residual_a
        previous = capacitor_voltage(node_v[index], float(input_v[index]))
    return TransientResult(time_s, input_v, node_v, iterations, residual_a)
