"""Имитация целочисленого ядра одной поправки Ньютона Q3.

Линейная часть и восстановление узлов пока остаются в float64. Цель
этого модуля — проверить разрядность сокращённой нелинейной системы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q3_model import (
    Q3Parameters,
    TransientResult,
    capacitor_voltage,
    operating_point,
    residual_and_jacobian,
)
from q3_reduced_model import (
    Reduction,
    nonlinear_currents,
    prepare_reduction,
    reconstruct_nodes,
    right_hand_side,
)


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class FixedFormats:
    """Число дробных разрядов для разных физических величин."""

    voltage: int = 30
    jacobian: int = 23
    normalized_influence: int = 30


@dataclass(frozen=True)
class FixedDiagnostics:
    maximum_voltage_integer: int
    maximum_jacobian_integer: int
    minimum_pivot_integer: int


def quantize(value: float, fractional_bits: int) -> int:
    """Округляет до знакового 32-разрядного формата и проверяет диапазон."""
    integer = int(round(float(value) * (1 << fractional_bits)))
    if integer < INT32_MIN or integer > INT32_MAX:
        raise OverflowError(
            f"Переполнение Q{32 - fractional_bits}.{fractional_bits}: {value}"
        )
    return integer


def dequantize(value: int, fractional_bits: int) -> float:
    return float(value) / float(1 << fractional_bits)


def rounded_shift(value: int, bits: int) -> int:
    half = 1 << (bits - 1)
    if value >= 0:
        return (value + half) >> bits
    return -((-value + half) >> bits)


def solve_3x3_fixed(
    matrix: list[list[int]],
    right: list[int],
    matrix_fractional_bits: int,
    right_fractional_bits: int,
) -> tuple[list[int], int]:
    """Решает 3×3 целочисленным методом Гаусса с выбором главного элемента."""
    a = [row.copy() for row in matrix]
    b = right.copy()
    minimum_pivot = INT32_MAX
    for column in range(3):
        pivot_row = max(range(column, 3), key=lambda row: abs(a[row][column]))
        if a[pivot_row][column] == 0:
            raise ArithmeticError("Нулевой опорный элемент")
        if pivot_row != column:
            a[column], a[pivot_row] = a[pivot_row], a[column]
            b[column], b[pivot_row] = b[pivot_row], b[column]
        pivot = a[column][column]
        minimum_pivot = min(minimum_pivot, abs(pivot))
        for row in range(column + 1, 3):
            factor = (a[row][column] << matrix_fractional_bits) // pivot
            for index in range(column, 3):
                a[row][index] -= rounded_shift(
                    factor * a[column][index], matrix_fractional_bits
                )
            b[row] -= rounded_shift(factor * b[column], matrix_fractional_bits)

    result = [0, 0, 0]
    for row in range(2, -1, -1):
        numerator = b[row]
        for column in range(row + 1, 3):
            numerator -= rounded_shift(
                a[row][column] * result[column], matrix_fractional_bits
            )
        result[row] = (numerator << matrix_fractional_bits) // a[row][row]
    return result, minimum_pivot


def one_correction_fixed(
    q_integer: list[int],
    linear_q_v: np.ndarray,
    reduction: Reduction,
    origin_q_v: np.ndarray,
    origin_current_a: np.ndarray,
    formats: FixedFormats,
) -> tuple[list[int], FixedDiagnostics]:
    """Одна поправка с раздельными масштабами напряжений и якобиана."""
    delta_q_v = np.array(
        [dequantize(value, formats.voltage) for value in q_integer],
        dtype=np.float64,
    )
    q_v = origin_q_v + delta_q_v
    currents, first, _ = nonlinear_currents(q_v, reduction.parameters)
    column_scale = np.max(np.abs(reduction.influence_matrix), axis=0)
    normalized_influence = reduction.influence_matrix / column_scale[np.newaxis, :]

    influence_integer = [
        [
            quantize(normalized_influence[row, column], formats.normalized_influence)
            for column in range(3)
        ]
        for row in range(3)
    ]
    current_integer = [
        quantize(
            column_scale[index] * (currents[index] - origin_current_a[index]),
            formats.voltage,
        )
        for index in range(3)
    ]
    first_integer = [
        quantize(column_scale[index] * first[index], formats.jacobian)
        for index in range(3)
    ]
    centered_linear_q_v = (
        linear_q_v - origin_q_v - reduction.influence_matrix @ origin_current_a
    )
    linear_integer = [
        quantize(centered_linear_q_v[index], formats.voltage)
        for index in range(3)
    ]

    residual: list[int] = []
    jacobian: list[list[int]] = []
    for row in range(3):
        residual_value = q_integer[row] - linear_integer[row]
        jacobian_row = []
        for column in range(3):
            residual_value += rounded_shift(
                influence_integer[row][column] * current_integer[column],
                formats.normalized_influence,
            )
            element = rounded_shift(
                influence_integer[row][column] * first_integer[column],
                formats.normalized_influence,
            )
            if row == column:
                element += 1 << formats.jacobian
            jacobian_row.append(element)
        residual.append(residual_value)
        jacobian.append(jacobian_row)

    correction, minimum_pivot = solve_3x3_fixed(
        jacobian,
        [-value for value in residual],
        formats.jacobian,
        formats.voltage,
    )
    updated = [q_integer[index] + correction[index] for index in range(3)]
    for value in updated:
        if value < INT32_MIN or value > INT32_MAX:
            raise OverflowError("Переполнение напряжения после поправки")
    diagnostics = FixedDiagnostics(
        max(abs(value) for value in updated + residual + current_integer),
        max(abs(value) for row in jacobian for value in row),
        minimum_pivot,
    )
    return updated, diagnostics


def simulate_fixed_reduced(
    duration_s: float = 5.0e-3,
    step_s: float = 1.0 / (48_000.0 * 16.0),
    input_peak_v: float = 50.0e-3,
    input_frequency_hz: float = 1_000.0,
    parameters: Q3Parameters | None = None,
    formats: FixedFormats | None = None,
) -> tuple[TransientResult, FixedDiagnostics]:
    """Считает переходной процесс с одной целочисленной поправкой на отсчёт."""
    parameters = parameters or Q3Parameters()
    formats = formats or FixedFormats()
    reduction = prepare_reduction(parameters, step_s)
    dc = operating_point(parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = input_peak_v * np.sin(2.0 * np.pi * input_frequency_hz * time_s)
    node_v = np.empty((count + 1, 6), dtype=np.float64)
    iterations = np.ones(count + 1, dtype=np.int32)
    residual_a = np.zeros(count + 1, dtype=np.float64)
    node_v[0] = dc.voltage_v
    origin_q_v = reduction.voltage_matrix @ node_v[0]
    origin_current_a, _, _ = nonlinear_currents(origin_q_v, parameters)
    q_integer = [0, 0, 0]
    previous_capacitor_v = capacitor_voltage(node_v[0], input_v[0])
    maximum_voltage_integer = max(abs(value) for value in q_integer)
    maximum_jacobian_integer = 0
    minimum_pivot_integer = INT32_MAX

    for index in range(1, count + 1):
        rhs = right_hand_side(reduction, previous_capacitor_v, float(input_v[index]))
        linear_q_v = reduction.voltage_matrix @ np.linalg.solve(
            reduction.linear_matrix, rhs
        )
        q_integer, diagnostics = one_correction_fixed(
            q_integer,
            linear_q_v,
            reduction,
            origin_q_v,
            origin_current_a,
            formats,
        )
        maximum_voltage_integer = max(
            maximum_voltage_integer, diagnostics.maximum_voltage_integer
        )
        maximum_jacobian_integer = max(
            maximum_jacobian_integer, diagnostics.maximum_jacobian_integer
        )
        minimum_pivot_integer = min(
            minimum_pivot_integer, diagnostics.minimum_pivot_integer
        )
        q_v = origin_q_v + np.array(
            [dequantize(value, formats.voltage) for value in q_integer],
            dtype=np.float64,
        )
        node_v[index] = reconstruct_nodes(q_v, rhs, reduction)
        physical_residual, _ = residual_and_jacobian(
            node_v[index], parameters, previous_capacitor_v,
            float(input_v[index]), step_s,
        )
        residual_a[index] = float(np.max(np.abs(physical_residual)))
        previous_capacitor_v = capacitor_voltage(node_v[index], input_v[index])
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")

    return TransientResult(time_s, input_v, node_v, iterations, residual_a), FixedDiagnostics(
        maximum_voltage_integer, maximum_jacobian_integer, minimum_pivot_integer
    )
