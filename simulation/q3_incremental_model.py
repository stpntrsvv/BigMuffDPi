"""Модель Q3 с рекуррентным обновлением нелинейных функций."""

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
    prepare_reduction,
    reconstruct_nodes,
    right_hand_side,
)


CORDIC_LIMIT = 1.118
Q30_SCALE = float(1 << 30)


@dataclass
class NonlinearCache:
    forward_exp: np.float32
    reverse_exp: np.float32
    diode_sinh: np.float32
    diode_cosh: np.float32


@dataclass(frozen=True)
class IncrementalDiagnostics:
    maximum_forward_increment: float
    maximum_reverse_increment: float
    maximum_diode_increment: float
    maximum_cache_relative_error: float
    fallback_count: int
    periodic_refresh_count: int


def scales(parameters: Q3Parameters) -> np.ndarray:
    return np.array(
        [
            parameters.forward_ideality * parameters.thermal_voltage_v,
            parameters.reverse_ideality * parameters.thermal_voltage_v,
            parameters.diode_ideality * parameters.thermal_voltage_v,
        ],
        dtype=np.float64,
    )


def rebuild_cache(q_v: np.ndarray, parameters: Q3Parameters) -> NonlinearCache:
    argument = np.clip(np.asarray(q_v, dtype=np.float64) / scales(parameters), -80.0, 80.0)
    return NonlinearCache(
        np.float32(np.exp(argument[0])),
        np.float32(np.exp(argument[1])),
        np.float32(np.sinh(argument[2])),
        np.float32(np.cosh(argument[2])),
    )


def cordic_direct(argument: float) -> tuple[np.float32, np.float32]:
    """Имитирует масштаб Q1.31/SCALE=1 без внутренней ошибки алгоритма CORDIC."""
    if abs(argument) > CORDIC_LIMIT:
        raise ValueError("Аргумент вне прямого диапазона CORDIC")
    cosh_integer = int(round(np.cosh(argument) * Q30_SCALE))
    sinh_integer = int(round(np.sinh(argument) * Q30_SCALE))
    return np.float32(sinh_integer / Q30_SCALE), np.float32(cosh_integer / Q30_SCALE)


def cached_currents(
    cache: NonlinearCache, parameters: Q3Parameters
) -> tuple[np.ndarray, np.ndarray]:
    scale = scales(parameters)
    saturation = np.array(
        [
            parameters.forward_saturation_current_a,
            parameters.reverse_saturation_current_a,
            2.0 * parameters.diode_saturation_current_a,
        ],
        dtype=np.float32,
    )
    current = np.array(
        [
            saturation[0] * (cache.forward_exp - np.float32(1.0)),
            saturation[1] * (cache.reverse_exp - np.float32(1.0)),
            saturation[2] * cache.diode_sinh,
        ],
        dtype=np.float32,
    )
    first = np.array(
        [
            saturation[0] * cache.forward_exp / np.float32(scale[0]),
            saturation[1] * cache.reverse_exp / np.float32(scale[1]),
            saturation[2] * cache.diode_cosh / np.float32(scale[2]),
        ],
        dtype=np.float32,
    )
    return current, first


def update_cache(
    cache: NonlinearCache,
    old_q_v: np.ndarray,
    new_q_v: np.ndarray,
    parameters: Q3Parameters,
) -> tuple[NonlinearCache, np.ndarray, bool]:
    increment = (new_q_v - old_q_v) / scales(parameters)
    if np.any(np.abs(increment) > CORDIC_LIMIT):
        return rebuild_cache(new_q_v, parameters), increment, True

    forward_sinh, forward_cosh = cordic_direct(float(increment[0]))
    diode_sinh_delta, diode_cosh_delta = cordic_direct(float(increment[2]))
    forward_exp = np.float32(
        cache.forward_exp * np.float32(forward_cosh + forward_sinh)
    )

    new_reverse_argument = float(new_q_v[1] / scales(parameters)[1])
    if new_reverse_argument <= -80.0:
        reverse_exp = np.float32(0.0)
    elif cache.reverse_exp == 0.0:
        reverse_exp = np.float32(np.exp(np.clip(new_reverse_argument, -80.0, 80.0)))
    else:
        reverse_sinh, reverse_cosh = cordic_direct(float(increment[1]))
        reverse_exp = np.float32(
            cache.reverse_exp * np.float32(reverse_cosh + reverse_sinh)
        )

    diode_sinh = np.float32(
        cache.diode_sinh * diode_cosh_delta
        + cache.diode_cosh * diode_sinh_delta
    )
    diode_cosh = np.float32(
        cache.diode_cosh * diode_cosh_delta
        + cache.diode_sinh * diode_sinh_delta
    )
    return NonlinearCache(forward_exp, reverse_exp, diode_sinh, diode_cosh), increment, False


def one_incremental_correction(
    q_v: np.ndarray,
    linear_q_v: np.ndarray,
    reduction: Reduction,
    cache: NonlinearCache,
) -> tuple[np.ndarray, NonlinearCache, np.ndarray, bool]:
    current, first = cached_currents(cache, reduction.parameters)
    q32 = np.asarray(q_v, dtype=np.float32)
    linear32 = np.asarray(linear_q_v, dtype=np.float32)
    influence32 = np.asarray(reduction.influence_matrix, dtype=np.float32)
    residual = q32 - linear32 + influence32 @ current
    jacobian = np.eye(3, dtype=np.float32) + influence32 * first[np.newaxis, :]
    correction = np.linalg.solve(jacobian, -residual).astype(np.float32)
    new_q_v = (q32 + correction).astype(np.float32)
    new_cache, increment, fallback = update_cache(
        cache, q32.astype(np.float64), new_q_v.astype(np.float64), reduction.parameters
    )
    return new_q_v.astype(np.float64), new_cache, increment, fallback


def cache_error(
    cache: NonlinearCache, q_v: np.ndarray, parameters: Q3Parameters
) -> float:
    reference = rebuild_cache(q_v, parameters)
    actual = np.array(
        [cache.forward_exp, cache.reverse_exp, cache.diode_sinh, cache.diode_cosh],
        dtype=np.float64,
    )
    expected = np.array(
        [reference.forward_exp, reference.reverse_exp, reference.diode_sinh, reference.diode_cosh],
        dtype=np.float64,
    )
    scale = np.maximum(np.abs(expected), 1.0)
    return float(np.max(np.abs(actual - expected) / scale))


def simulate_incremental_reduced(
    duration_s: float = 20.0e-3,
    step_s: float = 1.0 / (48_000.0 * 16.0),
    input_peak_v: float = 50.0e-3,
    input_frequency_hz: float = 1_000.0,
    refresh_period: int = 256,
    parameters: Q3Parameters | None = None,
) -> tuple[TransientResult, IncrementalDiagnostics]:
    parameters = parameters or Q3Parameters()
    reduction = prepare_reduction(parameters, step_s)
    dc = operating_point(parameters)
    count = int(round(duration_s / step_s))
    time_s = np.arange(count + 1, dtype=np.float64) * step_s
    input_v = input_peak_v * np.sin(2.0 * np.pi * input_frequency_hz * time_s)
    node_v = np.empty((count + 1, 6), dtype=np.float64)
    iterations = np.ones(count + 1, dtype=np.int32)
    residual_a = np.zeros(count + 1, dtype=np.float64)
    node_v[0] = dc.voltage_v
    q_v = reduction.voltage_matrix @ node_v[0]
    cache = rebuild_cache(q_v, parameters)
    previous_capacitor_v = capacitor_voltage(node_v[0], input_v[0])
    maximum_increment = np.zeros(3, dtype=np.float64)
    maximum_cache_error = 0.0
    fallback_count = 0
    periodic_refresh_count = 0

    for index in range(1, count + 1):
        rhs = right_hand_side(reduction, previous_capacitor_v, float(input_v[index]))
        linear_q_v = reduction.voltage_matrix @ np.linalg.solve(reduction.linear_matrix, rhs)
        q_v, cache, increment, fallback = one_incremental_correction(
            q_v, linear_q_v, reduction, cache
        )
        maximum_increment = np.maximum(maximum_increment, np.abs(increment))
        fallback_count += int(fallback)
        if refresh_period > 0 and index % refresh_period == 0:
            cache = rebuild_cache(q_v, parameters)
            periodic_refresh_count += 1
        maximum_cache_error = max(maximum_cache_error, cache_error(cache, q_v, parameters))
        node_v[index] = reconstruct_nodes(q_v, rhs, reduction)
        physical_residual, _ = residual_and_jacobian(
            node_v[index], parameters, previous_capacitor_v,
            float(input_v[index]), step_s,
        )
        residual_a[index] = float(np.max(np.abs(physical_residual)))
        previous_capacitor_v = capacitor_voltage(node_v[index], input_v[index])
        if not np.all(np.isfinite(node_v[index])):
            raise FloatingPointError(f"Нечисловой результат на шаге {index}")

    diagnostics = IncrementalDiagnostics(
        float(maximum_increment[0]),
        float(maximum_increment[1]),
        float(maximum_increment[2]),
        maximum_cache_error,
        fallback_count,
        periodic_refresh_count,
    )
    return TransientResult(time_s, input_v, node_v, iterations, residual_a), diagnostics
