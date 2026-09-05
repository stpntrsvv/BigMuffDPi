"""Сравнение одной поправки Ньютона и Галлея для диодной пары.

Проверяется скалярное уравнение эквивалента Тевенина:

    (v - u) / R + 2 Is sinh(v / (n Vt)) = 0.

Параметры диода по умолчанию условные. Это проверка численного продолжения
решения между близкими отсчётами, а не готовая модель 1N914 или Big Muff.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BOLTZMANN = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19


@dataclass(frozen=True)
class DiodePair:
    saturation_current_a: float
    ideality: float
    temperature_k: float
    source_resistance_ohm: float

    @property
    def thermal_voltage_v(self) -> float:
        return BOLTZMANN * self.temperature_k / ELECTRON_CHARGE

    @property
    def inverse_scaled_thermal_voltage(self) -> float:
        return 1.0 / (self.ideality * self.thermal_voltage_v)


@dataclass(frozen=True)
class Result:
    method: str
    waveform: str
    factor: int
    sample_rate_hz: float
    sample_count: int
    maximum_error_v: float
    rms_error_v: float
    maximum_residual_a: float
    rms_residual_a: float
    nonfinite_count: int


def residual(voltage_v: float, source_v: float, model: DiodePair) -> float:
    z = voltage_v * model.inverse_scaled_thermal_voltage
    diode_current_a = 2.0 * model.saturation_current_a * math.sinh(z)
    return (voltage_v - source_v) / model.source_resistance_ohm + diode_current_a


def residual_derivatives(
    voltage_v: float, source_v: float, model: DiodePair
) -> tuple[float, float, float]:
    del source_v
    a = model.inverse_scaled_thermal_voltage
    z = voltage_v * a
    sinh_z = math.sinh(z)
    cosh_z = math.cosh(z)
    scale = 2.0 * model.saturation_current_a
    first = 1.0 / model.source_resistance_ohm + scale * a * cosh_z
    second = scale * a * a * sinh_z
    return scale * sinh_z, first, second


def exact_root(source_v: float, model: DiodePair) -> float:
    """Находит единственный корень монотонного уравнения делением отрезка."""
    low = min(0.0, source_v)
    high = max(0.0, source_v)

    for _ in range(80):
        middle = 0.5 * (low + high)
        value = residual(middle, source_v, model)
        if value > 0.0:
            high = middle
        else:
            low = middle

    return 0.5 * (low + high)


def newton_step(voltage_v: float, source_v: float, model: DiodePair) -> float:
    value = residual(voltage_v, source_v, model)
    _, first, _ = residual_derivatives(voltage_v, source_v, model)
    return voltage_v - value / first


def halley_step(voltage_v: float, source_v: float, model: DiodePair) -> float:
    value = residual(voltage_v, source_v, model)
    _, first, second = residual_derivatives(voltage_v, source_v, model)
    denominator = 2.0 * first * first - value * second
    if abs(denominator) <= 1.0e-30:
        return voltage_v - value / first
    return voltage_v - 2.0 * value * first / denominator


def source_value(
    waveform: str,
    index: int,
    count: int,
    sample_rate_hz: float,
    frequency_hz: float,
    peak_v: float,
) -> float:
    if waveform == "sine":
        return peak_v * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate_hz)
    return 0.0 if index < count // 2 else peak_v


def simulate(
    method_name: str,
    step: Callable[[float, float, DiodePair], float],
    waveform: str,
    factor: int,
    base_sample_rate_hz: float,
    duration_s: float,
    frequency_hz: float,
    peak_v: float,
    model: DiodePair,
) -> Result:
    sample_rate_hz = base_sample_rate_hz * factor
    count = max(2, round(duration_s * sample_rate_hz))
    approximation_v = 0.0
    sum_error_squared = 0.0
    sum_residual_squared = 0.0
    maximum_error_v = 0.0
    maximum_residual_a = 0.0
    nonfinite_count = 0

    for index in range(count):
        source_v = source_value(
            waveform, index, count, sample_rate_hz, frequency_hz, peak_v
        )
        reference_v = exact_root(source_v, model)

        try:
            approximation_v = step(approximation_v, source_v, model)
            current_residual_a = abs(residual(approximation_v, source_v, model))
        except (OverflowError, ZeroDivisionError):
            approximation_v = math.nan
            current_residual_a = math.inf

        if not math.isfinite(approximation_v) or not math.isfinite(current_residual_a):
            nonfinite_count += 1
            approximation_v = reference_v
            continue

        error_v = abs(approximation_v - reference_v)
        maximum_error_v = max(maximum_error_v, error_v)
        maximum_residual_a = max(maximum_residual_a, current_residual_a)
        sum_error_squared += error_v * error_v
        sum_residual_squared += current_residual_a * current_residual_a

    finite_count = count - nonfinite_count
    denominator = max(1, finite_count)
    return Result(
        method=method_name,
        waveform=waveform,
        factor=factor,
        sample_rate_hz=sample_rate_hz,
        sample_count=count,
        maximum_error_v=maximum_error_v,
        rms_error_v=math.sqrt(sum_error_squared / denominator),
        maximum_residual_a=maximum_residual_a,
        rms_residual_a=math.sqrt(sum_residual_squared / denominator),
        nonfinite_count=nonfinite_count,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform", choices=("sine", "step"), default="sine")
    parser.add_argument("--factors", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--base-rate", type=float, default=48_000.0)
    parser.add_argument("--duration", type=float, default=0.005)
    parser.add_argument("--frequency", type=float, default=1_000.0)
    parser.add_argument("--peak", type=float, default=2.0)
    parser.add_argument("--resistance", type=float, default=10_000.0)
    parser.add_argument("--is", dest="saturation_current", type=float, default=2.0e-9)
    parser.add_argument("--n", dest="ideality", type=float, default=1.9)
    parser.add_argument("--temperature", type=float, default=300.15)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    positive = {
        "base-rate": args.base_rate,
        "duration": args.duration,
        "frequency": args.frequency,
        "resistance": args.resistance,
        "is": args.saturation_current,
        "n": args.ideality,
        "temperature": args.temperature,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} должно быть положительным конечным числом")
    if not math.isfinite(args.peak):
        raise ValueError("peak должен быть конечным числом")
    if any(factor <= 0 for factor in args.factors):
        raise ValueError("все множители частоты должны быть положительными")


def print_results(results: list[Result]) -> None:
    print(
        "method waveform factor rate_hz samples max_error_v rms_error_v "
        "max_residual_a rms_residual_a nonfinite"
    )
    for result in results:
        print(
            f"{result.method:6s} {result.waveform:8s} {result.factor:6d} "
            f"{result.sample_rate_hz:9.0f} {result.sample_count:7d} "
            f"{result.maximum_error_v:.6e} {result.rms_error_v:.6e} "
            f"{result.maximum_residual_a:.6e} {result.rms_residual_a:.6e} "
            f"{result.nonfinite_count:d}"
        )


def write_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=Result.__dataclass_fields__.keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> int:
    args = parse_arguments()
    validate_arguments(args)
    model = DiodePair(
        saturation_current_a=args.saturation_current,
        ideality=args.ideality,
        temperature_k=args.temperature,
        source_resistance_ohm=args.resistance,
    )

    methods = (("newton", newton_step), ("halley", halley_step))
    results = [
        simulate(
            method_name,
            step,
            args.waveform,
            factor,
            args.base_rate,
            args.duration,
            args.frequency,
            args.peak,
            model,
        )
        for factor in args.factors
        for method_name, step in methods
    ]

    print_results(results)
    if args.csv is not None:
        write_csv(args.csv, results)

    return 1 if any(result.nonfinite_count for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
