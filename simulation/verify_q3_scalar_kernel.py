"""Проверяет float32-рекуррентность скалярного ядра по данным прошивки."""

from __future__ import annotations

import struct

import numpy as np

from generate_q3_stream_fixture import make_fixture
from q3_incremental_model import cordic_direct


Q30 = float(1 << 30)
INVERSE_DIODE_SCALE = np.float32(20.348649694129275)
DIODE_IS_TWICE = np.float32(4.0e-9)


def bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def rebuild(value: np.float32) -> tuple[np.float32, np.float32]:
    argument = float(value * INVERSE_DIODE_SCALE)
    sinh = np.float32(round(np.sinh(argument) * Q30) / Q30)
    cosh = np.float32(round(np.cosh(argument) * Q30) / Q30)
    return sinh, cosh


def run(peak_v: float, refresh_period: int) -> tuple[int, int, int]:
    seed, _, linear, _, influence = make_fixture(peak_v)
    q = np.float32(seed[2])
    sinh, cosh = rebuild(q)
    h = np.float32(influence)
    fallback_count = 0
    large_count = 0
    fast_refresh_steps = 0
    for index, forcing in enumerate(linear, start=1):
        current = np.float32(DIODE_IS_TWICE * sinh)
        first = np.float32(
            np.float32(DIODE_IS_TWICE * cosh) * INVERSE_DIODE_SCALE
        )
        residual = np.float32(
            np.float32(q - np.float32(forcing)) + np.float32(h * current)
        )
        jacobian = np.float32(np.float32(1.0) + np.float32(h * first))
        correction = np.float32(-residual / jacobian)
        q = np.float32(q + correction)
        delta = np.float32(correction * INVERSE_DIODE_SCALE)
        magnitude = abs(float(delta))
        large = magnitude > 0.25
        fallback = magnitude > 1.118
        large_count += int(large)
        fallback_count += int(fallback)
        if fallback:
            sinh, cosh = rebuild(q)
        else:
            delta_sinh, delta_cosh = cordic_direct(float(delta))
            old_sinh, old_cosh = sinh, cosh
            sinh = np.float32(
                np.float32(old_sinh * delta_cosh)
                + np.float32(old_cosh * delta_sinh)
            )
            cosh = np.float32(
                np.float32(old_cosh * delta_cosh)
                + np.float32(old_sinh * delta_sinh)
            )
        if large:
            fast_refresh_steps = 32
        active_period = refresh_period or (8 if fast_refresh_steps else 32)
        if index % active_period == 0 and not fallback:
            sinh, cosh = rebuild(q)
        if fast_refresh_steps:
            fast_refresh_steps -= 1
    return bits(q), fallback_count, large_count


def main() -> int:
    nominal = run(0.05, 32)
    strong = run(1.0, 0)
    print(f"nominal_q2_bits={nominal[0]}")
    print(f"nominal_fallback_count={nominal[1]}")
    print(f"nominal_large_count={nominal[2]}")
    print(f"strong_q2_bits={strong[0]}")
    print(f"strong_fallback_count={strong[1]}")
    print(f"strong_large_count={strong[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
