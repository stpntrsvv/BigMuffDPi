"""Собирает сухой ми-мажорный аккорд из шести открытых гитарных нот.

Исходные записи выровнены по атаке. Мы добавляем задержку 7 мс на струну,
имитируя удар медиатором от шестой струны к первой. Результат сводится в моно,
пересчитывается на 48 кГц и нормируется к -3 дБ от полной шкалы.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "simulation" / "samples" / "e_major_chord"
SOURCE_DIR = SAMPLE_DIR / "source"
OUTPUT_RATE = 48_000
STRUM_DELAY_S = 0.007
OUTPUT_DURATION_S = 4.0
ATTACK_DURATION_S = 0.25
PEAK = 10.0 ** (-3.0 / 20.0)

SOURCE_NAMES = (
    "string-6-E-as-E2.wav",
    "string-5-B-as-B2.wav",
    "string-4-E-as-E3.wav",
    "string-3-Gx-as-Gx3.wav",
    "string-2-B-as-B3.wav",
    "string-1-E-as-E4.wav",
)


def read_pcm32_stereo(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 2 or stream.getsampwidth() != 4:
            raise ValueError(f"Ожидается стерео PCM32: {path}")
        rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    stereo = np.frombuffer(frames, dtype="<i4").reshape(-1, 2)
    mono = stereo.astype(np.float64).mean(axis=1) / 2_147_483_648.0
    return rate, mono


def resample_linear(signal: np.ndarray, source_rate: int) -> np.ndarray:
    output_count = int(round(len(signal) * OUTPUT_RATE / source_rate))
    source_position = np.arange(len(signal), dtype=np.float64)
    output_position = np.arange(output_count, dtype=np.float64) * source_rate / OUTPUT_RATE
    return np.interp(output_position, source_position, signal)


def write_pcm16(path: Path, signal: np.ndarray) -> None:
    pcm = np.int16(np.clip(signal, -1.0, 1.0) * 32767.0)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(OUTPUT_RATE)
        stream.writeframes(pcm.astype("<i2", copy=False).tobytes())


def main() -> int:
    notes: list[np.ndarray] = []
    for name in SOURCE_NAMES:
        rate, note = read_pcm32_stereo(SOURCE_DIR / name)
        notes.append(resample_linear(note, rate))

    output_count = int(round(OUTPUT_DURATION_S * OUTPUT_RATE))
    chord = np.zeros(output_count, dtype=np.float64)
    for string_index, note in enumerate(notes):
        start = int(round(string_index * STRUM_DELAY_S * OUTPUT_RATE))
        count = min(len(note), output_count - start)
        chord[start:start + count] += note[:count]

    # Мягкий срез предотвращает щелчок в конце файла.
    fade_count = int(round(0.05 * OUTPUT_RATE))
    chord[-fade_count:] *= np.linspace(1.0, 0.0, fade_count, endpoint=True)
    chord -= float(np.mean(chord))
    maximum = float(np.max(np.abs(chord)))
    if maximum == 0.0:
        raise ValueError("Исходные записи пусты")
    chord *= PEAK / maximum

    write_pcm16(SAMPLE_DIR / "e_major_dry.wav", chord)
    attack_count = int(round(ATTACK_DURATION_S * OUTPUT_RATE))
    write_pcm16(SAMPLE_DIR / "e_major_attack.wav", chord[:attack_count])
    print(f"Полный аккорд: {SAMPLE_DIR / 'e_major_dry.wav'}")
    print(f"Атака для расчёта: {SAMPLE_DIR / 'e_major_attack.wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
