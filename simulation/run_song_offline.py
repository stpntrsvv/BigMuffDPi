"""Пропускает пользовательскую гитарную WAV-дорожку через составное C-ядро."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RATE = 48_000


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        rate = stream.getframerate()
        width = stream.getsampwidth()
        frames = stream.getnframes()
        compression = stream.getcomptype()
        raw = stream.readframes(frames)
    if width not in (2, 3) or compression != "NONE":
        raise ValueError("нужен несжатый PCM16 или PCM24 WAV")
    if channels not in (1, 2):
        raise ValueError("поддерживается только mono или stereo WAV")
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        full_scale = 32768.0
    else:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (octets[:, 0].astype(np.int32)
                   | (octets[:, 1].astype(np.int32) << 8)
                   | (octets[:, 2].astype(np.int32) << 16))
        samples = np.where(samples & 0x800000, samples - 0x1000000, samples).astype(np.float64)
        full_scale = 8388608.0
    samples = samples.reshape(-1, channels) / full_scale
    return samples.mean(axis=1), rate


def write_pcm16(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.int16(np.clip(samples, -1.0, 1.0) * 32767.0)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.astype("<i2").tobytes())


def prepare_input(source: Path, destination: Path, percentile: float) -> tuple[float, float, float]:
    samples, rate = read_pcm16(source)
    if rate != RATE:
        duration = len(samples) / rate
        old_time = np.arange(len(samples), dtype=np.float64) / rate
        new_time = np.arange(round(duration * RATE), dtype=np.float64) / RATE
        samples = np.interp(new_time, old_time, samples)
    samples -= samples.mean()
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        raise ValueError("входной файл содержит только тишину")
    reference = float(np.percentile(np.abs(samples), percentile))
    if reference == 0.0:
        raise ValueError("рабочий уровень входного файла равен нулю")
    normalized = samples / peak * 0.95
    write_pcm16(destination, normalized)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return peak, rms, reference / peak


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True,
                          capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="чистая гитарная дорожка WAV")
    parser.add_argument("--input-peak-mv", type=float, default=100.0,
                        help="напряжение рабочего уровня на входе модели (по умолчанию 100)")
    parser.add_argument("--level-percentile", type=float, default=99.9,
                        help="процентиль модуля для рабочего уровня; 100 означает пик")
    parser.add_argument("--sustain", type=float, default=0.5)
    parser.add_argument("--tone", type=float, default=0.5)
    parser.add_argument("--volume", type=float, default=0.5)
    parser.add_argument("--q1-cache", choices=("safe", "direct", "legacy"), default="safe",
                        help="Q1 cache: safe=точный refresh/16, direct=хранить exp, legacy=старый sinh/cosh")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "simulation/experiments/song_offline/output")
    args = parser.parse_args()
    if args.input_peak_mv <= 0:
        parser.error("--input-peak-mv должен быть положительным")
    if not 0.0 < args.level_percentile <= 100.0:
        parser.error("--level-percentile должен быть в диапазоне (0; 100]")
    for name in ("sustain", "tone", "volume"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name} должен быть в диапазоне 0...1")
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"файл не найден: {source}")
    compiler = shutil.which("gcc")
    if compiler is None:
        parser.error("gcc не найден в PATH")

    output = args.output_dir.resolve()
    build = output / "build"
    build.mkdir(parents=True, exist_ok=True)
    dry = output / "dry_48k_mono.wav"
    original_peak, original_rms, reference_to_peak = prepare_input(
        source, dry, args.level_percentile)
    model_peak_mv = args.input_peak_mv / reference_to_peak
    fixture = build / "composed_frontend_fixture_bdf2.h"
    generator = ROOT / "simulation/generate_composed_frontend_fixture.py"
    generated = run([
        sys.executable, str(generator), "--bdf2",
        "--sustain", str(args.sustain), "--tone", str(args.tone),
        "--volume", str(args.volume), "--output", str(fixture),
    ])

    results = []
    q1_flags = ({"safe": ["-DCOMPOSED_SLOW_FULL_REFRESH"],
                 "direct": ["-DCOMPOSED_Q1_DIRECT_EXP"],
                 "legacy": []}[args.q1_cache])
    for variant, extra in (("legacy", []), ("matched", ["-DCOMPOSED_Q2_MATCHED"])):
        executable = build / f"song_muff_{variant}.exe"
        compile_result = run([
            compiler, "-O3", "-std=c11", "-DCOMPOSED_PEDAL_HOST", "-DCOMPOSED_BDF2",
            *q1_flags,
            *extra, "-I", str(build), "-I", str(ROOT / "include"),
            str(ROOT / "simulation/composed_chord_host.c"),
            str(ROOT / "src/composed_frontend_benchmark.c"),
            "-o", str(executable), "-lm",
        ])
        rendered = output / f"muff_center_{variant}.wav"
        process_result = subprocess.run([
            str(executable), str(dry), str(rendered),
            str(model_peak_mv / 1000.0),
        ], cwd=ROOT, text=True, capture_output=True)
        diagnostic = (process_result.stdout + process_result.stderr).strip()
        results.append((variant, rendered.name, process_result.returncode, diagnostic))

    report = [
        "# Офлайн-прослушивание гитарной партии", "",
        f"Источник: `{source}`", "",
        f"Исходный WAV: peak={original_peak:.6f} FS, RMS={original_rms:.6f} FS.",
        f"Модель: BDF2 1×, 48 кГц, Sustain={args.sustain:g}, Tone={args.tone:g}, "
        f"Volume={args.volume:g}.",
        f"Режим кэша Q1: `{args.q1_cache}`.",
        f"{args.level_percentile:g}-й процентиль модуля входа принят за "
        f"{args.input_peak_mv:g} мВ; сохранённый максимум соответствует {model_peak_mv:.3f} мВ.", "",
        "Выход каждого варианта нормирован отдельно до 0,95 FS только для прослушивания. "
        "Поэтому громкость между файлами не является физическим измерением.", "",
        "## Результаты", "",
    ]
    for variant, filename, returncode, stats in results:
        status = "готов" if returncode == 0 else "СРЫВ, WAV не создан"
        report += [f"- `{filename}` — {variant} Q2; **{status}**; `{stats}`"]
    report += ["", "`dry_48k_mono.wav` — подготовленный сухой контроль.", "",
               f"Генератор: `{generated.stdout.strip()}`", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Готово: {output}")
    for _, filename, returncode, stats in results:
        status = "OK" if returncode == 0 else "FAIL"
        print(f"  {filename} [{status}]: {stats}")
    return 1 if any(row[2] != 0 for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
