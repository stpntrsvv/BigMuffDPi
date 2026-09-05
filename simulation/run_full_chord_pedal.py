"""Прогоняет полный сухой аккорд через эталонную схему Big Muff Pi в ngspice."""

from __future__ import annotations

import argparse
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_reference import find_ngspice


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_dry.wav"
NGSPICE_DIR = ROOT / "simulation" / "ngspice"
RAW = ROOT / "simulation" / "raw" / "full_chord_pedal"
EXPERIMENT = ROOT / "simulation" / "experiments" / "full_chord_pedal"
FIGURES = EXPERIMENT / "figures"
RATE = 48_000
INPUT_PEAK_V = 0.100
MAXIMUM_STEP_S = 1.0 / 384_000.0


@dataclass(frozen=True)
class Setting:
    stem: str
    title: str
    sustain: float
    tone: float
    volume: float = 0.8


SETTINGS = (
    Setting("sustain_025_tone_050", "Sustain 0,25; Tone 0,50", 0.25, 0.50),
    Setting("sustain_050_tone_050", "Sustain 0,50; Tone 0,50", 0.50, 0.50),
    Setting("sustain_100_tone_000", "Sustain 1,00; Tone 0,00", 1.00, 0.00),
    Setting("sustain_100_tone_050", "Sustain 1,00; Tone 0,50", 1.00, 0.50),
    Setting("sustain_100_tone_100", "Sustain 1,00; Tone 1,00", 1.00, 1.00),
)


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != RATE
        ):
            raise ValueError(f"Ожидается моно PCM16, 48 кГц: {path}")
        frames = stream.readframes(stream.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0


def write_pcm16(path: Path, signal: np.ndarray, scale_v: float) -> None:
    pcm = np.int16(np.clip(signal / scale_v, -1.0, 1.0) * 32767.0)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(pcm.astype("<i2", copy=False).tobytes())


def build_circuit(path: Path, setting: Setting, input_v: np.ndarray) -> None:
    duration_s = (len(input_v) - 1) / RATE
    lines = [
        "* Full dry guitar chord through Big Muff Pi V3",
        f".param SUSTAIN={setting.sustain:.9g}",
        f".param TONE={setting.tone:.9g}",
        f".param VOLUME={setting.volume:.9g}",
        "VCC vcc 0 9",
        "VIN in 0 PWL(",
    ]
    lines.extend(
        f"+ {index / RATE:.12g} {value:.12g}"
        for index, value in enumerate(input_v)
    )
    output_table = f"../raw/full_chord_pedal/{setting.stem}.txt"
    lines.extend((
        ")",
        ".include models.lib",
        ".include big_muff_v3.inc",
        ".options temp=27 plotwinsize=0",
        ".control",
        "set noaskquit",
        "set wr_singlescale",
        "set wr_vecnames",
        f"tran {1 / RATE:.12g} {duration_s:.12g} 0 {MAXIMUM_STEP_S:.12g}",
        f"wrdata {output_table} v(out)",
        "quit",
        ".endc",
        ".end",
    ))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_setting(executable: Path, setting: Setting, input_v: np.ndarray) -> float:
    circuit = RAW / f"{setting.stem}.cir"
    log_path = RAW / f"{setting.stem}.log"
    build_circuit(circuit, setting, input_v)
    started = time.perf_counter()
    result = subprocess.run(
        [str(executable), "-b", str(circuit)],
        cwd=NGSPICE_DIR,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ngspice: код {result.returncode}; см. {log_path}")
    output_table = RAW / f"{setting.stem}.txt"
    if not output_table.is_file():
        raise RuntimeError(f"ngspice не создал {output_table}; см. {log_path}")
    return elapsed


def output_is_complete(setting: Setting, duration_s: float) -> bool:
    path = RAW / f"{setting.stem}.txt"
    if not path.is_file() or path.stat().st_size < 100:
        return False
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 1024))
        tail = stream.read().decode("ascii", errors="ignore")
    rows = tuple(line.split() for line in tail.splitlines() if line.strip())
    try:
        final_time = float(rows[-1][0])
    except (IndexError, ValueError):
        return False
    return final_time >= duration_s - 2.0 / RATE


def load_output(setting: Setting, count: int) -> np.ndarray:
    table = np.loadtxt(RAW / f"{setting.stem}.txt", skiprows=1)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"Неожиданный формат таблицы: {setting.stem}")
    target_time = np.arange(count) / RATE
    output = np.interp(target_time, table[:, 0], table[:, 1])
    output -= float(np.mean(output[-min(count, RATE // 10):]))
    fade_count = min(count, RATE // 100)
    output[-fade_count:] *= np.linspace(1.0, 0.0, fade_count)
    return output


def window_rms(signal: np.ndarray, window: int) -> np.ndarray:
    count = len(signal) // window
    framed = signal[:count * window].reshape(count, window)
    return np.sqrt(np.mean(framed * framed, axis=1))


def elapsed_text(seconds: float) -> str:
    return "ранее" if not np.isfinite(seconds) else f"{seconds:.1f}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", type=Path)
    parser.add_argument(
        "--quick", action="store_true",
        help="проверить первые 50 мс и одно положение",
    )
    parser.add_argument("--workers", type=int, default=4, help="число одновременных ngspice")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    RAW.mkdir(parents=True, exist_ok=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    executable = find_ngspice(args.ngspice)
    input_v = read_pcm16_mono(SAMPLE)
    input_v -= float(np.mean(input_v))
    input_v *= INPUT_PEAK_V / float(np.max(np.abs(input_v)))
    settings = SETTINGS
    if args.quick:
        input_v = input_v[: int(round(0.050 * RATE))]
        settings = (SETTINGS[3],)

    duration_s = (len(input_v) - 1) / RATE
    elapsed_by_stem: dict[str, float] = {}
    outputs: dict[str, np.ndarray] = {}
    pending = tuple(
        setting for setting in settings if not output_is_complete(setting, duration_s)
    )
    for setting in settings:
        if setting not in pending:
            elapsed_by_stem[setting.stem] = float("nan")
            print(f"Уже рассчитано: {setting.title}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(pending)))) as pool:
        futures = {
            pool.submit(run_setting, executable, setting, input_v): setting
            for setting in pending
        }
        for future in as_completed(futures):
            setting = futures[future]
            elapsed_by_stem[setting.stem] = future.result()
            print(f"Готово: {setting.title} — {elapsed_by_stem[setting.stem]:.1f} с", flush=True)

    for setting in settings:
        outputs[setting.stem] = load_output(setting, len(input_v))

    if args.quick:
        print("Быстрая проверка пройдена")
        return 0

    common_scale = 1.05 * max(float(np.max(np.abs(value))) for value in outputs.values())
    input_scale = 1.05 * float(np.max(np.abs(input_v)))
    write_pcm16(EXPERIMENT / "input_100mv.wav", input_v, input_scale)
    for setting in settings:
        write_pcm16(EXPERIMENT / f"{setting.stem}.wav", outputs[setting.stem], common_scale)
    pause = np.zeros(RATE // 2)
    montage = np.concatenate(tuple(
        part
        for setting in settings
        for part in (outputs[setting.stem], pause)
    ))
    write_pcm16(EXPERIMENT / "all_settings.wav", montage, common_scale)

    window = RATE // 100
    envelope_time = (np.arange(len(input_v) // window) + 0.5) * window / RATE
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(envelope_time, 1e3 * window_rms(input_v, window), color="black")
    axes[0].set_ylabel("Вход, мВ СКО")
    for setting in settings:
        axes[1].plot(
            envelope_time, window_rms(outputs[setting.stem], window), label=setting.title
        )
    axes[1].set_ylabel("Выход, В СКО")
    axes[1].set_xlabel("Время, с")
    axes[1].legend(ncol=2)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Полный сухой аккорд через Big Muff Pi V3")
    figure.tight_layout()
    figure.savefig(FIGURES / "level_envelopes.png", dpi=160)
    plt.close(figure)

    rows = "\n".join(
        f"| {setting.title} | {setting.volume:.2f} | "
        f"{1e3 * np.sqrt(np.mean(outputs[setting.stem] ** 2)):.2f} | "
        f"{np.max(np.abs(outputs[setting.stem])):.3f} | "
        f"{elapsed_text(elapsed_by_stem[setting.stem])} | [{setting.stem}.wav]({setting.stem}.wav) |"
        for setting in settings
    )
    report = f"""# Полный гитарный аккорд через педаль

Четырёхсекундный сухой ми-мажорный аккорд пропущен через полную эталонную
схему Big Muff Pi V3 в ngspice. Вход приведён к 100 мВ пик. Внутренний максимальный шаг
решателя равен 1/384000 с, выходные WAV — моно PCM16, 48 кГц.

Все варианты имеют единый масштаб амплитуды. Volume везде равен 0,8, поэтому разница
громкости между файлами сохранена.

| Sustain; Tone | Volume | Выход, мВ СКО | Пик, В | Расчёт, с | Звук |
|---|---:|---:|---:|---:|---|
{rows}

- [Сухой вход 100 мВ](input_100mv.wav)
- [Все пять положений подряд](all_settings.wav) — порядок совпадает с таблицей, между
  вариантами оставлено по 0,5 с тишины.

![Огибающие уровня](figures/level_envelopes.png)

Это эталонная SPICE-схема с предварительными моделями BC239 и 1N914, а не ещё не перенесённая
на STM32 сокращённая модель. Файлы показывают звук полной схемы и будут служить опорой
для последующего сравнения.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
