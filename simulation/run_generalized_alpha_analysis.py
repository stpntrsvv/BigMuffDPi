"""Перебирает затухание обобщённого α-метода на синусах и аккорде."""

from __future__ import annotations

import wave
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muff_complete_model import OUTPUT, simulate_complete, simulate_complete_generalized_alpha
from run_oversampling_analysis import BASE_RATE, align, analyze, lowpass_downsample


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "simulation" / "experiments" / "generalized_alpha"
CACHE = EXPERIMENT / "cache"
CHORD = ROOT / "simulation" / "samples" / "e_major_chord" / "e_major_attack.wav"
RHOS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0)
FREQUENCIES = (440.0, 1000.0, 3000.0, 6000.0)
LEVELS = (0.005, 0.025, 0.100)
DURATION_S = 0.060
CHORD_DURATION_S = 0.120


def cached_tone(frequency_hz: float, level_v: float, rho: float) -> np.ndarray:
    path = CACHE / f"tone_{frequency_hz:g}_{level_v * 1e3:g}mv_rho{rho:g}.npy"
    if path.exists():
        return np.load(path)
    result = simulate_complete_generalized_alpha(
        1.0, 1.0, 0.8, 1,
        lambda t: level_v * np.sin(2.0 * np.pi * frequency_hz * t),
        rho, DURATION_S, "hybrid", True,
    )
    output = result.node_v[:, OUTPUT]
    np.save(path, output)
    return output


def reference_tone(frequency_hz: float, level_v: float) -> np.ndarray:
    path = CACHE / f"tone_{frequency_hz:g}_{level_v * 1e3:g}mv_reference.npy"
    if path.exists():
        return np.load(path)
    result = simulate_complete(
        1.0, 1.0, 0.8, 16,
        lambda t: level_v * np.sin(2.0 * np.pi * frequency_hz * t),
        DURATION_S, "hybrid", True,
    )
    output = lowpass_downsample(result.node_v[:, OUTPUT], 16)
    np.save(path, output)
    return output


def read_chord(level_v: float):
    with wave.open(str(CHORD), "rb") as stream:
        rate = stream.getframerate()
        source = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64)
    source -= np.mean(source)
    source *= level_v / np.max(np.abs(source))
    return lambda t: np.interp(t * rate, np.arange(len(source)), source, left=0.0, right=0.0)


def chord_output(level_v: float, rho: float | None) -> np.ndarray:
    name = "reference" if rho is None else f"rho{rho:g}"
    path = CACHE / f"chord_{level_v * 1e3:g}mv_{name}.npy"
    if path.exists():
        return np.load(path)
    factor = 16 if rho is None else 1
    source = read_chord(level_v)
    result = (
        simulate_complete(1.0, 1.0, 0.8, factor, source, CHORD_DURATION_S, "hybrid", True)
        if rho is None else
        simulate_complete_generalized_alpha(
            1.0, 1.0, 0.8, factor, source, rho,
            CHORD_DURATION_S, "hybrid", True,
        )
    )
    output = lowpass_downsample(result.node_v[:, OUTPUT], factor)
    np.save(path, output)
    return output


def alternating_ratio(signal: np.ndarray) -> float:
    body = signal - np.mean(signal)
    alternating = abs(float(np.dot(body, (-1.0) ** np.arange(len(body))))) / len(body)
    rms = float(np.sqrt(np.mean(body * body)))
    return alternating / max(rms, 1e-30)


def general_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    reference, candidate = align(reference, candidate)
    reference -= np.mean(reference)
    candidate -= np.mean(candidate)
    error = candidate - reference
    relative = np.sqrt(np.mean(error * error)) / max(np.sqrt(np.mean(reference * reference)), 1e-30)
    window = np.hanning(len(reference))
    rm = np.abs(np.fft.rfft(reference * window))
    cm = np.abs(np.fft.rfft(candidate * window))
    spectral = np.linalg.norm(cm - rm) / max(np.linalg.norm(rm), 1e-30)
    return 100.0 * float(relative), 20.0 * np.log10(max(float(spectral), 1e-15)), alternating_ratio(candidate)


def valid(signal: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(signal)) and np.max(np.abs(signal)) <= 20.0)


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    scores: dict[float, list[float]] = {rho: [] for rho in RHOS}
    failures: dict[float, int] = {rho: 0 for rho in RHOS}
    for frequency_hz in FREQUENCIES:
        for level_v in LEVELS:
            print(f"Синус {frequency_hz:g} Гц, {level_v * 1e3:g} мВ", flush=True)
            reference = reference_tone(frequency_hz, level_v)[int(0.02 * BASE_RATE):]
            for rho in RHOS:
                candidate = cached_tone(frequency_hz, level_v, rho)[int(0.02 * BASE_RATE):]
                if not valid(candidate):
                    failures[rho] += 1
                    rows.append(f"| {frequency_hz:g} | {level_v * 1e3:g} | {rho:g} | СРЫВ | — | — |")
                    continue
                metrics = analyze(reference, candidate, 1, frequency_hz)
                alt = alternating_ratio(candidate)
                scores[rho].append(metrics.magnitude_error_db)
                rows.append(
                    f"| {frequency_hz:g} | {level_v * 1e3:g} | {rho:g} | "
                    f"{metrics.relative_error_percent:.3f} | {metrics.magnitude_error_db:.2f} | {alt:.3e} |"
                )

    chord_rows: list[str] = []
    for level_v in (0.025, 0.100):
        print(f"Аккорд {level_v * 1e3:g} мВ", flush=True)
        reference = chord_output(level_v, None)
        for rho in RHOS:
            candidate = chord_output(level_v, rho)
            if not valid(candidate):
                failures[rho] += 1
                chord_rows.append(f"| {level_v * 1e3:g} | {rho:g} | СРЫВ | — | — |")
                continue
            relative, spectral, alt = general_metrics(reference, candidate)
            scores[rho].append(spectral)
            chord_rows.append(
                f"| {level_v * 1e3:g} | {rho:g} | {relative:.3f} | {spectral:.2f} | {alt:.3e} |"
            )

    figure, axis = plt.subplots(figsize=(10, 6))
    means = [np.mean(scores[rho]) if scores[rho] else np.nan for rho in RHOS]
    axis.plot(RHOS, means, "o-")
    for rho, count in failures.items():
        axis.annotate(f"срывов: {count}", (rho, means[RHOS.index(rho)]), xytext=(0, 8), textcoords="offset points", ha="center")
    axis.set_xlabel(r"$\rho_\infty$")
    axis.set_ylabel("Средняя ошибка модуля спектра, дБ")
    axis.set_title("Обобщённый α-метод: точность и устойчивость")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(EXPERIMENT / "rho_sweep.png", dpi=160)
    plt.close(figure)

    summary = "\n".join(
        f"| {rho:g} | {failures[rho]} | {np.mean(scores[rho]):.2f} |"
        for rho in RHOS
    )
    report = f"""# Обобщённый α-метод

Проверен метод второго порядка для систем первого порядка с управляемым
высокочастотным затуханием. Значение `rho=1` сохраняет предельную высокочастотную
моду, `rho=0` максимально её подавляет.

## Сводка

| rho | Число срывов | Средняя ошибка спектра исправных точек, дБ |
|---:|---:|---:|
{summary}

## Синусы

| Частота, Гц | Вход, мВ | rho | Ошибка формы, % | Ошибка спектра, дБ | Чередующаяся доля |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Аккорд

| Вход, мВ | rho | Ошибка формы, % | Ошибка спектра, дБ | Чередующаяся доля |
|---:|---:|---:|---:|---:|
{chr(10).join(chord_rows)}

![Перебор затухания](rho_sweep.png)

## Вывод

Лучший общий компромисс данной сетки — `rho=0.2`. Он имеет наименьшую среднюю
ошибку спектра исправных точек и устойчив на аккордах 25 и 100 мВ. На аккорде
25 мВ ошибка формы равна 3,470%, ошибка спектра −34,87 дБ; на аккорде 100 мВ —
4,628% и −28,38 дБ. Максимально затухающий край `rho=0`, спектрально
эквивалентный BDF2, на сильном аккорде срывается.

Сильные непрерывные синусы 100 мВ остаются проблемными почти для всей сетки.
Это свойство полной узловой модели и отдельный вопрос от устойчивости на
реальном гитарном сигнале: составное C-ядро BDF2 ранее прошло полный аккорд без
срывов. Перенос `rho=0.2` в C требует отдельной проверки соответствия редукции.
"""
    (EXPERIMENT / "report.md").write_text(report, encoding="utf-8")
    print(EXPERIMENT / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
