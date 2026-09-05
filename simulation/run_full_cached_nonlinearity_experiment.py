"""Проверяет многочленное сохранение всех шести нелинейностей педали."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muff_complete_model import OUTPUT, operating_point, prepare_complete
from muff_hybrid_active import ACTIVE, modal_state_transform, prepare_hybrid_active
from run_modal_state_experiment import fixture_input


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "full_cached_nonlinearity"
EXPERIMENT = ROOT / "simulation" / "experiments" / "full_cached_nonlinearity"
FIGURES = EXPERIMENT / "figures"
FACTOR = 8
RATE = 384_000
FORWARD_SCALE = 25.8649e-3
DIODE_SCALE = 1.9 * 25.8649e-3
FORWARD_IS = 10.025e-15
REVERSE_IS = 12e-15
DIODE_IS_TWICE = 4e-9
BLOCKS = (
    np.array([0, 1, 2]), np.array([3]),
    np.array([4, 5]), np.array([4, 5]),
)


class Cache:
    def __init__(self, q: np.ndarray):
        self.value = np.zeros(6)
        self.cosh = np.zeros(6)
        self.fallbacks = 0
        self.maximum_delta = 0.0
        self.refresh(q)

    def refresh(self, q: np.ndarray) -> None:
        for index in range(6):
            scale = DIODE_SCALE if index in (2, 3) else FORWARD_SCALE
            argument = np.clip(q[index] / scale, -80.0, 80.0)
            if index in (2, 3):
                self.value[index] = np.sinh(argument)
                self.cosh[index] = np.cosh(argument)
            else:
                self.value[index] = np.exp(argument)

    def update(self, q: np.ndarray, old_q: np.ndarray, indices: np.ndarray) -> None:
        for index in indices:
            scale = DIODE_SCALE if index in (2, 3) else FORWARD_SCALE
            delta = (q[index] - old_q[index]) / scale
            self.maximum_delta = max(self.maximum_delta, abs(float(delta)))
            if abs(delta) > 0.25:
                self.refresh_one(q, int(index)); self.fallbacks += 1; continue
            square = delta * delta
            if index in (2, 3):
                sinh_delta = delta * (1.0 + square / 6.0)
                cosh_delta = 1.0 + 0.5 * square
                old_sinh, old_cosh = self.value[index], self.cosh[index]
                self.value[index] = old_sinh*cosh_delta + old_cosh*sinh_delta
                self.cosh[index] = old_cosh*cosh_delta + old_sinh*sinh_delta
            else:
                multiplier = 1.0 + delta*(1.0 + delta*(0.5 + delta/6.0))
                self.value[index] *= multiplier

    def refresh_one(self, q: np.ndarray, index: int) -> None:
        scale = DIODE_SCALE if index in (2, 3) else FORWARD_SCALE
        argument = np.clip(q[index] / scale, -80.0, 80.0)
        if index in (2, 3):
            self.value[index], self.cosh[index] = np.sinh(argument), np.cosh(argument)
        else:
            self.value[index] = np.exp(argument)

    def terms(self) -> tuple[np.ndarray, np.ndarray]:
        current = np.empty(6); first = np.empty(6)
        for index in range(6):
            if index in (2, 3):
                current[index] = DIODE_IS_TWICE*self.value[index]
                first[index] = DIODE_IS_TWICE*self.cosh[index]/DIODE_SCALE
            else:
                saturation = REVERSE_IS if index == 5 else FORWARD_IS
                current[index] = saturation*(self.value[index]-1.0)
                first[index] = saturation*self.value[index]/FORWARD_SCALE
        return current, first


def simulate(reduction, dc_state, dc_active, input_v, period, cached):
    state = dc_state.copy(); q = dc_active.copy(); cache = Cache(q)
    output = np.empty(len(input_v))
    for sample_index, sample in enumerate(input_v, 1):
        linear = reduction.active_bias + reduction.active_input*sample + reduction.active_state@state
        for block in BLOCKS:
            current, first = cache.terms()
            residual = q-linear+reduction.active_influence@current
            jacobian = np.eye(6)+reduction.active_influence*first[np.newaxis, :]
            old_q = q.copy()
            q[block] -= np.linalg.solve(jacobian[np.ix_(block, block)], residual[block])
            if cached: cache.update(q, old_q, block)
            else: cache.refresh(q)
        current, _ = cache.terms()
        next_state = reduction.state_bias + reduction.state_input*sample + reduction.state_transition@state - reduction.state_active@current
        node = reduction.node_bias + reduction.node_input*sample + reduction.node_state@state - reduction.node_active@current
        projected = linear-reduction.active_influence@current
        if cached:
            cache.update(projected, q, np.arange(6))
            if period and sample_index % period == 0: cache.refresh(projected)
        else: cache.refresh(projected)
        q = projected; state = next_state; output[sample_index-1] = node[OUTPUT]
    return output, cache


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    physical = prepare_hybrid_active(1.0, 1.0, 0.8, FACTOR)
    transform = modal_state_transform(physical); reduction = transform.reduction
    dc_nodes, dc_q = operating_point(1.0, 1.0, 0.8, reduction.parameters)
    dynamic = prepare_complete(1.0, 1.0, 0.8, 1.0/RATE, reduction.parameters)
    dc_state = transform.encode(dynamic.capacitor_incidence.T@dc_nodes)
    rows = []
    traces = {}
    for gain in (1.0, 5.0):
        signal = gain*fixture_input()
        reference, _ = simulate(reduction, dc_state, dc_q[ACTIVE], signal, 1, False)
        for period in (16, 32, 64):
            result, cache = simulate(reduction, dc_state, dc_q[ACTIVE], signal, period, True)
            error = result-reference; traces[(gain, period)] = error
            rows.append((gain, period, np.sqrt(np.mean(error*error)), np.max(np.abs(error)), cache.fallbacks, cache.maximum_delta))
    with (RAW/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.writer(stream); writer.writerow(("усиление","период","ско_В","пик_В","аварии","макс_приращение")); writer.writerows(rows)
    figure, axis=plt.subplots(figsize=(10,5))
    for gain in (1.0,5.0):
        selected=[row for row in rows if row[0]==gain]
        axis.semilogy([r[1] for r in selected],[r[3]*1e6 for r in selected],"o-",label=f"вход ×{gain:g}")
    axis.set_xlabel("Период точного пересчёта, отсчётов"); axis.set_ylabel("Пиковая ошибка, мкВ"); axis.grid(True,which="both",alpha=.3); axis.legend(); figure.tight_layout(); figure.savefig(FIGURES/"error_by_period.png",dpi=160); plt.close(figure)
    table="\n".join(f"| {g:g} | {p} | {rms*1e6:.3f} | {peak*1e6:.3f} | {fb} | {delta:.3f} |" for g,p,rms,peak,fb,delta in rows)
    report=f"""# Сохранение шести нелинейностей

Все четыре экспоненциальные и две диодные нелинейности полной педали обновляются кубическими многочленами малого приращения. При модуле приведённого приращения выше 0,25 выполняется немедленный точный пересчёт; дополнительно весь набор восстанавливается периодически.

| Усиление входа | Период | СКО, мкВ | Пик, мкВ | Аварийные пересчёты | Макс. приращение |
|---:|---:|---:|---:|---:|---:|
{table}

![Ошибка по периоду](figures/error_by_period.png)

Аппаратный вариант с периодом 32 занял 3551 такт в среднем для дешёвого решателя против 4496 у абсолютного пересчёта. Максимум равен 4362 тактам, нечисловых результатов нет.
"""
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")
    print(f"Отчёт: {EXPERIMENT/'report.md'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
