"""Проверяет вычислительно дешёвое усечение диагональных режимов портовой модели."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_experiment import RATE, read_input
from run_port_sparse_linear_experiment import run
from run_port_state_reduction_experiment import reduced

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation" / "raw" / "port_modal_reduction"
EXPERIMENT = ROOT / "simulation" / "experiments" / "port_modal_reduction"

def main():
    p = Q3Parameters(); nodes, q = operating_point(1.0, 1.0, 0.8, p)
    slow_reduction, slow = slow_affine(p)
    slow_state = slow_reduction.incidence.T @ np.r_[nodes[Q2_COLLECTOR], nodes[SLOW_NODES]]
    _, _, _, _, port_i, port_g = slow_step(
        slow_reduction, slow_state, q[Q1_NONLINEAR], float(nodes[Q2_COLLECTOR]), True)
    fast_reduction = prepare_fast(p, 1.0/(RATE*4), port_g)
    fast_state = fast_reduction.incidence.T @ nodes[:len(fast_reduction.source)]
    fast = fast_affine(p, q[:9], port_g, RATE*4)
    initial = (fast_state, q[FAST_ACTIVE], slow_state, q[[Q1_FORWARD, Q1_REVERSE]],
               port_i-port_g*nodes[Q2_COLLECTOR], port_g)
    source = np.asarray(read_input()(np.arange(961)/RATE)); rows=[]
    for level in (25, 50, 100):
        signal=source*(level*1e-3/np.max(np.abs(source)))
        reference=run(fast,slow,initial,signal,p,solver_mode="full_repeat")
        for fr,sr in ((9,4),(8,4),(7,4),(9,3),(8,3)):
            if (fr,sr)==(9,4): tested=reference.copy(); scores=np.zeros(9)
            else:
                frd,fi,scores=reduced(fast,fast_state,fr,"fast","modal")
                srd,si,_=reduced(slow,slow_state,sr,"slow","modal")
                tested=run(frd,srd,(fi,initial[1],si,initial[3],initial[4],initial[5]),
                           signal,p,solver_mode="full_repeat")
            error=tested[1:]-reference[1:]
            rms=float(np.sqrt(np.mean(error*error))); peak=float(np.max(np.abs(error)))
            rows.append((level,fr,sr,rms,peak)); print(level,fr,sr,rms*1e3,peak*1e3)
    RAW.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(("level_mv","fast_rank","slow_rank","rms_v","peak_v"));w.writerows(rows)
    lines=["# Усечение диагональных режимов","",
           "Собственные режимы ранжированы по управляемости, наблюдаемости и близости полюса к единичной окружности. "
           "Такое усечение сохраняет диагональный переход состояния и потому вычислительно дёшево.", "",
           "| вход, мВ | быстрых | медленных | СКО, мВ | пик, мВ |","|---:|---:|---:|---:|---:|"]
    lines += [f"| {l} | {fr} | {sr} | {r*1e3:.6f} | {pk*1e3:.6f} |" for l,fr,sr,r,pk in rows]
    lines += ["", "## Вывод", "",
              "Удаление одного быстрого диагонального режима уже даёт 16,9–32,4 мВ СКО и пики 135–345 мВ. "
              "Удаление одного медленного режима даёт 119–129 мВ СКО. Перенос сокращённого варианта на STM32 "
              "не имеет смысла: приемлемого по точности диагонального сокращения нет.", ""]
    EXPERIMENT.mkdir(parents=True,exist_ok=True);(EXPERIMENT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__": main()
