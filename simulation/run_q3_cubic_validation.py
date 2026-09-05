"""Проверяет кубический локальный закон Q3 на сетке Sustain/Tone и уровней."""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import active_terms, run

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"q3_cubic_validation"
EXPERIMENT=ROOT/"simulation"/"experiments"/"q3_cubic_validation"
RATE,DURATION_S=48_000,0.020
SETTINGS=((0.25,0.0),(0.25,1.0),(0.5,0.5),(1.0,0.0),(1.0,0.5),(1.0,1.0))
LEVELS=(25,50,100,200)


def main():
    p=Q3Parameters(); scale=p.forward_ideality*p.thermal_voltage_v; rows=[]
    start=int(round(0.003*RATE)); time=np.arange(int(round(DURATION_S*RATE))+1)/RATE
    for sustain,tone in SETTINGS:
        nodes,dc_q=operating_point(sustain,tone,0.8,p)
        sr,slow=slow_affine(p,tone,0.8)
        ss=sr.incidence.T@np.r_[nodes[Q2_COLLECTOR],nodes[SLOW_NODES]]
        _,_,_,_,pi,pg=slow_step(sr,ss,dc_q[Q1_NONLINEAR],float(nodes[Q2_COLLECTOR]),True)
        fast=fast_affine(p,dc_q[:9],pg,RATE*4,None,sustain)
        fr=prepare_fast(p,1.0/(RATE*4),pg,None,sustain)
        fs=fr.incidence.T@nodes[:len(fr.source)]
        initial=(fs,dc_q[FAST_ACTIVE],ss,dc_q[[Q1_FORWARD,Q1_REVERSE]],pi-pg*nodes[Q2_COLLECTOR],pg)
        dc=dc_q[FAST_ACTIVE]; i0,g0=active_terms(dc,p)
        def cubic(q,parameters):
            current,first=active_terms(q,parameters); d=q[1]-dc[1]; z=d/scale
            current[1]=i0[1]+g0[1]*d+0.5*g0[1]*d*z+g0[1]*d*z*z/6.0
            first[1]=g0[1]*(1.0+z+0.5*z*z)
            return current,first
        for level in LEVELS:
            signal=np.asarray(source_input(level*1e-3)(time))
            reference=run(fast,slow,initial,signal,p,"full_repeat")
            tested=run(fast,slow,initial,signal,p,"full_repeat",terms_function=cubic)
            error=tested[start:]-reference[start:]
            rows.append((sustain,tone,level,bool(np.all(np.isfinite(tested))),
                         np.sqrt(np.mean(error*error))*1e3,np.max(np.abs(error))*1e3))
            print(sustain,tone,level,rows[-1][-2],rows[-1][-1],flush=True)
    RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as s:
        w=csv.writer(s);w.writerow(("sustain","tone","level_mv","finite","rms_mv","peak_mv"));w.writerows(rows)
    table="\n".join(f"| {s:.2f} | {t:.2f} | {max(r[4] for r in rows if r[:2]==(s,t)):.6f} | {max(r[5] for r in rows if r[:2]==(s,t)):.6f} |" for s,t in SETTINGS)
    report=f"""# Проверка кубического закона Q3

Локальное кубическое разложение прямого перехода Q3 проверено относительно
экспоненциального `full_repeat` на атаке аккорда 25, 50, 100 и 200 мВ.

| Sustain | Tone | Худшее СКО, мВ | Худший пик, мВ |
|---:|---:|---:|---:|
{table}

Все конденсаторы, четыре портовые переменные и нелинейные связи сохранены.
Аппаратная экономия пока не измерена.
"""
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")


if __name__=="__main__":main()
