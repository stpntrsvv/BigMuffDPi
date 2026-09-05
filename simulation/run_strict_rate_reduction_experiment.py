"""Проверяет строгую портовую модель на сетках 1×, 2× и 4×."""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

from generate_port_multirate_fixture import FAST_ACTIVE, fast_affine, slow_affine
from muff_complete_model import Q1_FORWARD, Q1_REVERSE, operating_point
from muff_multirate_model import Q1_NONLINEAR, Q2_COLLECTOR, SLOW_NODES, prepare_fast, slow_step
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"strict_rate_reduction"
EXPERIMENT=ROOT/"simulation"/"experiments"/"strict_rate_reduction"
RATE,DURATION=48_000,0.020
SETTINGS=((0.25,0.0),(0.5,0.5),(1.0,1.0))
LEVELS=(25,50,100,200)
RATES=((1,1),(2,1),(2,2),(4,2))


def model(parameters,sustain,tone,fast_factor,slow_factor):
    nodes,dc_q=operating_point(sustain,tone,0.8,parameters)
    sr,slow=slow_affine(parameters,tone,0.8,RATE*slow_factor)
    ss=sr.incidence.T@np.r_[nodes[Q2_COLLECTOR],nodes[SLOW_NODES]]
    _,_,_,_,pi,pg=slow_step(sr,ss,dc_q[Q1_NONLINEAR],float(nodes[Q2_COLLECTOR]),True)
    fast=fast_affine(parameters,dc_q[:9],pg,RATE*fast_factor,None,sustain)
    fr=prepare_fast(parameters,1.0/(RATE*fast_factor),pg,None,sustain)
    fs=fr.incidence.T@nodes[:len(fr.source)]
    initial=(fs,dc_q[FAST_ACTIVE],ss,dc_q[[Q1_FORWARD,Q1_REVERSE]],pi-pg*nodes[Q2_COLLECTOR],pg)
    return fast,slow,initial


def main():
    p=Q3Parameters(); rows=[]; time=np.arange(int(round(DURATION*RATE))+1)/RATE
    start=int(round(0.003*RATE))
    for sustain,tone in SETTINGS:
        prepared={rate:model(p,sustain,tone,*rate) for rate in RATES}
        for level in LEVELS:
            signal=np.asarray(source_input(level*1e-3)(time))
            fast,slow,initial=prepared[(4,2)]
            reference=run(fast,slow,initial,signal,p,"full_repeat",4,2)
            for ff,sf in RATES:
                if (ff,sf)==(4,2): tested=reference
                else:
                    fast,slow,initial=prepared[(ff,sf)]
                    tested=run(fast,slow,initial,signal,p,"full_repeat",ff,sf)
                finite=bool(np.all(np.isfinite(tested)))
                error=tested[start:]-reference[start:]
                rms=float(np.sqrt(np.mean(error*error))*1e3) if finite else np.inf
                peak=float(np.max(np.abs(error))*1e3) if finite else np.inf
                stable=finite and peak<1000.0
                rows.append((sustain,tone,level,ff,sf,stable,rms,peak))
                print(sustain,tone,level,ff,sf,finite,rms,peak,flush=True)
    RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.writer(stream);w.writerow(("sustain","tone","level_mv","fast_factor","slow_factor","stable","rms_mv","peak_mv"));w.writerows(rows)
    lines=[]
    for ff,sf in RATES:
        group=[r for r in rows if r[3:5]==(ff,sf)]
        lines.append(f"| {ff}×/{sf}× | {max(r[6] for r in group):.3f} | {max(r[7] for r in group):.3f} | {'да' if all(r[5] for r in group) else 'нет'} | {ff/4:.2f} |")
    report="""# Строгое снижение частоты портовой модели

Коэффициенты обеих подсистем заново построены для каждого шага. На каждом шаге
выполняются две полные поправки; это не пропуск строк старой сетки. Эталон — 4×/2×.
Проверена атака аккорда 25–200 мВ при трёх положениях Sustain/Tone.

| Быстро/медленно | Худшее СКО, мВ | Худший пик, мВ | Всюду конечно | Относительное число быстрых шагов |
|---|---:|---:|---|---:|
"""+"\n".join(lines)+"\n"
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")


if __name__=="__main__":main()
