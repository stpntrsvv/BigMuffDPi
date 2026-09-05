"""Сравнивает точный линейный переход 1×/2× с портовым эталоном 4×."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from generate_port_multirate_fixture import FAST_ACTIVE,slow_affine
from muff_complete_model import Q1_FORWARD,Q1_REVERSE,operating_point
from muff_exact_linear_model import exact_fast_affine
from muff_multirate_model import Q1_NONLINEAR,Q2_COLLECTOR,SLOW_NODES,slow_step
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run
from run_strict_rate_reduction_experiment import RATE,model

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"exact_linear_rate"
EXPERIMENT=ROOT/"simulation"/"experiments"/"exact_linear_rate"
DURATION=0.250;CASES=((0.5,0.5,100),(1.0,0.5,100),(1.0,1.0,100),(1.0,1.0,200))
VARIANTS=((1,1,"full_quad"),(2,2,"full_repeat"),(2,2,"damped_half"),(2,2,"full_quad"))

def exact_model(p,sustain,tone,ff,sf):
    nodes,dc_q=operating_point(sustain,tone,0.8,p)
    sr,slow=slow_affine(p,tone,0.8,RATE*sf)
    ss=sr.incidence.T@np.r_[nodes[Q2_COLLECTOR],nodes[SLOW_NODES]]
    _,_,_,_,pi,pg=slow_step(sr,ss,dc_q[Q1_NONLINEAR],float(nodes[Q2_COLLECTOR]),True)
    fast=exact_fast_affine(p,dc_q[:9],pg,RATE*ff,sustain)
    # Состояния являются теми же физическими напряжениями конденсаторов.
    from muff_frontend_model import prepare_frontend
    fr=prepare_frontend(sustain,None,p);fs=fr.capacitor_incidence[:len(fr.source),:9].T@nodes[:len(fr.source)]
    initial=(fs,dc_q[FAST_ACTIVE],ss,dc_q[[Q1_FORWARD,Q1_REVERSE]],pi-pg*nodes[Q2_COLLECTOR],pg)
    return fast,slow,initial

def main():
    p=Q3Parameters();time=np.arange(int(round(DURATION*RATE))+1)/RATE;rows=[]
    for sustain,tone,level in CASES:
        signal=np.asarray(source_input(level*1e-3)(time))
        reference=run(*model(p,sustain,tone,4,2),signal,p,"full_repeat",4,2)
        for ff,sf,mode in VARIANTS:
            tested=run(*exact_model(p,sustain,tone,ff,sf),signal,p,mode,ff,sf)
            error=tested[1:]-reference[1:];finite=bool(np.all(np.isfinite(tested)))
            rms=float(np.sqrt(np.mean(error*error))*1e3) if finite else np.inf
            peak=float(np.max(np.abs(error))*1e3) if finite else np.inf
            rows.append((sustain,tone,level,ff,sf,mode,finite and peak<1000,rms,peak))
            print(rows[-1],flush=True)
    RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as s:
        w=csv.writer(s);w.writerow(("sustain","tone","level_mv","fast","slow","mode","stable","rms_mv","peak_mv"));w.writerows(rows)
    lines=[]
    for ff,sf,mode in VARIANTS:
        group=[r for r in rows if r[3:6]==(ff,sf,mode)]
        lines.append(f"| {ff}×/{sf}× | {mode} | {max(r[7] for r in group):.3f} | {max(r[8] for r in group):.3f} | {'да' if all(r[6] for r in group) else 'нет'} |")
    report="""# Точный линейный переход быстрого ядра

Девять конденсаторных напряжений переведены в непрерывную систему с алгебраическими
ограничениями. Линейный переход вычислен матричной экспонентой; четыре нелинейных
тока считаются постоянными внутри шага и определяются неявно по конечному напряжению.
Проверен полный 250-мс файл, эталон — обратный Эйлер 4×/2×.

| Частоты | Решатель | Худшее СКО, мВ | Худший пик, мВ | Устойчиво |
|---|---|---:|---:|---|
"""+"\n".join(lines)+"\n"
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")

if __name__=="__main__":main()
