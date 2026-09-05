"""Проверяет 2× при двух, трёх и четырёх поправках быстрого ядра."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run
from run_strict_rate_reduction_experiment import RATE,DURATION,model

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"rate_solver_depth"
EXPERIMENT=ROOT/"simulation"/"experiments"/"rate_solver_depth"
SETTINGS=((0.5,0.5),(1.0,1.0)); LEVELS=(50,100,200)
MODES=("full_repeat","damped_half","full_triple","damped_quarter","full_quad")

def main():
    p=Q3Parameters();time=np.arange(int(round(DURATION*RATE))+1)/RATE
    start=int(round(0.003*RATE));rows=[]
    for sustain,tone in SETTINGS:
        ref_model=model(p,sustain,tone,4,2); test_model=model(p,sustain,tone,2,2)
        for level in LEVELS:
            signal=np.asarray(source_input(level*1e-3)(time))
            reference=run(*ref_model,signal,p,"full_repeat",4,2)
            for mode in MODES:
                tested=run(*test_model,signal,p,mode,2,2)
                finite=bool(np.all(np.isfinite(tested)))
                error=tested[start:]-reference[start:]
                rms=float(np.sqrt(np.mean(error*error))*1e3) if finite else np.inf
                peak=float(np.max(np.abs(error))*1e3) if finite else np.inf
                practical=finite and peak<1000.0
                rows.append((sustain,tone,level,mode,practical,rms,peak))
                print(rows[-1],flush=True)
    RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as s:
        w=csv.writer(s);w.writerow(("sustain","tone","level_mv","mode","stable","rms_mv","peak_mv"));w.writerows(rows)
    lines=[]
    for mode in MODES:
        group=[r for r in rows if r[3]==mode]
        lines.append(f"| {mode} | {max(r[5] for r in group):.3f} | {max(r[6] for r in group):.3f} | {'да' if all(r[4] for r in group) else 'нет'} |")
    report="""# Глубина решения при частоте 2×

Проверены две, три и четыре полные поправки быстрого блока на каждом шаге 2×.
Эталон — две поправки при 4×. Пик более 1 В считается практической потерей устойчивости.

| Режим | Худшее СКО, мВ | Худший пик, мВ | Устойчиво всюду |
|---|---:|---:|---|
"""+"\n".join(lines)+"\n"
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")

if __name__=="__main__":main()
