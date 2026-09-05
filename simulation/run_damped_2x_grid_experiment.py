"""Полная сетка уровней и ручек для демпфированного быстрого ядра 2×."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run
from run_strict_rate_reduction_experiment import RATE,DURATION,model

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"damped_2x_grid"
EXPERIMENT=ROOT/"simulation"/"experiments"/"damped_2x_grid"
VALUES=(0.0,0.5,1.0); LEVELS=(25,50,100,200)

def high_rms(error):
    spectrum=np.fft.rfft(error); frequency=np.fft.rfftfreq(len(error),1.0/RATE)
    spectrum[frequency<10_000.0]=0.0
    return float(np.sqrt(np.mean(np.fft.irfft(spectrum,n=len(error))**2)))

def main():
    p=Q3Parameters(); time=np.arange(int(round(DURATION*RATE))+1)/RATE
    start=int(round(0.003*RATE)); rows=[]
    for sustain in VALUES:
        for tone in VALUES:
            reference_model=model(p,sustain,tone,4,2)
            tested_model=model(p,sustain,tone,2,2)
            for level in LEVELS:
                signal=np.asarray(source_input(level*1e-3)(time))
                reference=run(*reference_model,signal,p,"full_repeat",4,2)
                tested=run(*tested_model,signal,p,"damped_half",2,2)
                error=tested[start:]-reference[start:]
                finite=bool(np.all(np.isfinite(tested)))
                rms=float(np.sqrt(np.mean(error*error))*1e3) if finite else np.inf
                peak=float(np.max(np.abs(error))*1e3) if finite else np.inf
                high=high_rms(error)*1e3 if finite else np.inf
                stable=finite and peak<1000.0
                rows.append((sustain,tone,level,stable,rms,peak,high))
                print(rows[-1],flush=True)
    RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.writer(stream);w.writerow(("sustain","tone","level_mv","stable","rms_mv","peak_mv","high_rms_mv"));w.writerows(rows)
    table=[]
    for sustain in VALUES:
        for tone in VALUES:
            group=[r for r in rows if r[:2]==(sustain,tone)]
            table.append(f"| {sustain:.1f} | {tone:.1f} | {max(r[4] for r in group):.3f} | {max(r[5] for r in group):.3f} | {max(r[6] for r in group):.3f} | {'да' if all(r[3] for r in group) else 'нет'} |")
    report="""# Демпфированное быстрое ядро 2×

Две поправки Ньютона с множителями 0,5 и 1,0 проверены на полной сетке
Sustain/Tone 0, 0,5 и 1, входах 25, 50, 100 и 200 мВ. Эталон — две полные
поправки 4×; медленная подсистема в обоих случаях работает 2×.

| Sustain | Tone | Худшее СКО, мВ | Худший пик, мВ | Выше 10 кГц, мВ СКО | Устойчиво |
|---:|---:|---:|---:|---:|---|
"""+"\n".join(table)+"\n"
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")

if __name__=="__main__":main()
