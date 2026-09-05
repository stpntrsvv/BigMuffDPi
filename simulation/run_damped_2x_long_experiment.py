"""Длинная проверка демпфированного ядра 2× на полном файле атаки."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run
from run_strict_rate_reduction_experiment import RATE,model

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"damped_2x_long"
EXPERIMENT=ROOT/"simulation"/"experiments"/"damped_2x_long"
DURATION=0.250; CASES=((1.0,0.5,100),(1.0,0.5,200),(1.0,1.0,100),(1.0,1.0,200))

def high_rms(error):
    spectrum=np.fft.rfft(error);freq=np.fft.rfftfreq(len(error),1/RATE)
    spectrum[freq<10_000]=0
    return float(np.sqrt(np.mean(np.fft.irfft(spectrum,n=len(error))**2)))

def main():
    p=Q3Parameters();time=np.arange(int(round(DURATION*RATE))+1)/RATE;rows=[]
    for sustain,tone,level in CASES:
        signal=np.asarray(source_input(level*1e-3)(time))
        reference=run(*model(p,sustain,tone,4,2),signal,p,"full_repeat",4,2)
        tested=run(*model(p,sustain,tone,2,2),signal,p,"damped_half",2,2)
        error=tested[1:]-reference[1:];finite=bool(np.all(np.isfinite(tested)))
        rows.append((sustain,tone,level,finite and np.max(np.abs(error))<1,
                     np.sqrt(np.mean(error*error))*1e3,np.max(np.abs(error))*1e3,
                     high_rms(error)*1e3,float(error[-1]*1e3)))
        print(rows[-1],flush=True)
    RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
    with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as s:
        w=csv.writer(s);w.writerow(("sustain","tone","level_mv","stable","rms_mv","peak_mv","high_rms_mv","final_mv"));w.writerows(rows)
    table="\n".join(f"| {s:.1f} | {t:.1f} | {l} | {r:.3f} | {pk:.3f} | {h:.3f} | {last:.3f} | {'да' if ok else 'нет'} |" for s,t,l,ok,r,pk,h,last in rows)
    report=f"""# Длинная проверка демпфированного ядра 2×

Обработан полный 250-мс файл атаки аккорда. Эталон — `full_repeat` 4×/2×.

| Sustain | Tone | Вход, мВ | СКО, мВ | Пик, мВ | Выше 10 кГц, мВ СКО | Конечная ошибка, мВ | Устойчиво |
|---:|---:|---:|---:|---:|---:|---:|---|
{table}
"""
    (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")

if __name__=="__main__":main()
