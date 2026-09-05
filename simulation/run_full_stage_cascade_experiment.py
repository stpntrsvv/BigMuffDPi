"""Два независимых полных каскада Q3/Q2 без глобальной узловой связи."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from muff_complete_model import Q2_COLLECTOR,simulate_complete
from muff_frontend_model import SUSTAIN_WIPER
from q3_model import COLLECTOR,OUTPUT,Q3Parameters,simulate_input
from run_port_multirate_level_sweep import source_input

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"full_stage_cascade"
EXPERIMENT=ROOT/"simulation"/"experiments"/"full_stage_cascade"
RATE=48_000;FACTOR=4;DURATION=.250;CASES=((.5,.5,100),(1,.5,100),(1,1,100),(1,1,200))

def main():
 rows=[];p=Q3Parameters();step=1/(RATE*FACTOR)
 for sustain,tone,level in CASES:
  source=source_input(level*1e-3)
  reference=simulate_complete(sustain,tone,.8,FACTOR,source,DURATION,"hybrid",True)
  boundary=reference.node_v[:,SUSTAIN_WIPER]
  q3=simulate_input(boundary,step,p)
  q2=simulate_input(q3.node_v[:,OUTPUT],step,p)
  error=q2.node_v[:,COLLECTOR]-reference.node_v[:,Q2_COLLECTOR]
  start=int(.003/step);error=error[start:]
  rows.append((sustain,tone,level,True,float(np.sqrt(np.mean(error*error))*1e3),
               float(np.max(np.abs(error))*1e3),int(np.max(q3.iterations)),int(np.max(q2.iterations))))
  print(rows[-1],flush=True)
 RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
 with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(("sustain","tone","level_mv","stable","collector_rms_mv","collector_peak_mv","q3_iterations","q2_iterations"));w.writerows(rows)
 table="\n".join(f"| {s} | {t} | {l} | {r:.3f} | {pk:.3f} | {i3} | {i2} |" for s,t,l,_,r,pk,i3,i2 in rows)
 report=f"""# Каскад двух полных физических ступеней

Из полной модели взято только напряжение после Sustain. Далее два независимых
полных каскада Q3-типа соединены последовательно. Каждый сохраняет Ebers–Moll,
диодную пару Шокли, C5/C12/C6/C13 и полное схождение Ньютона. Межкаскадная
нагрузка заменена штатным `load_ohm = 110 кОм`; обратной связи назад нет.

| Sustain | Tone | Вход, мВ | Q2 коллектор, СКО мВ | Пик, мВ | Ньютон Q3 | Ньютон Q2 |
|---:|---:|---:|---:|---:|---:|---:|
{table}
"""
 (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")
if __name__=="__main__":main()
