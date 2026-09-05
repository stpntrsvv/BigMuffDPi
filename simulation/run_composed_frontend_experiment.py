"""Проверяет окончательный разрез Q4+Q3 → Q2 на полном сигнале."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from generate_port_multirate_fixture import slow_affine
from muff_complete_model import Q2_COLLECTOR,operating_point,simulate_complete
from muff_composed_frontend_model import simulate_composed
from muff_multirate_model import Q1_NONLINEAR,SLOW_NODES,slow_step
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/"simulation"/"raw"/"composed_frontend";EXPERIMENT=ROOT/"simulation"/"experiments"/"composed_frontend"
RATE=48_000;DURATION=.250;CASES=((.5,.5,100),(1,.5,100),(1,1,100),(1,1,200))
def main():
 p=Q3Parameters();rows=[]
 for sustain,tone,level in CASES:
  nodes,dc_q=operating_point(sustain,tone,.8,p);sr,_=slow_affine(p,tone,.8)
  ss=sr.incidence.T@np.r_[nodes[Q2_COLLECTOR],nodes[SLOW_NODES]]
  _,_,_,_,pi,pg=slow_step(sr,ss,dc_q[Q1_NONLINEAR],float(nodes[Q2_COLLECTOR]),True)
  po=pi-pg*nodes[Q2_COLLECTOR];source=source_input(level*1e-3)
  reference=simulate_complete(sustain,tone,.8,4,source,DURATION,"hybrid",True)
  tested,res=simulate_composed(np.asarray(source(reference.time_s)),p,sustain,tone,1/(RATE*4),pg,po,nodes,dc_q)
  start=int(.003*RATE*4);e=tested[start:]-reference.node_v[start:,Q2_COLLECTOR]
  rows.append((sustain,tone,level,float(np.sqrt(np.mean(e*e))*1e3),float(np.max(np.abs(e))*1e3),res));print(rows[-1],flush=True)
 RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
 with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(("sustain","tone","level_mv","rms_mv","peak_mv","residual_v"));w.writerows(rows)
 table="\n".join(f"| {s} | {t} | {l} | {r:.3f} | {pk:.3f} | {res:.2e} |" for s,t,l,r,pk,res in rows)
 (EXPERIMENT/"report.md").write_text("# Композиция Q4+Q3 → Q2\n\n| Sustain | Tone | Вход, мВ | СКО Q2, мВ | Пик, мВ | Невязка |\n|---:|---:|---:|---:|---:|---:|\n"+table+"\n",encoding="utf-8")
if __name__=="__main__":main()
