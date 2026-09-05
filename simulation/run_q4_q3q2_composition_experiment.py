"""Композиция полного Q4+Sustain и проверенной связанной ячейки Q3–Q2."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from muff_complete_model import Q2_COLLECTOR as REF_Q2,simulate_complete
from muff_q4_stage_model import simulate_q4
from q3_model import Q3Parameters
from two_clipping_stages_model import Q2_COLLECTOR,simulate_two_stages
from run_port_multirate_level_sweep import source_input
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/"simulation"/"raw"/"q4_q3q2_composition";EXPERIMENT=ROOT/"simulation"/"experiments"/"q4_q3q2_composition"
RATE=48_000;FACTOR=4;DURATION=.250;CASES=((.5,.5,100),(1,.5,100),(1,1,200))
def main():
 p=Q3Parameters();rows=[];time=np.arange(int(DURATION*RATE*FACTOR)+1)/(RATE*FACTOR)
 for sustain,tone,level in CASES:
  source=source_input(level*1e-3);signal=np.asarray(source(time))
  q4,q4res=simulate_q4(signal,1/(RATE*FACTOR),sustain,p)
  def drive(t):return np.interp(t,time,q4)
  block=simulate_two_stages("full_scalar",FACTOR,drive,DURATION,True,p)
  reference=simulate_complete(sustain,tone,.8,FACTOR,source,DURATION,"hybrid",True)
  error=block.node_v[:,Q2_COLLECTOR]-reference.node_v[:,REF_Q2];start=int(.003*RATE*FACTOR);e=error[start:]
  rows.append((sustain,tone,level,float(np.sqrt(np.mean(e*e))*1e3),float(np.max(np.abs(e))*1e3),q4res,block.maximum_reduced_residual_v));print(rows[-1],flush=True)
 RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
 with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(("sustain","tone","level_mv","rms_mv","peak_mv","q4_residual","q3q2_residual"));w.writerows(rows)
 table="\n".join(f"| {s} | {t} | {l} | {r:.3f} | {pk:.3f} | {a:.2e} | {b:.2e} |" for s,t,l,r,pk,a,b in rows)
 (EXPERIMENT/"report.md").write_text("# Композиция полного Q4 и Q3–Q2\n\n| Sustain | Tone | Вход, мВ | СКО Q2, мВ | Пик, мВ | Невязка Q4 | Невязка Q3–Q2 |\n|---:|---:|---:|---:|---:|---:|---:|\n"+table+"\n",encoding="utf-8")
if __name__=="__main__":main()
