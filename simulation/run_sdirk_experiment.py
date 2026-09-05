"""Последняя проверка нынешней узловой архитектуры: SDIRK2 при 2×."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from muff_sdirk_model import simulate_sdirk
from q3_model import Q3Parameters
from run_port_multirate_level_sweep import source_input
from run_port_sparse_linear_experiment import run
from run_strict_rate_reduction_experiment import RATE,model
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/"simulation"/"raw"/"sdirk_2x";EXPERIMENT=ROOT/"simulation"/"experiments"/"sdirk_2x"
DURATION=.250;CASES=((.5,.5,100),(1,.5,100),(1,1,100),(1,1,200))
def main():
 p=Q3Parameters();time=np.arange(int(DURATION*RATE)+1)/RATE;rows=[]
 for s,t,l in CASES:
  signal=np.asarray(source_input(l*1e-3)(time));ref=run(*model(p,s,t,4,2),signal,p,"full_repeat",4,2)
  for corrections in (2,3,4):
   tested=simulate_sdirk(signal,s,t,fast_factor=2,corrections=corrections);e=tested[1:]-ref[1:]
   finite=bool(np.all(np.isfinite(tested)));r=float(np.sqrt(np.mean(e*e))*1e3) if finite else np.inf;pk=float(np.max(np.abs(e))*1e3) if finite else np.inf
   rows.append((s,t,l,corrections,finite and pk<1000,r,pk));print(rows[-1],flush=True)
 RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
 with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(("sustain","tone","level_mv","corrections","stable","rms_mv","peak_mv"));w.writerows(rows)
 lines=[]
 for c in (2,3,4):
  g=[r for r in rows if r[3]==c];lines.append(f"| {c} | {max(r[5] for r in g):.3f} | {max(r[6] for r in g):.3f} | {'да' if all(r[4] for r in g) else 'нет'} |")
 (EXPERIMENT/"report.md").write_text("# SDIRK2 при 2×\n\n| Поправок на этап | Худшее СКО, мВ | Худший пик, мВ | Устойчиво |\n|---:|---:|---:|---|\n"+"\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":main()
