"""Проверяет готовую физически правильную ячейку Q3–Q2 в новой композиции."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from muff_complete_model import Q2_COLLECTOR as REF_Q2,simulate_complete
from muff_frontend_model import SUSTAIN_WIPER
from two_clipping_stages_model import Q2_COLLECTOR,simulate_two_stages
from run_port_multirate_level_sweep import source_input

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"simulation"/"raw"/"reused_q3q2_block"
EXPERIMENT=ROOT/"simulation"/"experiments"/"reused_q3q2_block"
RATE=48_000;DURATION=.250;CASES=((.5,.5,100),(1,.5,100),(1,1,100),(1,1,200))

def main():
 rows=[]
 for sustain,tone,level in CASES:
  source=source_input(level*1e-3)
  reference=simulate_complete(sustain,tone,.8,4,source,DURATION,"hybrid",True)
  boundary_time=reference.time_s;boundary=reference.node_v[:,SUSTAIN_WIPER]
  def drive(time): return np.interp(time,boundary_time,boundary)
  for factor in (1,2,4):
   for architecture in ("reference_full","full_scalar"):
    tested=simulate_two_stages(architecture,factor,drive,DURATION,True)
    target=np.interp(tested.time_s,boundary_time,reference.node_v[:,REF_Q2])
    start=int(.003*RATE*factor);error=tested.node_v[start:,Q2_COLLECTOR]-target[start:]
    rms=float(np.sqrt(np.mean(error*error))*1e3);peak=float(np.max(np.abs(error))*1e3)
    rows.append((sustain,tone,level,factor,architecture,rms,peak,tested.maximum_reduced_residual_v))
    print(rows[-1],flush=True)
 RAW.mkdir(parents=True,exist_ok=True);EXPERIMENT.mkdir(parents=True,exist_ok=True)
 with (RAW/"summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(("sustain","tone","level_mv","factor","architecture","rms_mv","peak_mv","residual_v"));w.writerows(rows)
 lines=[]
 for factor in (1,2,4):
  for architecture in ("reference_full","full_scalar"):
   g=[r for r in rows if r[3:5]==(factor,architecture)]
   lines.append(f"| {factor}× | {architecture} | {max(r[5] for r in g):.3f} | {max(r[6] for r in g):.3f} | {max(r[7] for r in g):.2e} |")
 report="""# Повторное использование связанной ячейки Q3–Q2

Ячейка содержит единственные C5/C13, настоящий R12, оба BJT, обе диодные пары
и линейную нагрузку Tone. Входом служит реальное напряжение после Sustain из
полной модели. Проверены полный Q3+Q2 и ранее измеренный полный Q3+скалярный Q2.

| Частота | Архитектура | Худшее СКО Q2, мВ | Худший пик, мВ | Невязка, В |
|---:|---|---:|---:|---:|
"""+"\n".join(lines)+"\n"
 (EXPERIMENT/"report.md").write_text(report,encoding="utf-8")
if __name__=="__main__":main()
