"""Проверяет TR-BDF2 1× на атаке гитарного аккорда."""
from __future__ import annotations
import time
import wave
from pathlib import Path
import numpy as np
from muff_complete_model import OUTPUT, simulate_complete
from run_bdf2_chord_analysis import magnitude_error

ROOT=Path(__file__).resolve().parents[1]
SAMPLE=ROOT/"simulation/samples/e_major_chord/e_major_attack.wav"
EXPERIMENT=ROOT/"simulation/experiments/tr_bdf2"
DURATION=.120

def wav(path:Path)->np.ndarray:
 with wave.open(str(path),"rb") as f:return np.frombuffer(f.readframes(f.getnframes()),dtype="<i2").astype(np.float64)

def source(level:float):
 raw=wav(SAMPLE);rate=48_000;raw-=np.mean(raw);raw*=level/np.max(np.abs(raw))
 return lambda t:np.interp(t*rate,np.arange(len(raw)),raw,left=0,right=0)

def reference(level:float)->np.ndarray:
 path=(ROOT/"simulation/experiments/bdf2_chord/euler_16x.wav" if level==.025
       else ROOT/"simulation/experiments/full_chord_pedal/sustain_100_tone_100.wav")
 return wav(path)[:int(DURATION*48_000)+1]

def main():
 EXPERIMENT.mkdir(parents=True,exist_ok=True);rows=[]
 for level in (.025,.100):
  for label,method in (("BDF2", "bdf2"),("TR-BDF2","tr_bdf2")):
   started=time.perf_counter();result=simulate_complete(1,1,.8,1,source(level),DURATION,"hybrid",True,integration_method=method);elapsed=time.perf_counter()-started
   candidate=result.node_v[:,OUTPUT];ref=reference(level);n=min(len(ref),len(candidate));ref=ref[:n];candidate=candidate[:n]
   ref-=np.mean(ref);candidate-=np.mean(candidate);gain=np.dot(ref,candidate)/np.dot(candidate,candidate);shape,spectral=magnitude_error(ref,candidate*gain)
   rows.append((level*1e3,label,shape,spectral,float(np.max(np.abs(result.node_v))),float(np.max(result.residual_v)),elapsed))
   if method=="tr_bdf2":np.savez_compressed(EXPERIMENT/f"chord_{int(level*1000)}mv.npz",output_v=result.node_v[:,OUTPUT],input_v=result.input_v)
   print(rows[-1],flush=True)
 table="\n".join(f"| {l:g} | {label} | {sh:.3f}% | {sp:.2f} | {pk:.3f} | {res:.2e} | {sec:.1f} |" for l,label,sh,sp,pk,res,sec in rows)
 (EXPERIMENT/"report.md").write_text("# TR-BDF2 1×\n\nКонтроль для 25 мВ — полная узловая модель с Эйлером 16×; для 100 мВ — сохранённый полный SPICE-прогон. Убраны постоянная составляющая, целая задержка и общий масштаб.\n\n| Вход | Метод | Ошибка формы | Ошибка спектра, дБ | Максимум узла, В | Невязка | Время, с |\n|---:|:---|---:|---:|---:|---:|---:|\n"+table+"\n",encoding="utf-8")


if __name__ == "__main__":
 main()
