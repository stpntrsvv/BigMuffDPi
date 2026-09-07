"""Сохраняет внутренние величины медленного блока перед отказом офлайн-тракта."""
from __future__ import annotations
import argparse
import ctypes
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
import numpy as np
from run_song_offline import ROOT, RATE, read_pcm16

FP = ctypes.POINTER(ctypes.c_float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--input-peak-mv", type=float, default=100.0)
    parser.add_argument("--level-percentile", type=float, default=99.9)
    parser.add_argument("--variant", choices=("legacy", "matched"), default="legacy")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--slow-refresh", action="store_true")
    parser.add_argument("--q1-direct-exp", action="store_true")
    args = parser.parse_args()
    samples, rate = read_pcm16(args.input.resolve())
    if rate != RATE:
        old_t=np.arange(len(samples))/rate
        samples=np.interp(np.arange(round(len(samples)*RATE/rate))/RATE,old_t,samples)
    samples-=samples.mean()
    reference=np.percentile(np.abs(samples),args.level_percentile)
    x=np.ascontiguousarray(samples*(args.input_peak_mv/1000.0/reference),dtype=np.float32)
    outdir=ROOT/"simulation/raw/song_fault";outdir.mkdir(parents=True,exist_ok=True)
    fixture=outdir/"composed_frontend_fixture_bdf2.h"
    subprocess.run([sys.executable,str(ROOT/"simulation/generate_composed_frontend_fixture.py"),
        "--bdf2","--sustain","0.5","--tone","0.5","--volume","0.5","--output",str(fixture)],check=True)
    wrapper=outdir/"diagnostic.c"
    wrapper.write_text('#include "composed_frontend_benchmark.c"\n',encoding="utf-8")
    build_suffix="_full" if args.full_refresh else "_slow" if args.slow_refresh else ""
    dll=outdir/f"diagnostic{build_suffix}.dll"
    flags=["-DCOMPOSED_Q2_MATCHED"] if args.variant=="matched" else []
    if args.full_refresh:flags.append("-DCOMPOSED_PEDAL_HOST_FULL_REFRESH")
    if args.slow_refresh:flags.append("-DCOMPOSED_SLOW_FULL_REFRESH")
    if args.q1_direct_exp:flags.append("-DCOMPOSED_Q1_DIRECT_EXP")
    subprocess.run(["gcc","-shared","-O3","-std=c11","-DCOMPOSED_PEDAL_HOST",
        "-DCOMPOSED_PEDAL_HOST_DIAGNOSTIC","-DCOMPOSED_BDF2",*flags,"-I",str(outdir),
        "-I",str(ROOT/"include"),"-I",str(ROOT/"src"),str(wrapper),"-o",str(dll),"-lm"],check=True)
    lib=ctypes.CDLL(str(dll));lib.composed_pedal_init()
    lib.composed_pedal_process_sample.argtypes=[ctypes.c_float];lib.composed_pedal_process_sample.restype=ctypes.c_float
    lib.composed_pedal_fault_stage.restype=ctypes.c_uint32
    lib.composed_pedal_debug.argtypes=[FP]
    ring=deque(maxlen=16);debug=np.zeros(32,dtype=np.float32)
    bad=0
    for i,value in enumerate(x):
        y=lib.composed_pedal_process_sample(value)
        lib.composed_pedal_debug(debug.ctypes.data_as(FP))
        ring.append({"sample":i,"time_s":i/RATE,"input_v":float(value),"output_v":float(y),"debug":debug.astype(float).tolist()})
        if lib.composed_pedal_fault_stage() or not np.isfinite(y):bad=i+1;break
    labels=["port","sq0_before","sq1_before","hs0","hs1","hs2","hs3","lq0","lq1",
        "p0_r0","p0_r1","p0_norm","p0_det","p0_dx0","p0_dx1","p0_invdet",
        "p1_r0","p1_r1","p1_norm","p1_det","p1_dx0","p1_dx1","p1_invdet",
        "sq0_after","sq1_after","current0","current1","next0","next1","next2","next3","output"]
    result={"bad_sample":bad,"bad_time_s":(bad-1)/RATE if bad else None,"variant":args.variant,
        "input_peak_mv":args.input_peak_mv,"level_percentile":args.level_percentile,
        "full_refresh":args.full_refresh,"slow_refresh":args.slow_refresh,
        "labels":labels,"window":list(ring)}
    suffix="_full_refresh" if args.full_refresh else "_slow_refresh" if args.slow_refresh else ""
    path=outdir/f"diagnostic_{args.variant}_{args.input_peak_mv:g}mv{suffix}.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(path);print(f"bad_sample={bad} time_s={result['bad_time_s']}")
    if bad:
        print("last_samples: sample input port sq0_before sq1_before pass0_norm pass0_dx0 pass0_dx1 pass1_norm output")
        selected=(0,1,2,11,13,14,18,31)
        for row in ring:
            d=row["debug"]
            values=" ".join(f"{d[index]:.9g}" for index in selected)
            print(f"{row['sample']} {row['input_v']:.9g} {values}")
        for label,value in zip(labels,ring[-1]["debug"]):print(f"{label}={value:.9g}")
    return 1 if bad else 0

if __name__=="__main__":raise SystemExit(main())
