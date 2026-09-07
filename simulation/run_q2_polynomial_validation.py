"""Проверяет полином Q2 в настоящем C-ядре; результаты в физических вольтах.

Нужен GCC в PATH. Новый полином включается через COMPOSED_Q2_MATCHED.
Точный Q2 — отдельная контрольная сборка на компьютере, не прошивка.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import subprocess
import wave
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from generate_q2_polynomial import ROOT, DS, DIS2, influence, exact, main as generate
from run_oversampling_analysis import align
from run_generalized_alpha_analysis import chord_output, reference_tone

EXPERIMENT = ROOT / "simulation/experiments/q2_polynomial"
RAW = ROOT / "simulation/raw/q2_polynomial"
RATE = 48000
FP = ctypes.POINTER(ctypes.c_float)
UP = ctypes.POINTER(ctypes.c_uint32)


def build(method, variant):
    stem = f"{method}_{variant}"
    source = (ROOT / "src/composed_frontend_benchmark.c").read_text(encoding="utf-8")
    if variant == "exact":
        old = "if(__builtin_fabsf(linear_q)<=54.0F*DS){"
        assert source.count(old) == 1
        source = source.replace(old, "if(0){")
    (RAW / f"{stem}_core.c").write_text(source, encoding="utf-8")
    wrapper = f'#include "{stem}_core.c"\n' + r'''
__declspec(dllexport) void sweep(const float *x,float *y,uint32_t n){
 for(uint32_t k=0;k<n;k++)y[k]=q2_analytic_predictor(x[k]);
}
__declspec(dllexport) uint32_t run(const float *x,float *y,uint32_t n,uint32_t *stats){
 composed_pedal_init();uint32_t bad=0;
 for(uint32_t k=0;k<n;k++){
  y[k]=composed_pedal_process_sample(x[k]);
  if(runtime_state.fault || !isfinite(y[k]) || fabsf(y[k])>20){bad=k+1;break;}
 }
 stats[0]=runtime_state.fallbacks;stats[1]=runtime_state.adaptive;
 stats[2]=runtime_state.holds;stats[3]=runtime_state.fault;
 stats[4]=runtime_state.q2_bracketed;
 return bad;
}
'''
    path = RAW / f"{stem}.c"
    path.write_text(wrapper, encoding="utf-8")
    dll = RAW / f"{stem}.dll"
    flags = {"bdf2": ["-DCOMPOSED_BDF2"], "alpha02": ["-DCOMPOSED_GENERALIZED_ALPHA02"]}[method]
    if variant != "legacy":
        flags += ["-DCOMPOSED_Q2_MATCHED"]
    command = ["gcc", "-shared", "-O3", "-std=c11", "-DCOMPOSED_PEDAL_HOST", *flags,
               "-I", str(ROOT / "include"), str(path), "-o", str(dll), "-lm"]
    subprocess.run(command, check=True, capture_output=True)
    lib = ctypes.CDLL(str(dll))
    lib.run.argtypes = [FP, FP, ctypes.c_uint32, UP]
    lib.run.restype = ctypes.c_uint32
    lib.sweep.argtypes = [FP, FP, ctypes.c_uint32]
    lib.sweep.restype = None
    return lib


def process(lib, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.zeros_like(x)
    stats = np.zeros(5, dtype=np.uint32)
    bad = lib.run(x.ctypes.data_as(FP), y.ctypes.data_as(FP), len(x), stats.ctypes.data_as(UP))
    return y.astype(np.float64), int(bad), stats.tolist()


def read_input(name, level):
    with wave.open(str(ROOT / f"simulation/samples/e_major_chord/{name}.wav"), "rb") as f:
        assert f.getframerate() == RATE and f.getnchannels() == 1 and f.getsampwidth() == 2
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(float)
    x -= x.mean()
    return x * (level / np.max(np.abs(x)))


def metrics(ref, y):
    ref, y = align(ref, y)
    ref, y = ref - ref.mean(), y - y.mean()
    rms = np.linalg.norm(y-ref) / np.linalg.norm(ref) * 100
    gain = float(np.dot(ref, y) / np.dot(y, y))
    shape = np.linalg.norm(gain*y-ref) / np.linalg.norm(ref) * 100
    window = np.hanning(len(ref))
    a, b = np.abs(np.fft.rfft(window*ref)), np.abs(np.fft.rfft(window*y))
    spectral = 20*np.log10(max(np.linalg.norm(b-a)/np.linalg.norm(a), 1e-15))
    return [float(rms), float(shape), float(spectral), gain]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    generate()
    libs = {(m,v): build(m,v) for m in ("bdf2", "alpha02") for v in ("legacy", "fixed", "exact")}
    grid = np.linspace(-54*DS, 54*DS, 200001, dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    scalar = []
    for method in ("bdf2", "alpha02"):
        truth = np.sign(grid) * DS * exact(np.abs(grid.astype(float))/DS, influence(method)*DIS2/DS)
        for variant in ("legacy", "fixed"):
            y = np.zeros_like(grid)
            libs[method,variant].sweep(grid.ctypes.data_as(FP), y.ctypes.data_as(FP), len(grid))
            error = y-truth
            residual = y.astype(float) + influence(method)*DIS2*np.sinh(y.astype(float)/DS)-grid
            scalar.append([method,variant,float(np.max(np.abs(error))*1000),float(np.max(np.abs(residual))*1000)])
            axes[0].plot(grid,error*1000,label=f"{method} {variant}")
            if variant == "fixed": axes[1].plot(grid,error*1000,label=method)
    for ax in axes:
        ax.set(xlabel="Линейное напряжение Q2, В",ylabel="Ошибка корня, мВ")
        ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(EXPERIMENT/"scalar_error.png",dpi=150);plt.close(fig)
    print("scalar",scalar,flush=True)
    if any(error > .01 or residual > .2 for _,variant,error,residual in scalar if variant == "fixed"):
        raise RuntimeError("Полином не прошёл локальные пределы: 10 мкВ корень, 0,2 мВ невязка")

    cases=[]
    # Эти контрольные данные получены без WAV-нормировки и затухания хвоста.
    for level in (.025,.1):
        reference=chord_output(level,None)
        x=read_input("e_major_attack",level)[:len(reference)]
        cases.append((f"attack_{int(level*1000)}mv",x,reference,"полная модель Эйлера 16×"))
    table=np.loadtxt(ROOT/"simulation/raw/full_chord_pedal/sustain_100_tone_100.txt",skiprows=1)
    x=read_input("e_major_dry",.1)
    ref=np.interp(np.arange(len(x))/RATE,table[:,0],table[:,1])
    cases.append(("full_100mv_spice",x,ref,"ngspice, исходная таблица в вольтах"))
    for frequency in (440,1000,3000,6000):
        for level in (.005,.025,.1):
            reference=reference_tone(frequency,level)
            x=level*np.sin(2*np.pi*frequency*np.arange(len(reference))/RATE)
            cases.append((f"sine_{frequency}_{int(level*1000)}mv",x,reference,"полная модель Эйлера 16×"))
    rows=[];outputs={};failures=[]
    for name,x,reference,reference_name in cases:
        for method in ("bdf2","alpha02"):
            for variant in ("legacy","fixed","exact"):
                y,bad,stats=process(libs[method,variant],x)
                values=None if bad else metrics(reference,y)
                row=dict(case=name,method=method,variant=variant,bad_sample=bad,stats=stats,
                         metrics=values,reference=reference_name,peak=float(np.max(np.abs(y))))
                rows.append(row);outputs[name,method,variant]=y
                if bad: failures.append([name,method,variant,bad])
                print(name,method,variant,"bad",bad,"metrics",values,flush=True)
    stability=[]
    for level in (.025,.05,.1,.2):
        x=read_input("e_major_dry",level)
        for method in ("bdf2","alpha02"):
            for variant in ("legacy","fixed","exact"):
                y,bad,stats=process(libs[method,variant],x)
                stability.append([level,method,variant,bad,stats,float(np.max(np.abs(y)))])
                print("full stability",stability[-1],flush=True)
    comparisons=[]
    for name,_,_,_ in cases:
        for method in ("bdf2","alpha02"):
            exact_y=outputs[name,method,"exact"]
            for variant in ("legacy","fixed"):
                y=outputs[name,method,variant]
                subset=[r for r in rows if r['case']==name and r['method']==method and r['variant'] in (variant,'exact')]
                if any(r['bad_sample'] for r in subset):continue
                comparisons.append([name,method,variant,float(np.sqrt(np.mean((y-exact_y)**2))*1000)])
    # Полный аккорд и прослушивание с общим масштабом, без независимой нормировки.
    name="full_100mv_spice"
    ys=[outputs[name,"bdf2",v] for v in ("legacy","fixed","exact")]
    scale=max(np.max(np.abs(y)) for y in [ref,*ys])*1.05
    for variant,y in zip(("legacy","fixed","exact"),ys):
        with wave.open(str(EXPERIMENT/f"bdf2_{variant}_100mv.wav"),"wb") as f:
            f.setparams((1,2,RATE,0,"NONE","not compressed"))
            f.writeframes(np.int16(np.clip(y/scale,-1,1)*32767).tobytes())
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    t=np.arange(2400)/RATE*1000
    axes[0].plot(t,ref[:2400],label="ngspice",alpha=.7)
    for variant,y in zip(("legacy","fixed","exact"),ys):
        axes[0].plot(t,y[:2400],label=variant,alpha=.7)
        window=np.hanning(len(y));freq=np.fft.rfftfreq(len(y),1/RATE)
        axes[1].plot(freq,20*np.log10(np.maximum(np.abs(np.fft.rfft(y*window))/window.sum(),1e-12)),label=variant)
    axes[0].set(xlabel="Время, мс",ylabel="Выход, В")
    axes[1].set(xlabel="Частота, Гц",ylabel="Спектр, дБВ",xlim=(0,24000),ylim=(-120,0))
    for ax in axes:ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(EXPERIMENT/"full_chord.png",dpi=150);plt.close(fig)
    result=dict(scalar=scalar,rows=rows,stability=stability,exact_comparison=comparisons,
                source_sha256=hashlib.sha256((ROOT/"src/composed_frontend_benchmark.c").read_bytes()).hexdigest())
    (RAW/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    report=["# Полином Q2 для выбранного интегратора", "",
            "48 кГц; Sustain=1, Tone=1, Volume=0,8. C-ядро собрано GCC -O3 на компьютере.",
            "legacy — прежний полином Эйлера; fixed — пять новых полиномов той же степени;",
            "exact — прежний защищённый Ньютон для каждого Q2, контроль на компьютере.", "",
            "## Локальная точность Q2", "",
            "200001 точка в диапазоне −54…54 Vd; вычисление полинома выполняет C float.", "",
            "| Метод | Вариант | Максимальная ошибка корня, мВ | Максимальная невязка, мВ |",
            "|:---|:---|---:|---:|"]
    report += [f"| {m} | {v} | {e:.6f} | {r:.6f} |" for m,v,e,r in scalar]
    report += ["", "![Ошибка Q2](scalar_error.png)", "", "## Ошибка выхода", "",
               "Сигналы взяты до нормировки WAV. Удалены целая задержка (±24 отсчёта) и DC.",
               "СКО и спектр сохраняют физический масштаб. Столбец формы дополнительно",
               "исключает общий коэффициент усиления, найденный ПОСЛЕ выравнивания.",
               "Атака: первые 120 мс отдельного e_major_attack; полный аккорд: e_major_dry, 4 с.",
               "Эталон атак и синусов — полная модель Эйлера 16× с общим FIR;",
               "полного аккорда 100 мВ — исходная таблица ngspice, без затухания и WAV-квантизации.","",
               "| Сигнал | Метод | Вариант | СКО, % | Форма, % | Спектр, дБ |",
               "|:---|:---|:---|---:|---:|---:|"]
    for r in rows:
        values="СРЫВ | — | —" if r['bad_sample'] else " | ".join(f"{v:.4f}" for v in r['metrics'][:3])
        report.append(f"| {r['case']} | {r['method']} | {r['variant']} | {values} |")
    report += ["", "## Отличие от точного Q2 того же C-ядра", "",
               "Без выравнивания, удаления DC и подгонки усиления; СКО в мВ.","",
               "| Сигнал | Метод | Вариант | СКО, мВ |", "|:---|:---|:---|---:|"]
    report += [f"| {n} | {m} | {v} | {r:.6f} |" for n,m,v,r in comparisons]
    report += ["", "## Устойчивость полного аккорда", "",
               "| Вход, мВ | Метод | Вариант | Первый плохой отсчёт (0 = нет) | Удержания | Пик, В |",
               "|---:|:---|:---|---:|---:|---:|"]
    report += [f"| {l*1000:g} | {m} | {v} | {b} | {s[2]} | {pk:.6f} |" for l,m,v,b,s,pk in stability]
    report += ["", "![Полный аккорд](full_chord.png)", "",
               "Прослушивание, единый масштаб:","",
               "- [Прежний BDF2](bdf2_legacy_100mv.wav)",
               "- [Исправленный полином](bdf2_fixed_100mv.wav)",
               "- [Точный Q2](bdf2_exact_100mv.wav)","",
               "Воспроизведение: `python simulation/run_q2_polynomial_validation.py`.",
               "Сырые результаты и временные сборки: `simulation/raw/q2_polynomial/`.",
               "Новые аппаратные такты этим опытом не измеряются.", "",
               "## Вывод и статус", "",
               "Полином исправляет локальную несогласованность интегратора. На коротких атаках",
               "относительно полной модели Эйлера 16× ошибка уменьшается. На полном аккорде",
               "100 мВ относительно ngspice ошибка увеличивается; точный Q2 даёт почти тот же",
               "результат, что новый полином. Значит, дальнейшее уточнение полинома этот",
               "разрыв не устранит. Причина остаточного расхождения всего тракта этим опытом",
               "не установлена; возможная компенсация ошибок требует отдельной проверки.", "",
               "Рабочая сборка сохраняет прежний полином. Новый включается только макросом",
               "`COMPOSED_Q2_MATCHED`; отдельный стенд — `composed_bdf2_q2_block_benchmark`.",
               "Сетка полного аккорда обнаружила срывы BDF2 на 50 и 200 мВ и до исправления.",
               "При наличии срыва исправленного BDF2 программа завершится ненулевым кодом,",
               "предварительно сохранив все результаты. Это отрицательный итог общей проверки,",
               "а не потеря отчёта.", "",
               "Пять интервалов, степень 5 и структура Горнера сохранены. Локальная цена",
               "полинома должна остаться близкой, но цена всего блока может измениться из-за",
               "другой траектории и числа уточнений Tone–Q1. Нулевую доплату по времени всего",
               "тракта без нового аппаратного измерения заявлять нельзя.", "",
               "Этот опыт использует отдельную 120-мс атаку и полный четырёхсекундный файл",
               "явно раздельно. Проценты формы рассчитаны после выравнивания, в отличие от",
               "некоторых прежних программ, подбиравших масштаб до него. Для будущего",
               "сравнения использовать одинаковый вход и исходный эталон в вольтах."]
    (EXPERIMENT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    # Отказ исправленного рабочего BDF2 считается ошибкой проверки.
    if any(b or s[2] for _,m,v,b,s,_ in stability if m=="bdf2" and v=="fixed"):
        raise RuntimeError("Исправленный BDF2 не прошёл полный сигнал")


if __name__ == "__main__":
    main()
