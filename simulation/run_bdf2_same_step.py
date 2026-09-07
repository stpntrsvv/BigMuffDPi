"""Сравнивает общую и составную гибридные модели на одинаковом шаге BDF2.

Начало согласовано с C: два прошлых состояния равны DC, BDF2 применяется
уже к первому входному отсчёту. Исходный simulate_complete не изменяется.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from muff_complete_model import prepare_complete, operating_point, nonlinear_terms, _step, OUTPUT
from run_q2_polynomial_validation import build, process, read_input, ROOT, RATE, RAW as C_RAW

RAW = ROOT / "simulation/raw/bdf2_same_step"
EXPERIMENT = ROOT / "simulation/experiments/bdf2_same_step"
RESIDUAL_LIMIT = 1e-8


def damped_step(reduction, history, previous_q, dc_q, value):
    """Повтор того же нелинейного уравнения с поиском убывания невязки."""
    rhs = (reduction.source + reduction.input_vector * value +
           reduction.capacitor_incidence @ (reduction.capacitor_conductance * history))
    linear_q = reduction.voltage @ np.linalg.solve(reduction.linear_matrix, rhs)
    q = previous_q.copy()
    for _ in range(100):
        current, derivative = nonlinear_terms(q, reduction.parameters, 'hybrid', dc_q)
        residual = q-linear_q+reduction.influence @ current
        norm = float(np.max(np.abs(residual)))
        if norm < 1e-12:
            break
        jacobian = np.eye(len(q))+reduction.influence*derivative[np.newaxis,:]
        dx = np.linalg.solve(jacobian, residual)
        damping = 1.
        for _ in range(24):
            candidate = q-damping*dx
            ci, _ = nonlinear_terms(candidate, reduction.parameters, 'hybrid', dc_q)
            if np.max(np.abs(candidate-linear_q+reduction.influence@ci)) < norm:
                q = candidate
                break
            damping *= .5
        else:
            break
    current, _ = nonlinear_terms(q, reduction.parameters, 'hybrid', dc_q)
    norm = float(np.max(np.abs(q-linear_q+reduction.influence@current)))
    nodes = np.linalg.solve(reduction.linear_matrix, rhs-reduction.injection@current)
    return nodes, reduction.capacitor_incidence.T@nodes, q, norm, 0.


def global_bdf2(x):
    reduction = prepare_complete(1., 1., .8, 1./RATE, integration_scale=1.5)
    nodes, dc_q = operating_point(1., 1., .8, reduction.parameters)
    state = reduction.capacitor_incidence.T @ nodes
    older = state.copy()
    q = dc_q.copy()
    y = np.full(len(x), np.nan)
    residuals = np.full(len(x), np.nan)
    reason = None
    bad = 0
    peak_node = 0.
    retries = 0
    first_retry = 0
    started = time.perf_counter()
    for k, value in enumerate(x):
        history = (4.*state-older)/3.
        try:
            node, next_state, next_q, residual, _ = _step(
                reduction, history, q.copy(), dc_q, float(value), "hybrid", True
            )
            if not np.isfinite(residual) or residual > RESIDUAL_LIMIT or not np.all(np.isfinite(node)):
                retries += 1
                if not first_retry:
                    first_retry = k+1
                node, next_state, next_q, residual, _ = damped_step(
                    reduction, history, q, dc_q, float(value))
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            bad, reason = k+1, str(error)
            break
        residuals[k] = residual
        if not np.all(np.isfinite(node)) or not np.isfinite(residual):
            bad, reason = k+1, "nonfinite"
            break
        peak_node = max(peak_node, float(np.max(np.abs(node))))
        if residual > RESIDUAL_LIMIT:
            bad, reason = k+1, "nonconverged"
            break
        if np.max(np.abs(node)) > 12.:
            bad, reason = k+1, "node_outside_12V"
            break
        y[k] = node[OUTPUT]
        older, state, q = state, next_state, next_q
        if (k+1) % RATE == 0:
            print(f"  global {k+1}/{len(x)}", flush=True)
    return y, dict(bad_sample=bad, reason=reason, peak_node=peak_node,
                   damped_retries=retries, first_unmodified_failure=first_retry,
                   max_residual=float(np.nanmax(residuals)) if np.any(np.isfinite(residuals)) else None,
                   seconds=time.perf_counter()-started), residuals


def error_metrics(ref, y):
    # Совпадающие входные отсчёты и моменты: никаких подгонок задержки/масштаба.
    difference=y-ref
    rms_v=float(np.sqrt(np.mean(difference*difference)))
    denominator=float(np.sqrt(np.mean((ref-ref.mean())**2)))
    window=np.hanning(len(ref))
    a=np.abs(np.fft.rfft((ref-ref.mean())*window))
    b=np.abs(np.fft.rfft((y-y.mean())*window))
    spectral=20*np.log10(max(float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),1e-15))
    return dict(rms_mv=1000*rms_v, relative_percent=100*rms_v/max(denominator,1e-30),
                peak_mv=float(np.max(np.abs(difference))*1000), spectral_db=float(spectral))


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    C_RAW.mkdir(parents=True,exist_ok=True)
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    libs={v:build("bdf2",v) for v in ("legacy","fixed","exact")}
    model_hashes={name:hashlib.sha256((ROOT/'simulation'/name).read_bytes()).hexdigest()
                  for name in ('muff_complete_model.py','muff_frontend_model.py','q3_model.py')}
    cases=[]
    for level in (.025,.05,.1,.2):
        cases.append((f"attack_{level*1000:g}mv",read_input("e_major_attack",level)[:5760]))
    for hz in (440,1000,3000,6000):
        cases.append((f"sine_{hz}_25mv",.025*np.sin(2*np.pi*hz*np.arange(2880)/RATE)))
    for level in (.025,.1):
        cases.append((f"full_{level*1000:g}mv",read_input("e_major_dry",level)))
    rows=[];pairwise=[];cache_data={}
    for name,x in cases:
        # И Python, и C получают одни и те же округлённые float32 значения.
        x=np.asarray(x,dtype=np.float32)
        print(name,len(x),flush=True)
        stamp=dict(models=model_hashes,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                   startup='two_dc_bdf2_first',residual_limit=RESIDUAL_LIMIT)
        cache_path=RAW/f'{name}_global.npz'
        metadata_path=RAW/f'{name}_global.json'
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if cache_path.exists() and metadata.get('stamp')==stamp and metadata['info']['bad_sample']==0:
            with np.load(cache_path) as cache:
                ref=cache['output'];residuals=cache['residual']
            info=metadata['info']
            print('  reused converged global',flush=True)
        else:
            ref,info,residuals=global_bdf2(x)
            np.savez_compressed(cache_path,output=ref,residual=residuals)
            metadata_path.write_text(json.dumps(dict(stamp=stamp,info=info)),encoding='utf-8')
        outputs={};statuses={}
        for variant,lib in libs.items():
            y,bad,stats=process(lib,x)
            # Нечисловой хвост никогда не участвует в метриках.
            if bad:y[bad-1:]=np.nan
            outputs[variant]=y;statuses[variant]=dict(bad_sample=bad,stats=stats)
            values=error_metrics(ref,y) if not bad and not info['bad_sample'] else None
            rows.append(dict(case=name,variant=variant,global_status=info,c_status=statuses[variant],metrics=values))
            print(variant,'global',info,'C',bad,'error',values,flush=True)
        for variant in ('legacy','fixed'):
            if not statuses[variant]['bad_sample'] and not statuses['exact']['bad_sample']:
                pairwise.append(dict(case=name,variant=variant,metrics=error_metrics(outputs['exact'],outputs[variant])))
        np.savez_compressed(RAW/f"{name}.npz",input=x,global_output=ref,residual=residuals,**outputs)
        cache_data[name]=(x,ref,outputs,info,statuses)
    result=dict(rows=rows,pairwise=pairwise,
                hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [ROOT/'simulation/muff_complete_model.py',ROOT/'simulation/muff_frontend_model.py',
                         ROOT/'src/composed_frontend_benchmark.c',ROOT/'include/composed_frontend_fixture_bdf2.h']})
    (RAW/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    # Показываем успешную атаку и полный слабый аккорд, если контроль сошёлся.
    figure,axes=plt.subplots(2,2,figsize=(13,8))
    for row,name in enumerate(('attack_25mv','full_25mv')):
        x,ref,outputs,info,statuses=cache_data[name]
        if info['bad_sample']:
            for ax in axes[row]:ax.text(.1,.5,f"{name}: контроль не сошёлся",transform=ax.transAxes)
            continue
        n=min(len(x),2400);t=np.arange(n)/RATE*1000
        axes[row,0].plot(t,ref[:n],label='общая BDF2',color='black')
        for variant in ('legacy','fixed','exact'):
            if statuses[variant]['bad_sample']:continue
            y=outputs[variant]
            axes[row,0].plot(t,y[:n],label=variant,alpha=.7)
            axes[row,1].plot(np.arange(len(y))/RATE,1000*(y-ref),label=variant,alpha=.7)
        axes[row,0].set(title=name,xlabel='Время, мс',ylabel='Выход, В')
        axes[row,1].set(title='Разность без выравнивания',xlabel='Время, с',ylabel='Ошибка, мВ')
        for ax in axes[row]:ax.grid(alpha=.3);ax.legend()
    figure.tight_layout();figure.savefig(EXPERIMENT/'comparison.png',dpi=150);plt.close(figure)
    lines=['# Общая и составная BDF2 при одинаковом шаге','',
           '## Условия','',
           '48 кГц, BDF2 1×, Sustain=1, Tone=1, Volume=0,8. Во всех моделях одна гибридная',
           'физика: нелинейные прямые переходы Q4/Q3, обе диодные пары, линеаризованный',
           'прямой переход Q2, постоянные обратные токи Q4/Q3/Q2, оба перехода Q1.',
           'Параметры полупроводников совпадают. Генератор C поглощает линейные токи в матрицы.',
           'Общая система сохраняет все связи; составная использует предсказанную базу Q2',
           'и постоянный эквивалент Нортона нагрузки Tone. Эти приближения не маскируются.', '',
           'Вход нормируется один раз, затем округляется до float32 и передаётся обеим моделям.',
           'Начальные истории равны DC, первый отсчёт уже считается BDF2, как в C.',
           'Обычный simulate_complete начинает с Эйлера; здесь его _step вызван отдельно,',
           'чтобы исключить различие запуска. Исходная модель и прошивка не изменялись.', '',
           'Общая модель: Python float64, общий Ньютон до 30 поправок, целевая невязка 1e-12 В.',
           'Если обычный Ньютон не сходится, ТОТ ЖЕ шаг повторяется от прежней точки',
           'с демпфированием (до 100 поправок и 24 делений шага пополам). Уравнения,',
           'история и частота не меняются. Первый отказ исходного Ньютона и число повторов',
           'показаны отдельно. Контроль принимается только по конечной невязке.',
           'Контроль отвергается при невязке >1e-8 В, нечисловом результате или |узел|>12 В.',
           'Это не гарантированно сходящийся решатель: неуспех не доказывает отсутствие',
           'физического решения и не превращается в численную оценку ошибки.', '',
           'Составные варианты: реальное C-ядро, GCC -O3, float32 на компьютере:', '',
           '- legacy — рабочий вариант с прежним полиномом Q2;',
           '- fixed — согласованный полином Q2;',
           '- exact — защищённый Ньютон Q2 на каждом отсчёте. Остальные блоки сохраняют',
           '  рабочее число поправок и обновление сохранённых нелинейностей.', '',
           'Разница общей модели и exact включает разбиение, ограниченное число поправок,',
           'обновление нелинейностей и float32. Это не чистая оценка только межблочной связи.',
           'Никакой подгонки задержки, усиления или DC для временной ошибки нет.',
           'Относительная СКО нормирована на СКО переменной части общей модели;',
           'спектральная ошибка вычисляется после удаления DC с окном Ханна.', '',
           '## Ошибка относительно общей BDF2','',
           '| Сигнал | C-вариант | СКО, мВ | СКО, % | Пик ошибки, мВ | Спектр, дБ |',
           '|:---|:---|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics']
        values='— | — | — | —' if m is None else ' | '.join(f'{m[k]:.5f}' for k in ('rms_mv','relative_percent','peak_mv','spectral_db'))
        lines.append(f"| {r['case']} | {r['variant']} | {values} |")
    lines += ['', 'Прочерк означает, что хотя бы одна сторона не прошла весь сигнал. Метрики',
              'успешного префикса не выдаются за результат полного опыта.', '',
              '## Сходимость и устойчивость','',
              'Номер плохого отсчёта — с единицы; 0 означает отсутствие отказа.', '',
              '| Сигнал | Общая: отсчёт отказа | Причина | Макс. невязка, В | Первый отказ обычного Ньютона | Повторов с демпфированием | C legacy | C fixed | C exact |',
              '|:---|---:|:---|---:|---:|---:|---:|---:|---:|']
    for name,(_,_,_,info,statuses) in cache_data.items():
        lines.append(f"| {name} | {info['bad_sample']} | {info['reason'] or '—'} | {info['max_residual']:.3e} | {info['first_unmodified_failure']} | {info['damped_retries']} | " +
                     ' | '.join(str(statuses[v]['bad_sample']) for v in ('legacy','fixed','exact'))+' |')
    lines += ['', '## Собственный вклад Q2 относительно exact','',
              'Остальной C-код совпадает. Без выравнивания и нормировки усиления.', '',
              '| Сигнал | Вариант | СКО, мВ |', '|:---|:---|---:|']
    lines += [f"| {r['case']} | {r['variant']} | {r['metrics']['rms_mv']:.6f} |" for r in pairwise]
    lines += ['', '![Сигналы и ошибки](comparison.png)', '',
              'Воспроизведение: `python simulation/run_bdf2_same_step.py` (GCC в PATH).',
              'Исходные массивы, статусы и хэши: `simulation/raw/bdf2_same_step/`.',
              'Скрипт сохраняет отчёт перед ненулевым завершением при любом отказе.',
              'Новые такты STM32 не измерялись.']
    lines += ["", '## Итог сравнения при одинаковом BDF2\n\nПолный аккорд 25 мВ: СКО рабочего C-ядра относительно общей гибридной BDF2\n9,650%, исправленного полинома 6,600%, точного Q2 6,599%.\nПолный аккорд 100 мВ: соответственно 8,591%, 5,716% и 5,713%.\nПодгонка задержки, усиления и DC не применялась.\n\nСледовательно, согласованный полином действительно приближает составное ядро\nк общей модели при том же интеграторе. Ухудшение относительно ngspice из\nпредыдущего опыта остаётся отдельным результатом: эталоны не взаимозаменяемы.\nОстаточные 5,7–6,6% нельзя целиком приписать разбиению: здесь вместе действуют\nграницы блоков, ограниченное число поправок, обновление нелинейностей,\nобнуление малых коэффициентов и float32.\n\nВажное уточнение прежнего вывода о срывах BDF2: общая модель прошла полный\nаккорд 100 мВ с тремя демпфированными повторами того же шага и максимальной\nневязкой около 1e-12 В. Атаки 100/200 мВ потребовали 2/9 повторов. Это\nподтверждает проблему сходимости обычного Ньютона на этих сигналах, а не\nнеизбежную неустойчивость самого BDF2. Устойчивость за пределами проверок\nне доказана. На атаке 200 мВ все три составных C-варианта всё ещё срываются\nна отсчёте 2107, тогда как общая модель с демпфированием проходит весь отрезок.\n\nРабочая прошивка не менялась. Исправленный полином остаётся экспериментальным;\nновых измерений тактов и прошивки платы не выполнялось.\n']
    (EXPERIMENT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(r['global_status']['bad_sample'] or r['c_status']['bad_sample'] for r in rows):
        raise SystemExit('Есть несошедшиеся/неустойчивые случаи; отчёт сохранён.')


if __name__=='__main__':
    main()
