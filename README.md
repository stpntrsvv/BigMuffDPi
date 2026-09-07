# Big Muff STM32

Физическая цифровая реализация транзисторного Big Muff Pi V3 на плате
NUCLEO-G474RE.

В репозитории находятся эталонная схема ngspice и приведённая физическая модель
всей педали: входной Q4, Sustain, два каскада ограничения, темброблок,
выходной Q1 и Volume. Это не набор фильтров с waveshaper: модель сохраняет
состояния конденсаторов, нелинейности транзисторов и диодов и межкаскадные
обратные связи.

Текущая составная архитектура работает с BDF2 одним проходом при 48 кГц:

`Q4 + Sustain + Q3 (3x3) -> Q2 scalar -> Tone + Q1 (2x2)`.

На полном четырёхсекундном аккорде при размере DMA-полубуфера 32 получено
`0/6000` опозданий: максимум 111783 такта при бюджете 113312
(запас около 1,35%). Это сохранённый замер BDF2 на аккорде 100 мВ при
Sustain=1, Tone=1, Volume=0,8, а не проверка всей сетки параметров.
Настоящий ADC/DMA/DSP/DAC runtime собран с живым ADC-входом; для его аппаратной
проверки требуется описанная в документации внешняя обвязка со смещением 1,65 В.
Проверенный ранее режим цифрового нуля остаётся только диагностическим.

Подробное актуальное состояние: [передача следующему этапу](docs/09%20Передача%20следующему%20чату.md).

Начальная страница проекта: [docs/00 Начало.md](docs/00%20Начало.md).

Одна из основных техник проекта — перенос сохранённых экспонент и
гиперболических функций по малой поправке Ньютона вместо полного пересчёта:
[метод малого приращения нелинейностей](docs/10%20Метод%20малого%20приращения%20нелинейностей.md).

## Сборка

```powershell
platformio run -e composed_bdf2_block_benchmark
platformio run -e composed_audio_runtime
```

Загрузка через встроенный ST-Link:

```powershell
platformio run -e composed_audio_runtime --target upload
```

Отдельные измерительные сборки полной модели:

```powershell
platformio run -e full_pedal_cheap
platformio run -e full_pedal_robust
```

## Исследовательские стенды

Команды ниже воспроизводят отдельные, в том числе исторические опыты.
`composed_block_benchmark` сохраняет контрольный вариант Эйлера.
Карта каталогов и сборок: [устройство проекта](docs/02%20Устройство%20проекта.md).
Последние исследования: [коррекции границ](simulation/experiments/boundary_correction/report.md)
и [TR-BDF2](simulation/experiments/tr_bdf2/report.md); в рабочую прошивку они не включены.

## Разделение стоимости Q3

Повторно построить отчёт и графики по измеренному на плате разделению физики и арифметики:

```powershell
..\.venv\Scripts\python.exe simulation\plot_q3_breakdown.py
```

Результат: [simulation/experiments/q3_breakdown/report.md](simulation/experiments/q3_breakdown/report.md).

Сравнение отдельного вызова Q3 с ядром, встроенным в цикл:

```powershell
..\.venv\Scripts\python.exe simulation\plot_q3_inline_stream.py
```

Результат: [simulation/experiments/q3_inline_stream/report.md](simulation/experiments/q3_inline_stream/report.md).

Сравнение машинного кода, версий компилятора и порядка арифметики:

```powershell
..\.venv\Scripts\python.exe simulation\plot_q3_asm_compiler.py
```

Результат: [simulation/experiments/q3_asm_compiler/report.md](simulation/experiments/q3_asm_compiler/report.md).

Локальная проверка арифметики невязки и шага Ньютона:

```powershell
..\.venv\Scripts\python.exe simulation\plot_q3_local_arithmetic.py
```

Результат: [simulation/experiments/q3_local_arithmetic/report.md](simulation/experiments/q3_local_arithmetic/report.md).

Сравнение физики каскада и частоты расчёта:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_physics_sweep.py
```

Результат: [simulation/experiments/q3_physics_sweep/report.md](simulation/experiments/q3_physics_sweep/report.md).

Сравнение двух связанных каскадов ограничения:

```powershell
..\.venv\Scripts\python.exe simulation\run_two_clipping_stages_experiment.py
```

Результат: [simulation/experiments/two_clipping_stages/report.md](simulation/experiments/two_clipping_stages/report.md).

Входной Q4, Sustain, ЛАЧХ/ЛФЧХ и устойчивость дискретного шага:

```powershell
..\.venv\Scripts\python.exe simulation\run_frontend_stability_experiment.py
```

Результат: [simulation/experiments/frontend_stability/report.md](simulation/experiments/frontend_stability/report.md).

Глобальные возмущения и сверка с ngspice:

```powershell
..\.venv\Scripts\python.exe simulation\run_frontend_global_experiment.py
```

Результат: [simulation/experiments/frontend_global/report.md](simulation/experiments/frontend_global/report.md).

Полная цепь с темброблоком, Q1 и Volume:

```powershell
..\.venv\Scripts\python.exe simulation\run_complete_chain_experiment.py
```

Результат: [simulation/experiments/complete_chain/report.md](simulation/experiments/complete_chain/report.md).

Сравнение способов расчёта Q1:

```powershell
..\.venv\Scripts\python.exe simulation\run_q1_nonlinearity_experiment.py
```

Результат: [simulation/experiments/q1_nonlinearity/report.md](simulation/experiments/q1_nonlinearity/report.md).

Адаптивный переход Q1 и звуковые файлы:

```powershell
..\.venv\Scripts\python.exe simulation\run_q1_adaptive_experiment.py
```

Результат: [simulation/experiments/q1_adaptive/report.md](simulation/experiments/q1_adaptive/report.md).

Сухой гитарный аккорд и его прогон через полную и адаптивную модели:

```powershell
..\.venv\Scripts\python.exe simulation\build_guitar_chord_sample.py
..\.venv\Scripts\python.exe simulation\run_guitar_chord_experiment.py
```

Исходник: [simulation/samples/e_major_chord/README.md](simulation/samples/e_major_chord/README.md).
Результат: [simulation/experiments/guitar_chord/report.md](simulation/experiments/guitar_chord/report.md).

Полный четырёхсекундный аккорд через эталонную схему при пяти положениях ручек:

```powershell
..\.venv\Scripts\python.exe simulation\run_full_chord_pedal.py
```

Результат: [simulation/experiments/full_chord_pedal/report.md](simulation/experiments/full_chord_pedal/report.md).

Точное исключение линейных узлов и неактивных нелинейных переменных полной модели:

```powershell
..\.venv\Scripts\python.exe simulation\run_complete_reduction_experiment.py
..\.venv\Scripts\python.exe simulation\run_hybrid_active_experiment.py
```

Результаты: [линейное сокращение](simulation/experiments/complete_reduction/report.md) и
[шесть активных нелинейностей](simulation/experiments/hybrid_active/report.md).

Выбор каскадного решателя вместо плотной системы `6×6`:

```powershell
..\.venv\Scripts\python.exe simulation\run_cascade_solver_experiment.py
```

Результат: [simulation/experiments/cascade_solver/report.md](simulation/experiments/cascade_solver/report.md).
