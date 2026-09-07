# Проверочные расчёты

Каталог предназначен для воспроизводимых численных опытов до переноса
алгоритмов на STM32.

## Установка зависимостей

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-simulation.txt
```

Используется окружение Python, расположенное рядом с каталогом проекта:

```powershell
..\.venv\Scripts\python.exe simulation\verify_diode_solver.py
```

Проверка резкого перепада:

```powershell
..\.venv\Scripts\python.exe simulation\verify_diode_solver.py --waveform step
```

`verify_diode_solver.py` сравнивает одну поправку Ньютона и Галлея с точно
доведённым решением скалярного уравнения для эквивалента Тевенина и
встречно-параллельной диодной пары. Начальные параметры диода условные и не
считаются моделью 1N914 — программа проверяет численный метод, а не звучание.

Построение графиков этого опыта:

```powershell
..\.venv\Scripts\python.exe simulation\plot_diode_solver.py
```

Запуск полной эталонной схемы ngspice и построение её графиков:

```powershell
..\.venv\Scripts\python.exe simulation\run_reference.py
```

Сопоставление изолированного каскада ограничения Q3 с независимым узловым
решателем Python:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_experiment.py
```

Точное сокращение Q3 и сравнение одной поправки Ньютона/Галлея:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_reduction_experiment.py
```

Сборка измерителя тактов Q3 для подключённой NUCLEO-G474RE:

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run -e q3_benchmark
```

Прошивка выводит результаты через USART2 виртуального последовательного порта
ST-Link со скоростью 115200 бод. Построение отчёта по сохранённому журналу:

```powershell
..\.venv\Scripts\python.exe simulation\plot_q3_cycle_benchmark.py
```

Проверка раздельных форматов фиксированной точки:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_fixed_point_experiment.py
```

Проверка обновления нелинейных функций по малому приращению, накопления
ошибки и аварийного восстановления:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_incremental_experiment.py
```

Подготовка периодической последовательности для потоковой проверки на плате:

```powershell
..\.venv\Scripts\python.exe simulation\generate_q3_stream_fixture.py
```

Построение отчёта по сохранённому потоковому журналу платы:

```powershell
..\.venv\Scripts\python.exe simulation\plot_q3_stream_benchmark.py
```

Проверка повторного использования обратного якобиана и обновляемого
обратного знаменателя:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_quasi_newton_experiment.py
```

Отдельная сборка измерителя целочисленного решения 3×3:

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run -e q3_fixed_benchmark
```

Описание схемы: [ngspice/README.md](ngspice/README.md).

Отчёты опытов:

- [эталонная схема](experiments/reference_baseline/report.md);
- [узловая модель каскада Q3](experiments/q3_reference/report.md);
- [сокращение Q3 и одна поправка](experiments/q3_reduction/report.md);
- [измерение тактов Q3 на STM32](experiments/q3_cycle_benchmark/report.md);
- [фиксированная точка для ядра Q3](experiments/q3_fixed_point/report.md);
- [нелинейности Q3 по малому приращению](experiments/q3_incremental/report.md);
- [длительный поток Q3 на STM32](experiments/q3_stream_benchmark/report.md);
- [повторное использование обратного якобиана](experiments/q3_quasi_newton/report.md);
- [Ньютон и Галлей](experiments/diode_solver/report.md);
- [атака сухого гитарного аккорда](experiments/guitar_chord/report.md);
- [полный аккорд при пяти положениях ручек](experiments/full_chord_pedal/report.md).
- [точное исключение линейных узлов](experiments/complete_reduction/report.md);
- [шесть активных нелинейностей гибридной модели](experiments/hybrid_active/report.md);
- [каскадное решение шести нелинейностей](experiments/cascade_solver/report.md).
- [ресурсы полного шага педали на STM32](experiments/full_pedal_cycle_benchmark/report.md).

Сборка сухого ми-мажорного аккорда из открытых записей и его численная проверка:

```powershell
..\.venv\Scripts\python.exe simulation\build_guitar_chord_sample.py
..\.venv\Scripts\python.exe simulation\run_guitar_chord_experiment.py
```

Прогон полного четырёхсекундного аккорда через эталонную схему ngspice:

```powershell
..\.venv\Scripts\python.exe simulation\run_full_chord_pedal.py
```

Сокращение полной модели перед переносом в C:

```powershell
..\.venv\Scripts\python.exe simulation\run_complete_reduction_experiment.py
..\.venv\Scripts\python.exe simulation\run_hybrid_active_experiment.py
..\.venv\Scripts\python.exe simulation\run_cascade_solver_experiment.py
..\.venv\Scripts\python.exe simulation\generate_full_pedal_fixture.py
```

Сборка измерителя полного шага:

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run -e full_pedal_benchmark
```

Проверка диагональных координат и сохранённых нелинейностей полной педали:

```powershell
..\.venv\Scripts\python.exe simulation\run_modal_state_experiment.py
..\.venv\Scripts\python.exe simulation\run_full_cached_nonlinearity_experiment.py
..\.venv\Scripts\python.exe simulation\verify_fixed_4x4.py
```

Проверка распределения каскадов по частотам расчёта:

```powershell
..\.venv\Scripts\python.exe simulation\run_multirate_partition_experiment.py
```

Результаты: [разнос блоков по частотам](experiments/multirate_partition/report.md).

Физическое портовое разбиение на коллекторе Q2:

```powershell
..\.venv\Scripts\python.exe simulation\run_port_multirate_experiment.py
```

Результаты: [портовая многоскоростная модель](experiments/port_multirate/report.md).

Проверка портовой модели на уровнях 25–200 мВ:

```powershell
..\.venv\Scripts\python.exe simulation\run_port_multirate_level_sweep.py
```

Результаты: [уровни портовой модели](experiments/port_multirate_levels/report.md).

Создание коэффициентов и сборка аппаратного измерителя 4×/2×:

```powershell
..\.venv\Scripts\python.exe simulation\generate_port_multirate_fixture.py
& "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run -e port_multirate_benchmark
```

Результаты: [такты портовой модели](experiments/port_multirate_cycle_benchmark/report.md).

Подбор безопасного порога разрежения быстрой линейной части:

```powershell
..\.venv\Scripts\python.exe simulation\run_port_sparse_linear_experiment.py
```

Результаты: [разрежение быстрой линейной части](experiments/port_sparse_linear/report.md).

Сравнение блочных решателей быстрого ядра:

```powershell
..\.venv\Scripts\python.exe simulation\run_port_solver_dimension_experiment.py
```

Результаты: [размерность быстрого решателя](experiments/port_solver_dimension/report.md).

Чередование полного шага, прогноза и адаптивной невязки:

```powershell
..\.venv\Scripts\python.exe simulation\run_port_alternating_solver_experiment.py
```

Результаты: [полный шаг и прогноз](experiments/port_alternating_solver/report.md).

Все новые программы должны:

- иметь описанные единицы измерения и значения по умолчанию;
- отделять условные параметры от измеренных;
- печатать невязку, а не только разницу с эталонным сигналом;
- завершаться ненулевым кодом при нечисловом результате;
- не требовать сторонних пакетов без инженерной причины.
python simulation/run_port_partition_rate_experiment.py

pio run -e fmac_benchmark -t upload

- `simulation/experiments/port_partition_rate/report.md` — проверка индивидуальной частоты четырёх нелинейных портов быстрого ядра.
- `simulation/experiments/fmac_benchmark/report.md` — аппаратная цена FMAC при опросе и непрерывной передаче DMA.
- `simulation/experiments/port_state_reduction/report.md` — сбалансированное сокращение пространства состояний.
- `simulation/experiments/port_modal_reduction/report.md` — дешёвое усечение диагональных режимов.
- `simulation/run_fast_capacitor_simplification_experiment.py` — строгое удаление C10/C12/C11 при сохранении четырёх нелинейных портов.
- `simulation/experiments/fast_capacitor_simplification/report.md` — ошибка и оценка числа линейных произведений после удаления малых ёмкостей.
- `simulation/run_fast_port_sensitivity_experiment.py` — диапазоны и чувствительность четырёх активных портов быстрого ядра.
- `simulation/run_bjt_local_law_experiment.py` — линейный, квадратичный и кубический локальные законы BJT.
- `simulation/run_q3_cubic_validation.py` — проверка кубического закона Q3 по уровням и положениям Sustain/Tone.
- `simulation/run_strict_rate_reduction_experiment.py` — строгая перестройка коэффициентов для сеток 1×/2×/4×.
- `simulation/run_rate_solver_depth_experiment.py` — глубина и демпфирование Ньютона при быстром ядре 2×.
- `simulation/run_damped_2x_grid_experiment.py` — полная короткая сетка ручек и уровней для 2×.
- `simulation/run_damped_2x_long_experiment.py` — накопление ошибки на полном 250-мс файле атаки.
- `simulation/muff_exact_linear_model.py` — непрерывный линейный скелет и точный переход состояний.
- `simulation/experiments/exact_linear_rate/report.md` — закрытый путь разделённого экспоненциального шага.
- `simulation/muff_sdirk_model.py` — совместный L-устойчивый SDIRK2 для быстрого ядра.
- `simulation/experiments/sdirk_2x/report.md` — решающая проверка SDIRK2 и закрытие текущей архитектуры.
- `simulation/run_full_stage_cascade_experiment.py` — первый опыт композиции независимых полных каскадов.
- `simulation/run_composed_frontend_experiment.py` — длинная проверка разреза `Q4+Q3 → Q2` с предиктором напряжения базы.
- `simulation/run_reused_q3q2_block_experiment.py` — длинная проверка готовой связанной ячейки Q3–Q2.
- `simulation/muff_q4_stage_model.py` — отрицательная проверка полного Q4 без обратной нагрузки C5.

## Актуальное составное ядро BDF2 1×

Текущая realtime-кандидатура работает одним проходом при 48 кГц:

`Q4 + Sustain + Q3 (3x3) -> Q2 scalar -> Tone + Q1 (2x2)`.

Связанные эксперименты и результаты:

- `simulation/experiments/composed_frontend_cycle_benchmark/report.md` — такты короткого и полного сигналов на STM32;
- `simulation/experiments/composed_runtime_chord/report.md` — проверка полного аккорда и WAV для прослушивания;
- `src/composed_chord_fixture.S` — встроенный полный тестовый аккорд;
- `src/composed_frontend_benchmark.c` под `COMPOSED_BLOCK_BENCHMARK` — блоковый deadline benchmark;
- `src/audio_io.c` — ADC/DMA/DAC runtime на регистрах.

Рабочий BDF2 1×: полный аккорд 100 мВ, блок 32 — 0/6000 опозданий,
максимум 111783 такта при бюджете 113312. Настоящий DMA с цифровым нулём:
0/9000 опозданий, максимум 95847. Прежний максимум 107126 относится к Эйлеру.
Сборка: `platformio run -e composed_bdf2_block_benchmark`.
Условия: [BDF2 на STM32](experiments/bdf2_stm32/report.md).

## Анализ необходимости oversampling

- `simulation/run_oversampling_analysis.py` — полная модель на
  1x/2x/4x/8x/16x, FIR-приведение к 48 кГц, THD, временная и спектральная
  ошибка относительно 16x;
- `simulation/experiments/oversampling_analysis/report.md` — таблица пилота;
- `simulation/experiments/oversampling_analysis/figures/` — FFT выхода и
  спектры ошибки.

Дорогие результаты кэшируются в игнорируемой папке `cache/`. Быстрый запуск:

```powershell
..\.venv\Scripts\python.exe simulation\run_oversampling_analysis.py --quick
```
- `simulation/run_bdf2_analysis.py` — сравнение Эйлера и BDF2 в полной модели; отчёт и графики находятся в `simulation/experiments/bdf2_analysis/`.
- `simulation/run_bdf2_chord_analysis.py` — сравнение Эйлера и BDF2 на атаке настоящего гитарного аккорда со звуковыми файлами для прослушивания.
- `simulation/experiments/bdf2_stm32/report.md` — перенос BDF2 1× в составное ядро и замеры на STM32G474.
- `simulation/run_generalized_alpha_analysis.py` — перебор управляемого высокочастотного затухания обобщённого α-метода; лучший модельный компромисс найден около `rho=0.2`.
- `simulation/experiments/generalized_alpha_composed/report.md` — проверка `rho=0.2` в составном C-ядре; выигрыш полной модели не сохранился на сильном аккорде.

- [Коррекции границ](experiments/boundary_correction/report.md) — запаздывающие
  поправки и мгновенное согласование 2×2; в рабочую сборку не включены.
- `simulation/run_tr_bdf2_analysis.py` — TR-BDF2 в полной модели;
  [отчёт](experiments/tr_bdf2/report.md). Переноса на STM32 нет.

Не сравнивать ошибки разных отчётов без сверки эталона, длительности и
нормировки. Актуальное состояние:
[передача контекста](../docs/09%20Передача%20следующему%20чату.md).

## Согласование полинома Q2 с интегратором

`generate_q2_polynomial.py` создаёт отдельные полиномы BDF2 и α-метода.
`run_q2_polynomial_validation.py` сравнивает старый, исправленный и точный Q2
в одном C-ядре, проверяет локальную невязку, синусы и полный аккорд.
Требуется GCC в PATH; таблица ngspice берётся из `simulation/raw/full_chord_pedal/`
(при отсутствии восстановить через `run_full_chord_pedal.py`).

[Отчёт, графики и звук](experiments/q2_polynomial/report.md).
Исправление улучшает локальную точность, но не даёт общего выигрыша к ngspice.
Рабочая сборка не меняется; эксперимент — `composed_bdf2_q2_block_benchmark`.
Ненулевой код проверки отражает найденные срывы на 50/200 мВ, отчёт сохраняется.

## Общая и составная BDF2 на одном шаге

`python simulation/run_bdf2_same_step.py` сравнивает одинаковые входы, шаг и
начальные истории. Общая гибридная система использует повтор несошедшегося
шага с демпфированием Ньютона, без изменения интегратора.
[Отчёт](experiments/bdf2_same_step/report.md). Ненулевое завершение отражает
срыв составных вариантов на атаке 200 мВ; результаты сохраняются.
