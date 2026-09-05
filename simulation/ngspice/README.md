# Эталонная схема ngspice

## Состав

- `big_muff_v3.inc` — полная топология Big Muff Pi V3 по сохранённой схеме
  ElectroSmash;
- `models.lib` — начальные модели BC239 и 1N914;
- `reference.cir` — рабочая точка, частотный и переходный расчёты всей педали;
- `q3_reference.cir` — изолированный первый каскад ограничения Q3 с явно
  записанными уравнениями Эберса—Молла и диодной пары;
- `frontend_equation_reference.cir` — Q4, Sustain, Q3, Q2 и нагрузка
  темброблока с теми же явными уравнениями, что и в Python.

Пассивные элементы и соединения повторяют схему статьи. Параметры
полупроводников пока предварительные: они нужны для работающего исходного
эталона, но позже должны быть заменены проверенными моделями или результатами
измерения конкретных деталей.

## Положения регуляторов

Параметры `SUSTAIN`, `TONE` и `VOLUME` лежат в диапазоне от 0 до 1:

- `SUSTAIN=1` — максимальный сигнал на каскады ограничения;
- `TONE=0` — низкочастотная сторона;
- `TONE=1` — высокочастотная сторона;
- `VOLUME=1` — максимальный выходной уровень.

В исходном опыте: `SUSTAIN=1`, `TONE=0.5`, `VOLUME=1`, вход — синус 1 кГц,
200 мВ пик-пик.

## Запуск

Из корня проекта:

```powershell
..\.venv\Scripts\python.exe simulation\run_reference.py
```

Изолированный Q3 и его сопоставление с узловой моделью Python:

```powershell
..\.venv\Scripts\python.exe simulation\run_q3_experiment.py
```

Входной тракт, глобальные возмущения и сверка с явными уравнениями ngspice:

```powershell
..\.venv\Scripts\python.exe simulation\run_frontend_global_experiment.py
```

Программа ищет ngspice в `PATH`, в переменной `NGSPICE_EXE` и в локальном
каталоге `..\.tools\ngspice-47\Spice64\bin\ngspice_con.exe`.

Официальная страница загрузки:
[ngspice downloads](https://ngspice.sourceforge.io/download.html).

Исходные таблицы и журнал ngspice пишутся в `simulation/raw/` и не входят в
Git. Графики и краткие выводы хранятся в `simulation/experiments/`.
