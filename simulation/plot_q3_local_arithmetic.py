from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "q3_local_arithmetic" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

names = ["Строгий", "Невязка", "Ньютон", "Оба участка", "Равная производная"]
nominal = np.array([183, 195, 182, 194, 183])
strong = np.array([191, 201, 184, 197, 191])
x = np.arange(len(names))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
bars_a = ax.bar(x - width / 2, nominal, width, label="Обычный сигнал")
bars_b = ax.bar(x + width / 2, strong, width, label="Сильный сигнал")
ax.bar_label(bars_a, padding=3)
ax.bar_label(bars_b, padding=3)
ax.set_xticks(x, names)
ax.set_ylabel("Тактов на внутренний шаг")
ax.set_title("Локальные преобразования арифметики Q3")
ax.set_ylim(160, 210)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.savefig(OUT / "comparison.png", dpi=160)
