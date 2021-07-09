import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

game = "xdo"
data = pd.read_csv(f"{game}_kuhn_exploitability")
maximal_time = 100
maxi = max(np.where(data.iloc[:, 1] < maximal_time)[0])
mini = 2
plt.plot(data.iloc[mini:maxi, 1], data.iloc[mini:maxi, 2])

game = "cfr"
data = pd.read_csv(f"{game}_kuhn_exploitability")
maxi = max(np.where(data.iloc[:, 1] < maximal_time)[0])
plt.plot(data.iloc[mini:maxi, 1], data.iloc[mini:maxi, 2])

plt.title(f"Algorithms on Kuhn")
plt.legend(["xdo", "cfr"])
plt.ylabel("Exploitabtility")
plt.xlabel("Time(s)")
plt.savefig(f"kuhn.png")
plt.show()
