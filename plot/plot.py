import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

algos = ['xdo', 'cfr', 'oxdo']
game = 'kuhn'
for i, algo in enumerate(algos):
    data = pd.read_csv(f"{algo}_{game}_exploitability")
    maximal_time = 100
    maxi = max(np.where(data.iloc[:, 1] < maximal_time)[0])
    mini = 0
    plt.plot(data.iloc[mini:maxi, 1], data.iloc[mini:maxi, 2])

plt.title(f"Algorithms on {game}")
plt.legend(algos)
plt.ylabel("Exploitabtility")
plt.xlabel("Time(s)")
plt.savefig(f"{game}.png")
plt.show()
