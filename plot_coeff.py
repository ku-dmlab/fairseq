

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.5, 1.0, 100)

plt.figure(figsize=(5,4))

def get_y(x, c):
    return 1 / (2 * (1 - x + (2 * x - 1) * c))
cs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
colors = plt.cm.rainbow(cs)
for i, c in enumerate(cs):
    plt.plot(x, get_y(x, c), color=colors[i], label="$\\rho_{G>Q}$ = "+f"{c}")
plt.xlabel("$\\eta$")
plt.ylabel("$c_3$")
plt.legend()
plt.tight_layout()
plt.savefig("coeff.pdf", bbox_inches='tight', pad_inches=0.0)