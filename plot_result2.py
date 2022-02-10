

import pickle

with open("score.pkl", "rb") as handle:
    score = pickle.load(handle)
print(score)

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(20, 40))
gs = GridSpec(2, len(score))

all_all_perfs = []
for fig_num, (key, value) in enumerate(score.items()):
    all_perfs = [[], [], [], []]
    ax = fig.add_subplot(gs[0, fig_num])
    ks = sorted(list(value.keys()))
    for i, k in enumerate(ks):
        v = value[k]
        res = np.array(list(v.items()))
        x, y = res[:, 0], res[:, 1]
        ax.plot(np.arange(len(x)), y, label=key + '_' + k, marker="o", markerfacecolor="none", c=plt.cm.viridis(i / len(ks)))
        perfs = sorted(y, reverse=True)
        if "last" not in k and "best" not in k:
            all_perfs[0].append(k)
            all_perfs[1].append(perfs[0])
            all_perfs[2].append(np.average(perfs[:3]))
            all_perfs[3].append(np.average(perfs))
    ax.set_xticklabels(["Q only"] + list(x[1:]))

    ax.hlines(34.68, xmin=0, xmax=len(x)-1, label="baseline", color="k", linestyle="dotted")
    ax.hlines(34.84, xmin=0, xmax=len(x)-1, label="REINFORCE", color="r", linestyle="dotted")
    ax.set_ylim(bottom=34)
    ax.legend()
    ax.set_xlabel("Q mixing ratio")
    ax.set_ylabel("Test BLEU")
    all_all_perfs.append([key, all_perfs])
ax = fig.add_subplot(gs[1, :])
for (key, all_perfs), color in zip(all_all_perfs, ["b", "r", "k", "cyan"]):
    ax.plot(np.arange(len(all_perfs[0])), all_perfs[1], label=f"{key} max", c=color, linestyle="-")
    ax.plot(np.arange(len(all_perfs[0])), all_perfs[2], label=f"{key} top 3", c=color, linestyle="--")
    ax.plot(np.arange(len(all_perfs[0])), all_perfs[3], label=f"{key} avg", c=color, linestyle="-.")
    ax.set_xticklabels(all_perfs[0])
ax.set_title(f"all_max: {max([max(all_perfs[1]) for (key, all_perfs) in all_all_perfs])}")
ax.legend()

fig.tight_layout()
fig.savefig("plot2.png")

# tau는 0.9 ~ 0.95 정도로
# 최소한 72정도까지는 돌려봐야할듯?


