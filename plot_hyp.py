
import matplotlib.pyplot as plt

res = [28.24, 28.7, 28.95, 29.03, 29.04, 28.64]
alphas = [0.1, 0.3, 1, 3, 10, 30]

etas = [0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.999]
res2 = [29.0,28.78,28.74,28.83,28.78,28.83,28.84,28.95,29.06]


fig = plt.figure(figsize=(5,4))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(alphas, res, marker="o", color="b", label="Varying $\\alpha$, $\\tau=0.99$")
ax1.set_xlabel("$\\alpha$")
#ax1.set_ylim(25)
ax1.set_xscale("log")
ax1.set_ylabel("Test BLEU")

ax2.plot(etas, res2, marker="x", color="g", label="Varying $\\eta$, $\\alpha=1.0$")
ax2.set_xlabel("$\\eta$")
fig.legend(bbox_to_anchor=[0.95, 0.35])
fig.tight_layout()
fig.savefig("hyp.png", bbox_inches='tight', pad_inches=0.0)