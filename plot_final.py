
import matplotlib.pyplot as plt

def plot(is_cross, plot_all):
    results1 = [
        [14.1875, 14.27, 14.15, 14.145, 13.6275],
        [14.6675, 14.5825, 14.4725, 14.385, 14.1],
        [14.9125, 14.87, 14.7875, 14.705, 14.125],
        [14.9975, 14.895, 14.8125, 14.7, 14.08]
    ]
    results2 = [
        [26.6575, 26.6425, 26.4875, 26.5675, 26.4925],
        [27.4575, 27.45, 27.4575, 27.4375, 27.305],
        [26.775, 26.755, 26.7675, 26.7725, 26.4925],
        [27.55, 27.5675, 27.55, 27.48, 27.2775]
    ]
    results = results1 if is_cross else results2
    baseline = [13.625] if is_cross else [25.925]
    # BC, CER(H), GOLD-\delta, CER(R)

    fig = plt.figure(figsize=(5,4))
    ax1 = fig.add_subplot(111)
    if is_cross:
        noise = [0, 1, 3, 5, 7]
        ax1.set_xticks([0, 1, 3, 5, 7])
        ax1.set_xticklabels([0, 1, 3, 5, "mpnet"])
        ax1.set_xlabel("Noise Variance")
    else:
        noise = [0, 1, 2, 3, 4]
        ax1.set_xticks([0, 1, 2, 3, 4])
        ax1.set_xticklabels([0, 1, 2, 3, "mpnet"])
        ax1.set_xlabel("Top Removed")

    labels = ["$BC$", "$CER(\mathcal{A})$", "$GOLD$", "$CER(\mathcal{R})$"]
    for i, res in enumerate(results):
        ax1.plot(noise, res, marker="o", label=labels[i])
        ax1.set_ylabel("Test BLEU")
    ax1.axhline(baseline[0], c="gray", linestyle="dashed", label="Baseline")

    fig.legend(bbox_to_anchor=(0.41, 0.5))#bbox_to_anchor=[0.95, 0.35])
    fig.tight_layout()
    if is_cross:
        fig.savefig("final_cross.png", bbox_inches='tight', pad_inches=0.0)
    else:
        fig.savefig("final_in.png", bbox_inches='tight', pad_inches=0.0)

plot(True)