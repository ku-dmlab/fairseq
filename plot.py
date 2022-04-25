import numpy as np
import matplotlib.pyplot as plt
import itertools


def bar_plot():

    algs = ["Baseline", "Reinforce", "Ours(online)", "Ours(offline)", "Ours(imitation)"]
    vals = [
        [27.81, 27.97, 27.8, 28.08],
        [27.91, 28.14, 27.81, 27.88],
        [28.14, 28.3, 28.3, 28.46],
        [28.22, 28.21, 28.4, 28.48],
        [28.17,	28.2, 28.18, 28.28]
    ]
    means = np.mean(vals, axis=1)
    errors = np.std(vals, axis=1) / np.sqrt(len(vals[0])) * 1.96

    x_pos = np.arange(len(algs))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=errors, align='center', color=colors) #alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Test BLEU')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algs)
    ax.set_ylim(27.5, 29)
    ax.set_title('Performances of algorithms (IWSLT14 en-de)')

    for rect, alg, mean, error in zip(ax.patches, algs, means, errors):
        label = f"{mean:.3f}(±{error:.3f})"
        if alg != "Baseline":
            label += f"\n Gain={mean - means[0]:.3f}"
        x_value = rect.get_x() + rect.get_width() / 2
        y_value = rect.get_height()
        ax.annotate(label, (x_value, y_value), xytext = (0, 40), textcoords="offset points", ha="center", va="bottom")

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot.png')
    plt.show()

def hyper_plot():
    alphas = [0, 0.1, 0.3, 1, 3]
    taus = [0.5, 0.75, 0.9, 0.95, 0.99]
    a_vals_on = [27.88, 27.93, 28.16, 28.14, 28.1]
    t_vals_on = [28.1, 28.13, 28.14, 28.23, 28.2]
    a_vals_off = [27.83, 27.81, 28.27, 28.28, 28.25]
    t_vals_off = [28.37, 28.34, 28.28, 28.25, 28.16]
    t_vals_off_a0 = [27.78, 27.83, 27.83, 27.83, 27.81]
    a_vals_im = [28, 28.15, 28.17]
    fig, axes = plt.subplots(2)
    axes[0].plot(np.arange(5), a_vals_on, label="Online", marker="o")
    axes[0].plot(np.arange(5), a_vals_off, label="Offline", marker="o")
    axes[0].plot(np.arange(2,5), a_vals_im, label="Imitation", marker="o")
    axes[0].set_xticks(np.arange(5))
    axes[0].set_xticklabels(alphas)
    axes[0].set_ylabel("Test BLEU")
    axes[0].set_xlabel("$\\alpha$")
    axes[0].legend()
    axes[0].set_title('Performances over varying $\\alpha$ (IWSLT14 en-de, $\\tau=0.9$)')

    axes[1].plot(np.arange(5), t_vals_on, marker="o", label="Online($\\alpha=1.0$)")
    axes[1].plot(np.arange(5), t_vals_off, marker="o", label="Offline($\\alpha=1.0$)")
    axes[1].plot(np.arange(5), t_vals_off_a0, marker="o", color="#d62728", label="Offline($\\alpha=0$)")
    axes[1].set_xticks(np.arange(5))
    axes[1].set_xticklabels(taus)
    axes[1].set_ylabel("Test BLEU")
    axes[1].set_xlabel("$\\tau$")
    axes[1].legend()
    axes[1].set_title('Performances over varying $\\tau$ (IWSLT14 en-de)')
    plt.tight_layout()
    plt.savefig("hyper_plot.png")

def mixed_plot():
    baselines = [
        [27.81, 27.97, 27.8, 28.08],
        [27.78, 27.73, 27.71, 27.98],
        [26.7, 26.65, 26.69, 26.7],
        [22.89, 4.5, 4.35, 22.27]
    ]
    ours1 = [
        [28.22, 28.21, 28.4, 28.48],
        [28.19, 28.22, 28.16, 28.31],
        [28.05, 28.12, 28.01, 28.21],
        [27.84, 27.81, 27.76, 28.03]
    ]
    ours2 = [
        [28.22, 28.21, 28.4, 28.48],
        [27.85, 27.97, 27.86, 28.18],
        [26.98, 27.16, 26.9, 27.39],
        [22.87, 4.52, 4.27, 22.23]
    ]
    fig, ax = plt.subplots()
    x = np.arange(len(baselines))
    plt.errorbar(x, np.mean(baselines, axis=1), yerr=np.std(baselines, axis=1)/2*1.96, label="baseline")
    plt.errorbar(x, np.mean(ours1, axis=1), yerr=np.std(ours1, axis=1)/2*1.96, label="agent 1")
    plt.errorbar(x, np.mean(ours2, axis=1), yerr=np.std(ours2, axis=1)/2*1.96, label="agent 2")
    plt.xticks(x, ["GT", "(5, 10, 15, 20, GT)", "(2, 5, 7, 10, GT)", "(1, 2, 3, 4, GT)"])

    # Save the figure and show
    plt.ylim(10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mixed_plot.png')
    #plt.show()

mixed_plot()