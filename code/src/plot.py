import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(s, w=20):
    return s.rolling(w, min_periods=1, center=True).mean()


def plot_ep(ax, ep, col, title, ylabel, limit=None):
    ax.plot(ep.global_step, ep[col], alpha=0.2, linewidth=0.8)
    ax.plot(ep.global_step, smooth(ep[col]), linewidth=2)
    if limit:
        ax.axhline(
            limit, color="red", linestyle="--", linewidth=1.2, label="safe limit"
        )
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))


# Curves from logs
def plot_training(episodes_path, iters_path, output_path=None, F=1.0):
    ep = pd.read_csv(episodes_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("PPO-CBF-GP Training", fontsize=13)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    plot_ep(axes[0, 0], ep, "episode_return", "Episode Return", "Return")
    plot_ep(axes[0, 1], ep, "mean_u_cbf", "Mean u_CBF (vanishing)", "u_CBF")
    plot_ep(
        axes[1, 0],
        ep,
        "max_angle_deg",
        "Max Angle per Episode",
        "Degrees",
        limit=np.degrees(F),
    )
    plot_ep(axes[1, 1], ep, "safety_violations", "Safety Violations", "Steps")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.show()
