from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPO_LOG = (
    PROJECT_ROOT
    / "results"
    / "comparison_ppo_ppo-cbf-gp"
    / "ppo"
    / "log_episodes_ppo.csv"
)
DEFAULT_CBF_LOG = (
    PROJECT_ROOT
    / "results"
    / "comparison_ppo_ppo-cbf-gp"
    / "ppo_cbf_gp"
    / "log_episodes.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "comparison_ppo_ppo-cbf-gp"
    / "comparison_ppo_vs_cbf.png"
)
SAFE_LIMIT_RAD = 1.0
FIGSIZE = (13, 4)

REQUIRED_COLUMNS = {
    "global_step",
    "episode_return",
    "max_angle_deg",
    "safety_violations",
}


def parse_args():
    parser = ArgumentParser(description="Plot PPO vs PPO-CBF-GP training logs.")
    parser.add_argument(
        "--ppo-log",
        type=Path,
        default=DEFAULT_PPO_LOG,
        help="Path to PPO baseline episode log CSV.",
    )
    parser.add_argument(
        "--cbf-log",
        type=Path,
        default=DEFAULT_CBF_LOG,
        help="Path to PPO-CBF-GP episode log CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the comparison figure will be saved.",
    )
    return parser.parse_args()


def smooth(series, window=20):
    return series.rolling(window, min_periods=1, center=True).mean()


def validate_log(df, path):
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_cols}")


def format_axis(ax, plt):
    ax.grid(alpha=0.3)
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))


def main():
    args = parse_args()
    if not args.ppo_log.exists():
        raise FileNotFoundError(f"PPO log not found: {args.ppo_log}")
    if not args.cbf_log.exists():
        raise FileNotFoundError(f"PPO-CBF-GP log not found: {args.cbf_log}")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Load logs for comparison
    ppo = pd.read_csv(args.ppo_log)
    cbf = pd.read_csv(args.cbf_log)
    validate_log(ppo, args.ppo_log)
    validate_log(cbf, args.cbf_log)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.suptitle("PPO vs PPO-CBF-GP", fontsize=13)
    fig.subplots_adjust(wspace=0.35)

    # Return
    axes[0].plot(ppo.global_step, smooth(ppo.episode_return), label="PPO")
    axes[0].plot(cbf.global_step, smooth(cbf.episode_return), label="PPO-CBF-GP")
    axes[0].set_title("Episode Return")
    axes[0].set_ylabel("Return")
    format_axis(axes[0], plt)

    # Max angle
    axes[1].plot(ppo.global_step, smooth(ppo.max_angle_deg), label="PPO")
    axes[1].plot(cbf.global_step, smooth(cbf.max_angle_deg), label="PPO-CBF-GP")
    axes[1].axhline(
        np.degrees(SAFE_LIMIT_RAD),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="safe limit",
    )
    axes[1].set_title("Max Angle per Episode")
    axes[1].set_ylabel("Degrees")
    format_axis(axes[1], plt)

    # Safety violations
    axes[2].plot(ppo.global_step, smooth(ppo.safety_violations), label="PPO")
    axes[2].plot(cbf.global_step, smooth(cbf.safety_violations), label="PPO-CBF-GP")
    axes[2].set_title("Safety Violations")
    axes[2].set_ylabel("Steps per Episode")
    format_axis(axes[2], plt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved comparison plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
