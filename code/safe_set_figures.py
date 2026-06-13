from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from src.cbf_geom import F_LIM, GAMMA, LAM, SPEED_MAX, U_MAX, nominal_u_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEG = 180.0 / np.pi


# Safe state set C = { |theta| + LAM |theta_dot| <= F }, state-only, fixed in training
def plot_safe_state_set(output_path):
    th_int = F_LIM  # theta intercept in rad
    td_int = F_LIM / LAM  # theta_dot intercept in rad/s
    # diamond vertices, counter-clockwise
    verts_rad = np.array(
        [
            [th_int, 0.0],
            [0.0, td_int],
            [-th_int, 0.0],
            [0.0, -td_int],
        ]
    )
    verts = verts_rad.copy()
    verts[:, 0] *= DEG

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill(
        verts[:, 0],
        verts[:, 1],
        facecolor="#aed4ea",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.7,
        label="safe state set C",
    )
    ax.set_title(r"Safe state set:  $|\theta| + 0.01\,|\dot{\theta}| \leq 1$")
    ax.set_xlabel(r"angle $\theta$ (deg)")
    ax.set_ylabel(r"angular velocity $\dot{\theta}$ (rad/s)")
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-1.15 * td_int, 1.15 * td_int)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"Saved {output_path}")
    plt.close(fig)


# Width u_hi - u_lo of the nominal control set K_cbf over the state plane.
# Uses nominal_u_bounds (sigma = 0) so the figure is the theoretical set
# before the GP uncertainty margin.
def plot_safe_control_set(output_path, n_theta=400, n_thetadot=400):
    th_deg = np.linspace(-90, 90, n_theta)
    td = np.linspace(-40, 40, n_thetadot)
    TH, TD = np.meshgrid(th_deg / DEG, td)

    width = np.full(TH.shape, np.nan)
    for i in range(TH.shape[0]):
        for j in range(TH.shape[1]):
            lo, hi = nominal_u_bounds(TH[i, j], TD[i, j])
            if hi > lo:
                width[i, j] = hi - lo

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(
        th_deg,
        td,
        width,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=2 * U_MAX,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"safe torque interval width $u_{hi}-u_{lo}$ (N.m)")
    ax.set_title(r"Safe control set $K_{cbf}(\theta, \dot{\theta})$ width (nominal)")
    ax.set_xlabel(r"angle $\theta$ (deg)")
    ax.set_ylabel(r"angular velocity $\dot{\theta}$ (rad/s)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    p = ArgumentParser(
        description="Generate the theoretical (nominal) safe-set figures for the report."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "cbf_plots",
        help="Directory for the output PNGs.",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Parameters: LAM={LAM}, F={F_LIM}, gamma_b={GAMMA}, "
        f"u_max={U_MAX}, speed_max={SPEED_MAX}"
    )
    plot_safe_state_set(args.output_dir / "safe_state_set.png")
    plot_safe_control_set(args.output_dir / "safe_control_set.png")


if __name__ == "__main__":
    main()
