import sys
from argparse import ArgumentParser as ArgP
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as Poly3D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(CODE_DIR / "src"))

from dynamics_gp import get_nominal_dynamics as nom_dyn  # noqa: E402

H = [
    np.array([1, 0.01]),
    np.array([1, -0.01]),
    np.array([-1, 0.01]),
    np.array([-1, -0.01]),
]
F = 1.0
GAM = 0.5
TB = 15.0
MS = 60.0
LAM = 0.01

CG = "#2BA89B"
SS = "#cfe8e3"
FS = (15, 9.5)
PSZ = (9.2, 4.2)
PPAD = {"left": 0.10, "right": 0.70, "top": 0.88, "bottom": 0.16}
GW = [1.45, 1]
GWS = 0.16
GHS = 0.30
XL_ST = (-88, 88)
YL_ST = (-32, 32)
XL_U = (-88, 88)
YL_U = (-23, 23)
EL = 18
AZ = -58

L_SAFE = "state safe set"
L_NOM_ST = "nominal admissible states"
L_NOM_U = r"nominal interval at $\dot{\theta}=0$"
L_NOM3 = "nominal CBF set (reference)"
L_TH = r"$\theta$ (deg)"
L_TD = r"$\dot{\theta}$ (rad/s)"
L_U = r"$u$ (N.m)"
L_MU = r"mean control $u$ (N.m)"
L_K = r"training step $k$"
L_PROP = r"proposed $u_{\mathrm{PPO}}+u_{\mathrm{BAR}}$"
L_DEP = r"deployed $u_{\mathrm{total}}$"
L_CORR = r"CBF projection ($u_{\mathrm{CBF}}$)"


# Nom CBF bounds
def u_bounds(th, td):
    obs = np.array([np.cos(th), np.sin(th), td])
    f, g, x = nom_dyn(obs, 0.0)
    lo, hi = -TB, TB

    for h_vec in H:
        hg = float(np.dot(h_vec, g))
        rhs = GAM * F + float(np.dot(h_vec, f)) - (1 - GAM) * float(np.dot(h_vec, x))
        coef = -hg
        if abs(coef) < 1e-12:
            continue
        if coef > 0:
            hi = min(hi, rhs / coef)
        else:
            lo = max(lo, rhs / coef)

    f1, g1 = float(f[1]), float(g[1])
    if abs(g1) > 1e-12:
        b1 = (MS - f1) / g1
        b2 = (-MS - f1) / g1
        hi = min(hi, max(b1, b2))
        lo = max(lo, min(b1, b2))

    return lo, hi


# Grid CBF bounds
def grid_bounds(th, td):
    lo_g = np.full_like(th, np.nan)
    hi_g = np.full_like(th, np.nan)
    m = np.zeros_like(th)

    for i in range(th.shape[0]):
        for j in range(th.shape[1]):
            lo, hi = u_bounds(th[i, j], td[i, j])
            if hi > lo:
                lo_g[i, j] = lo
                hi_g[i, j] = hi
                m[i, j] = 1

    return lo_g, hi_g, m


# Build plotting grids
def make_grids():
    th = np.linspace(np.radians(-88), np.radians(88), 140)
    td = np.linspace(-32, 32, 140)
    th_g, td_g = np.meshgrid(th, td)
    lo, hi, m = grid_bounds(th_g, td_g)

    thc = np.linspace(np.radians(-72), np.radians(72), 90)
    tdc = np.linspace(-32, 32, 90)
    thc_g, tdc_g = np.meshgrid(thc, tdc)
    loc, hic, mc = grid_bounds(thc_g, tdc_g)

    cfig = plt.figure()
    cax = cfig.add_subplot(111)
    ct = cax.contour(
        np.degrees(thc_g),
        tdc_g,
        mc,
        levels=[0.5],
    )
    plt.close(cfig)

    th_line = np.linspace(-88, 88, 300)
    ulo = []
    uhi = []
    for x in th_line:
        lo_i, hi_i = u_bounds(np.radians(x), 0.0)
        if hi_i > lo_i:
            ulo.append(lo_i)
            uhi.append(hi_i)
        else:
            ulo.append(np.nan)
            uhi.append(np.nan)

    return {
        "th": np.degrees(th_g),
        "td": td_g,
        "lo": lo,
        "hi": hi,
        "m": m,
        "thc": np.degrees(thc_g),
        "tdc": tdc_g,
        "loc": loc,
        "hic": hic,
        "mc": mc,
        "seg": ct.allsegs[0],
        "th_line": th_line,
        "ulo": np.array(ulo),
        "uhi": np.array(uhi),
    }


# Wrap pendulum angle
def wrap_angle(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


# Load control log
def load_ctrl(path):
    df = pd.read_csv(path)
    req = {"global_step", "theta", "theta_dot", "u_ppo", "u_bar", "u_total"}
    miss = req.difference(df.columns)
    if miss:
        miss_s = ", ".join(sorted(miss))
        raise ValueError(f"{path} is missing required columns: {miss_s}")

    df = df.copy()
    df["th"] = np.degrees(df.theta)
    df["reward"] = -(
        wrap_angle(df.theta.values) ** 2
        + 0.1 * df.theta_dot.values**2
        + 0.001 * df.u_total.values**2
    )
    df["prop"] = df.u_ppo + df.u_bar
    df["proj"] = df.u_total - df.prop
    return df


# Draw nominal CBF cage
def draw_cage(
    ax,
    grids,
    a_surf=0.07,
    a_wall=0.06,
    lw_c=1.8,
    a_in=0.0,
    step=7,
):
    ax.plot_surface(
        grids["thc"],
        grids["tdc"],
        grids["hic"],
        color=CG,
        alpha=a_surf,
        linewidth=0,
        shade=False,
    )
    ax.plot_surface(
        grids["thc"],
        grids["tdc"],
        grids["loc"],
        color=CG,
        alpha=a_surf,
        linewidth=0,
        shade=False,
    )
    ax.contour(
        grids["thc"],
        grids["tdc"],
        grids["mc"],
        levels=[0.5],
        colors=CG,
        linewidths=lw_c,
        offset=-TB,
    )
    ax.contour(
        grids["thc"],
        grids["tdc"],
        grids["mc"],
        levels=[0.5],
        colors=CG,
        linewidths=lw_c,
        offset=TB,
    )
    for seg in grids["seg"]:
        for k in range(0, len(seg) - 1, 3):
            x0, y0 = seg[k]
            x1, y1 = seg[min(k + 3, len(seg) - 1)]
            ax.add_collection3d(
                Poly3D(
                    [[[x0, y0, -TB], [x1, y1, -TB], [x1, y1, TB], [x0, y0, TB]]],
                    alpha=a_wall,
                    facecolor=CG,
                    edgecolor="none",
                )
            )

    if a_in > 0:
        th = grids["thc"]
        td = grids["tdc"]
        lo = grids["loc"]
        hi = grids["hic"]
        m = grids["mc"]
        for i in range(0, th.shape[0], step):
            for j in range(0, th.shape[1], step):
                if m[i, j] <= 0 or not np.isfinite(lo[i, j]):
                    continue
                ax.plot(
                    [th[i, j], th[i, j]],
                    [td[i, j], td[i, j]],
                    [lo[i, j], hi[i, j]],
                    color=CG,
                    lw=0.8,
                    alpha=a_in,
                )


# Label 3-D axes
def label_3d(ax):
    ax.set_xlabel(L_TH, labelpad=8)
    ax.set_ylabel(L_TD, labelpad=8)
    ax.set_zlabel(L_U, labelpad=4)
    ax.set_zlim(*YL_U)
    ax.view_init(EL, AZ)


# Save figure as PNG
def save_png(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# Save one 2-D panel
def save_panel(path, dpi, draw):
    fig, ax = plt.subplots(figsize=PSZ)
    fig.subplots_adjust(**PPAD)
    draw(ax)
    save_png(fig, path, dpi)


# Draw state regions
def draw_st_bg(ax, grids):
    th = np.linspace(XL_ST[0], XL_ST[1], 400)
    thr = np.radians(th)
    w = (F - np.abs(thr)) / LAM
    ok = w >= 0
    ax.fill_between(
        th,
        -w,
        w,
        where=ok,
        color=SS,
        alpha=0.65,
    )
    ax.contourf(
        grids["th"],
        grids["td"],
        grids["m"],
        levels=[0.5, 1.5],
        colors=[CG],
        alpha=0.38,
    )


# State legend handles
def st_h():
    return [
        Patch(facecolor=SS, edgecolor="none", alpha=0.65, label=L_SAFE),
        Patch(
            facecolor=CG,
            edgecolor="none",
            alpha=0.38,
            label=L_NOM_ST,
        ),
    ]


# Control legend handle
def u_h():
    return Patch(
        facecolor=CG,
        edgecolor="none",
        alpha=0.22,
        label=L_NOM_U,
    )


# Dot legend handle
def dot_h(label):
    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#444444",
        markersize=6,
        label=label,
    )


# Smooth curve
def smooth(values, window):
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


# Fit plot limits
def fit_lim(*vals, base, span0, padf=0.10):
    arr = [
        np.asarray(v, dtype=float).ravel() for v in vals if v is not None and len(v) > 0
    ]
    if not arr:
        return base

    v = np.concatenate(arr)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return base

    lo0 = float(np.min(v))
    hi0 = float(np.max(v))
    mid = 0.5 * (lo0 + hi0)
    span = max(hi0 - lo0, float(span0))
    pad = padf * span
    lo = mid - 0.5 * span - pad
    hi = mid + 0.5 * span + pad
    return max(base[0], lo), min(base[1], hi)


# Bin control traces
def time_data(df, k_zoom, bin, win):
    grp = df.groupby(df.global_step // bin)
    avg = grp.agg(
        step=("global_step", "mean"),
        prop=("prop", "mean"),
        dep=("u_total", "mean"),
    ).reset_index(drop=True)

    if k_zoom is not None and k_zoom > 0:
        avg = avg[avg.step <= k_zoom]
        ttl = f"first {int(k_zoom / 1000)}k steps"
    else:
        ttl = "full training"

    return (
        avg,
        smooth(avg.prop, win),
        smooth(avg.dep, win),
        ttl,
    )


# RL multiview figure
def plot_rl(df, grids, path, skip, dpi, cby):
    smp = df.iloc[::skip].copy()
    smp = smp.sort_values(cby, ascending=True)

    if cby == "global_step":
        col = smp[cby] / smp.global_step.max()
        cmap = "cividis_r"
        clab = "training progress"
        vmin = None
        vmax = None
        ttl = "RL controls over training: step-k progression"
    else:
        col = smp[cby]
        cmap = "viridis_r"
        clab = "per-step reward (0 = best)"
        vmin = smp.reward.quantile(0.02)
        vmax = 0
        ttl = "RL controls over training: per-step reward"

    xl = fit_lim(
        smp.th,
        base=XL_ST,
        span0=120,
        padf=0.08,
    )
    yl = fit_lim(
        smp.theta_dot,
        base=YL_ST,
        span0=10,
        padf=0.18,
    )
    yu = fit_lim(
        smp.u_total,
        base=YL_U,
        span0=22,
        padf=0.10,
    )

    fig = plt.figure(figsize=FS)
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=GW,
        wspace=GWS,
        hspace=GHS,
    )

    ax = fig.add_subplot(gs[:, 0], projection="3d")
    draw_cage(ax, grids)
    sc = ax.scatter(
        smp.th,
        smp.theta_dot,
        smp.u_total,
        s=6,
        c=col,
        cmap=cmap,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
    )
    label_3d(ax)
    ax.set_title(r"3-D: ($\theta$, $\dot{\theta}$, $u$)", fontsize=12)
    ax.legend(
        handles=[
            Line2D([0], [0], color=CG, lw=2, label=L_NOM3),
            dot_h(L_DEP),
        ],
        fontsize=9,
        markerscale=1.2,
        loc="upper left",
    )

    ax_s = fig.add_subplot(gs[0, 1])
    draw_st_bg(ax_s, grids)
    ax_s.scatter(
        smp.th,
        smp.theta_dot,
        s=6,
        c=col,
        cmap=cmap,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
    )
    ax_s.set_xlabel(L_TH)
    ax_s.set_ylabel(L_TD)
    ax_s.set_title(r"State plane $\theta$-$\dot{\theta}$", fontsize=11)
    ax_s.set_xlim(*xl)
    ax_s.set_ylim(*yl)
    ax_s.grid(alpha=0.3)
    ax_s.legend(
        handles=st_h() + [dot_h("sampled states")],
        fontsize=7.5,
        loc="upper right",
    )

    ax_u = fig.add_subplot(gs[1, 1])
    ax_u.fill_between(
        grids["th_line"],
        grids["ulo"],
        grids["uhi"],
        color=CG,
        alpha=0.22,
    )
    ax_u.scatter(
        smp.th,
        smp.u_total,
        s=6,
        c=col,
        cmap=cmap,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
    )
    ax_u.set_xlabel(L_TH)
    ax_u.set_ylabel(L_U)
    ax_u.set_title(r"Control vs angle $\theta$-$u$", fontsize=11)
    ax_u.set_xlim(*xl)
    ax_u.set_ylim(*yu)
    ax_u.grid(alpha=0.3)
    ax_u.legend(
        handles=[u_h(), dot_h(L_DEP)],
        fontsize=7.5,
        loc="upper right",
    )

    cbar = fig.colorbar(sc, ax=[ax_s, ax_u], shrink=0.7, pad=0.02, location="right")
    cbar.set_label(clab)
    fig.suptitle(ttl, fontsize=13, y=0.98)
    save_png(fig, path, dpi)


# Time-series panel
def plot_time(
    ax,
    df,
    k_zoom,
    bin,
    win,
):
    avg, up, ud, title = time_data(df, k_zoom, bin, win)

    corr = ax.fill_between(
        avg.step,
        up,
        ud,
        color="#7A7A7A",
        alpha=0.18,
        label=r"CBF correction gap ($u_{\mathrm{CBF}}$)",
    )
    h_prop = ax.plot(
        avg.step,
        up,
        color="#D81E1E",
        lw=1.25,
        label=L_PROP,
    )[0]
    h_dep = ax.plot(
        avg.step,
        ud,
        color="#1f3fd0",
        lw=1.25,
        label=L_DEP,
    )[0]
    ax.set_ylabel(L_MU)
    ax.set_title(f"Control over time ({title})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))

    hs = [corr, h_prop, h_dep]
    ax.legend(
        handles=hs,
        fontsize=7.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )


# Check GP columns
def gp_cols(df):
    return {"gp_interval_lo", "gp_interval_hi"}.issubset(df.columns)


# Samples changed by CBF
def proj_cases(df, gp=False):
    use_gp = gp and gp_cols(df)
    mod = df[df.proj.abs() > 0.3]
    rows = []
    for _, row in mod.iterrows():
        if use_gp:
            lo, hi = float(row.gp_interval_lo), float(row.gp_interval_hi)
        else:
            lo, hi = u_bounds(row.theta, row.theta_dot)
        if hi < lo or (hi - lo) >= 2 * TB - 0.5:
            continue
        prop_out = row.prop < lo - 1e-6 or row.prop > hi + 1e-6
        dep_in = lo - 0.3 <= row.u_total <= hi + 0.3
        if prop_out and dep_in:
            rows.append((row.th, row.theta_dot, row.prop, row.u_total, lo, hi))

    return pd.DataFrame(rows, columns=["th", "td", "prop", "dep", "lo", "hi"])


# Samples needing no CBF
def safe_cases(
    df,
    gp=False,
    tol=0.1,
    k0=100000,
):
    use_gp = gp and gp_cols(df)
    cand = df[df.proj.abs() <= tol]
    if "global_step" in cand.columns:
        cand = cand[cand.global_step >= k0]

    rows = []
    for _, row in cand.iterrows():
        if use_gp:
            lo, hi = float(row.gp_interval_lo), float(row.gp_interval_hi)
        else:
            lo, hi = u_bounds(row.theta, row.theta_dot)
        if hi < lo:
            continue
        prop_in = lo - 1e-6 <= row.prop <= hi + 1e-6
        dep_in = lo - 0.3 <= row.u_total <= hi + 0.3
        if prop_in and dep_in:
            rows.append((row.th, row.theta_dot, row.prop, row.u_total, lo, hi))

    return pd.DataFrame(rows, columns=["th", "td", "prop", "dep", "lo", "hi"])


# Downsample cases
def thin(cases, nmax):
    n = len(cases)
    if n > nmax:
        idx = np.linspace(0, n - 1, nmax, dtype=int)
        cases = cases.iloc[idx]
    return cases, n


# State projection panel
def plot_state2d(
    ax,
    grids,
    act,
    safe,
    c_prop,
    c_dep,
    c_safe,
    xlim,
    ylim,
):
    draw_st_bg(ax, grids)
    ax.scatter(
        act.th,
        act.td,
        s=36,
        c=c_prop,
        marker="^",
        alpha=0.75,
        label=L_PROP,
    )
    ax.scatter(
        act.th,
        act.td,
        s=20,
        c=c_dep,
        alpha=0.85,
        label=L_DEP,
    )
    ax.scatter(
        safe.th,
        safe.td,
        s=26,
        c=c_safe,
        marker="D",
        alpha=0.80,
        label="safe proposal (without CBF projection)",
    )
    ax.set_xlabel(L_TH)
    ax.set_ylabel(L_TD)
    ax.set_title("State-plane locations", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(
        handles=st_h()
        + [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=c_prop,
                label=L_PROP,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c_dep,
                label=L_DEP,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=c_safe,
                label="safe proposal (without CBF projection)",
            ),
        ],
        fontsize=7.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )


# Control projection panel
def plot_u2d(
    ax,
    grids,
    act,
    safe,
    use_gp,
    c_seg,
    c_int,
    c_prop,
    c_dep,
    c_safe,
    xlim,
    ylim,
):
    ax.fill_between(
        grids["th_line"],
        grids["ulo"],
        grids["uhi"],
        color=CG,
        alpha=0.22,
    )
    for _, row in act.iterrows():
        if use_gp:
            ax.plot(
                [row.th, row.th],
                [row.lo, row.hi],
                color=c_int,
                lw=3.2,
                alpha=0.70,
                solid_capstyle="round",
            )
        ax.plot(
            [row.th, row.th],
            [row.prop, row.dep],
            color=c_seg,
            lw=1.3,
            alpha=0.85,
            zorder=3,
        )
    ax.scatter(
        act.th,
        act.prop,
        s=38,
        c=c_prop,
        marker="^",
        edgecolor="k",
        linewidth=0.2,
        zorder=4,
    )
    ax.scatter(
        act.th,
        act.dep,
        s=24,
        c=c_dep,
        edgecolor="k",
        linewidth=0.2,
        zorder=4,
    )
    ax.scatter(
        safe.th,
        safe.dep,
        s=30,
        c=c_safe,
        marker="D",
        edgecolor="k",
        linewidth=0.2,
        alpha=0.85,
        zorder=4,
    )
    ax.set_xlabel(L_TH)
    ax.set_ylabel(L_U)
    ax.set_title(r"Control projection in the $\theta$-$u$ plane", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)

    hs = [u_h()]
    if use_gp:
        hs.append(Line2D([0], [0], color=c_int, lw=3, label="logged GP-CBF interval"))
    hs.extend(
        [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=c_prop,
                label=L_PROP,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c_dep,
                label=L_DEP,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=c_safe,
                label="safe proposal (without CBF projection)",
            ),
        ]
    )
    ax.legend(
        handles=hs,
        fontsize=7.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )


# Split 2-D output paths
def panels(p):
    suffix = p.suffix or ".png"
    stem = p.stem
    return {
        "state": p.with_name(f"{stem}_state{suffix}"),
        "control": p.with_name(f"{stem}_control{suffix}"),
        "timeseries": p.with_name(f"{stem}_timeseries{suffix}"),
    }


# Projection figures
def plot_proj(ctrl, grids, args):
    out, dpi = args.out, args.dpi
    c_seg = "#888888"
    c_int = "#E8A23A"
    c_prop = "#D81E1E"
    c_dep = "#1f3fd0"
    c_safe = "#6A3D9A"
    use_gp = gp_cols(ctrl)

    act = proj_cases(ctrl, gp=True)
    act, n_proj = thin(act, args.n_proj)
    safe = safe_cases(
        ctrl,
        gp=True,
        tol=args.tol,
        k0=args.k0,
    )
    safe, n_safe = thin(safe, args.n_safe)

    fig_3d = plt.figure(figsize=(10.5, 8.4))
    ax = fig_3d.add_subplot(111, projection="3d")
    draw_cage(ax, grids)

    for _, row in act.iterrows():
        up = np.clip(row.prop, -22, 22)
        if use_gp:
            ax.plot(
                [row.th, row.th],
                [row.td, row.td],
                [row.lo, row.hi],
                color=c_int,
                lw=2.8,
                alpha=0.85,
            )
        ax.plot(
            [row.th, row.th],
            [row.td, row.td],
            [up, row.dep],
            color=c_seg,
            lw=0.7,
            alpha=0.6,
        )

    ax.scatter(
        act.th,
        act.td,
        np.clip(act.prop, -22, 22),
        s=18,
        c=c_prop,
        marker="^",
        alpha=0.8,
    )
    ax.scatter(
        act.th,
        act.td,
        act.dep,
        s=14,
        c=c_dep,
        alpha=0.85,
    )
    ax.scatter(
        safe.th,
        safe.td,
        safe.dep,
        s=20,
        c=c_safe,
        marker="D",
        alpha=0.85,
    )

    ax.set_xlabel(L_TH, labelpad=8)
    ax.set_ylabel(L_TD, labelpad=8)
    ax.set_zlabel(L_U, labelpad=4)
    ax.set_zlim(*YL_U)
    ax.view_init(EL, AZ)
    hs = [
        Line2D([0], [0], color=CG, lw=2, label=L_NOM3),
    ]
    if use_gp:
        hs.append(
            Line2D(
                [0],
                [0],
                color=c_int,
                lw=3,
                alpha=0.85,
                label="logged GP-CBF interval",
            )
        )
    hs.extend(
        [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=c_prop,
                label=L_PROP,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c_dep,
                label=L_DEP,
            ),
            Line2D(
                [0],
                [0],
                color=c_seg,
                lw=1,
                label=L_CORR,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=c_safe,
                label="safe proposal (without CBF projection)",
            ),
        ]
    )
    ax.legend(
        handles=hs,
        fontsize=9,
        markerscale=1.3,
        loc="upper left",
    )
    src = "logged GP-robust intervals" if use_gp else "nominal intervals"
    ax.set_title(
        "3-D: CBF projection at fixed state",
        fontsize=12,
    )

    fig_3d.suptitle(
        f"CBF projection relative to {src}",
        fontsize=13,
        y=0.96,
    )
    save_png(fig_3d, out / "projection3d.png", dpi)

    xl = fit_lim(
        act.th,
        safe.th,
        base=XL_ST,
        span0=120,
        padf=0.08,
    )
    yl = fit_lim(
        act.td,
        safe.td,
        base=YL_ST,
        span0=10,
        padf=0.18,
    )
    yu = fit_lim(
        act.prop,
        act.dep,
        act.lo,
        act.hi,
        safe.dep,
        base=YL_U,
        span0=22,
        padf=0.10,
    )

    paths = panels(out / "projection2d.png")

    def draw_state(ax):
        plot_state2d(
            ax,
            grids,
            act,
            safe,
            c_prop,
            c_dep,
            c_safe,
            xl,
            yl,
        )

    def draw_u(ax):
        plot_u2d(
            ax,
            grids,
            act,
            safe,
            use_gp,
            c_seg,
            c_int,
            c_prop,
            c_dep,
            c_safe,
            xl,
            yu,
        )

    def draw_time(ax):
        plot_time(
            ax,
            ctrl,
            args.k_zoom,
            args.bin,
            args.win,
        )
        ax.set_xlabel(L_K)

    save_panel(paths["state"], dpi, draw_state)
    save_panel(paths["control"], dpi, draw_u)
    save_panel(paths["timeseries"], dpi, draw_time)

    print(
        f"projection3d plotted projs: {len(act)} of {n_proj}; "
        f"no-CBF: {len(safe)} of {n_safe}"
    )


# Command-line options
def parse_args():
    p = ArgP(description="Generate CBF/RL figures from log_controls.csv.")
    p.add_argument(
        "--log-controls",
        dest="log",
        type=Path,
        default=ROOT / "log_controls.csv",
        help="Path to the per-step control CSV.",
    )
    p.add_argument(
        "--output-dir",
        dest="out",
        type=Path,
        default=ROOT / "cbf_plots",
        help="Directory where PNG figures will be saved.",
    )
    p.add_argument(
        "--sample-every",
        dest="skip",
        type=int,
        default=200,
        help="Use one point every N rows for scatter plots.",
    )
    p.add_argument(
        "--projection3d-max-samples",
        dest="n_proj",
        type=int,
        default=30,
        help="Max projected samples.",
    )
    p.add_argument(
        "--projection3d-safe-max-samples",
        dest="n_safe",
        type=int,
        default=100,
        help="Max no-CBF samples.",
    )
    p.add_argument(
        "--projection3d-no-cbf-tol",
        dest="tol",
        type=float,
        default=0.1,
        help="No-CBF tol for |u_total - u_prop|.",
    )
    p.add_argument(
        "--projection3d-safe-min-step",
        dest="k0",
        type=int,
        default=100000,
        help="Min step for no-CBF samples.",
    )
    p.add_argument(
        "--control-zoom-steps",
        dest="k_zoom",
        type=int,
        default=25000,
        help="Initial steps in time plot; 0 uses all.",
    )
    p.add_argument(
        "--control-bin-size",
        dest="bin",
        type=int,
        default=200,
        help="Steps per time bin.",
    )
    p.add_argument(
        "--control-smooth-window",
        dest="win",
        type=int,
        default=15,
        help="Rolling window in bins.",
    )
    p.add_argument("--dpi", type=int, default=145, help="PNG output DPI.")
    p.add_argument(
        "--figures",
        dest="figs",
        nargs="+",
        choices=[
            "all",
            "rl-progress",
            "rl-reward",
            "projection3d",
        ],
        default=["all"],
        help="Choose which figures to generate.",
    )
    return p.parse_args()


# Main script
def main():
    args = parse_args()
    if args.skip <= 0:
        raise ValueError("--sample-every must be greater than 0")
    if args.n_proj <= 0:
        raise ValueError("--projection3d-max-samples must be > 0")
    if args.n_safe <= 0:
        raise ValueError("--projection3d-safe-max-samples must be > 0")
    if args.tol < 0:
        raise ValueError("--projection3d-no-cbf-tol must be non-negative")
    if args.k0 < 0:
        raise ValueError("--projection3d-safe-min-step must be non-negative")
    if args.k_zoom < 0:
        raise ValueError("--control-zoom-steps must be non-negative")
    if args.bin <= 0:
        raise ValueError("--control-bin-size must be > 0")
    if args.win <= 0:
        raise ValueError("--control-smooth-window must be > 0")
    if not args.log.exists():
        raise FileNotFoundError(f"log_controls CSV not found: {args.log}")

    selected = set(args.figs)
    if "all" in selected:
        selected = {
            "rl-progress",
            "rl-reward",
            "projection3d",
        }

    ctrl = load_ctrl(args.log)
    need_grid = bool({"rl-progress", "rl-reward", "projection3d"} & selected)
    grids = make_grids() if need_grid else None

    if "rl-progress" in selected:
        plot_rl(
            ctrl,
            grids,
            args.out / "rl_multiview_progress.png",
            args.skip,
            args.dpi,
            "global_step",
        )
    if "rl-reward" in selected:
        plot_rl(
            ctrl,
            grids,
            args.out / "rl_multiview_reward.png",
            args.skip,
            args.dpi,
            "reward",
        )
    if "projection3d" in selected:
        plot_proj(ctrl, grids, args)


if __name__ == "__main__":
    main()
