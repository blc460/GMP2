"""The six figures of the plan. Colour = scanner, marker = platform (tripod / bridge)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import add_bins, pass_shares, passes

MARKERS = {"tripod": "o", "bridge": "^"}


def _colours(df):
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {sc: cyc[i % len(cyc)] for i, sc in enumerate(sorted(df["scanner"].unique()))}


def _scatter_with_median(ax, df, x, y, bins, col):
    for sc, g in df.groupby("scanner"):
        for plat, gp in g.groupby("platform"):
            ax.scatter(gp[x], gp[y], s=10, alpha=0.5, color=col[sc], marker=MARKERS.get(plat, "s"),
                       label=f"{sc}, {plat}")
        b = pd.cut(g[x], bins, right=False)
        med = g.groupby(b, observed=True)[[x, y]].median()
        ax.plot(med[x], med[y], color=col[sc], lw=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def _save(fig, folder, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(folder) / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def make_all(df, lim, cfg, folder):
    for strip, d in df.dropna(subset=["sigma_z_mm"]).groupby("strip"):
        suffix = "" if strip == "sok" else f"_{strip}"
        l = lim[lim["strip"] == strip] if len(lim) else lim
        point_noise(d, cfg, folder, suffix)
        height_precision(d, l, cfg, folder, suffix)
        density(d, cfg, folder, suffix)
        pass_share(d, cfg, folder, suffix)
        limit_map(d, cfg, folder, suffix)
        along_track(d, folder, suffix)


def point_noise(df, cfg, folder, suffix=""):
    col = _colours(df)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _scatter_with_median(ax[0], df, "d_m", "sigma_z_mm", cfg["bins"]["distance"], col)
    _scatter_with_median(ax[1], df, "theta_deg", "sigma_z_mm", cfg["bins"]["incidence"], col)
    ax[0].set(xlabel="horizontal distance d [m]", ylabel="point noise sigma_z (MAD) [mm]")
    ax[1].set(xlabel="incidence angle theta [deg]", xlim=(40, 90))
    ax[0].legend(fontsize=8)
    fig.suptitle("Point noise per 0.5 m segment")
    _save(fig, folder, f"fig1_point_noise{suffix}")


def height_precision(df, lim, cfg, folder, suffix=""):
    col = _colours(df)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _scatter_with_median(ax[0], df, "d_m", "sigma_zbar_mm", cfg["bins"]["distance"], col)
    _scatter_with_median(ax[1], df, "theta_deg", "sigma_zbar_mm", cfg["bins"]["incidence"], col)
    for a, qty in ((ax[0], "d_max_m"), (ax[1], "theta_max_deg")):
        for c in cfg["limits"]["criteria"]:
            a.axhline(c["sigma_req_mm"], color="0.4", ls="--", lw=0.8)
            if a is ax[0]:
                a.annotate(c["name"], (0.01, c["sigma_req_mm"]), xycoords=("axes fraction", "data"),
                           fontsize=7, va="bottom", color="0.3")
            if len(lim):
                sel = lim[(lim["criterion"] == c["name"]) & (lim["quantity"] == qty)]
                for _, r in sel.iterrows():
                    if np.isfinite(r["limit"]):
                        a.plot(r["limit"], c["sigma_req_mm"], marker="v", ms=9, color=col.get(r["scanner"], "k"),
                               mec="k", zorder=5)
    ax[0].set(xlabel="horizontal distance d [m]", ylabel="rail height precision sigma_z/sqrt(n) [mm]")
    ax[1].set(xlabel="incidence angle theta [deg]", xlim=(40, 90))
    ax[0].legend(fontsize=8)
    fig.suptitle("Rail height precision per segment, criteria (dashed) and limits (triangles)")
    _save(fig, folder, f"fig2_height_precision{suffix}")


def density(df, cfg, folder, suffix=""):
    col = _colours(df)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _scatter_with_median(ax, df, "d_m", "n", cfg["bins"]["distance"], col)
    ax.axhline(cfg["segments"]["min_points"], color="0.3", ls="--", lw=1)
    ax.annotate("n_min", (0.01, cfg["segments"]["min_points"]), xycoords=("axes fraction", "data"),
                fontsize=8, va="bottom")
    ax.set(xlabel="horizontal distance d [m]", ylabel="points per segment n")
    ax.legend(fontsize=8)
    _save(fig, folder, f"fig3_density{suffix}")


def pass_share(df, cfg, folder, suffix=""):
    col = _colours(df)
    crit = cfg["limits"]["criteria"]
    fig, ax = plt.subplots(len(crit), 2, figsize=(11, 2.6 * len(crit)), sharey=True, squeeze=False)
    for j, by in enumerate(("d_bin", "theta_bin")):
        sh = pass_shares(df, cfg, by)
        for i, c in enumerate(crit):
            a = ax[i, j]
            for sc, g in sh[sh["criterion"] == c["name"]].groupby("scanner"):
                g = g.groupby(by, observed=True)["pass_share"].mean()
                a.plot([iv.mid for iv in g.index], g.to_numpy(), "o-", color=col[sc], label=sc)
            a.axhline(cfg["limits"]["p_pass"], color="0.3", ls="--", lw=1)
            a.set(ylim=(-0.05, 1.05), title=f"{c['name']} ({c['sigma_req_mm']} mm)")
            a.grid(alpha=0.3)
        ax[-1, j].set_xlabel("horizontal distance d [m]" if by == "d_bin" else "incidence angle theta [deg]")
    for a in ax[:, 0]:
        a.set_ylabel("share passing")
    ax[0, 0].legend(fontsize=8)
    _save(fig, folder, f"fig4_pass_share{suffix}")


def limit_map(df, cfg, folder, suffix=""):
    df = passes(df, cfg)
    crit = cfg["limits"]["criteria"]
    scanners = sorted(df["scanner"].unique())
    fig, ax = plt.subplots(len(crit), len(scanners), figsize=(4.5 * len(scanners), 2.6 * len(crit)),
                           sharex=True, sharey=True, squeeze=False)
    for i, c in enumerate(crit):
        for j, sc in enumerate(scanners):
            g = df[df["scanner"] == sc]
            ok = g[f"pass_{c['name']}"]
            a = ax[i, j]
            a.scatter(g.loc[ok, "d_m"], g.loc[ok, "h_m"], s=10, color="tab:green", label="pass")
            a.scatter(g.loc[~ok, "d_m"], g.loc[~ok, "h_m"], s=10, color="tab:red", marker="x", label="fail")
            a.set_title(f"{sc}: {c['name']}", fontsize=9)
            a.grid(alpha=0.3)
    for a in ax[-1, :]:
        a.set_xlabel("horizontal distance d [m]")
    for a in ax[:, 0]:
        a.set_ylabel("scanner height above rail h [m]")
    ax[0, 0].legend(fontsize=8)
    _save(fig, folder, f"fig5_limit_map{suffix}")


def along_track(df, folder, suffix=""):
    """One row per dataset and rail."""
    keys = sorted(df.groupby(["dataset", "rail"]).groups)
    fig, ax = plt.subplots(len(keys), 2, figsize=(11, 2.2 * len(keys)), squeeze=False)
    for i, (lab, rail) in enumerate(keys):
        g = df[(df["dataset"] == lab) & (df["rail"] == rail)].sort_values("s_mid")
        ax[i, 0].plot(g["s_mid"], g["sigma_z_mm"], ".-")
        ax[i, 1].plot(g["s_mid"], g["n"], ".-")
        for a in ax[i]:
            a.axvline(g["scanner_s_m"].iloc[0], color="k", lw=1, ls=":")
            a.grid(alpha=0.3)
        ax[i, 0].set(ylabel="sigma_z [mm]", title=f"{lab}, rail {rail}")
        ax[i, 1].set(ylabel="points per segment", title=f"{lab}, rail {rail}")
    for a in ax[-1]:
        a.set_xlabel("station s [m] (dotted: closest to the scanner)")
    _save(fig, folder, f"fig6_along_track{suffix}")


def rail_plan(pts, rails, junctions, origin, label, folder):
    """Plan view of the detected rails (centrelines, ids), junctions and the scanner."""
    rng = np.random.default_rng(0)
    i = rng.choice(len(pts), min(len(pts), 200_000), replace=False)
    dx, dy = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    fig, ax = plt.subplots(figsize=(14, min(14, max(3, 14 * dy / max(dx, 1e-9)) + 1)))
    ax.scatter(pts[i, 0], pts[i, 1], s=0.2, color="0.75", rasterized=True)
    for r in rails:
        ax.plot(r.line[:, 0], r.line[:, 1], lw=1.2)
        m = r.line[len(r.line) // 2]
        ax.annotate(r.id, m, fontsize=8, ha="center", va="bottom")
    if len(junctions):
        ax.plot(junctions[:, 0], junctions[:, 1], "rx", ms=8, label="junction")
    ax.plot(origin[0], origin[1], "k*", ms=10, label="scanner")
    ax.set(aspect="equal", xlabel="x [m]", ylabel="y [m]", title=f"{label}: {len(rails)} rails, {len(junctions)} junctions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(folder) / f"fig0_rails_{label}.png", dpi=200)
    plt.close(fig)
