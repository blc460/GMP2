"""Plots: metric vs measurement distance and vs incidence angle per scanner, railhead check."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fixed categorical order (colour-blind checked palette); scanners are sorted by name,
# so a scanner keeps its colour as long as the set of scanners is the same
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]


def scanner_styles(scanners):
    names = sorted(set(scanners))
    if len(names) > len(COLORS):
        raise ValueError(f"{len(names)} scanners; at most {len(COLORS)} can be told apart, use --scanner")
    return {s: (COLORS[i], MARKERS[i]) for i, s in enumerate(names)}


def _style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)


def plot_vs_range_incidence(df, bins, value, ylabel, path, title, models=None, logy=False):
    """Two panels: value vs range and vs incidence angle, one colour and marker per scanner.

    Dots = segments, line with band = bin median and interquartile range,
    dashed = fitted model `models[scanner][by](x)` (by = range | incidence).
    """
    styles = scanner_styles(df["scanner"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (x, by, xlabel, xmed) in zip(axs, (
            ("range_m", "range", "measurement distance [m]", "range_median_m"),
            ("incidence_deg", "incidence", "angle of incidence [deg]", "incidence_median_deg"))):
        for sc, (color, mk) in styles.items():
            d = df[df["scanner"] == sc]
            n_set = d["setup"].nunique()
            ax.scatter(d[x], d[value], s=12, marker=mk, color=color, alpha=0.35, lw=0,
                       label=f"{sc} ({n_set} setup{'s' if n_set > 1 else ''}, {len(d)} segments)")
            b = bins[(bins["scanner"] == sc) & (bins["by"] == by)].sort_values(xmed)
            ax.fill_between(b[xmed], b[f"{value}_p25"], b[f"{value}_p75"], color=color, alpha=0.15, lw=0)
            ax.plot(b[xmed], b[f"{value}_median"], "-", marker=mk, color=color, lw=2, ms=8,
                    markeredgecolor="white", markeredgewidth=1)
            if models and sc in models and by in models[sc]:
                xs = np.linspace(d[x].min(), d[x].max(), 200)
                ax.plot(xs, models[sc][by](xs), "--", color=color, lw=1.2)
        ax.set_xlabel(xlabel)
        if logy:
            ax.set_yscale("log")
        _style(ax)
    axs[0].set_ylabel(ylabel)
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="grey", lw=2, marker="o", markeredgecolor="white"),
          plt.Line2D([], [], color="grey", lw=1.2, ls="--")]
    l += ["bin median (band: IQR)", "fit"]
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08 + 0.04 * (len(l) > 4), 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_height_consistency(df, bins, path, title):
    """Two panels: empirical vs formal sigma_z vs range and vs incidence angle, one colour per scanner.

    Dots = |e| of the segments, solid line with band = bin RMS of e (empirical
    sigma_z) with 95 % CI, dashed = bin RMS of the formal sigma_z times sqrt(VF).
    """
    styles = scanner_styles(df["scanner"])
    floor = 0.3 * np.nanmin(bins["sigma_formal_vf_mm"])   # |e| near 0 would stretch the log axis
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (x, by, xlabel, xmed) in zip(axs, (
            ("range_m", "range", "measurement distance [m]", "range_median_m"),
            ("incidence_deg", "incidence", "angle of incidence [deg]", "incidence_median_deg"))):
        for sc, (color, mk) in styles.items():
            d = df[df["scanner"] == sc]
            n_set = d["setup"].nunique()
            ax.scatter(d[x], d["abs_e_mm"].clip(lower=floor), s=12, marker=mk, color=color, alpha=0.35, lw=0,
                       label=f"{sc} ({n_set} setup{'s' if n_set > 1 else ''}, {len(d)} segments)")
            b = bins[(bins["scanner"] == sc) & (bins["by"] == by) & (bins["n_used"] >= 2)].sort_values(xmed)
            ax.fill_between(b[xmed], b["sigma_emp_ci_lo_mm"], b["sigma_emp_ci_hi_mm"], color=color, alpha=0.15,
                            lw=0)
            ax.plot(b[xmed], b["sigma_emp_mm"], "-", marker=mk, color=color, lw=2, ms=8,
                    markeredgecolor="white", markeredgewidth=1)
            ax.plot(b[xmed], b["sigma_formal_vf_mm"], "--", color=color, lw=1.2)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        _style(ax)
    axs[0].set_ylabel("rail height error [mm]")
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="grey", lw=2, marker="o", markeredgecolor="white"),
          plt.Line2D([], [], color="grey", lw=1.2, ls="--")]
    l += ["empirical σ_z per bin (band: 95 % CI)", "formal σ_z · √VF"]
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08 + 0.04 * (len(l) > 4), 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_height_profile(seg, cross, path, label):
    """Normalised leave-one-out residual e along the track per rail and for the cross level."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for j, r in enumerate(("L", "R")):
        d = seg[seg["rail"] == r].sort_values("s_mid")
        axs[0].plot(d["s_mid"], d["e_mm"], ".-", ms=4, lw=1, color=COLORS[j], label=f"rail {r}")
        axs[0].fill_between(d["s_mid"], -2 * d["sigma_formal_vf_mm"], 2 * d["sigma_formal_vf_mm"],
                            color=COLORS[j], alpha=0.15, lw=0)
    axs[0].set_ylabel("height residual e [mm]")
    axs[0].legend(fontsize=8, title="band: ±2 formal σ_z · √VF", title_fontsize=7)
    if len(cross):
        d = cross.sort_values("s_mid")
        axs[1].plot(d["s_mid"], d["e_mm"], ".-", ms=4, lw=1, color=COLORS[2])
        axs[1].fill_between(d["s_mid"], -2 * d["sigma_formal_vf_mm"], 2 * d["sigma_formal_vf_mm"],
                            color=COLORS[2], alpha=0.15, lw=0)
    axs[1].set_ylabel("cross level residual e [mm]\n(L − R)")
    axs[1].set_xlabel("station along the track s [m]")
    for ax in axs:
        ax.axhline(0, color="0.3", lw=0.8)
        _style(ax)
    axs[0].set_title(f"{label}: leave-one-out residuals of the segment heights", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_railhead_check(p, path):
    """Lateral profile of both railheads with the measured head edges and the band used."""
    fig, axs = plt.subplots(2, 2, figsize=(9, 5), sharex="col")
    for j, side in enumerate(("L", "R")):
        b = p.bands[side]
        prof = b.profile
        qmm = np.array(prof["q"]) * 1e3
        axs[0, j].plot(qmm, np.array(prof["median_dz"]) * 1e3, ".-", ms=3, color=COLORS[0])
        axs[1, j].bar(qmm, prof["count"], width=1.8, color=COLORS[0])
        for a in axs[:, j]:
            a.axvspan(b.q_lo * 1e3, b.q_hi * 1e3, color=COLORS[2], alpha=0.2, label="band used")
            for e in (prof["head_lo"], prof["head_hi"]):
                a.axvline(e * 1e3, color="0.3", lw=1, ls="--")
            _style(a)
        axs[0, j].set_title(f"{p.label}: rail {side}, head {1e3 * p.head_width[side]:.0f} mm, "
                            f"band {1e3 * (b.q_hi - b.q_lo):.0f} mm", fontsize=9)
        axs[0, j].set_ylabel("median height [mm]\n(rel. to rail trend)")
        axs[1, j].set_ylabel("points per 2 mm bin")
        axs[1, j].set_xlabel("lateral offset from rail centre line [mm]")
    axs[0, 0].plot([], [], color="0.3", lw=1, ls="--", label="head edge (quantile)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
