"""All plots of the numbered scripts: one colour and marker per scanner, consistent axes and units."""
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


def plot_height_residuals(seg, cross, path, label):
    """Leave-one-out residual e of the rail heights along the track, per rail and for the cross level."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for j, r in enumerate(("L", "R")):
        d = seg[seg["rail"] == r].sort_values("s_mid")
        axs[0].plot(d["s_mid"], d["e_mm"], ".-", ms=4, lw=1, color=COLORS[j], label=f"rail {r}")
        axs[0].fill_between(d["s_mid"], -2 * d["sigma_z_station_mm"], 2 * d["sigma_z_station_mm"],
                            color=COLORS[j], alpha=0.15, lw=0)
    axs[0].set_ylabel("height residual e [mm]")
    axs[0].legend(fontsize=8, title="band: ±2 σ_z,station", title_fontsize=7)
    if len(cross):
        d = cross.sort_values("s_mid")
        axs[1].plot(d["s_mid"], d["e_mm"], ".-", ms=4, lw=1, color=COLORS[2])
        axs[1].fill_between(d["s_mid"], -2 * d["sigma_z_station_mm"], 2 * d["sigma_z_station_mm"],
                            color=COLORS[2], alpha=0.15, lw=0)
    axs[1].set_ylabel("cross level residual e [mm]\n(L − R)")
    axs[1].set_xlabel("station along the track s [m]")
    for ax in axs:
        ax.axhline(0, color="0.3", lw=0.8)
        _style(ax)
    axs[0].set_title(f"{label}: leave-one-out residuals of the rail heights", fontsize=10)
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
            for e in (prof.get("head_lo"), prof.get("head_hi")):
                if e is not None:
                    a.axvline(e * 1e3, color="0.3", lw=1, ls="--")
            if np.isfinite(b.groove_edge):
                a.axvline(b.groove_edge * 1e3, color=COLORS[7], lw=1, label="groove edge")
            _style(a)
        head = f"head {1e3 * p.head_width[side]:.0f} mm, " if np.isfinite(p.head_width[side]) else ""
        axs[0, j].set_title(f"{p.label}: rail {side}, {head}band {1e3 * (b.q_hi - b.q_lo):.0f} mm", fontsize=9)
        axs[0, j].set_ylabel("median height [mm]\n(rel. to rail trend)")
        axs[1, j].set_ylabel("points per 2 mm bin")
        axs[1, j].set_xlabel("lateral offset from rail centre line [mm]")
    if p.band_method == "railhead":
        axs[0, 0].plot([], [], color="0.3", lw=1, ls="--", label="head edge (quantile)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


REQ_STYLES = {"K2": ("0.25", "--"), "K3.6": ("0.25", ":")}


def _requirement_lines(ax, reqs):
    """Horizontal lines of the required sigma_z, reqs = {key: (value_mm, label)}."""
    for key, (v, lab) in reqs.items():
        c, ls = REQ_STYLES.get(key, ("0.25", "-."))
        ax.axhline(v, color=c, ls=ls, lw=1.2, label=lab)


def plot_sigma_z(st, bins, by, path, reqs, title):
    """sigma_z of the rail heights vs distance to the scanner (by = range) or angle of incidence (by = incidence).

    Dots = station-wise sigma_z (1.4826 MAD of the plane residuals times sqrt of the
    cofactor of the height and sqrt(VF)); line with band = empirical sigma_z per bin,
    1.4826 MAD of the leave-one-out height residuals with 95 % CI; horizontal lines =
    requirements.
    """
    x, xmed, xlabel = {"range": ("range_m", "range_median_m", "distance to the scanner [m]"),
                       "incidence": ("incidence_deg", "incidence_median_deg", "angle of incidence [deg]")}[by]
    styles = scanner_styles(st["scanner"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for sc, (color, mk) in styles.items():
        d = st[st["scanner"] == sc]
        n_set = d["setup"].nunique()
        ax.scatter(d[x], d["sigma_z_station_mm"], s=14, marker=mk, color=color, alpha=0.4, lw=0,
                   label=f"{sc}: σ_z,station ({n_set} setup{'s' if n_set > 1 else ''}, {len(d)} heights)")
        b = bins[(bins["scanner"] == sc) & (bins["by"] == by) & (bins["n_used"] >= 3)].sort_values(xmed)
        if len(b):
            ax.fill_between(b[xmed], b["sigma_z_emp_ci_lo_mm"], b["sigma_z_emp_ci_hi_mm"], color=color,
                            alpha=0.15, lw=0)
            ax.plot(b[xmed], b["sigma_z_emp_mm"], "-", marker=mk, color=color, lw=2, ms=8,
                    markeredgecolor="white", markeredgewidth=1, label=f"{sc}: σ_z,emp per bin (95 % CI)")
    _requirement_lines(ax, reqs)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("standard deviation of a rail height σ_z [mm]")
    _style(ax)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_track_geometry(g, path, label, T_vpf, T_n):
    """Rail heights, cant, twist and vertical versine along the track with +-2 sigma bands.

    Dark band = 2 sigma from the station-wise sigma_z, light band = 2 sigma from the
    empirical sigma_z of the scan; red dashed = attention values T.
    """
    g = g.sort_values("s")
    fig, axs = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True)
    s = g["s"].to_numpy()
    H = (0.5 * (g["zL_mm"] + g["zR_mm"])).to_numpy()
    ok = np.isfinite(H)
    trend = np.polyval(np.polyfit(s[ok], H[ok], 1), s) if ok.sum() > 1 else 0.0
    for j, r in enumerate(("L", "R")):
        z = g[f"z{r}_mm"].to_numpy() - trend
        sz = g[f"sigma_z{r}_emp_mm"].to_numpy()
        axs[0].plot(s, z, ".-", ms=4, lw=1, color=COLORS[j], label=f"rail {r}")
        axs[0].fill_between(s, z - 2 * sz, z + 2 * sz, color=COLORS[j], alpha=0.15, lw=0)
    axs[0].set_ylabel("rail height minus\nlinear trend [mm]")
    axs[0].legend(fontsize=8, ncol=2, title="band: ±2 σ_z,emp", title_fontsize=7)
    for ax, name, unit, T, ylab in ((axs[1], "cant", "mm", None, "cant ü [mm]"),
                                    (axs[2], "twist", "permille", T_n, "twist N [‰]\n(base 5 m)"),
                                    (axs[3], "vpf", "mm", T_vpf, "vertical versine vPf [mm]\n(chord 10 m)")):
        v = g[f"{name}_{unit}"].to_numpy()
        for src, alpha, lab in (("emp", 0.15, "±2σ (empirical σ_z)"), ("station", 0.35, "±2σ (station σ_z)")):
            sg = g[f"sigma_{name}_{src}_{unit}"].to_numpy()
            ax.fill_between(s, v - 2 * sg, v + 2 * sg, color=COLORS[2], alpha=alpha, lw=0, label=lab)
        ax.plot(s, v, ".-", ms=4, lw=1, color=COLORS[2])
        if T is not None:
            for t in (-T, T):
                ax.axhline(t, color=COLORS[7], ls="--", lw=1)
            ax.plot([], [], color=COLORS[7], ls="--", lw=1,
                    label=f"attention value ±{T:g} {'‰' if unit == 'permille' else unit}"
                          + (" (applies to the change)" if name == "vpf" else ""))
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axs:
        _style(ax)
    axs[-1].set_xlabel("station along the track s [m]")
    axs[0].set_title(f"{label}: track geometry from the rail heights", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_limits(d, curves, reqs, path, scanner):
    """Station-wise sigma_z vs distance and vs incidence with the fitted model and the limits."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    color, mk = COLORS[0], MARKERS[0]
    for ax, (by, x, xlabel) in zip(axs, (("range", "range_m", "distance to the scanner [m]"),
                                        ("incidence", "incidence_deg", "angle of incidence [deg]"))):
        c = curves[by]
        ax.axvspan(d[x].min(), d[x].max(), color="0.5", alpha=0.1, lw=0, label="observed range")
        ax.scatter(d[x], d["sigma_z_station_mm"], s=12, marker=mk, color=color, alpha=0.35, lw=0,
                   label="σ_z,station")
        ax.plot(c["x"], c["model"], "--", color=color, lw=1.2, label="fit to σ_z,station")
        ax.plot(c["x"], c["total"], "-", color=COLORS[1], lw=1.8, label="fit ⊕ σ_sys")
        for key, xm in c["limits"].items():
            if xm is not None and np.isfinite(xm):
                ax.axvline(xm, color="0.25", ls=REQ_STYLES.get(key, ("", "-."))[1], lw=1)
        _requirement_lines(ax, reqs)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        _style(ax)
    axs[0].set_ylim(0.5 * d["sigma_z_station_mm"].min(), 10 * max(v for v, _ in reqs.values()))
    axs[0].set_ylabel("σ_z [mm]")
    axs[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle(f"{scanner}: operational limits (vertical lines: largest value that meets the requirement)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
