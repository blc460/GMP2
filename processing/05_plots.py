"""Step 5: figures per scanner and station, and per scanner pooled over the stations (marker = station).

- density_theta / density_r: point density [pts/m2] over the incidence angle / slant range, right axis = mean
  point spacing 1000 / sqrt(density) [mm]; dots = cells, line = median per bin, dashed = expected from the scan grid.
- sigma_r_theta: the cells at their r and theta_ground, colour = sigma_z (log scale 0.3 to 3 mm in all figures).
- sigma_z_r_and_theta: sigma_z over r and over theta_ground, one dot per cell (linear axis, no median line).
- sigma_Z_<method>_r_and_theta: sigma_Z of the cell height over r and theta_ground, one dot per cell, with the
  required values (`requirements`) as horizontal lines; method = sqrt_n, bins or boot (see precision.py).
Axes are the same in all figures of a rail_class.

Input: results/<sok|lip>/01_segments.csv. Output: results/<sok|lip>/figures/05_<scanner>_<station>_<plot>.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from cells import OUT, cfg

FOLDER = OUT / "figures"
EDGES = dict(theta_ground_deg=np.array(cfg["incidence"]), r_m=np.array(cfg["distance"]))
LABELS = dict(theta_ground_deg=r"incidence angle $\theta$ [deg]", r_m="slant range $r$ [m]")
STYLE = {"S1": ("tab:blue", "o"), "S2": ("tab:orange", "s"), "S3": ("tab:green", "^"), "S4": ("tab:purple", "D")}
METHODS = {"sqrt_n": r"$\sigma_Z = \sigma_z/\sqrt{n}$", "bins": rf"$\sigma_Z$ from {cfg['n_bins']} bins ($s/\sqrt{{{cfg['n_bins']}}}$)",
           "boot": r"$\sigma_Z$ from block bootstrap"}
NORM = LogNorm(0.3, 3.0)


def save(fig, name, plot):
    fig.tight_layout()
    fig.savefig(FOLDER / f"05_{name}_{plot}.png", dpi=200)
    plt.close(fig)


def scatter_cells(a, g, x, y, ymax=None):
    """One dot per cell, colour and marker per station; values above ymax as open triangles at the top edge."""
    for st, gs in g.groupby("setup"):
        colour, marker = STYLE.get(st, ("0.4", "o"))
        val = gs[y] if ymax is None else gs[y].clip(upper=ymax)
        a.scatter(gs[x], val, s=9, alpha=0.45, color=colour, marker=marker, linewidths=0, label=f"cells {st}")
        if ymax is not None and (gs[y] > ymax).any():
            over = gs[gs[y] > ymax]
            a.scatter(over[x], np.full(len(over), 0.985 * ymax), marker="^", s=22, facecolors="none",
                      edgecolors=colour, linewidths=0.8)


def density(g, x, name, plot):
    fig, a = plt.subplots(figsize=(6.8, 4.5))
    scatter_cells(a, g, x, "density_pts_m2")
    med = g.groupby(pd.cut(g[x], EDGES[x], right=False), observed=True)[[x, "density_pts_m2", "rho_exp_pts_m2"]].median()
    a.plot(med[x], med["density_pts_m2"], "o-", color="navy", lw=2, label="median measured")
    a.plot(med[x], med["rho_exp_pts_m2"], "k--", label="expected from the scan grid")
    a.set(yscale="log", ylim=D_LIM, xlim=(EDGES[x][0], EDGES[x][-1]), xlabel=LABELS[x], ylabel="point density [pts/m2]")
    sec = a.secondary_yaxis("right", functions=(lambda v: 1000 / np.sqrt(np.maximum(v, 1e-9)),
                                                lambda s: 1e6 / np.square(np.maximum(s, 1e-9))))
    sec.set_ylabel("mean point spacing [mm]")
    sec.set_yticks([1, 2, 3, 5, 10, 20, 30, 50], ["1", "2", "3", "5", "10", "20", "30", "50"])
    sec.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=7)
    save(fig, name, plot)


def sigma_map(g, name):
    fig, a = plt.subplots(figsize=(7, 4.8))
    for st, gs in g.groupby("setup"):
        m = a.scatter(gs["r_m"], gs["theta_ground_deg"], c=gs["sigma_z_mm"], norm=NORM, cmap="viridis", s=14,
                      marker=STYLE.get(st, ("", "o"))[1], edgecolors="none")
    cb = fig.colorbar(m, ax=a, label=r"$\sigma_z$ [mm] (log scale)")
    cb.set_ticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3], labels=["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
    cb.ax.minorticks_off()
    a.set(xlim=(EDGES["r_m"][0], EDGES["r_m"][-1]), ylim=(0, 90), xlabel=LABELS["r_m"], ylabel=LABELS["theta_ground_deg"])
    a.grid(alpha=0.3)
    if g["setup"].nunique() > 1:
        a.legend(handles=[plt.Line2D([], [], ls="", marker=STYLE.get(st, ("", "o"))[1], color="0.4", label=st)
                          for st in sorted(g["setup"].unique())], fontsize=7, title="station")
    save(fig, name, "sigma_r_theta")


def r_and_theta(g, y, ylabel, ymax, name, plot, requirements=()):
    """y over r (left) and theta_ground (right), one dot per cell."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, x in zip(ax, ("r_m", "theta_ground_deg")):
        for req, ls in zip(requirements, ("--", ":", "-.")):
            a.axhline(req, color="0.25", ls=ls, lw=1.1, label=f"requirement {req} mm")
        scatter_cells(a, g, x, y, ymax)
        a.set(xlim=(EDGES[x][0], EDGES[x][-1]), ylim=(0, ymax), xlabel=LABELS[x])
        a.grid(alpha=0.3)
    ax[0].set_ylabel(ylabel)
    ax[0].legend(fontsize=7, loc="upper left", markerscale=1.5)
    above = int((g[y] > ymax).sum())
    if above:
        ax[0].text(0.98, 0.03, f"{above} cell{'s' if above > 1 else ''} above {ymax:g} mm (open triangles at top)",
                   transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7)
    save(fig, name, plot)


def name_of(*parts):
    return "_".join(parts).replace("/", "-").replace(" ", "_")


if __name__ == "__main__":
    df = pd.read_csv(OUT / "01_segments.csv")
    FOLDER.mkdir(exist_ok=True)
    D_LIM = (df["density_pts_m2"].min() / 2, df["density_pts_m2"].max() * 2)
    groups = [(name_of(sc, st), g) for (sc, st), g in df.groupby(["scanner", "setup"])]
    groups += [(name_of(sc, "pooled"), g) for sc, g in df.groupby("scanner")]
    for name, g in groups:
        density(g, "theta_ground_deg", name, "density_theta")
        density(g, "r_m", name, "density_r")
        sigma_map(g, name)
        r_and_theta(g, "sigma_z_mm", r"$\sigma_z$ [mm]", cfg["sigma_z_max"], name, "sigma_z_r_and_theta")
        for method, label in METHODS.items():
            r_and_theta(g, f"sigma_Z_{method}_mm", f"{label} [mm]", cfg["sigma_Z_max"], name,
                        f"sigma_Z_{method}_r_and_theta", cfg["requirements"])
    print(f"{7 * len(groups)} figures in {FOLDER}")
