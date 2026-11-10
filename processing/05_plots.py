"""Step 5: figures per scanner and station, and per scanner pooled over the stations (three separate plots each).

- density_theta: point density [pts/dm2 = points per 10 x 10 cm] over the incidence angle theta_ground;
  dots = cells, line = median per 10 deg bin, dashed = expected from the scan grid (rho_exp). The density
  of a cell is n / (length * width), both from 2 % to 98 % of its points along and across the rail.
- density_r: the same over the slant range r (5 m bins).
- sigma_r_theta: the cells at their slant range r and angle theta_ground, colour = sigma_z (log scale,
  0.3 to 3 mm, the same in all figures).
In pooled figures the marker shape (and in the density plots the colour) shows the station. Axes are the same in all figures of a rail_class.

Input: results/<sok|lip>/01_segments.csv. Output: results/<sok|lip>/figures/05_<scanner>_<station>_<plot>.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from cells import OUT, cfg

df = pd.read_csv(OUT / "01_segments.csv")
df["density_dm2"] = df["density_pts_m2"] / 100                               # pts per dm2 (10 x 10 cm)
df["expected_dm2"] = df["rho_exp_pts_m2"] / 100
folder = OUT / "figures"
folder.mkdir(exist_ok=True)
EDGES = dict(theta_ground_deg=np.array(cfg["incidence"]), r_m=np.array(cfg["distance"]))
LABELS = dict(theta_ground_deg="incidence angle theta_ground [deg]", r_m="slant range r [m]")
MARKERS = dict(zip(sorted(df["setup"].unique()), "os^D"))
COLOURS = dict(zip(sorted(df["setup"].unique()), ["tab:blue", "tab:orange", "tab:green", "tab:purple"]))
D_LIM = (df["density_dm2"].min() / 2, df["density_dm2"].max() * 2)
NORM = LogNorm(0.3, 3.0)
CLASS = {1: "railhead (SOK)", 2: "lip"}.get(cfg["rail_class"], f"class {cfg['rail_class']}")


def save(fig, name, plot):
    fig.tight_layout()
    fig.savefig(folder / f"05_{name}_{plot}.png", dpi=150)
    plt.close(fig)


def density(g, x, title, name, plot):
    fig, a = plt.subplots(figsize=(6.8, 4.5))
    for st, gs in g.groupby("setup"):
        a.scatter(gs[x], gs["density_dm2"], s=8, alpha=0.35, color=COLOURS[st], marker=MARKERS[st],
                  label=f"cells {st}" if g["setup"].nunique() > 1 else "cells")
    med = g.groupby(pd.cut(g[x], EDGES[x], right=False), observed=True)[[x, "density_dm2", "expected_dm2"]].median()
    a.plot(med[x], med["density_dm2"], "o-", color="navy", lw=2, label="median measured")
    a.plot(med[x], med["expected_dm2"], "k--", label="expected from the scan grid")
    a.set(yscale="log", ylim=D_LIM, xlim=(EDGES[x][0], EDGES[x][-1]), xlabel=LABELS[x],
          ylabel="point density [pts/dm2]", title=title)
    a.secondary_yaxis("right", functions=(lambda v: v * 100, lambda v: v / 100)).set_ylabel("point density [pts/m2]")
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=7)
    save(fig, name, plot)


def sigma(g, title, name):
    fig, a = plt.subplots(figsize=(7, 4.8))
    for st, gs in g.groupby("setup"):
        m = a.scatter(gs["r_m"], gs["theta_ground_deg"], c=gs["sigma_z_mm"], norm=NORM, cmap="viridis", s=14,
                      marker=MARKERS[st], edgecolors="none", label=st)
    cb = fig.colorbar(m, ax=a, label="sigma_z [mm] (log scale)")
    cb.set_ticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3])
    cb.set_ticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
    cb.ax.minorticks_off()
    a.set(xlim=(EDGES["r_m"][0], EDGES["r_m"][-1]), ylim=(EDGES["theta_ground_deg"][0], 90),
          xlabel=LABELS["r_m"], ylabel=LABELS["theta_ground_deg"], title=title)
    a.grid(alpha=0.3)
    if g["setup"].nunique() > 1:
        a.legend(handles=[plt.Line2D([], [], ls="", marker=MARKERS[st], color="0.4", label=st)
                          for st in sorted(g["setup"].unique())], fontsize=7, title="station")
    save(fig, name, "sigma_r_theta")


def name_of(*parts):
    return "_".join(parts).replace("/", "-").replace(" ", "_")


groups = [(f"{sc}, station {st}", name_of(sc, st), g) for (sc, st), g in df.groupby(["scanner", "setup"])]
groups += [(f"{sc}, pooled", name_of(sc, "pooled"), g) for sc, g in df.groupby("scanner")]
for title, name, g in groups:
    title = f"{title}, {CLASS} ({len(g)} cells)"
    density(g, "theta_ground_deg", title, name, "density_theta")
    density(g, "r_m", title, name, "density_r")
    sigma(g, title, name)
print(f"{3 * len(groups)} figures in {folder}")
