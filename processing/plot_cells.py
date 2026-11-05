"""Optional: figure that explains how the cells along the rails are made and how switches are dropped.

Usage: python processing/plot_cells.py Bu_RTC-2   (any part of the file name of one row of datasets.csv)
Output: results/cells_explained.png and .pdf. Not part of the pipeline; the zoom windows below fit RTC-2.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from cells import HERE, _axis, cells, cfg, level, read, width

SWITCH = (17.0, 37.0, 3.4, 10.4)       # zoom windows (x0, x1, y0, y1) in the levelled frame [m]
CURVE = (8.5, 19.0, -7.6, 3.6)

datasets = pd.read_csv(HERE / cfg["datasets"], keep_default_na=False)
ds = datasets[datasets["path"].str.contains(sys.argv[1], regex=False)].iloc[0]
_, p0, origin0 = read(ds)
p, origin, tilt = level(p0, origin0)
rail, piece, cell = cells(p)

# status per cell as in 01_segments.py: kept, dropped (switch) or too few points
order = np.argsort(cell)
status, w_cell = {}, {}
for idx in np.split(order, np.flatnonzero(np.diff(cell[order])) + 1):
    c = cell[idx[0]]
    if c < 0:
        continue
    if len(idx) < cfg["min_points"]:
        status[c] = "few"
        continue
    w_cell[c] = width(p[idx, :2])
    status[c] = "drop" if w_cell[c] > cfg["max_width"] else "keep"
st = np.array([status.get(c, "few") for c in cell])
rng = np.random.default_rng(0)
tab = plt.get_cmap("tab10")


def sample(box, k=60000):
    x0, x1, y0, y1 = box
    i = np.flatnonzero((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1))
    return rng.choice(i, min(len(i), k), replace=False)


def cell_plot(a, box, title):
    i = sample(box)
    colour = np.where(cell[i] % 2 == 0, "tab:blue", "tab:orange")
    colour = np.where(st[i] == "drop", "red", np.where(st[i] == "few", "0.75", colour))
    a.scatter(p[i, 0], p[i, 1], c=colour, s=1.5, linewidths=0)
    for q in np.unique(piece[i]):                       # PCA direction of every piece with its centre in view
        ip = np.flatnonzero(piece == q)
        ax_, c = _axis(p[ip, :2]), p[ip, :2].mean(axis=0)
        if len(ip) < 100 or not (box[0] < c[0] < box[1] and box[2] < c[1] < box[3]):
            continue
        a.plot(*np.array([c - 0.8 * ax_, c + 0.8 * ax_]).T, color="k", lw=1.2)
        a.plot(*c, "ko", ms=2.5)
    a.set(xlim=box[:2], ylim=box[2:], xlabel="x [m]", ylabel="y [m]", title=title, aspect="equal")


fig = plt.figure(figsize=(12, 15))
gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 1.25, 1.25, 0.9])

# (a) overview: rail groups
a = fig.add_subplot(gs[0, :])
i = rng.choice(len(p), min(len(p), 150000), replace=False)
a.scatter(p[i, 0], p[i, 1], c=tab(rail[i] % 10), s=0.5, linewidths=0)
a.plot(*origin[:2], "k*", ms=12, label="scanner")
for box, name in ((SWITCH, "b"), (CURVE, "c")):
    a.add_patch(plt.Rectangle(box[::2], box[1] - box[0], box[3] - box[2], fill=False, ls="--", lw=1))
    a.text(box[0], box[3], f" ({name})", va="bottom")
a.set(xlabel="x [m]", ylabel="y [m]", aspect="equal",
      title=f"(a) Levelled railhead points (tilt {tilt:.2f} deg), colour = rail group "
            f"(points closer than {cfg['rail_gap']} m)")
a.legend(loc="lower left", fontsize=8)

legend = [Patch(color="tab:blue", label="cell (0.5 m), alternating"), Patch(color="tab:orange", label="next cell"),
          Patch(color="red", label=f"dropped: wider than {cfg['max_width']} m (switch)"),
          Patch(color="0.75", label=f"skipped: < {cfg['min_points']} points")]
cell_plot(fig.add_subplot(gs[1, :]), SWITCH,
          f"(b) Switch area: {cfg['window']:.0f} m pieces with their PCA direction (black), cells along each piece")
fig.axes[-1].legend(handles=legend + [plt.Line2D([], [], color="k", marker="o", ms=2.5,
                                                 label="PCA direction at the piece centre")],
                    loc="upper left", fontsize=8, ncol=3)
cell_plot(fig.add_subplot(gs[2, 0]), CURVE, "(c) Curve: followed piece by piece")

# (d) two cells across the rail: a normal one and a dropped one
a = fig.add_subplot(gs[2, 1])
kept = [c for c in w_cell if status[c] == "keep" and abs(w_cell[c] - 0.042) < 0.002]
drop = [c for c in w_cell if status[c] == "drop" and 0.15 < w_cell[c] < 0.35]
for c, colour, dy in ((kept[len(kept) // 2], "tab:blue", 0.0), (drop[len(drop) // 2], "red", 0.6)):
    xy = p[cell == c, :2]
    ax_ = _axis(xy)
    s, t = (xy - xy.mean(axis=0)) @ ax_, (xy - xy.mean(axis=0)) @ np.array([-ax_[1], ax_[0]])
    lo, hi = np.quantile(t, [0.02, 0.98])
    a.scatter(s + dy, t, s=2, color=colour, linewidths=0)
    for v in (lo, hi):
        a.plot([dy - 0.27, dy + 0.27], [v, v], color="k", lw=0.8)
    a.text(dy, hi + 0.01, f"width {hi - lo:.3f} m", ha="center", va="bottom", fontsize=8)
    a.text(dy, -0.29, "kept" if colour == "tab:blue" else "dropped (switch)", ha="center", fontsize=9, color=colour)
a.axhspan(-cfg["max_width"] / 2, cfg["max_width"] / 2, color="0.9", zorder=0)
a.text(0.3, cfg["max_width"] / 2, f" max_width {cfg['max_width']} m", fontsize=8, va="bottom", ha="center")
a.set(xlim=(-0.35, 0.95), ylim=(-0.32, 0.32), aspect="equal", xlabel="along the cell s [m]",
      ylabel="across the cell t [m]", title="(d) Two cells in their own PCA frame,\nwidth = 2 to 98 % of t")

# (e) all cell widths
a = fig.add_subplot(gs[3, :])
w = np.array(list(w_cell.values()))
a.hist(np.clip(w, 0, 0.6), bins=np.arange(0, 0.61, 0.01), color="tab:blue")
a.axvline(cfg["max_width"], color="red", ls="--", label=f"max_width = {cfg['max_width']} m")
a.set(yscale="log", xlabel="cell width across the rail [m]", ylabel="cells",
      title=f"(e) Width of all cells: {np.sum(w <= cfg['max_width'])} kept, {np.sum(w > cfg['max_width'])} dropped "
            f"(single railhead: median {np.median(w):.3f} m)")
a.legend(fontsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / cfg["results"] / f"cells_explained.{ext}", dpi=200)
print(f"results/cells_explained.png: {len(w_cell)} cells, {np.sum(w > cfg['max_width'])} dropped")
