"""Step 7: Allan deviation of the height residuals along the rail: are the bins of a cell independent?

Usage: python processing/07_allan.py

The 4 bins per cell (sigma_Z_bins) and the bootstrap blocks (sigma_Z_boot) assume that the noise is uncorrelated
beyond the bin / block length. This is checked along whole 5 m pieces of rail (more data than one 0.5 m cell):

1. Per piece (cells.py; only points of kept cells, no switches): residuals about the rail surface
   z = a + b u + c u^2 + d v + e v^2 (u along, v across; `allan_u_order` = 2 allows a vertical curve).
2. For each bin length tau (`allan_tau`): cut the piece into bins of tau along the rail, take the mean residual per
   bin and the Allan variance  sigma_A^2(tau) = 1/2 < (mean_{k+1} - mean_k)^2 >  over adjacent bins
   (Allan 1966; non-overlapping). Pooled over all pieces of a scanner and station.
3. Reference for uncorrelated (white) noise: var(e) / n_bin, i.e. sigma_A falls with tau^-1/2.
   ratio = sigma_A / white. ratio ~ 1: the bin means are independent at this tau; ratio > 1: correlated noise (or
   rail geometry that is not in the surface model) at this scale.

Read: the bins of a cell are fine if the ratio is ~1 at tau = cell / n_bins (0.125 m); blocks of the bootstrap
should be at least as long as the tau where the ratio stops growing. At tau close to the piece length the
parabola along the rail absorbs part of the variation, so the largest tau are biased low.

Input: point clouds (datasets.csv). Output: results/<sok|lip>/07_allan.csv and 07_allan.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cells import OUT, along_across, cells, cfg, datasets, group_keys, iter_cells, level, read
from precision import allan, residuals

TAU = np.array(cfg["allan_tau"], float)


def process(ds):
    las, p, origin = read(ds)
    p, origin, _ = level(p, origin)
    _, piece, cell = cells(p)
    keep = np.zeros(len(p), bool)
    for c in iter_cells(p, cell):
        if c.status == "keep":
            keep[c.idx] = True
    a_sum, a_cnt, w_sum, pieces = np.zeros(len(TAU)), np.zeros(len(TAU)), np.zeros(len(TAU)), 0
    for q in np.unique(piece[keep]):
        idx = np.flatnonzero(keep & (piece == q))
        if len(idx) < 100:
            continue
        u, v = along_across(p[idx, :2])
        e = residuals(u, v, p[idx, 2], cfg["surface"], cfg["outlier_k"], cfg["allan_u_order"])
        a, n, w = allan(u, e, TAU)
        a_sum, a_cnt, w_sum, pieces = a_sum + a, a_cnt + n, w_sum + w, pieces + 1
    return a_sum, a_cnt, w_sum, pieces


def table():
    acc = {}
    for ds in datasets():
        key = tuple(ds[k] for k in group_keys())
        r = process(ds)
        acc[key] = [x + y for x, y in zip(acc.get(key, [0, 0, 0, 0]), r)]
        print(f"{ds['path']}: {r[3]} pieces")
    rows = []
    for key, (a, n, w, pieces) in acc.items():
        for i, tau in enumerate(TAU):
            if n[i] == 0:
                continue
            rows.append(dict(zip(group_keys(), key), tau_m=tau, pieces=pieces, differences=int(n[i]),
                             allan_dev_mm=1000 * np.sqrt(a[i] / n[i]), white_mm=1000 * np.sqrt(w[i] / n[i])))
    t = pd.DataFrame(rows)
    t["ratio"] = t["allan_dev_mm"] / t["white_mm"]
    return t


def figure(t):
    keys = group_keys()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (key, g) in enumerate(t.groupby(keys)):
        label = ", ".join(key) if isinstance(key, tuple) else key
        colour = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]
        ax[0].loglog(g["tau_m"], g["allan_dev_mm"], "o-", color=colour, ms=3, label=label)
        ax[0].loglog(g["tau_m"], g["white_mm"], ":", color=colour, lw=0.8)
        ax[1].semilogx(g["tau_m"], g["ratio"], "o-", color=colour, ms=3, label=label)
    bin_len = cfg["cell"] / cfg["n_bins"]
    for a in ax:
        a.axvline(bin_len, color="0.3", ls="--", lw=1)
        a.axvline(cfg["boot_block"], color="0.3", ls="-.", lw=1)
        a.set_xlabel(r"bin length $\tau$ [m]")
        a.grid(alpha=0.3, which="both")
    ax[0].set_ylabel(r"Allan deviation $\sigma_A(\tau)$ [mm]  (dotted: white noise)")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_ylabel(r"$\sigma_A$ / white noise")
    ax[1].text(bin_len, ax[1].get_ylim()[1], " bin", va="top", fontsize=7)
    ax[1].text(cfg["boot_block"], ax[1].get_ylim()[1], " block", va="top", fontsize=7, ha="right")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "07_allan.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t = table()
    t.round(4).to_csv(OUT / "07_allan.csv", index=False)
    print(t.round(3).to_string(index=False))
    figure(t)
