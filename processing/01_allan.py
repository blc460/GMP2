"""Step 1: Allan deviation of the heights along the rail, and from it the number of bins per cell (Allan rule).

Usage: python processing/01_allan.py

sigma_Z_bins (02) assumes that the bin heights of a cell are independent, i.e. that the bins are at least as long as
the correlation of the noise. The correlation length is read from the Allan deviation along whole 5 m pieces of
rail (more data than one 0.5 m cell):

1. Per piece (cells.py; only points of kept cells, no switches): the levelled heights as they are (no model).
2. For each bin length tau (`allan_tau`): cut the piece into bins of tau along the rail, take the mean height per
   bin and the Allan variance  sigma_A^2(tau) = 1/2 < (mean_{k+1} - mean_k)^2 >  over adjacent bins
   (Allan 1966; non-overlapping). Pooled over all pieces of a scanner and station, and of a scanner.
3. Reference for uncorrelated (white) noise: sigma_z^2 / n_bin (sigma_z of the cells, as in 02), i.e. sigma_A falls
   with tau^-1/2. ratio = sigma_A / white: ~1 = the bin means are independent at this tau; > 1 = correlated noise
   or rail geometry (gradient, cant, crown; not removed) at this scale. A slope b adds b tau / sqrt(2).
4. Allan rule, per scanner (pooled over its stations): tau0 = the largest tau for which the ratio stays at or below
   `allan_max_ratio` (1.2) for all tau from `allan_min_tau` (3 cm) up to tau0. Shorter tau are not used: bins of
   1 to 2 cm are shorter than the spacing of the scan lines at flat angles and the reference is unreliable there.
   Number of bins m = floor(cell / tau0), limited to [min_bins, max_bins]; the bins are then >= tau0 long.

Input: point clouds (datasets.csv).
Output: results/<sok|lip>/01_allan.csv (curve per scanner and station, and per scanner as station "all"),
        01_bins.csv (tau0, m and bin length per scanner, read by 02) and 01_allan.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cells import OUT, along_across, cells, cfg, datasets, group_keys, iter_cells, level, read
from precision import allan, mad_sigma

TAU = np.array(cfg["allan_tau"], float)


def process(ds):
    las, p, origin = read(ds)
    p, origin, _ = level(p, origin)
    _, piece, cell = cells(p)
    keep = np.zeros(len(p), bool)
    var = np.full(len(p), np.nan)                         # sigma_z^2 of the point's cell
    for c in iter_cells(p, cell):
        if c.status == "keep":
            keep[c.idx] = True
            var[c.idx] = mad_sigma(p[c.idx, 2]) ** 2
    a_sum, a_cnt, w_sum, pieces = np.zeros(len(TAU)), np.zeros(len(TAU)), np.zeros(len(TAU)), 0
    for q in np.unique(piece[keep]):
        idx = np.flatnonzero(keep & (piece == q))
        if len(idx) < 100:
            continue
        u, _ = along_across(p[idx, :2])
        a, n, w = allan(u, p[idx, 2], TAU, np.mean(var[idx]))
        a_sum, a_cnt, w_sum, pieces = a_sum + a, a_cnt + n, w_sum + w, pieces + 1
    return a_sum, a_cnt, w_sum, pieces


def rows_of(key, keys, acc):
    a, n, w, pieces = acc
    out = []
    for i, tau in enumerate(TAU):
        if n[i] > 0:
            out.append(dict(zip(keys, key), tau_m=tau, pieces=pieces, differences=int(n[i]),
                            allan_dev_mm=1000 * np.sqrt(a[i] / n[i]), white_mm=1000 * np.sqrt(w[i] / n[i])))
    return out


def table():
    keys = group_keys()
    acc, per_scanner = {}, {}
    for ds in datasets():
        r = process(ds)
        key = tuple(ds[k] for k in keys)
        acc[key] = [x + y for x, y in zip(acc.get(key, [0, 0, 0, 0]), r)]
        per_scanner[ds["scanner"]] = [x + y for x, y in zip(per_scanner.get(ds["scanner"], [0, 0, 0, 0]), r)]
        print(f"{ds['path']}: {r[3]} pieces")
    rows = [row for key, x in acc.items() for row in rows_of(key, keys, x)]
    rows += [dict(row, **{k: "all" for k in ["setup"] + keys[1:]})
             for sc, x in per_scanner.items() for row in rows_of((sc,), ["scanner"], x)]
    t = pd.DataFrame(rows)
    t["ratio"] = t["allan_dev_mm"] / t["white_mm"]
    return t


def allan_rule(t):
    """tau0 and the number of bins per scanner (rows of station 'all'), see module doc."""
    rows = []
    for sc, g in t[t["setup"] == "all"].groupby("scanner"):
        g = g[g["tau_m"] >= cfg["allan_min_tau"] - 1e-9].sort_values("tau_m")
        bad = np.flatnonzero(g["ratio"].to_numpy() > cfg["allan_max_ratio"])
        i = (bad[0] - 1) if len(bad) else len(g) - 1
        tau0 = g["tau_m"].iloc[max(i, 0)]
        m = int(np.clip(np.floor(cfg["cell"] / tau0 + 1e-9), cfg["min_bins"], cfg["max_bins"]))
        rows.append(dict(scanner=sc, tau0_m=tau0, ratio_at_tau0=g["ratio"].iloc[max(i, 0)],
                         first_tau_above_m=g["tau_m"].iloc[bad[0]] if len(bad) else np.nan,
                         n_bins=m, bin_length_m=cfg["cell"] / m))
    return pd.DataFrame(rows)


def figure(t, rule):
    keys = group_keys()
    scanners = sorted(t["scanner"].unique())
    fig, ax = plt.subplots(2, len(scanners), figsize=(4.6 * len(scanners), 7.5), squeeze=False, sharex=True)
    for j, sc in enumerate(scanners):
        for i, (key, g) in enumerate(t[t["scanner"] == sc].groupby(keys)):
            label = ", ".join(key[1:]) if isinstance(key, tuple) else key
            pooled = "all" in key
            style = dict(color="k", lw=2, ms=3) if pooled else dict(color=f"C{i}", lw=1, ms=2)
            ax[0, j].loglog(g["tau_m"], g["allan_dev_mm"], "o-", label=label, **style)
            if pooled:
                ax[0, j].loglog(g["tau_m"], g["white_mm"], "k:", lw=1, label="white noise (all)")
            ax[1, j].semilogx(g["tau_m"], g["ratio"], "o-", label=label, **style)
        r = rule[rule["scanner"] == sc].iloc[0]
        for a in ax[:, j]:
            a.axvline(r["tau0_m"], color="r", ls="--", lw=1)
            a.axvline(r["bin_length_m"], color="r", ls=":", lw=1)
            a.grid(alpha=0.3, which="both")
        ax[1, j].axhline(1, color="k", lw=0.8)
        ax[1, j].axhline(cfg["allan_max_ratio"], color="r", lw=0.8)
        ax[0, j].set_title(f"{sc}: tau0 = {100 * r['tau0_m']:.1f} cm, m = {r['n_bins']} bins "
                           f"({100 * r['bin_length_m']:.1f} cm)", fontsize=9)
        ax[1, j].set_xlabel(r"bin length $\tau$ [m]")
        ax[0, j].legend(fontsize=6)
    ax[0, 0].set_ylabel(r"Allan deviation $\sigma_A(\tau)$ [mm]")
    ax[1, 0].set_ylabel(r"$\sigma_A$ / white noise")
    fig.tight_layout()
    fig.savefig(OUT / "01_allan.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t = table()
    t.round(4).to_csv(OUT / "01_allan.csv", index=False)
    rule = allan_rule(t)
    rule.round(4).to_csv(OUT / "01_bins.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(rule.round(3).to_string(index=False))
    figure(t, rule)
