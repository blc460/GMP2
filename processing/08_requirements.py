"""Step 8: core result. How far out, under which angle and with which scanner is the required sigma_Z reached?

Usage: python processing/08_requirements.py

Per scanner (all stations together), per class of incidence angle theta_ground (`incidence`) and slant range r
(`distance`), from sigma_Z_bins of the cells (02):
- cells, median sigma_Z_bins and its 95 % interval (bootstrap over the cells of the class, `table_boot_runs`),
  90 % quantile, median sigma_z;
- a class meets a requirement (`requirements`, 0.67 and 1.2 mm) if it has at least `min_cells` cells and the upper
  limit of the 95 % interval of the median is at or below the requirement.
- r_max per scanner, angle class and requirement: the outer edge of the last range class that meets it, walking
  outwards from the scanner over the classes with at least `min_cells` cells and stopping at the first that fails.
  limited_by = "sigma_Z" (a class failed: the requirement limits the range) or "data" (every class with enough
  cells passed: the measured ranges limit r_max, the scanner may reach further).
The same is done for each rail side (near / far / single, see 02) and each beam direction to the rail
(longitudinal / oblique / perpendicular), and for all angles together ("all").

sigma_Z_bins is the random part of the uncertainty of one cell height within one scan. Errors common to a whole
scan (levelling, scanner calibration and position) are not included, so r_max is an upper limit of the usable range.

Input: results/<sok|lip>/02_cells.csv.
Output: results/<sok|lip>/08_requirements_table.csv (all classes), 08_r_max.csv, 08_<scanner>.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cells import OUT, cfg

RNG = np.random.default_rng(2026)
Y = "sigma_Z_bins_mm"
SPLITS = {"all": None, "rail_side": "rail_side", "direction": "direction"}


def median_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = np.median(x[RNG.integers(0, len(x), size=(cfg["table_boot_runs"], len(x)))], axis=1)
    return tuple(np.quantile(m, [(1 - level) / 2, (1 + level) / 2]))


def classes(df, split):
    """Table per scanner x split x theta class (incl. 'all') x r class."""
    d = df.assign(split=df[split] if split else "all")
    d = pd.concat([d, d.assign(theta_bin="all")])
    rows = []
    for (sc, sp, th, rb), g in d.groupby(["scanner", "split", "theta_bin", "r_bin"], observed=True, sort=True):
        y = g[Y].dropna()
        lo, hi = median_ci(y)
        row = dict(scanner=sc, split_by=split or "all", split=sp, theta_bin=th, r_bin=rb, r_hi=rb.right,
                   cells=len(y), stations="+".join(sorted(g["setup"].unique())),
                   sigma_Z_median_mm=y.median(), sigma_Z_ci_lo_mm=lo, sigma_Z_ci_hi_mm=hi,
                   sigma_Z_q90_mm=y.quantile(0.9) if len(y) else np.nan, sigma_z_median_mm=g["sigma_z_mm"].median())
        for req in cfg["requirements"]:
            row[f"meets_{req}"] = bool(len(y) >= cfg["min_cells"] and hi <= req)
        rows.append(row)
    return pd.DataFrame(rows)


def r_max(t):
    """Outer edge of the last range class that meets the requirement, walking outwards from the scanner over the
    classes with at least min_cells cells (classes with fewer cells are skipped); the walk stops at the first such
    class that fails. limited_by: 'sigma_Z' (a class failed) or 'data' (all classes with enough cells passed)."""
    rows = []
    for key, g in t.groupby(["scanner", "split_by", "split", "theta_bin"], sort=False):
        g = g[g["cells"] >= cfg["min_cells"]].sort_values("r_hi")
        row = dict(zip(["scanner", "split_by", "split", "theta_bin"], key),
                   cells=int(t.loc[g.index, "cells"].sum()) if len(g) else 0, classes=len(g),
                   r_data_min_m=g["r_hi"].min() - np.diff(cfg["distance"]).min() if len(g) else np.nan, r_data_max_m=g["r_hi"].max())
        for req in cfg["requirements"]:
            ok = g[f"meets_{req}"].to_numpy()
            n = len(ok) if ok.all() else int(np.argmin(ok))
            row[f"r_max_{req}_m"] = g["r_hi"].iloc[n - 1] if n else np.nan
            row[f"limited_by_{req}"] = "no data" if not len(g) else ("data" if ok.all() else "sigma_Z")
        rows.append(row)
    return pd.DataFrame(rows)


def figure(df, rm, sc):
    d = df[df["scanner"] == sc]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    markers = dict(zip(cfg["direction_names"], "o^s"))
    colours = {"near": "tab:blue", "far": "tab:red", "single": "0.5"}
    for a, x in zip(ax, ("r_m", "theta_ground_deg")):
        for (side, dirn), g in d.groupby(["rail_side", "direction"]):
            a.scatter(g[x], g[Y].clip(upper=cfg["sigma_Z_max"]), s=8, alpha=0.4, linewidths=0,
                      color=colours.get(side, "k"), marker=markers.get(dirn, "o"), label=f"{side}, {dirn}")
        edges = np.array(cfg["distance"] if x == "r_m" else cfg["incidence"], float)
        grp = d.groupby(pd.cut(d[x], edges, right=False), observed=True)[Y]
        med = grp.median()
        ci = grp.apply(lambda s: median_ci(s))
        mid = [iv.mid for iv in med.index]
        a.plot(mid, med.to_numpy(), "k-o", ms=3, lw=1.5, label="median")
        a.fill_between(mid, [c[0] for c in ci], [c[1] for c in ci], color="k", alpha=0.15, label="95 % of median")
        for req, ls in zip(cfg["requirements"], ("--", ":")):
            a.axhline(req, color="0.2", ls=ls, lw=1, label=f"requirement {req} mm")
        a.set(xlim=(edges[0], edges[-1]), ylim=(0, cfg["sigma_Z_max"]),
              xlabel="slant range $r$ [m]" if x == "r_m" else r"incidence angle $\theta$ [deg]")
        a.grid(alpha=0.3)
    r = rm[(rm["scanner"] == sc) & (rm["split_by"] == "all") & (rm["theta_bin"] == "all")].iloc[0]
    for req, ls in zip(cfg["requirements"], ("--", ":")):
        ax[0].axvline(r[f"r_max_{req}_m"], color="g", ls=ls, lw=1.2,
                      label=f"r_max {req} mm = {r[f'r_max_{req}_m']:g} m ({r[f'limited_by_{req}']})")
    ax[0].set_ylabel(r"$\sigma_Z$ from bins [mm]")
    ax[0].legend(fontsize=6, ncol=2, loc="upper left")
    ax[0].set_title(f"{sc}: {d[Y].notna().sum()} cells, stations {'+'.join(sorted(d['setup'].unique()))}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"08_{sc.replace('/', '-').replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "02_cells.csv")
    df["r_bin"] = pd.cut(df["r_m"], cfg["distance"], right=False)
    df["theta_bin"] = pd.cut(df["theta_ground_deg"], cfg["incidence"], right=False).astype(str)
    t = pd.concat([classes(df, split) for split in SPLITS.values()], ignore_index=True)
    rm = r_max(t)
    t.assign(r_bin=t["r_bin"].astype(str)).round(4).to_csv(OUT / "08_requirements_table.csv", index=False)
    rm.round(3).to_csv(OUT / "08_r_max.csv", index=False)
    show = rm[rm["split_by"] == "all"].drop(columns=["split_by", "split"])
    print(show.to_string(index=False))
    for sc in sorted(df["scanner"].unique()):
        figure(df, rm, sc)
