"""Shared steps of the quality criteria.

1. Load one dataset (one scanner setup): E57 / PLY / ASCII, either one clip with
   both rails or one clip per rail, and the scanner origin (E57 pose by default).
2. Track axis (reused from --axis, or fitted and saved to <out>/axis.json) and
   running band per rail (`core.detect_running_band`, see <out>/band_profile.png).
3. Running band cut into segments of --seg-len along the rail, one plane fit
   per segment; range and incidence angle per segment (`core.segment_geometry`).
4. Every criterion writes, in <out> = <--out>/<label>/:
     <criterion>.csv        one row per segment: metric + range + incidence
     <criterion>_bins.csv   metric per range bin, per incidence bin and per
                            range x incidence cell
     <criterion>_fit.json   model fits against range and incidence
     <criterion>.png        metric vs range and vs incidence
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import core
from ..process_setup import prepare, plot_profiles


@dataclass
class Dataset:
    label: str
    out: Path
    origin: np.ndarray
    axis: core.TrackAxis
    bands: dict            # side -> RunningBand
    segments: list         # core.Segment of both rails
    geometry: list         # dict per segment (None if no fit): range, incidence, ...
    settings: dict


def add_dataset_args(ap: argparse.ArgumentParser):
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cloud", help="one clip containing both rails (E57/PLY/ASCII)")
    src.add_argument("--left", help="clip of the left rail; needs --right")
    ap.add_argument("--right")
    ap.add_argument("--origin", nargs=3, type=float, default=None,
                    help="scanner origin x y z in the cloud frame (default for E57: pose translation)")
    ap.add_argument("--axis", default=None,
                    help="track axis JSON to reuse (default: fitted from this cloud, saved to <out>/axis.json)")
    ap.add_argument("--label", default=None, help="dataset name (default: file name of the cloud)")
    ap.add_argument("--out", default="results", help="results folder; files go to <out>/<label>/")
    ap.add_argument("--seg-len", type=float, default=0.5, help="segment length along the rail [m]")
    ap.add_argument("--band-width", type=float, default=0.030, help="running band width [m]")
    ap.add_argument("--edge-margin", type=float, default=0.005, help="band start beyond the groove edge [m]")
    ap.add_argument("--range-bin", type=float, default=1.0, help="bin width of the range tables [m]")
    ap.add_argument("--incidence-bin", type=float, default=5.0, help="bin width of the incidence tables [deg]")


def load_dataset(a) -> Dataset:
    if (a.left is None) != (a.right is None):
        raise SystemExit("--left and --right must be given together")
    src = Path(a.cloud or a.left)
    label = a.label or (src.stem if a.cloud else src.stem.removesuffix("_L"))
    out = Path(a.out) / label
    out.mkdir(parents=True, exist_ok=True)
    axis_path = Path(a.axis) if a.axis else out / "axis.json"
    _, _, origin, axis, bands, sq = prepare(a.left, a.right, a.origin, axis_path,
                                            a.band_width, a.edge_margin, a.cloud)
    plot_profiles(bands, out / "band_profile.png", label)
    segments, geometry = [], []
    for side in ("L", "R"):
        P, s, q = sq[side]
        for seg in core.band_segments(P, s, q, bands[side], a.seg_len):
            segments.append(seg)
            geometry.append(core.segment_geometry(seg, origin, axis) if seg.fit is not None else None)
    settings = dict(
        source=str(src), right=a.right, origin=origin.tolist(), axis=str(axis_path),
        seg_len=a.seg_len, band_width=a.band_width, edge_margin=a.edge_margin,
        range_bin=a.range_bin, incidence_bin=a.incidence_bin,
        band={k: dict(q_lo=b.q_lo, q_hi=b.q_hi, groove_edge=b.groove_edge) for k, b in bands.items()},
    )
    return Dataset(label, out, origin, axis, bands, segments, geometry, settings)


def segment_table(ds: Dataset, metric, keep=lambda seg: True) -> pd.DataFrame:
    """One row per fitted segment that passes `keep`: id, geometry and metric(seg)."""
    rows = []
    for seg, geo in zip(ds.segments, ds.geometry):
        if seg.fit is None or not keep(seg):
            continue
        rows.append(dict(label=ds.label, rail=seg.rail, s_from=seg.s_from, s_to=seg.s_to,
                         n_raw=seg.n_raw, n=seg.fit.n, **geo, **metric(seg)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tables
def _edges(x, width):
    return np.arange(np.floor(x.min() / width) * width, x.max() + width, width)


def bin_table(df, value, range_bin, incidence_bin, extra=None) -> pd.DataFrame:
    """Statistics of `value` per range bin, per incidence bin and per range x incidence cell.

    Column `by` = range | incidence | range_x_incidence. Every row also gives the
    median of the other variable, so each value comes with both its distance and
    its incidence angle. `extra(group) -> dict` adds criterion-specific columns.
    """
    d = df.copy()
    r_e, i_e = _edges(d["range_m"], range_bin), _edges(d["incidence_deg"], incidence_bin)
    d["r_bin"] = np.digitize(d["range_m"], r_e) - 1
    d["i_bin"] = np.digitize(d["incidence_deg"], i_e) - 1
    rows = []
    for by, keys in (("range", ["r_bin"]), ("incidence", ["i_bin"]), ("range_x_incidence", ["r_bin", "i_bin"])):
        for k, g in d.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            row = dict(by=by)
            if "r_bin" in keys:
                row.update(range_lo_m=r_e[k[0]], range_hi_m=r_e[k[0]] + range_bin)
            if "i_bin" in keys:
                row.update(incidence_lo_deg=i_e[k[-1]], incidence_hi_deg=i_e[k[-1]] + incidence_bin)
            v = g[value]
            row.update(n_segments=len(g), n_points=int(g["n"].sum()),
                       range_median_m=g["range_m"].median(), incidence_median_deg=g["incidence_deg"].median(),
                       median=v.median(), mean=v.mean(), std=v.std(ddof=1), p25=v.quantile(0.25),
                       p75=v.quantile(0.75), min=v.min(), max=v.max())
            if extra is not None:
                row.update(extra(g))
            rows.append(row)
    cols = ["by", "range_lo_m", "range_hi_m", "incidence_lo_deg", "incidence_hi_deg"]
    t = pd.DataFrame(rows)
    return t[cols + [c for c in t.columns if c not in cols]].rename(
        columns={c: f"{value}_{c}" for c in ("median", "mean", "std", "p25", "p75", "min", "max")})


def linear_fit(x, y):
    """OLS y = a + b x with standard errors and R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    A = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    dof = len(y) - 2
    s2 = r @ r / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return dict(a=float(c[0]), b=float(c[1]), se_a=float(se[0]), se_b=float(se[1]),
                r2=float(1 - r @ r / np.sum((y - y.mean()) ** 2)), n=int(len(y)), rmse=float(np.sqrt(s2)))


def range_incidence_correlation(df):
    """On a horizontal surface cos(incidence) = height difference / range, so range and
    incidence are almost collinear; their separate effects can then not be told apart."""
    return float(np.corrcoef(df["range_m"], df["incidence_deg"])[0, 1])


def write_criterion(ds: Dataset, name, df, bins, fit, value, ylabel, logy=False, model=None):
    """Write <name>.csv, <name>_bins.csv, <name>_fit.json and <name>.png to ds.out.

    model: optional dict {"range": f(r), "incidence": f(theta)} of fitted curves to draw.
    """
    df.to_csv(ds.out / f"{name}.csv", index=False)
    bins.to_csv(ds.out / f"{name}_bins.csv", index=False)
    with open(ds.out / f"{name}_fit.json", "w") as f:
        json.dump(dict(label=ds.label, criterion=name, settings=ds.settings, **fit), f, indent=2)
    plot_criterion(ds, name, df, bins, value, ylabel, logy, model)
    print(f"{name}: {len(df)} segments -> {ds.out / name}.csv, _bins.csv, _fit.json, .png")


def plot_criterion(ds, name, df, bins, value, ylabel, logy, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (x, by, xlabel, cvar, clabel) in zip(axs, (
            ("range_m", "range", "range [m]", "incidence_deg", "incidence [deg]"),
            ("incidence_deg", "incidence", "incidence angle [deg]", "range_m", "range [m]"))):
        for rail, mk in (("L", "o"), ("R", "^")):
            d = df[df["rail"] == rail]
            sc = ax.scatter(d[x], d[value], c=d[cvar], marker=mk, s=16, cmap="viridis",
                            vmin=df[cvar].min(), vmax=df[cvar].max())
            ax.scatter([], [], marker=mk, s=16, c="grey", label=f"segment, rail {rail}")
        b = bins[bins["by"] == by]
        ax.errorbar(b[f"{x.split('_')[0]}_median_{x.split('_')[1]}"], b[f"{value}_median"],
                    yerr=[b[f"{value}_median"] - b[f"{value}_p25"], b[f"{value}_p75"] - b[f"{value}_median"]],
                    fmt="s-", color="tab:red", ms=4, lw=1, capsize=2, label="bin median (IQR)")
        if model and by in model:
            xs = np.linspace(df[x].min(), df[x].max(), 200)
            ax.plot(xs, model[by](xs), "k--", lw=1, label="fit")
        ax.set_xlabel(xlabel)
        fig.colorbar(sc, ax=ax, label=clabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel(ylabel)
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{ds.label}: {name}")
    fig.tight_layout()
    fig.savefig(ds.out / f"{name}.png", dpi=150)
    plt.close(fig)
