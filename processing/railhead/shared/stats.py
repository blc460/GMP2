"""Bin tables and fits per scanner, and writing the result files of a criterion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _edges(x, width):
    return np.arange(np.floor(x.min() / width) * width, x.max() + width, width)


def bin_table(df, value, range_bin, incidence_bin, extra=None) -> pd.DataFrame:
    """Statistics of `value` per scanner and range bin, incidence bin and range x incidence cell.

    Column `by` = range | incidence | range_x_incidence. The bin edges are the
    same for all scanners, so the rows can be compared directly. Every row also
    gives the median of the other variable. `extra(group) -> dict` adds
    criterion-specific columns.
    """
    d = df.copy()
    r_e, i_e = _edges(d["range_m"], range_bin), _edges(d["incidence_deg"], incidence_bin)
    d["r_bin"] = np.digitize(d["range_m"], r_e) - 1
    d["i_bin"] = np.digitize(d["incidence_deg"], i_e) - 1
    rows = []
    for by, keys in (("range", ["r_bin"]), ("incidence", ["i_bin"]), ("range_x_incidence", ["r_bin", "i_bin"])):
        for k, g in d.groupby(["scanner"] + keys):
            row = dict(scanner=k[0], by=by)
            if "r_bin" in keys:
                row.update(range_lo_m=r_e[k[1]], range_hi_m=r_e[k[1]] + range_bin)
            if "i_bin" in keys:
                row.update(incidence_lo_deg=i_e[k[-1]], incidence_hi_deg=i_e[k[-1]] + incidence_bin)
            v = g[value]
            row.update(n_setups=g["setup"].nunique(), n_segments=len(g), n_points=int(g["n"].sum()),
                       range_median_m=g["range_m"].median(), incidence_median_deg=g["incidence_deg"].median(),
                       median=v.median(), mean=v.mean(), std=v.std(ddof=1), p25=v.quantile(0.25),
                       p75=v.quantile(0.75), min=v.min(), max=v.max())
            if extra is not None:
                row.update(extra(g))
            rows.append(row)
    cols = ["scanner", "by", "range_lo_m", "range_hi_m", "incidence_lo_deg", "incidence_hi_deg"]
    t = pd.DataFrame(rows)
    t = t[[c for c in cols if c in t] + [c for c in t.columns if c not in cols]]
    return t.rename(columns={c: f"{value}_{c}" for c in ("median", "mean", "std", "p25", "p75", "min", "max")})


def linear_fit(x, y):
    """OLS y = a + b x with standard errors and R^2.

    The standard errors treat segments as independent; segments of one setup
    are not (same instrument, atmosphere, registration), so read them together
    with the number of setups.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(a=np.nan, b=np.nan, se_a=np.nan, se_b=np.nan, r2=np.nan, n=int(len(x)), rmse=np.nan)
    A = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = r @ r / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return dict(a=float(c[0]), b=float(c[1]), se_a=float(se[0]), se_b=float(se[1]),
                r2=float(1 - r @ r / np.sum((y - y.mean()) ** 2)), n=int(len(y)), rmse=float(np.sqrt(s2)))


def coverage(g):
    """Range and incidence span of one scanner and their correlation.

    On a horizontal railhead cos(incidence) = height difference / range, so for
    a fixed instrument height range and incidence are almost collinear and
    their effects cannot be separated (corr close to 1).
    """
    return dict(n_setups=int(g["setup"].nunique()), setups=sorted(g["setup"].unique().tolist()),
                n_segments=int(len(g)), n_points=int(g["n"].sum()),
                range_m=[float(g["range_m"].min()), float(g["range_m"].max())],
                incidence_deg=[float(g["incidence_deg"].min()), float(g["incidence_deg"].max())],
                corr_range_incidence=float(np.corrcoef(g["range_m"], g["incidence_deg"])[0, 1])
                if len(g) > 2 else np.nan)


def write_results(out, name, df, bins, fits, summary, settings):
    """<out>/<name>_segments.csv, _bins.csv, _summary.csv (one row per scanner), _fit.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_segments.csv", index=False)
    bins.to_csv(out / f"{name}_bins.csv", index=False)
    pd.DataFrame(summary).to_csv(out / f"{name}_summary.csv", index=False)
    with open(out / f"{name}_fit.json", "w") as f:
        json.dump(dict(criterion=name, settings=settings, scanners=fits), f, indent=2, default=float)
    print(f"{name}: {len(df)} segments of {df['label'].nunique()} scans, {df['scanner'].nunique()} scanner(s) "
          f"-> {out}")
