"""Criterion: point-cloud noise on the rail running surface vs range and incidence angle.

    python -m trackqc.quality.noise --cloud ../pointclouds/track-test.e57 [--out results]

Per segment (at least --min-points inliers): scatter of the points about a
least-squares plane with 3 sigma (MAD) outlier rejection, i.e. the noise normal
to the running surface. Columns: sigma_pc with 95 % chi-square CI, robust
sigma from the MAD, outlier share and the block variance factor (>> 1 means
correlated residuals or surface shape, so sigma/sqrt(n) is too optimistic).

Per bin the pooled sigma sqrt(sum((n-3) sigma^2) / sum(n-3)) with its CI is the
best single noise value for that range or incidence. Fits: linear in range,
linear in incidence, and a power law in range.
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import core
from . import common

NAME = "noise"
VALUE = "sigma_pc_mm"


def pooled(g):
    dof = (g["n"] - 3).to_numpy()
    s = float(np.sqrt(np.sum(dof * g[VALUE] ** 2) / dof.sum()))
    lo, hi = core.sigma_ci(s, dof.sum())
    return dict(sigma_pooled_mm=s, sigma_pooled_ci_lo_mm=lo, sigma_pooled_ci_hi_mm=hi,
                sigma_mad_mm_median=g["sigma_mad_mm"].median(), outlier_share_median=g["outlier_share"].median(),
                var_factor_median=g["var_factor"].median())


def evaluate(ds: common.Dataset, min_points=30):
    df = common.segment_table(ds, core.noise_metrics, keep=lambda seg: seg.fit.n >= min_points)
    if df.empty:
        raise RuntimeError(f"no segment with at least {min_points} points; check the clip")
    a = ds.settings
    bins = common.bin_table(df, VALUE, a["range_bin"], a["incidence_bin"], extra=pooled)
    lr = common.linear_fit(df["range_m"], df[VALUE])
    li = common.linear_fit(df["incidence_deg"], df[VALUE])
    pr = common.linear_fit(np.log(df["range_m"]), np.log(df[VALUE]))
    fit = dict(
        n_segments=int(len(df)), n_points=int(df["n"].sum()), min_points=min_points,
        range_m=[float(df["range_m"].min()), float(df["range_m"].max())],
        incidence_deg=[float(df["incidence_deg"].min()), float(df["incidence_deg"].max())],
        corr_range_incidence=common.range_incidence_correlation(df),
        linear_range=dict(formula="sigma_pc_mm = a + b * range_m", **lr),
        linear_incidence=dict(formula="sigma_pc_mm = a + b * incidence_deg", **li),
        power_law_range=dict(formula="ln(sigma_pc_mm) = a + b ln(range_m)", **pr),
        per_rail={r: pooled(g) for r, g in df.groupby("rail")},
        summary=dict(**pooled(df), median=float(df[VALUE].median()),
                     p05=float(df[VALUE].quantile(0.05)), p95=float(df[VALUE].quantile(0.95))),
    )
    model = {"range": lambda r: lr["a"] + lr["b"] * r, "incidence": lambda t: li["a"] + li["b"] * t}
    common.write_criterion(ds, NAME, df, bins, fit, VALUE, "point noise σ_pc [mm]", model=model)
    return df, bins, fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_dataset_args(ap)
    ap.add_argument("--min-points", type=int, default=30, help="minimum inliers per segment")
    a = ap.parse_args()
    evaluate(common.load_dataset(a), a.min_points)


if __name__ == "__main__":
    main()
