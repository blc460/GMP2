"""Criterion: point density on the rail running surface vs range and incidence angle.

    python -m trackqc.quality.density --cloud ../pointclouds/track-test.e57 [--out results]

Per segment (only segments the clip covers completely): points per cm^2 of the
running band, mean point spacing and completeness (occupied 5 mm cells).

Model: a scanner with a constant angular step delta puts
    density = cos(incidence) / (range^2 * delta^2)
points per unit area on a surface. k = density * range^2 / cos(incidence) is
fitted (median over the segments) and gives the effective angular step
delta = 1 / sqrt(k), which can be checked against the scan resolution setting.
On a horizontal surface with height difference h, cos(incidence) = h / range,
so density = k h / range^3 (log-log slope -3) and density = k cos^3(inc) / h^2.
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import core
from . import common

NAME = "density"
VALUE = "density_pts_cm2"


def evaluate(ds: common.Dataset, cell=0.005):
    df = common.segment_table(ds, lambda seg: core.density_metrics(seg, cell), keep=lambda seg: seg.full)
    if df.empty:
        raise RuntimeError("no complete segment with a plane fit; check the clip")
    a = ds.settings
    bins = common.bin_table(df, VALUE, a["range_bin"], a["incidence_bin"], extra=lambda g: dict(
        mean_spacing_mm_median=g["mean_spacing_mm"].median(), completeness_median=g["completeness"].median()))

    cos_i = np.cos(np.radians(df["incidence_deg"]))
    k_cm2 = df[VALUE] * df["range_m"] ** 2 / cos_i        # [pts/cm^2 * m^2]
    df["k_pts_cm2_m2"] = k_cm2
    k = float(k_cm2.median())
    delta = 1.0 / np.sqrt(k * 1e4)                         # [rad]; density in pts/m^2 = 1e4 * pts/cm^2
    h = float(df["height_diff_m"].median())
    pr = common.linear_fit(np.log(df["range_m"]), np.log(df[VALUE]))
    pc = common.linear_fit(np.log(cos_i), np.log(df[VALUE]))
    fit = dict(
        n_segments=int(len(df)), n_points=int(df["n"].sum()),
        range_m=[float(df["range_m"].min()), float(df["range_m"].max())],
        incidence_deg=[float(df["incidence_deg"].min()), float(df["incidence_deg"].max())],
        corr_range_incidence=common.range_incidence_correlation(df),
        median_height_diff_m=h,
        model_cos_over_r2=dict(
            formula="density = k * cos(incidence) / range^2",
            k_pts_cm2_m2=k, k_iqr=[float(k_cm2.quantile(0.25)), float(k_cm2.quantile(0.75))],
            angular_step_mrad=delta * 1e3, spacing_at_10m_normal_mm=delta * 10.0 * 1e3,
        ),
        power_law_range=dict(formula="ln(density) = a + b ln(range)", expected_b=-3.0, **pr),
        power_law_cos_incidence=dict(formula="ln(density) = a + b ln(cos(incidence))", **pc),
        summary=dict(median=float(df[VALUE].median()), p05=float(df[VALUE].quantile(0.05)),
                     p95=float(df[VALUE].quantile(0.95)),
                     median_spacing_mm=float(df["mean_spacing_mm"].median())),
    )
    model = {"range": lambda r: k * h / r ** 3,
             "incidence": lambda t: k * np.cos(np.radians(t)) ** 3 / h ** 2}
    common.write_criterion(ds, NAME, df, bins, fit, VALUE, "point density [pts/cm²]", logy=True, model=model)
    return df, bins, fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_dataset_args(ap)
    ap.add_argument("--cell", type=float, default=0.005, help="grid cell for completeness [m]")
    a = ap.parse_args()
    evaluate(common.load_dataset(a), a.cell)


if __name__ == "__main__":
    main()
