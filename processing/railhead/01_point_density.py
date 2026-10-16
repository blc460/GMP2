"""01 - Point density on the railhead per scanner, vs measurement distance and angle of incidence.

    python railhead/01_point_density.py [--scanner FARO RTC360] [--seg-len 0.5] [--cell 0.005]

Per segment (only segments the rail clip covers completely): inlier points per
cm^2 of the railhead band, mean point spacing 1/sqrt(density) and completeness
(share of occupied --cell grid cells), with range and incidence angle.

Model: a scanner with a constant angular step delta puts
    density = cos(incidence) / (range^2 * delta^2)
points per unit area on a surface. k = density * range^2 / cos(incidence) is
estimated per scanner (median over the segments) and gives the effective
angular step delta = 1 / sqrt(k), to be checked against the resolution setting.
On a horizontal railhead cos(incidence) = h / range for a height difference h,
so with one instrument height density ~ h / range^3 (log-log slope -3).
Fitted per scanner: ln(density) vs ln(range) and vs ln(cos(incidence)).

Writes to <results>/01_point_density/:
  density_segments.csv   one row per segment
  density_bins.csv       per scanner and range bin / incidence bin / range x incidence cell
  density_summary.csv    one row per scanner
  density_fit.json       models and fits per scanner, settings
  density.png            density vs range and vs incidence, one colour per scanner
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared import config, data, plots, segments, stats
from trackqc import core

NAME = "density"
VALUE = "density_pts_cm2"


def scanner_fit(g):
    cos_i = np.cos(np.radians(g["incidence_deg"]))
    k = g["k_pts_cm2_m2"]
    km = float(k.median())
    delta = 1.0 / np.sqrt(km * 1e4)                       # k in pts/m^2 * m^2 -> rad
    return dict(
        **stats.coverage(g),
        model_cos_over_r2=dict(formula="density = k * cos(incidence) / range^2",
                               k_pts_cm2_m2=km, k_iqr=[float(k.quantile(0.25)), float(k.quantile(0.75))],
                               angular_step_mrad=delta * 1e3, spacing_at_10m_normal_mm=delta * 10.0 * 1e3),
        power_law_range=dict(formula="ln(density) = a + b ln(range)", expected_b_fixed_height=-3.0,
                             **stats.linear_fit(np.log(g["range_m"]), np.log(g[VALUE]))),
        power_law_cos_incidence=dict(formula="ln(density) = a + b ln(cos(incidence))",
                                     **stats.linear_fit(np.log(cos_i), np.log(g[VALUE]))),
        summary=dict(median=float(g[VALUE].median()), p05=float(g[VALUE].quantile(0.05)),
                     p95=float(g[VALUE].quantile(0.95)), median_spacing_mm=float(g["mean_spacing_mm"].median()),
                     median_completeness=float(g["completeness"].median())),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_analysis_args(ap)
    ap.add_argument("--cell", type=float, default=0.005, help="grid cell for completeness [m]")
    a = ap.parse_args()
    preps = data.load_prepared(a.results, a.scanner)
    out = Path(a.results) / "01_point_density"

    df = segments.segment_table(preps, lambda seg: core.density_metrics(seg, a.cell), a.seg_len,
                                keep=lambda seg: seg.full)
    if df.empty:
        raise SystemExit("no complete segment with a plane fit; check the prepared scans")
    df["k_pts_cm2_m2"] = df[VALUE] * df["range_m"] ** 2 / np.cos(np.radians(df["incidence_deg"]))
    bins = stats.bin_table(df, VALUE, a.range_bin, a.incidence_bin, extra=lambda g: dict(
        mean_spacing_mm_median=g["mean_spacing_mm"].median(), completeness_median=g["completeness"].median()))

    fits, summary, models = {}, [], {}
    for sc, g in df.groupby("scanner"):
        f = fits[sc] = scanner_fit(g)
        pr, pc = f["power_law_range"], f["power_law_cos_incidence"]
        models[sc] = {"range": lambda r, pr=pr: np.exp(pr["a"]) * r ** pr["b"],
                      "incidence": lambda t, pc=pc: np.exp(pc["a"]) * np.cos(np.radians(t)) ** pc["b"]}
        summary.append(dict(scanner=sc, n_setups=f["n_setups"], n_segments=f["n_segments"],
                            range_min_m=f["range_m"][0], range_max_m=f["range_m"][1],
                            incidence_min_deg=f["incidence_deg"][0], incidence_max_deg=f["incidence_deg"][1],
                            density_median_pts_cm2=f["summary"]["median"], density_p05_pts_cm2=f["summary"]["p05"],
                            density_p95_pts_cm2=f["summary"]["p95"],
                            spacing_median_mm=f["summary"]["median_spacing_mm"],
                            completeness_median=f["summary"]["median_completeness"],
                            k_pts_cm2_m2=f["model_cos_over_r2"]["k_pts_cm2_m2"],
                            angular_step_mrad=f["model_cos_over_r2"]["angular_step_mrad"],
                            slope_log_range=pr["b"], slope_log_range_se=pr["se_b"]))
    settings = dict(results=str(a.results), scans=[p.label for p in preps], seg_len=a.seg_len, cell=a.cell,
                    range_bin=a.range_bin, incidence_bin=a.incidence_bin)
    stats.write_results(out, NAME, df, bins, fits, summary, settings)
    plots.plot_vs_range_incidence(df, bins, VALUE, "point density [pts/cm²]", out / f"{NAME}.png",
                                  "Point density on the railhead", models, logy=True)


if __name__ == "__main__":
    main()
