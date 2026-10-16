"""02 - Point-cloud noise on the railhead per scanner, vs measurement distance and angle of incidence.

    python railhead/02_point_noise.py [--scanner FARO RTC360] [--seg-len 0.5] [--min-points 30]

Per segment (at least --min-points inliers): scatter of the points about a
least-squares plane with 3 sigma (MAD) outlier rejection, i.e. the noise normal
to the railhead, sigma_pc, with its 95 % chi-square CI, the robust sigma from
the MAD, the outlier share and the block variance factor (>> 1: correlated
residuals or surface shape, so sigma/sqrt(n) is too optimistic).

sigma_range_equiv = sigma_pc / cos(incidence) is the range noise that would
produce sigma_pc if all noise were along the beam; it is the value to compare
with the manufacturer's range noise. At grazing incidence the footprint and
angular noise also contribute, so read it as an upper bound there.

Per scanner and bin the pooled sigma sqrt(sum((n-3) sigma^2) / sum(n-3)) with
its CI is the best single noise value. Fits per scanner: sigma_pc as power law
of range and of cos(incidence) (exponent 1 for pure beam noise), and
sigma_range_equiv linear in range (the form of a range-noise specification)
and in incidence (a remaining slope = incidence effect beyond the geometry).

Writes to <results>/02_point_noise/:
  noise_segments.csv   one row per segment
  noise_bins.csv       per scanner and range bin / incidence bin / range x incidence cell
  noise_summary.csv    one row per scanner (pooled sigma and CI)
  noise_fit.json       fits per scanner, settings
  noise.png            sigma_pc vs range and vs incidence, one colour per scanner
  noise_range_equiv.png  the same for sigma_range_equiv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared import config, data, plots, segments, stats
from trackqc import core

NAME = "noise"
VALUE = "sigma_pc_mm"


def pooled(g, value=VALUE):
    dof = (g["n"] - 3).to_numpy()
    s = float(np.sqrt(np.sum(dof * g[value] ** 2) / dof.sum()))
    lo, hi = core.sigma_ci(s, dof.sum())
    return dict(sigma_pooled_mm=s, sigma_pooled_ci_lo_mm=float(lo), sigma_pooled_ci_hi_mm=float(hi),
                sigma_range_equiv_pooled_mm=float(np.sqrt(np.sum(dof * g["sigma_range_equiv_mm"] ** 2) / dof.sum())),
                sigma_mad_mm_median=float(g["sigma_mad_mm"].median()),
                outlier_share_median=float(g["outlier_share"].median()),
                var_factor_median=float(g["var_factor"].median()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_analysis_args(ap)
    ap.add_argument("--min-points", type=int, default=30, help="minimum inliers per segment")
    a = ap.parse_args()
    preps = data.load_prepared(a.results, a.scanner)
    out = Path(a.results) / "02_point_noise"

    df = segments.segment_table(preps, core.noise_metrics, a.seg_len, keep=lambda seg: seg.fit.n >= a.min_points)
    if df.empty:
        raise SystemExit(f"no segment with at least {a.min_points} points; check the prepared scans")
    df["sigma_range_equiv_mm"] = df[VALUE] / np.cos(np.radians(df["incidence_deg"]))
    bins = stats.bin_table(df, VALUE, a.range_bin, a.incidence_bin, extra=pooled)

    fits, summary, models, models_eq = {}, [], {}, {}
    for sc, g in df.groupby("scanner"):
        cos_i = np.cos(np.radians(g["incidence_deg"]))
        pr = stats.linear_fit(np.log(g["range_m"]), np.log(g[VALUE]))
        pc = stats.linear_fit(np.log(cos_i), np.log(g[VALUE]))
        er = stats.linear_fit(g["range_m"], g["sigma_range_equiv_mm"])
        ei = stats.linear_fit(g["incidence_deg"], g["sigma_range_equiv_mm"])
        fits[sc] = dict(**stats.coverage(g),
                        power_law_range=dict(formula="ln(sigma_pc_mm) = a + b ln(range_m)", **pr),
                        power_law_cos_incidence=dict(formula="ln(sigma_pc_mm) = a + b ln(cos(incidence))",
                                                     expected_b_beam_noise=1.0, **pc),
                        range_equiv_linear_range=dict(formula="sigma_range_equiv_mm = a + b * range_m", **er),
                        range_equiv_linear_incidence=dict(
                            formula="sigma_range_equiv_mm = a + b * incidence_deg", **ei),
                        per_rail={r: pooled(gr) for r, gr in g.groupby("rail")},
                        per_setup={s: pooled(gs) for s, gs in g.groupby("setup")},
                        summary=dict(**pooled(g), median=float(g[VALUE].median()),
                                     p05=float(g[VALUE].quantile(0.05)), p95=float(g[VALUE].quantile(0.95))))
        models[sc] = {"range": lambda r, f=pr: np.exp(f["a"]) * r ** f["b"],
                      "incidence": lambda t, f=pc: np.exp(f["a"]) * np.cos(np.radians(t)) ** f["b"]}
        models_eq[sc] = {"range": lambda r, f=er: f["a"] + f["b"] * r,
                         "incidence": lambda t, f=ei: f["a"] + f["b"] * t}
        f = fits[sc]
        summary.append(dict(scanner=sc, n_setups=f["n_setups"], n_segments=f["n_segments"],
                            range_min_m=f["range_m"][0], range_max_m=f["range_m"][1],
                            incidence_min_deg=f["incidence_deg"][0], incidence_max_deg=f["incidence_deg"][1],
                            **f["summary"], slope_log_cos=pc["b"], slope_log_cos_se=pc["se_b"],
                            range_equiv_at_0m_mm=er["a"], range_equiv_per_m_mm=er["b"],
                            range_equiv_per_m_se_mm=er["se_b"]))
    settings = dict(results=str(a.results), scans=[p.label for p in preps], seg_len=a.seg_len,
                    min_points=a.min_points, range_bin=a.range_bin, incidence_bin=a.incidence_bin)
    stats.write_results(out, NAME, df, bins, fits, summary, settings)
    plots.plot_vs_range_incidence(df, bins, VALUE, "noise normal to the railhead σ_pc [mm]", out / f"{NAME}.png",
                                  "Point-cloud noise on the railhead", models)
    eq = stats.bin_table(df, "sigma_range_equiv_mm", a.range_bin, a.incidence_bin)
    plots.plot_vs_range_incidence(df, eq, "sigma_range_equiv_mm", "σ_pc / cos(incidence) [mm]",
                                  out / f"{NAME}_range_equiv.png", "Range-equivalent noise on the railhead", models_eq)


if __name__ == "__main__":
    main()
