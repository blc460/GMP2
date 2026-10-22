"""03 - Empirical height precision on the railhead per scanner, from the consistency of the height profile.

    python railhead/03_height_consistency.py [--scanner FARO RTC360] [--seg-len 0.5] [--window 4] [--degree 2]

Per segment (rail clip covers it completely, at least --min-points inliers):
rail height z_i = value of the fitted plane at the segment centre and band
centre (as trackqc.core.rail_heights), with its formal std sigma_z_formal =
sigma_pc / sqrt(n) and the same times sqrt(variance factor).

A rail cannot change height arbitrarily from one segment to the next. Per rail
a polynomial of --degree (2: constant vertical curvature) is fitted to the
heights of the neighbouring segments within +-window/2, leaving segment i out,
and predicts z_i. With equal height std of all segments
    Var(z_i - zhat_i) = sigma_z^2 (1 + h_i),  h_i = [(A'A)^-1]_00
so e_i = (z_i - zhat_i) / sqrt(1 + h_i) is one sample of the height error, and
the RMS of e over segments is the empirical sigma_z. Only segments with
neighbours on both sides are used (no extrapolation).

The same is done for the cross level (height difference left - right at the
same station): scanner tilt and other errors common to both rails cancel, and
sigma_cross / sqrt(2) is a second estimate of sigma_z.

The empirical sigma_z contains everything that varies from segment to segment:
point noise, incidence and range dependent bias, surface effects, stray points
in the band, and also the real short-wave unevenness of the rail (welds,
joints, corrugation). It is therefore an upper bound for the scanner and is
comparable between scanners only on the same track with the same settings.
Errors that are constant or vary smoothly along the track (levelling,
vertical angle, range offset, registration) are absorbed by the polynomial and
are not seen; they need a second setup or a reference (trackqc/compare.py).
empirical / formal >> 1 means the formal precision is too optimistic.

Pooled per scanner, rail, setup and range / incidence bin: sigma = RMS(e) with
95 % chi-square CI (dof = number of segments) and the robust sigma 1.4826 MAD(e).

Writes to <results>/03_height_consistency/:
  height_segments.csv    one row per rail segment (height, prediction, e)
  height_cross.csv       one row per station with both rails (cross level)
  height_bins.csv        per series (height | cross), scanner and range bin / incidence bin / cell
  height_summary.csv     one row per scanner
  height_fit.json        pooled values per scanner, rail and setup, settings
  height.png             |e| and empirical vs formal sigma_z vs range and incidence
  <label>_profile.png    e along the track per rail and for the cross level
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config, data, plots, segments, stats
from trackqc import core

NAME = "height"
VALUE = "abs_e_mm"


def height_metrics(seg):
    m = core.noise_metrics(seg)
    vf = max(m["var_factor"], 1.0) if np.isfinite(m["var_factor"]) else 1.0
    return dict(z_m=float(seg.fit.coef[0]), s_mid=0.5 * (seg.s_from + seg.s_to), sigma_pc_mm=m["sigma_pc_mm"],
                var_factor=m["var_factor"], sigma_formal_mm=m["sigma_z_formal_mm"],
                sigma_formal_vf_mm=m["sigma_z_formal_mm"] * np.sqrt(vf))


def loo_residuals(s, z, half_window, degree):
    """Leave-one-out residual of z_i against a local polynomial in s and its variance factor 1 + h_i."""
    s, z = np.asarray(s, float), np.asarray(z, float)
    zhat, h = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for i in range(len(s)):
        nb = np.abs(s - s[i]) <= half_window + 1e-9
        nb[i] = False
        x = s[nb] - s[i]
        if nb.sum() < degree + 2 or not ((x < 0).any() and (x > 0).any()):
            continue
        A = np.vander(x, degree + 1, increasing=True)
        c, *_ = np.linalg.lstsq(A, z[nb], rcond=None)
        zhat[i], h[i] = c[0], np.linalg.inv(A.T @ A)[0, 0]
    return zhat, h


def add_residuals(d, value_m, half_window, degree):
    d = d.sort_values("s_mid").copy()
    zhat, h = loo_residuals(d["s_mid"], d[value_m], half_window, degree)
    d["zhat_m"], d["h"] = zhat, h
    d["e_mm"] = (d[value_m] - zhat) / np.sqrt(1 + h) * 1e3
    d[VALUE] = d["e_mm"].abs()
    return d


def cross_table(df):
    """Cross level per station where both rails have a segment: L - R, mean range / incidence."""
    cols = ["scanner", "setup", "label", "s_from", "s_to", "s_mid"]
    L, R = (df[df["rail"] == r] for r in ("L", "R"))
    m = L.merge(R, on=cols, suffixes=("_L", "_R"))
    return pd.DataFrame({**{c: m[c] for c in cols}, "rail": "cross",
                         "n": m["n_L"] + m["n_R"],
                         "range_m": 0.5 * (m["range_m_L"] + m["range_m_R"]),
                         "incidence_deg": 0.5 * (m["incidence_deg_L"] + m["incidence_deg_R"]),
                         "cross_m": m["z_m_L"] - m["z_m_R"],
                         "sigma_formal_mm": np.hypot(m["sigma_formal_mm_L"], m["sigma_formal_mm_R"]),
                         "sigma_formal_vf_mm": np.hypot(m["sigma_formal_vf_mm_L"], m["sigma_formal_vf_mm_R"])})


def pooled(g):
    e = g["e_mm"].dropna().to_numpy()
    if len(e) < 2:
        return dict(n_used=len(e), sigma_emp_mm=np.nan, sigma_emp_ci_lo_mm=np.nan, sigma_emp_ci_hi_mm=np.nan,
                    sigma_emp_robust_mm=np.nan, sigma_formal_mm=np.nan, sigma_formal_vf_mm=np.nan,
                    ratio_emp_formal_vf=np.nan)
    s = float(np.sqrt(np.mean(e ** 2)))
    lo, hi = core.sigma_ci(s, len(e))
    gu = g[g["e_mm"].notna()]
    f = float(np.sqrt(np.mean(gu["sigma_formal_mm"] ** 2)))
    fv = float(np.sqrt(np.mean(gu["sigma_formal_vf_mm"] ** 2)))
    return dict(n_used=int(len(e)), sigma_emp_mm=s, sigma_emp_ci_lo_mm=float(lo), sigma_emp_ci_hi_mm=float(hi),
                sigma_emp_robust_mm=float(1.4826 * np.median(np.abs(e - np.median(e)))),
                sigma_formal_mm=f, sigma_formal_vf_mm=fv, ratio_emp_formal_vf=s / fv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_analysis_args(ap)
    ap.add_argument("--min-points", type=int, default=30, help="minimum inliers per segment")
    ap.add_argument("--window", type=float, default=4.0,
                    help="length of the neighbourhood along the rail for the local polynomial [m]")
    ap.add_argument("--degree", type=int, default=2, help="degree of the local polynomial (2 = vertical curve)")
    a = ap.parse_args()
    preps = data.load_prepared(a.results, a.scanner)
    out = Path(a.results) / "03_height_consistency"
    hw = a.window / 2

    seg = segments.segment_table(preps, height_metrics, a.seg_len,
                                 keep=lambda sg: sg.full and sg.fit.n >= a.min_points)
    if seg.empty:
        raise SystemExit(f"no complete segment with at least {a.min_points} points; check the prepared scans")
    seg = pd.concat([add_residuals(g, "z_m", hw, a.degree) for _, g in seg.groupby(["label", "rail"])],
                    ignore_index=True)
    cross = cross_table(seg)
    if not cross.empty:
        cross = pd.concat([add_residuals(g, "cross_m", hw, a.degree) for _, g in cross.groupby("label")],
                          ignore_index=True)
    if seg["e_mm"].notna().sum() < 3:
        raise SystemExit(f"fewer than 3 segments with neighbours on both sides within {a.window} m; "
                         "the clips are too short or have gaps")

    bins = []
    for series, d in (("height", seg), ("cross", cross)):
        d = d[d["e_mm"].notna()] if len(d) else d
        if len(d):
            b = stats.bin_table(d, VALUE, a.range_bin, a.incidence_bin, extra=pooled)
            b.insert(0, "series", series)
            bins.append(b)
    bins = pd.concat(bins, ignore_index=True)

    fits, summary = {}, []
    for sc, g in seg.groupby("scanner"):
        gu = g[g["e_mm"].notna()]
        c = cross[cross["scanner"] == sc] if len(cross) else cross
        pc = pooled(c) if len(c) else pooled(g.iloc[:0])
        fits[sc] = dict(**stats.coverage(gu),
                        height=pooled(g),
                        per_rail={r: pooled(gr) for r, gr in g.groupby("rail")},
                        per_setup={s: pooled(gs) for s, gs in g.groupby("setup")},
                        cross_level=dict(**pc, sigma_z_from_cross_mm=pc["sigma_emp_mm"] / np.sqrt(2)),
                        median_sigma_pc_mm=float(g["sigma_pc_mm"].median()))
        f = fits[sc]
        summary.append(dict(scanner=sc, n_setups=f["n_setups"], n_segments=f["n_segments"],
                            range_min_m=f["range_m"][0], range_max_m=f["range_m"][1],
                            incidence_min_deg=f["incidence_deg"][0], incidence_max_deg=f["incidence_deg"][1],
                            **{k: v for k, v in f["height"].items() if k != "n_used"},
                            **{f"sigma_emp_mm_{r}": v["sigma_emp_mm"] for r, v in f["per_rail"].items()},
                            sigma_cross_emp_mm=pc["sigma_emp_mm"],
                            sigma_z_from_cross_mm=f["cross_level"]["sigma_z_from_cross_mm"]))
    settings = dict(results=str(a.results), scans=[p.label for p in preps], seg_len=a.seg_len,
                    min_points=a.min_points, window=a.window, degree=a.degree,
                    range_bin=a.range_bin, incidence_bin=a.incidence_bin)
    stats.write_results(out, NAME, seg, bins, fits, summary, settings)
    cross.to_csv(out / f"{NAME}_cross.csv", index=False)
    plots.plot_height_consistency(seg[seg["e_mm"].notna()], bins[bins["series"] == "height"],
                                  out / f"{NAME}.png", "Empirical height precision on the railhead")
    for label, g in seg.groupby("label"):
        plots.plot_height_profile(g, cross[cross["label"] == label] if len(cross) else cross,
                                  out / f"{label}_profile.png", label)


if __name__ == "__main__":
    main()
