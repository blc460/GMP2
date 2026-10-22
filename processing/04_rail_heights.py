"""04 - Rail heights and their standard deviation sigma_z, vs distance to the scanner and angle of incidence.

    python 04_rail_heights.py [--scanner FARO RTC360] [--seg-len 0.5] [--window 4] [--degree 2]

Rail height (WP5). Per rail and segment of --seg-len (rail clip covers it
completely, at least --min-points inliers) the plane of step 03 gives the
height z_i = c0 at the station s_i (segment centre) and band centre, the
scanner analogue of a levelled point on the rail head.

sigma_z is estimated in two independent ways, both with the MAD:

1. Station-wise (from the points of one segment):
       sigma_MAD          = 1.4826 MAD of the plane residuals (all points, outliers included)
       sigma_z,MAD        = sigma_MAD sqrt(q00),  q00 = [(A'A)^-1]_00 ~ 1/n
       sigma_z,station    = sigma_z,MAD sqrt(max(VF, 1))
   VF = block variance factor (03): > 1 if the residuals are correlated along
   the rail, then sqrt(q00) ~ 1/sqrt(n) is too optimistic. For comparison the
   least-squares value sigma_z,formal = sigma_pc sqrt(q00) is also given.

2. Empirical (from the consistency of neighbouring heights): a rail cannot
   change height arbitrarily within a few metres. A polynomial of --degree
   (2: constant vertical curvature) through the heights within +-window/2,
   leaving station i out, predicts z_i. With equal sigma_z of all stations
       Var(z_i - zhat_i) = sigma_z^2 (1 + h_i),   h_i = [(B'B)^-1]_00 of the polynomial,
   so e_i = (z_i - zhat_i) / sqrt(1 + h_i) is one sample of the height error and
       sigma_z,emp = 1.4826 MAD(e)
   over a group of stations (scanner, rail, setup, distance bin, incidence
   bin). Its 95 % CI uses the chi-square interval with 0.37 n degrees of
   freedom (efficiency of the MAD); the RMS of e with its chi-square CI is
   given as cross-check. Only stations with neighbours on both sides are used.
   The same for the cross level L - R (scanner tilt and other errors common to
   both rails cancel): sigma_cross,emp / sqrt(2) is a second estimate.

   sigma_z,emp contains everything that varies from station to station: point
   noise, incidence- and range-dependent errors, stray points, and also the
   real short-wave unevenness of the rail (welds, joints, corrugation), so it
   is an upper bound for the scanner. Errors that are constant or vary
   smoothly along the track (levelling, vertical angle, range offset,
   registration) are absorbed by the polynomial; they need a second scan
   (step 06) or a reference. sigma_sys = sqrt(sigma_z,emp^2 - RMS(sigma_z,station)^2)
   is the part the station-wise value does not explain (input of step 07).
   Neighbouring e_i share heights, so they are slightly correlated and the
   CIs are somewhat too narrow.

Requirements (report, quality criteria; b = 1 m, l = 5 m, --speed-class):
sigma_z,req for K = 2 (VBZ two standard deviations) and K = 3.605 (alpha = beta = 5 %).

Writes to <results>/04_rail_heights/:
  heights_segments.csv   one row per rail and station: z, sigma_z (formal, MAD, station), e, range, incidence
  heights_cross.csv      one row per station with both rails: cross level and its residual
  heights_bins.csv       per series (height | cross), scanner and distance / incidence bin / cell
  heights_summary.csv    one row per scanner
  heights_fit.json       pooled values per scanner, rail and setup, requirements, settings, provenance
  sigma_z_vs_distance.png, sigma_z_vs_incidence.png
  <label>_height_residuals.png   e along the track per rail and for the cross level
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from shared import config, fit, io, plots, stats

NAME = "heights"
VALUE = "abs_e_mm"


def height_metrics(seg):
    f = seg.fit
    m = fit.noise_metrics(seg)
    vf = max(m["var_factor"], 1.0) if np.isfinite(m["var_factor"]) else 1.0
    return dict(z_m=float(f.coef[0]), q_center_m=seg.q_center, sigma_pc_mm=m["sigma_pc_mm"],
                sigma_mad_mm=m["sigma_mad_mm"], var_factor=m["var_factor"], outlier_share=m["outlier_share"],
                sigma_z_formal_mm=f.sigma_c0 * 1e3, sigma_z_mad_mm=f.sigma_c0_mad * 1e3,
                sigma_z_station_mm=f.sigma_c0_mad * np.sqrt(vf) * 1e3)


def loo_residuals(s, z, half_window, degree):
    """Leave-one-out prediction of z_i from a local polynomial in s, and its variance factor h_i."""
    s, z = np.asarray(s, float), np.asarray(z, float)
    zhat, h = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for i in range(len(s)):
        nb = np.abs(s - s[i]) <= half_window + 1e-9
        nb[i] = False
        x = s[nb] - s[i]
        if nb.sum() < degree + 2 or not ((x < 0).any() and (x > 0).any()):
            continue
        B = np.vander(x, degree + 1, increasing=True)
        c, *_ = np.linalg.lstsq(B, z[nb], rcond=None)
        zhat[i], h[i] = c[0], np.linalg.inv(B.T @ B)[0, 0]
    return zhat, h


def add_residuals(d, value_m, half_window, degree):
    d = d.sort_values("s_mid").copy()
    zhat, h = loo_residuals(d["s_mid"], d[value_m], half_window, degree)
    d["zhat_m"], d["h"] = zhat, h
    d["e_mm"] = (d[value_m] - zhat) / np.sqrt(1 + h) * 1e3
    d[VALUE] = d["e_mm"].abs()
    return d


def cross_table(df):
    """Cross level per station where both rails have a height: L - R, mean range / incidence."""
    cols = ["scanner", "setup", "label", "s_from", "s_to", "s_mid"]
    L, R = (df[df["rail"] == r] for r in ("L", "R"))
    m = L.merge(R, on=cols, suffixes=("_L", "_R"))
    return pd.DataFrame({**{c: m[c] for c in cols}, "rail": "cross",
                         "n": m["n_L"] + m["n_R"],
                         "range_m": 0.5 * (m["range_m_L"] + m["range_m_R"]),
                         "incidence_deg": 0.5 * (m["incidence_deg_L"] + m["incidence_deg_R"]),
                         "cross_m": m["z_m_L"] - m["z_m_R"],
                         "sigma_z_formal_mm": np.hypot(m["sigma_z_formal_mm_L"], m["sigma_z_formal_mm_R"]),
                         "sigma_z_mad_mm": np.hypot(m["sigma_z_mad_mm_L"], m["sigma_z_mad_mm_R"]),
                         "sigma_z_station_mm": np.hypot(m["sigma_z_station_mm_L"], m["sigma_z_station_mm_R"])})


def pooled(g):
    """Empirical sigma_z (1.4826 MAD of e, with CI), RMS cross-check and the station-wise values."""
    gu = g[g["e_mm"].notna()] if len(g) else g
    e = gu["e_mm"].to_numpy() if len(gu) else np.array([])
    n = len(e)
    out = dict(n_used=int(n))
    if n < 2:
        return out | {k: np.nan for k in ("sigma_z_emp_mm", "sigma_z_emp_ci_lo_mm", "sigma_z_emp_ci_hi_mm",
                                          "sigma_z_emp_rms_mm", "sigma_z_emp_rms_ci_lo_mm",
                                          "sigma_z_emp_rms_ci_hi_mm", "mean_e_mm", "sigma_z_formal_rms_mm",
                                          "sigma_z_mad_rms_mm", "sigma_z_station_rms_mm",
                                          "ratio_emp_station", "sigma_sys_mm")}
    s = fit.mad_sigma(e)
    lo, hi = fit.mad_sigma_ci(s, n)
    rms = float(np.sqrt(np.mean(e ** 2)))
    rlo, rhi = fit.sigma_ci(rms, n)
    station = float(np.sqrt(np.mean(gu["sigma_z_station_mm"] ** 2)))
    return out | dict(sigma_z_emp_mm=s, sigma_z_emp_ci_lo_mm=float(lo), sigma_z_emp_ci_hi_mm=float(hi),
                      sigma_z_emp_rms_mm=rms, sigma_z_emp_rms_ci_lo_mm=float(rlo), sigma_z_emp_rms_ci_hi_mm=float(rhi),
                      mean_e_mm=float(e.mean()),
                      sigma_z_formal_rms_mm=float(np.sqrt(np.mean(gu["sigma_z_formal_mm"] ** 2))),
                      sigma_z_mad_rms_mm=float(np.sqrt(np.mean(gu["sigma_z_mad_mm"] ** 2))),
                      sigma_z_station_rms_mm=station, ratio_emp_station=s / station,
                      sigma_sys_mm=float(np.sqrt(max(s ** 2 - station ** 2, 0.0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_segment_args(ap)
    ap.add_argument("--window", type=float, default=config.HEIGHT_WINDOW,
                    help="length of the neighbourhood along the rail for the local polynomial [m]")
    ap.add_argument("--degree", type=int, default=config.HEIGHT_DEGREE,
                    help="degree of the local polynomial (2 = vertical curve)")
    ap.add_argument("--speed-class", default=config.SPEED_CLASS, choices=list(config.ATTENTION),
                    help="speed class of the attention values (default: %(default)s km/h)")
    a = ap.parse_args()
    preps = io.load_prepared(a.results, a.scanner)
    out = config.step_dir(a.results, "heights")
    hw = a.window / 2
    reqs = config.requirements(a.speed_class)

    seg = fit.segment_table(preps, height_metrics, a.seg_len, keep=lambda sg: sg.full and sg.fit.n >= a.min_points)
    if seg.empty:
        raise SystemExit(f"no complete segment with at least {a.min_points} points; check the prepared scans")
    axes = {p.label: p.axis for p in preps}
    xy = np.vstack([axes[lab].to_xy(g["s_mid"], g["q_center_m"]) for lab, g in seg.groupby("label", sort=False)])
    order = np.concatenate([g.index.to_numpy() for _, g in seg.groupby("label", sort=False)])
    seg.loc[order, "x_station"], seg.loc[order, "y_station"] = xy[:, 0], xy[:, 1]
    seg = pd.concat([add_residuals(g, "z_m", hw, a.degree) for _, g in seg.groupby(["label", "rail"])],
                    ignore_index=True)
    cross = cross_table(seg)
    if not cross.empty:
        cross = pd.concat([add_residuals(g, "cross_m", hw, a.degree) for _, g in cross.groupby("label")],
                          ignore_index=True)
    if seg["e_mm"].notna().sum() < 3:
        raise SystemExit(f"fewer than 3 stations with neighbours on both sides within {a.window} m; "
                         "the clips are too short or have gaps")
    # empirical sigma_z per scan and rail, used by step 05 for the variance propagation
    emp = {k: pooled(g)["sigma_z_emp_mm"] for k, g in seg.groupby(["label", "rail"])}
    seg["sigma_z_emp_mm"] = [emp[(lab, r)] for lab, r in zip(seg["label"], seg["rail"])]
    for key, (v, _) in reqs.items():
        seg[f"station_meets_{key}"] = seg["sigma_z_station_mm"] <= v

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
        c = cross[cross["scanner"] == sc] if len(cross) else cross
        ph, pc = pooled(g), pooled(c)
        fits[sc] = dict(**stats.coverage(g[g["e_mm"].notna()]),
                        height=ph,
                        per_rail={r: pooled(gr) for r, gr in g.groupby("rail")},
                        per_setup={s: pooled(gs) for s, gs in g.groupby("setup")},
                        cross_level=dict(**pc, sigma_z_from_cross_mm=pc.get("sigma_z_emp_mm", np.nan) / np.sqrt(2)),
                        station=dict(median_sigma_z_formal_mm=float(g["sigma_z_formal_mm"].median()),
                                     median_sigma_z_mad_mm=float(g["sigma_z_mad_mm"].median()),
                                     median_sigma_z_station_mm=float(g["sigma_z_station_mm"].median()),
                                     p95_sigma_z_station_mm=float(g["sigma_z_station_mm"].quantile(0.95)),
                                     median_var_factor=float(g["var_factor"].median())),
                        requirements={k: dict(sigma_z_req_mm=v,
                                              share_stations_meeting=float(g[f"station_meets_{k}"].mean()),
                                              empirical_meets=bool(ph.get("sigma_z_emp_ci_hi_mm", np.inf) <= v))
                                      for k, (v, _) in reqs.items()})
        f = fits[sc]
        summary.append(dict(scanner=sc, n_setups=f["n_setups"], n_heights=len(g), n_used=ph["n_used"],
                            range_min_m=f["range_m"][0], range_max_m=f["range_m"][1],
                            incidence_min_deg=f["incidence_deg"][0], incidence_max_deg=f["incidence_deg"][1],
                            **{k: v for k, v in ph.items() if k != "n_used"},
                            **{f"sigma_z_emp_mm_{r}": v["sigma_z_emp_mm"] for r, v in f["per_rail"].items()},
                            sigma_cross_emp_mm=pc.get("sigma_z_emp_mm", np.nan),
                            sigma_z_from_cross_mm=f["cross_level"]["sigma_z_from_cross_mm"],
                            **f["station"],
                            **{f"share_stations_meeting_{k}": v["share_stations_meeting"]
                               for k, v in f["requirements"].items()}))
    settings = dict(results=str(a.results), scans=[p.label for p in preps], seg_len=a.seg_len,
                    min_points=a.min_points, window=a.window, degree=a.degree, range_bin=a.range_bin,
                    incidence_bin=a.incidence_bin, speed_class=a.speed_class,
                    requirements_mm={k: v for k, (v, _) in reqs.items()}, mad_to_sigma=fit.MAD2SIGMA)
    stats.write_results(out, NAME, seg, bins, fits, summary, settings, io.provenance(__file__, a))
    cross.to_csv(out / f"{NAME}_cross.csv", index=False)
    hb = bins[bins["series"] == "height"]
    plots.plot_sigma_z(seg, hb, "range", out / "sigma_z_vs_distance.png", reqs,
                       "Rail height precision vs distance to the scanner")
    plots.plot_sigma_z(seg, hb, "incidence", out / "sigma_z_vs_incidence.png", reqs,
                       "Rail height precision vs angle of incidence")
    for label, g in seg.groupby("label"):
        plots.plot_height_residuals(g, cross[cross["label"] == label] if len(cross) else cross,
                                    out / f"{label}_height_residuals.png", label)
    for s in summary:
        print(f"{s['scanner']}: sigma_z,emp = 1.4826 MAD(e) = {s['sigma_z_emp_mm']:.3f} mm "
              f"(95 % CI {s['sigma_z_emp_ci_lo_mm']:.3f}-{s['sigma_z_emp_ci_hi_mm']:.3f}, n = {s['n_used']}), "
              f"RMS(e) = {s['sigma_z_emp_rms_mm']:.3f} mm, median sigma_z,station = "
              f"{s['median_sigma_z_station_mm']:.3f} mm, sigma_sys = {s['sigma_sys_mm']:.3f} mm")


if __name__ == "__main__":
    main()
