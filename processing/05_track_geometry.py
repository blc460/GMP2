"""05 - Track geometry parameters from the rail heights: cant, twist and vertical versine with their precision.

    python 05_track_geometry.py [--scanner RTC360] [--speed-class 48-60] [--b 1.0]

Input: rail heights z_L,i and z_R,i of step 04 at the stations s_i (spacing =
segment length). Per scan (report, quality criteria; VBZ definitions):

    cant      ü_i   = b0 / b (z_R,i - z_L,i)                         b0 = 1 m
    twist     N_i   = (ü_i - ü_i+k) / l                              l = 5 m, k = l / spacing      [mm/m = per mille]
    versine   vPf_i = H_i - (H_i-m + H_i+m) / 2,  H = (z_L + z_R) / 2   chord 10 m, m = 5 m / spacing
              (without the constant design versine Pf_Rv, which cancels in the monitored change)

b = distance between the band centres of both rails at the station (measured,
--b to fix it, e.g. 1 m as in the report). Stations without a height on both
rails, or whose partner station is missing, give no value (no interpolation).

Variance propagation with the individual sigma_z of every height (uncorrelated):

    sigma_ü^2   = (b0 / b)^2 (sigma_zL^2 + sigma_zR^2)
    sigma_N^2   = (sigma_ü,i^2 + sigma_ü,i+k^2) / l^2
    sigma_vPf^2 = sigma_H,i^2 + (sigma_H,i-m^2 + sigma_H,i+m^2) / 4,    sigma_H^2 = (sigma_zL^2 + sigma_zR^2) / 4
    sigma_dvPf  = sqrt(2) sigma_vPf    (change between two epochs with equal precision)

once with the station-wise sigma_z,station (lower bound: point noise only) and
once with the empirical sigma_z,emp of the scan and rail (upper bound:
includes the rail's own short-wave unevenness). With equal sigma_z this gives
the report's sigma_N = 2 b0 sigma_z / (b l) and sigma_dvPf = sqrt(3/2) sigma_z.

Criterion K sigma <= T per station for K = 2 and K = 3.605 (twist: one epoch;
versine: change of two epochs), with the attention values T of --speed-class.
The measured twist is compared with T_N (VBZ: total twist of each epoch).

Writes to <results>/05_track_geometry/:
  <label>_geometry.csv     one row per station: heights, cant, twist, versine, sigmas, criterion flags
  geometry_summary.csv     one row per scan
  geometry_fit.json        summary per scan, settings, provenance
  <label>_geometry.png     heights, cant, twist and versine along the track with +-2 sigma bands
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from shared import config, io, plots, stats

SRC = ("station", "emp")   # sigma_z,station (step 04, MAD x sqrt(VF)) and sigma_z,emp (step 04, MAD of e)


def station_table(g, seg_len):
    """Both rails side by side on the full station grid of one scan (missing stations = NaN)."""
    g = g.copy()
    g["k"] = np.round(g["s_mid"] / seg_len - 0.5).astype(int)
    cols = dict(z_m="z", sigma_z_station_mm="sigma_z_station", sigma_z_emp_mm="sigma_z_emp",
                sigma_z_formal_mm="sigma_z_formal", q_center_m="q", range_m="range", incidence_deg="incidence",
                x_station="x", y_station="y", n="n")
    k = np.arange(g["k"].min(), g["k"].max() + 1)
    t = pd.DataFrame(index=k)
    for r in ("L", "R"):
        d = g[g["rail"] == r].set_index("k")
        for c, short in cols.items():
            t[f"{short}_{r}"] = d[c].reindex(k)
    t["s"] = (k + 0.5) * seg_len
    return t.reset_index(drop=True)


def shift(a, n):
    """a[i + n] for every i (NaN outside)."""
    out = np.full(len(a), np.nan)
    if n >= 0:
        out[:len(a) - n] = a[n:]
    else:
        out[-n:] = a[:len(a) + n]
    return out


def geometry(t, seg_len, b_fixed=None, l=config.TWIST_BASE, chord=config.VERSINE_BASE, b0=config.B0):
    k_t = int(round(l / seg_len))
    k_v = int(round(chord / 2 / seg_len))
    if not np.isclose(k_t * seg_len, l) or not np.isclose(2 * k_v * seg_len, chord):
        raise SystemExit(f"segment length {seg_len} m does not divide the twist base {l} m / half chord {chord / 2} m")
    zL, zR = t["z_L"].to_numpy(), t["z_R"].to_numpy()
    b = np.full(len(t), b_fixed) if b_fixed else np.abs(t["q_L"] - t["q_R"]).to_numpy()
    g = pd.DataFrame(dict(s=t["s"], x=0.5 * (t["x_L"] + t["x_R"]), y=0.5 * (t["y_L"] + t["y_R"]), b_m=b,
                          zL_mm=zL * 1e3, zR_mm=zR * 1e3))
    cant = b0 / b * (zR - zL) * 1e3                       # [mm]
    H = 0.5 * (zL + zR) * 1e3
    g["cant_mm"] = cant
    g["twist_permille"] = (cant - shift(cant, k_t)) / l    # mm / m = per mille
    g["vpf_mm"] = H - 0.5 * (shift(H, -k_v) + shift(H, k_v))
    for src in SRC:
        sL, sR = t[f"sigma_z_{src}_L"].to_numpy(), t[f"sigma_z_{src}_R"].to_numpy()
        g[f"sigma_zL_{src}_mm"], g[f"sigma_zR_{src}_mm"] = sL, sR
        sc = b0 / b * np.hypot(sL, sR)
        sH = 0.5 * np.hypot(sL, sR)
        g[f"sigma_cant_{src}_mm"] = sc
        g[f"sigma_twist_{src}_permille"] = np.hypot(sc, shift(sc, k_t)) / l
        g[f"sigma_vpf_{src}_mm"] = np.sqrt(sH ** 2 + 0.25 * (shift(sH, -k_v) ** 2 + shift(sH, k_v) ** 2))
        g[f"sigma_dvpf_{src}_mm"] = np.sqrt(2) * g[f"sigma_vpf_{src}_mm"]
    for r in ("L", "R"):
        g[f"range_{r}_m"], g[f"incidence_{r}_deg"] = t[f"range_{r}"], t[f"incidence_{r}"]
    return g


def criteria(g, speed_class):
    T_vpf, T_n = config.ATTENTION[speed_class]
    for src in SRC:
        for key, K in config.K_FACTORS.items():
            for name, sig, T in (("twist", f"sigma_twist_{src}_permille", T_n), ("dvpf", f"sigma_dvpf_{src}_mm", T_vpf)):
                v = g[sig]
                g[f"ok_{name}_{src}_{key}"] = np.where(v.notna(), K * v <= T, np.nan)
    g["twist_exceeds_T"] = np.where(g["twist_permille"].notna(), g["twist_permille"].abs() > T_n, np.nan)
    return g


def summarise(label, g, speed_class):
    def share(c):
        v = g[c].dropna()
        return float(v.mean()) if len(v) else np.nan

    def med(c):
        v = g[c].dropna()
        return float(v.median()) if len(v) else np.nan
    tw, vp = g["twist_permille"].dropna(), g["vpf_mm"].dropna()
    row = dict(label=label, n_stations=int(g[["zL_mm", "zR_mm"]].notna().all(axis=1).sum()),
               s_min_m=float(g["s"].min()), s_max_m=float(g["s"].max()),
               b_median_m=med("b_m"), cant_mean_mm=med("cant_mm"), cant_std_mm=float(g["cant_mm"].std()),
               n_twist=int(len(tw)), twist_min_permille=float(tw.min()) if len(tw) else np.nan,
               twist_max_permille=float(tw.max()) if len(tw) else np.nan,
               n_twist_exceeding_T=int(np.nansum(g["twist_exceeds_T"])),
               n_vpf=int(len(vp)), vpf_min_mm=float(vp.min()) if len(vp) else np.nan,
               vpf_max_mm=float(vp.max()) if len(vp) else np.nan, speed_class=speed_class)
    for src in SRC:
        for name, unit in (("cant", "mm"), ("twist", "permille"), ("dvpf", "mm")):
            row[f"sigma_{name}_{src}_median_{unit}"] = med(f"sigma_{name}_{src}_{unit}")
        for key in config.K_FACTORS:
            row[f"share_twist_ok_{src}_{key}"] = share(f"ok_twist_{src}_{key}")
            row[f"share_dvpf_ok_{src}_{key}"] = share(f"ok_dvpf_{src}_{key}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(ap)
    ap.add_argument("--speed-class", default=config.SPEED_CLASS, choices=list(config.ATTENTION),
                    help="speed class of the attention values (default: %(default)s km/h)")
    ap.add_argument("--b", type=float, default=None,
                    help="fixed distance of the height points across the track [m] (default: measured)")
    a = ap.parse_args()
    seg = io.read_step_csv(a.results, "heights", "heights_segments.csv")
    with open(config.step_dir(a.results, "heights") / "heights_fit.json") as f:
        seg_len = json.load(f)["settings"]["seg_len"]
    if a.scanner:
        seg = seg[seg["scanner"].isin(a.scanner)]
    out = config.step_dir(a.results, "geometry")
    out.mkdir(parents=True, exist_ok=True)
    T_vpf, T_n = config.ATTENTION[a.speed_class]

    rows, fits = [], {}
    for label, gl in seg.groupby("label"):
        t = station_table(gl, seg_len)
        g = criteria(geometry(t, seg_len, a.b), a.speed_class)
        length = float(g.loc[g[["zL_mm", "zR_mm"]].notna().all(axis=1), "s"].agg(np.ptp))
        if length < config.TWIST_BASE:
            print(f"{label}: heights on both rails cover {length:.1f} m < twist base {config.TWIST_BASE} m: no twist")
        elif length < config.VERSINE_BASE:
            print(f"{label}: heights cover {length:.1f} m < versine chord {config.VERSINE_BASE} m: no versine")
        g.insert(0, "label", label)
        g.insert(1, "scanner", gl["scanner"].iloc[0])
        g.insert(2, "setup", gl["setup"].iloc[0])
        g.to_csv(out / f"{label}_geometry.csv", index=False)
        plots.plot_track_geometry(g, out / f"{label}_geometry.png", label, T_vpf, T_n)
        rows.append(summarise(label, g, a.speed_class))
        fits[label] = rows[-1]
        r = rows[-1]
        print(f"{label}: {r['n_stations']} stations, b = {r['b_median_m']:.3f} m, cant {r['cant_mean_mm']:+.2f} mm, "
              f"{r['n_twist']} twists ({r['twist_min_permille']:+.3f}..{r['twist_max_permille']:+.3f} per mille), "
              f"{r['n_vpf']} versines; median sigma_N {r['sigma_twist_station_median_permille']:.4f} (station) / "
              f"{r["sigma_twist_emp_median_permille"]:.4f} per mille (empirical)")
    pd.DataFrame(rows).to_csv(out / "geometry_summary.csv", index=False)
    stats.write_json(out / "geometry_fit.json", dict(
        settings=dict(results=str(a.results), seg_len=seg_len, twist_base_m=config.TWIST_BASE,
                      versine_chord_m=config.VERSINE_BASE, b0_m=config.B0, b_m=a.b or "measured",
                      speed_class=a.speed_class, attention_values=dict(T_vpf_mm=T_vpf, T_N_permille=T_n),
                      K=config.K_FACTORS),
        provenance=io.provenance(__file__, a), scans=fits))
    print(f"geometry: {len(rows)} scan(s) -> {out}")


if __name__ == "__main__":
    main()
