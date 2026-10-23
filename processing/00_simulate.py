"""00 - Validation of the pipeline on simulated railhead scans with known rail heights.

    python 00_simulate.py                 # two simulated setups, full pipeline 01-07, errors against the truth
    python 00_simulate.py --mc 30         # in addition: Monte-Carlo check of sigma_z (01 and 04, 30 runs)
    python 00_simulate.py --mc 30 --z-step 0.001 --out results_sim_1mm   # heights stored in 1 mm steps

The simulated railheads are two flat 55 mm wide strips, 30 m long, 1.0 m
apart, with known heights (longitudinal gradient, 4 mm settlement trough,
cant ramp = constant twist of -0.4 per mille). A scanner with a constant
angular step samples them (density cos(theta) / (r delta)^2), with range
noise along the beam that grows with distance and incidence and 1 % gross
outliers. The values are illustrative and NOT properties of a real scanner;
they only test whether the estimators do what they claim when their
assumptions hold:

- rail heights z (04) unbiased: mean error ~ 0;
- station-wise sigma_z (04, 1.4826 MAD x sqrt(q00)): matches the scatter of
  the height errors over Monte-Carlo runs (ratio ~ 1);
- empirical sigma_z (04, 1.4826 MAD of the leave-one-out residuals): matches
  1.4826 MAD of the true height errors, plus the misfit of the local
  polynomial to the trough (which makes it an upper bound);
- twist (05) = -0.4 per mille.

Writes to <out> (default results_sim/, not part of the real results):
  clips/SIM_S1.npy, clips/SIM_S2.npy, clips/datasets.csv   simulated clips and list file for 01
  results/01_prepare ... results/07_operational_limits    pipeline output
  validation.json                                          errors against the truth (and Monte-Carlo)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from shared import config, fit, stats

HERE = Path(__file__).resolve().parent
HEAD = 0.055               # railhead width [m]
Y_RAIL = 0.5               # |y| of the head centres -> b = 1.0 m
SETUPS = {"S1": (12.0, -4.0, 1.5), "S2": (20.0, 6.0, 1.5)}
TWIST_TRUE = -0.4          # per mille


def true_heights(x):
    """Known rail heights [m] at x: gradient + 4 mm settlement trough, cant ramp (twist -0.4 per mille)."""
    cant = 0.002 + 0.0004 * x
    trough = -0.004 * np.exp(-0.5 * ((x - 15.0) / 5.0) ** 2)
    base = 0.01 * x / 30.0
    return base + trough - cant / 2, base + trough + cant / 2


def simulate_rail(side, origin, rng, x_range=(0.0, 30.0), dphi=0.6e-3, sigma0=0.0006, k_r=0.00003,
                  cell=0.002, p_outlier=0.01):
    xs = np.arange(*x_range, cell) + cell / 2
    yr = np.arange(-HEAD / 2, HEAD / 2, cell) + cell / 2
    X, YR = (a.ravel() for a in np.meshgrid(xs, yr, indexing="ij"))
    zL, zR = true_heights(X)
    Z = zL if side == "L" else zR
    P = np.column_stack([X, (Y_RAIL if side == "L" else -Y_RAIL) + YR, Z])
    beam = P - origin
    r = np.linalg.norm(beam, axis=1)
    cos_t = np.abs(beam[:, 2]) / r
    lam = cos_t / (r * dphi) ** 2 * cell ** 2
    idx = np.repeat(np.arange(len(P)), rng.poisson(np.minimum(lam, 50)))
    pts = P[idx] + np.column_stack([rng.uniform(-cell / 2, cell / 2, (len(idx), 2)), np.zeros(len(idx))])
    pts[:, 2] = true_heights(pts[:, 0])[0 if side == "L" else 1]
    u = (pts - origin) / np.linalg.norm(pts - origin, axis=1, keepdims=True)
    sig = (sigma0 + k_r * r[idx]) / np.maximum(cos_t[idx], 0.05) ** 0.5
    pts += u * (rng.normal(0, 1, len(idx)) * sig)[:, None]
    bad = rng.random(len(idx)) < p_outlier
    pts[bad, 2] += rng.normal(0, 0.01, bad.sum())
    return pts


def make_clips(folder, setups, seed, z_step=0.0, sigma0=0.0006):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, o) in enumerate(setups.items()):
        rng = np.random.default_rng(seed + i)
        o = np.asarray(o, float)
        pts = np.vstack([simulate_rail(side, o, rng, sigma0=sigma0) for side in ("L", "R")])
        if z_step:      # heights stored on a grid in the scanner frame, as in RTC360 / REGISTER 360 exports
            pts[:, 2] = o[2] + np.round((pts[:, 2] - o[2]) / z_step) * z_step
        np.save(folder / f"SIM_{name}.npy", pts)
        rows.append(dict(path=f"SIM_{name}.npy", scanner="SIM", setup=name,
                         origin_x=o[0], origin_y=o[1], origin_z=o[2]))
    with open(folder / "datasets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return folder / "datasets.csv"


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def height_errors(results):
    """Heights of step 04 with the true height at the station and the error [mm]."""
    seg = pd.read_csv(config.step_dir(results, "heights") / "heights_segments.csv")
    zL, zR = true_heights(seg["x_station"].to_numpy())
    seg["z_true_m"] = np.where(seg["rail"] == "L", zL, zR)
    seg["err_mm"] = (seg["z_m"] - seg["z_true_m"]) * 1e3
    return seg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results_sim"), help="output folder (default: %(default)s)")
    ap.add_argument("--mc", type=int, default=0, help="number of Monte-Carlo runs for the sigma_z check")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--sigma0", type=float, default=0.0006,
                    help="range noise at 0 m along the beam [m] (default: %(default)s)")
    ap.add_argument("--z-step", type=float, default=0.0,
                    help="store the simulated heights on a grid of this step [m], e.g. 0.001 (default: no grid)")
    a = ap.parse_args()
    out = Path(a.out)
    res = out / "results"
    lst = make_clips(out / "clips", SETUPS, a.seed, a.z_step, a.sigma0)
    print(run("01_prepare.py", "--list", lst, "--results", res, "--refit-axis").strip().splitlines()[-1])
    for script in ("02_point_density.py", "03_point_noise.py", "04_rail_heights.py", "05_track_geometry.py",
                   "06_compare_scans.py", "07_operational_limits.py"):
        run(script, "--results", res)
    seg = height_errors(res)
    val = dict(truth=dict(twist_permille=TWIST_TRUE, setups=SETUPS), z_step_m=a.z_step, sigma0_m=a.sigma0, scans={})
    for label, g in seg.groupby("label"):
        geo = pd.read_csv(config.step_dir(res, "geometry") / f"{label}_geometry.csv")
        e = g["err_mm"].to_numpy()
        v = val["scans"][label] = dict(
            n_heights=len(g), height_bias_mm=float(e.mean()), height_bias_se_mm=float(e.std(ddof=1) / np.sqrt(len(e))),
            true_error_mad_sigma_mm=fit.mad_sigma(e), true_error_rms_mm=float(np.sqrt(np.mean(e ** 2))),
            sigma_z_station_rms_mm=float(np.sqrt(np.mean(g["sigma_z_station_mm"] ** 2))),
            sigma_z_emp_mm=float(g["sigma_z_emp_mm"].mean()),
            twist_mean_permille=float(geo["twist_permille"].mean()),
            twist_rmse_permille=float(np.sqrt(np.nanmean((geo["twist_permille"] - TWIST_TRUE) ** 2))),
            sigma_twist_station_median_permille=float(geo["sigma_twist_station_permille"].median()))
        print(f"{label}: height bias {v['height_bias_mm']:+.4f} mm (SE {v['height_bias_se_mm']:.4f}), "
              f"true error 1.4826 MAD {v['true_error_mad_sigma_mm']:.4f} mm, RMS sigma_z,station "
              f"{v['sigma_z_station_rms_mm']:.4f} mm, sigma_z,emp {v['sigma_z_emp_mm']:.4f} mm; "
              f"twist {v['twist_mean_permille']:+.3f} per mille (true {TWIST_TRUE}), RMSE {v['twist_rmse_permille']:.4f}")

    if a.mc:
        errs = []
        for k in range(a.mc):
            mres = out / "mc" / "results"
            mlst = make_clips(out / "mc" / "clips", {"S1": SETUPS["S1"]}, a.seed + 1000 + 10 * k, a.z_step, a.sigma0)
            run("01_prepare.py", "--list", mlst, "--results", mres, "--refit-axis")
            run("04_rail_heights.py", "--results", mres)
            g = height_errors(mres)[["rail", "s_mid", "err_mm", "sigma_z_station_mm", "sigma_z_mad_mm",
                                     "sigma_z_formal_mm", "sigma_z_emp_mm"]]
            g["run"] = k
            errs.append(g)
            print(f"Monte Carlo run {k + 1}/{a.mc}", end="\r")
        e = pd.concat(errs, ignore_index=True)
        per = e.groupby(["rail", "s_mid"]).agg(n=("err_mm", "size"), bias=("err_mm", "mean"), std=("err_mm", "std"),
                                               rms=("err_mm", lambda x: np.sqrt(np.mean(x ** 2))),
                                               station=("sigma_z_station_mm", lambda x: np.sqrt(np.mean(x ** 2))),
                                               mad=("sigma_z_mad_mm", lambda x: np.sqrt(np.mean(x ** 2))),
                                               formal=("sigma_z_formal_mm", lambda x: np.sqrt(np.mean(x ** 2))))
        per = per[per["n"] >= max(5, a.mc // 2)]
        runs = e.groupby("run").agg(emp=("sigma_z_emp_mm", "mean"), true_mad=("err_mm", fit.mad_sigma))
        q = lambda x: [float(np.percentile(x, 5)), float(np.median(x)), float(np.percentile(x, 95))]
        val["monte_carlo"] = dict(
            runs=a.mc, stations=len(per), median_bias_mm=float(per["bias"].median()),
            ratio_true_std_to_sigma_z_station_p05_p50_p95=q(per["std"] / per["station"]),
            ratio_true_rms_to_sigma_z_station_p05_p50_p95=q(per["rms"] / per["station"]),
            rms_true_error_mm=float(np.sqrt(np.mean(e["err_mm"] ** 2))),
            rms_sigma_z_station_mm=float(np.sqrt(np.mean(e["sigma_z_station_mm"] ** 2))),
            share_stations_with_bias_above_2se=float(np.mean(np.abs(per["bias"]) > 2 * per["std"] / np.sqrt(per["n"]))),
            ratio_true_std_to_sigma_z_mad_p05_p50_p95=q(per["std"] / per["mad"]),
            ratio_true_std_to_sigma_z_formal_p05_p50_p95=q(per["std"] / per["formal"]),
            ratio_sigma_z_emp_to_true_mad_p05_p50_p95=q(runs["emp"] / runs["true_mad"]))
        m = val["monte_carlo"]
        print(f"\nMonte Carlo ({a.mc} runs, {len(per)} stations): median bias {m['median_bias_mm']:+.4f} mm; "
              f"RMS true error {m['rms_true_error_mm']:.4f} mm vs RMS sigma_z,station {m['rms_sigma_z_station_mm']:.4f} mm; "
              f"std(true error) / sigma_z,station 5/50/95 %: "
              f"{'/'.join(f'{x:.2f}' for x in m['ratio_true_std_to_sigma_z_station_p05_p50_p95'])}; "
              f"/ sigma_z,MAD: {'/'.join(f'{x:.2f}' for x in m['ratio_true_std_to_sigma_z_mad_p05_p50_p95'])}; "
              f"sigma_z,emp / MAD(true error): "
              f"{'/'.join(f'{x:.2f}' for x in m['ratio_sigma_z_emp_to_true_mad_p05_p50_p95'])}")
    stats.write_json(out / "validation.json", val)
    print(f"validation -> {out / 'validation.json'}")


if __name__ == "__main__":
    main()
