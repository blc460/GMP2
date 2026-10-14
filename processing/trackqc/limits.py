"""Operational limits: up to which distance does a scanner meet sigma_z <= sigma_z,req?

Uses all <label>_segments.csv files (labels SCANNER_SETUP_CONDITION). Per scanner
and surface condition it fits

    log(sigma_pc) = a0 + a1 log(r) + a2 alpha
    log(density)  = c0 + c1 log(r) + c2 alpha

with r = range and alpha = horizontal angle between line of sight and track.
Segments of one setup are not independent (same instrument position, same
atmosphere, neighbouring rail), so standard errors are clustered by setup.
For a horizontal running surface seen from a fixed instrument height h, the
incidence angle follows from the distance (cos theta = h / r), so r and theta
cannot be separated; theory predicts density ~ cos(theta) / r^2 = h / r^3, i.e. c1 ~ -3.

The predicted rail-height std for a cross-section window of area A is

    sigma_z(r) = sqrt( VF * sigma_pc(r)^2 / (density(r) * A) + sigma_sys^2 )

(VF = median variance factor from the segment diagnostics), and r_max is the
largest r with sigma_z(r) <= sigma_z,req. Its uncertainty is obtained by
resampling whole setups (cluster bootstrap); with 5 to 7 setups this interval
is wide, which must be stated.

    python -m trackqc.limits results/ --sigma-sys 0.3 --req 1.22
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_segments(folder):
    df = pd.concat([pd.read_csv(p) for p in Path(folder).glob("*_segments.csv")], ignore_index=True)
    df = df.dropna(subset=["sigma_pc_mm"])
    parts = df["label"].str.split("_", expand=True)
    df["scanner"], df["setup"], df["condition"] = parts[0], parts[1], parts[2]
    df["log_sigma"] = np.log(df["sigma_pc_mm"])
    df["log_density"] = np.log(df["density_pts_cm2"])
    df["log_r"] = np.log(df["range_m"])
    return df


def fit_models(d):
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["setup"])[0]})
    m_sig = smf.ols("log_sigma ~ log_r + horiz_angle_deg", d).fit(**kw)
    m_den = smf.ols("log_density ~ log_r + horiz_angle_deg", d).fit(**kw)
    return m_sig, m_den


def predict_sigma_z(m_sig, m_den, r, alpha, area_cm2, vf=1.0, sigma_sys=0.0):
    X = pd.DataFrame({"log_r": np.log(r), "horiz_angle_deg": alpha})
    sig = np.exp(m_sig.predict(X))
    den = np.exp(m_den.predict(X))
    return np.sqrt(vf * sig ** 2 / (den * area_cm2) + sigma_sys ** 2)


def r_max(m_sig, m_den, alpha, area_cm2, req, vf, sigma_sys, r_grid):
    sz = predict_sigma_z(m_sig, m_den, r_grid, np.full_like(r_grid, alpha), area_cm2, vf, sigma_sys)
    ok = sz <= req
    if not ok[0]:
        return np.nan
    return float(r_grid[np.argmin(ok)] if not ok.all() else r_grid[-1])


def analyse(folder, req=1.22, sigma_sys=0.0, window_len=0.5, band_width=0.030, n_boot=500, seed=1):
    df = load_segments(folder)
    area = window_len * band_width * 1e4
    r_grid = np.linspace(1, 80, 800)
    rows = []
    for (scanner, cond), d in df.groupby(["scanner", "condition"]):
        if d["setup"].nunique() < 2:
            continue
        m_sig, m_den = fit_models(d)
        vf = float(np.nanmedian(d["var_factor"])) if "var_factor" in d else 1.0
        vf = max(vf, 1.0)
        alpha = float(d["horiz_angle_deg"].median())
        rm = r_max(m_sig, m_den, alpha, area, req, vf, sigma_sys, r_grid)
        rng = np.random.default_rng(seed)
        setups = d["setup"].unique()
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(setups, len(setups))
            db = pd.concat([d[d["setup"] == s] for s in pick])
            if db["setup"].nunique() < 2:
                continue
            try:
                bs, bd = fit_models(db)
                boot.append(r_max(bs, bd, alpha, area, req, vf, sigma_sys, r_grid))
            except Exception:
                continue
        boot = np.array(boot)
        rows.append(dict(
            scanner=scanner, condition=cond, n_segments=len(d), n_setups=int(d["setup"].nunique()),
            slope_sigma_log_r=m_sig.params["log_r"], se_slope_sigma=m_sig.bse["log_r"],
            slope_density_log_r=m_den.params["log_r"], se_slope_density=m_den.bse["log_r"],
            var_factor=vf, alpha_deg=alpha, r_max_m=rm,
            r_max_ci95=np.nanpercentile(boot, [2.5, 97.5]).round(2).tolist() if np.isfinite(boot).any() else None,
        ))
    return df, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--req", type=float, default=1.22, help="required sigma_z [mm]")
    ap.add_argument("--sigma-sys", type=float, default=0.0, help="sigma_sys [mm] from trackqc.compare")
    ap.add_argument("--band-width", type=float, default=0.030)
    a = ap.parse_args()
    _, res = analyse(a.folder, a.req, a.sigma_sys, band_width=a.band_width)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
