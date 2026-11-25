"""Step 6: combine all results.

Usage: python processing/06_combine.py

1. 06_cells.csv: one row per 0.5 m cell over all scans, with scanner, setting (resolution, quality),
   station, scanner height h above the railhead plane (h = r cos theta_ground), r, theta_ground,
   sigma_z (residuals) and sigma_z_raw, the three sigma_Z (see precision.py), point count and density.
2. 06_r_theta_table.csv: sigma_z per range x angle bin, per scanner/setting (no pooling over r or theta
   alone): number of cells, stations in the bin, median, 25 % and 75 % quantile.
   Range bins every 5 m; angle bins every 10 deg up to 60 deg, then every 5 deg (most cells lie there).
3. 06_fit.csv: trial fits of log(sigma_z) per scanner/setting (least squares on all cells):
     r      : log s = a + b log r
     theta  : log s = a + c log(1/cos theta)
     r+theta: log s = a + b log r + c log(1/cos theta)
     r+st   : log s = a + b log r + one offset per station
   Within one station cos theta = h / r, so log(1/cos theta) = log r - log h: the angle term is the
   same as a term in the station height. r+theta and r+station can therefore not tell angle and
   station (height, platform, position) apart; they only show how much is left after range.

Input: results/<sok|lip>/01_segments.csv. Output: results/<sok|lip>/06_*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
try:
    from cells import OUT as out
except Exception:                                   # run without cells.py / config next to it
    out = HERE / "results" / "sok"

SETTING = {"RTC360": ("high", "-"), "FARO 1/1 Q3": ("1/1", "3"), "FARO 1/2 Q4": ("1/2", "4")}

c = pd.read_csv(out / "01_segments.csv")
c["resolution"] = c["scanner"].map(lambda s: SETTING.get(s, ("?", "?"))[0])
c["quality"] = c["scanner"].map(lambda s: SETTING.get(s, ("?", "?"))[1])
c["h_m"] = c["r_m"] * np.cos(np.radians(c["theta_ground_deg"]))
cols = ["dataset", "scanner", "resolution", "quality", "setup", "condition", "rail", "cell", "x", "y",
        "h_m", "r_m", "theta_ground_deg", "sigma_z_mm", "sigma_z_raw_mm", "sigma_Z_sqrt_n_mm", "sigma_Z_bins_mm",
        "sigma_Z_bins_lo_mm", "sigma_Z_bins_hi_mm", "n_bins", "sigma_Z_boot_mm", "n", "length_m", "width_m",
        "density_pts_m2", "rho_exp_pts_m2", "return_rate"]
c[cols].to_csv(out / "06_cells.csv", index=False, float_format="%.5g")

# 2. sigma_z per r x theta bin
r_edges = np.arange(0, 65, 5)
t_edges = np.r_[np.arange(0, 60, 10), np.arange(60, 91, 5)]
c["r_bin"] = pd.cut(c["r_m"], r_edges, right=False)
c["theta_bin"] = pd.cut(c["theta_ground_deg"], t_edges, right=False)
g = c.groupby(["scanner", "r_bin", "theta_bin"], observed=True)
table = g["sigma_z_mm"].agg(cells="size", median="median",
                            q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).reset_index()
table["stations"] = g["setup"].agg(lambda s: "+".join(sorted(s.unique()))).values
table["r_bin"] = table["r_bin"].astype(str); table["theta_bin"] = table["theta_bin"].astype(str)
table.round(3).to_csv(out / "06_r_theta_table.csv", index=False)

# 3. trial fits on log(sigma_z)
def fit(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return b, r2, mad, res

rows = []
for sc, s in c[c["sigma_z_mm"] > 0].groupby("scanner"):    # sigma_z = 0: a cell of one scan line (1 mm height grid)
    y = np.log(s["sigma_z_mm"].to_numpy())
    lr = np.log(s["r_m"].to_numpy())
    lt = -np.log(np.cos(np.radians(s["theta_ground_deg"].to_numpy())))
    st = sorted(s["setup"].unique())
    dummies = [(s["setup"] == k).to_numpy(float) for k in st[1:]]
    for model, X in [("r", [lr]), ("theta", [lt]), ("r+theta", [lr, lt]), ("r+station", [lr] + dummies)]:
        b, r2, mad, res = fit(y, X)
        row = dict(scanner=sc, model=model, cells=len(y), stations="+".join(st), a=b[0],
                   R2=r2, resid_factor=np.exp(mad))
        if model in ("r", "r+theta", "r+station"): row["b_r"] = b[1]
        if model == "theta": row["c_theta"] = b[1]
        if model == "r+theta": row["c_theta"] = b[2]
        if model == "r+station":
            for k, v in zip(st[1:], b[2:]): row[f"factor_{k}_vs_{st[0]}"] = np.exp(v)
            # residual median per station and range bin: where does the model fail?
            s2 = s.assign(res=res)
            worst = s2.groupby(["setup", "r_bin"], observed=True)["res"].agg(["median", "size"])
            worst = worst[worst["size"] >= 5]["median"].abs().sort_values().tail(1)
            row["worst_bin"] = f"{worst.index[0][0]} {worst.index[0][1]}: x{np.exp(worst.iloc[0]):.2f}"
        rows.append(row)
pd.DataFrame(rows).round(3).to_csv(out / "06_fit.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
