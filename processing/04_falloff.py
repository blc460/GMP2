"""Step 4: how fast point density, intensity and noise fall off with the slant range r, per scanner and station.

Per r bin: median density, expected density rho_exp, return rate, intensity, sigma_z and the noise along
the beam sigma_r = sigma_z / cos(theta_ground) (approximate: assumes all of sigma_z comes from range noise;
it grows without bound at grazing incidence). Per scanner line: log-log slope of density and intensity
against r (least squares over the cells) and the first r bin in which the median return rate is below 0.5.

Input: results/<sok|lip>/01_segments.csv.
Output: results/<sok|lip>/04_falloff.csv, 04_falloff_summary.csv (and 04_falloff_pooled.csv), 04_falloff.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cells import OUT, cfg, group_keys

df = pd.read_csv(OUT / "01_segments.csv")
df["sigma_r_mm"] = df["sigma_z_mm"] / np.cos(np.radians(df["theta_ground_deg"]))
df["r_bin"] = pd.cut(df["r_m"], cfg["distance"], right=False)
cols = ["density_pts_m2", "rho_exp_pts_m2", "return_rate", "intensity", "sigma_z_mm", "sigma_r_mm"]
def falloff(keys):
    return (df.groupby(keys + ["r_bin"], observed=True)
            .agg(cells=("n", "size"), r_m=("r_m", "median"), **{c: (c, "median") for c in cols})
            .round(3).reset_index())


keys = group_keys()
table = falloff(keys)


def slope(g, y):
    return np.polyfit(np.log(g["r_m"]), np.log(g[y]), 1)[0]


rows = []
for key, g in df.groupby(keys):
    t = table[(table[keys] == key).all(axis=1)]
    low = t[t["return_rate"] < 0.5]
    rows.append(dict(zip(keys, key), cells=len(g), density_slope=slope(g, "density_pts_m2"),
                     intensity_slope=slope(g, "intensity"),
                     return_rate_below_05_from=str(low["r_bin"].iloc[0]) if len(low) else "never"))
summary = pd.DataFrame(rows).round(2)
table.to_csv(OUT / "04_falloff.csv", index=False)
if cfg["by_setup"]:
    falloff(group_keys(by_setup=False)).to_csv(OUT / "04_falloff_pooled.csv", index=False)
summary.to_csv(OUT / "04_falloff_summary.csv", index=False)
print(table.to_string(index=False))
print(summary.to_string(index=False))

colours = dict(zip(sorted(df["scanner"].unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
markers = dict(zip(sorted(df["setup"].unique()), "osD^"))
fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for key, t in table.groupby(keys):
    key = dict(zip(keys, key))
    sc = key["scanner"]
    k = dict(color=colours[sc], marker=markers.get(key.get("setup"), "o"), ms=4)
    ax[0, 0].plot(t["r_m"], t["density_pts_m2"], label=", ".join(key.values()), **k)
    ax[0, 0].plot(t["r_m"], t["rho_exp_pts_m2"], color=colours[sc], ls="--", lw=0.8)
    ax[0, 1].plot(t["r_m"], t["return_rate"], **k)
    ax[1, 0].plot(t["r_m"], t["intensity"], **k)
    ax[1, 1].plot(t["r_m"], t["sigma_z_mm"], **k)
    ax[1, 1].plot(t["r_m"], t["sigma_r_mm"], color=colours[sc], marker=k["marker"], ms=3, ls=":", mfc="none")
ax[0, 0].set(yscale="log", ylabel="points per m2", title="density (dashed: expected from the scan grid)")
ax[0, 1].set(ylabel="measured / expected", title="return rate")
ax[0, 1].axhline(0.5, color="0.5", lw=0.8)
ax[1, 0].set(yscale="log", ylabel="median intensity (as stored)", title="intensity (scales differ per scanner)")
ax[1, 1].set(ylabel="mm", title="noise: sigma_z (filled), sigma_r = sigma_z / cos(theta_ground) (open)")
for a in ax[1]:
    a.set_xlabel("slant range r [m]")
for a in ax.ravel():
    a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"Falloff with range, rail_class {cfg['rail_class']} (median per {OUT.name} cell bin)")
fig.tight_layout()
fig.savefig(OUT / "04_falloff.png", dpi=200)
