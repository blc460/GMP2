"""Step 2: table of sigma_z per scanner, distance bin and incidence bin (theta_ground, to the ground plane).

Input: results/01_segments.csv. Output: results/02_sigma_z_table.csv and results/02_sigma_z.png
"""
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
with open(HERE / "config.toml", "rb") as f:
    cfg = tomllib.load(f)
out = HERE / cfg["results"]

df = pd.read_csv(out / "01_segments.csv")
df["d_bin"] = pd.cut(df["d_m"], cfg["distance"], right=False)
df["theta_bin"] = pd.cut(df["theta_ground_deg"], cfg["incidence"], right=False)
keys = ["scanner"] + (["condition"] if cfg["by_condition"] else [])

table = (df.groupby(keys + ["d_bin", "theta_bin"], observed=True)["sigma_z_mm"]
         .agg(segments="size", sigma_z_median_mm="median",
              sigma_z_q25_mm=lambda x: x.quantile(0.25), sigma_z_q75_mm=lambda x: x.quantile(0.75))
         .round(3).reset_index())
table.to_csv(out / "02_sigma_z_table.csv", index=False)
print(table.to_string(index=False))

# figure: sigma_z over d and over theta_ground, colour per scanner, line = median per bin
colours = dict(zip(sorted(df["scanner"].unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, x, b in ((ax[0], "d_m", "d_bin"), (ax[1], "theta_ground_deg", "theta_bin")):
    for sc, g in df.groupby("scanner"):
        a.scatter(g[x], g["sigma_z_mm"], s=6, alpha=0.3, color=colours[sc])
        med = g.groupby(b, observed=True)[[x, "sigma_z_mm"]].median()
        a.plot(med[x], med["sigma_z_mm"], marker="o", color=colours[sc], label=sc)
    a.grid(alpha=0.3)
ax[0].set(xlabel="horizontal distance d [m]", ylabel="sigma_z (MAD) [mm]", yscale="log")
ax[1].set(xlabel="incidence angle to the ground plane theta_ground [deg]")
ax[0].legend(fontsize=8)
fig.suptitle("Point noise sigma_z per 0.5 m cell (points) and median per bin (lines)")
fig.tight_layout()
fig.savefig(out / "02_sigma_z.png", dpi=200)
