"""Step 2: table of sigma_z per scanner and station (setup), distance bin and incidence bin (theta_ground).

With by_setup, the main table has one line per scanner and station; a pooled table per scanner is
written as a summary.

Input: results/<sok|lip>/01_segments.csv.
Output: results/<sok|lip>/02_sigma_z_table.csv (and 02_sigma_z_table_pooled.csv) and 02_sigma_z.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cells import OUT as out, cfg, group_keys

df = pd.read_csv(out / "01_segments.csv")
df["d_bin"] = pd.cut(df["d_m"], cfg["distance"], right=False)
df["theta_bin"] = pd.cut(df["theta_ground_deg"], cfg["incidence"], right=False)


def table(keys):
    g = df.groupby(keys + ["d_bin", "theta_bin"], observed=True)
    return (g["sigma_z_mm"].agg(segments="size", sigma_z_median_mm="median",
                                sigma_z_q25_mm=lambda x: x.quantile(0.25), sigma_z_q75_mm=lambda x: x.quantile(0.75))
            .join(g["sigma_z_med_mm"].agg(sigma_z_med_median_mm="median"))      # variant without plane
            .round(3).reset_index())


main = table(group_keys())
main.to_csv(out / "02_sigma_z_table.csv", index=False)
print(main.to_string(index=False))
if cfg["by_setup"]:
    table(group_keys(by_setup=False)).to_csv(out / "02_sigma_z_table_pooled.csv", index=False)

# figure: sigma_z over d and over theta_ground; colour per scanner, line style per station; line = median per bin
colours = dict(zip(sorted(df["scanner"].unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
styles = dict(zip(sorted(df["setup"].unique()), ["-", "--", ":", "-."]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for a, x, b in ((ax[0], "d_m", "d_bin"), (ax[1], "theta_ground_deg", "theta_bin")):
    for (sc, st), g in df.groupby(["scanner", "setup"]):
        a.scatter(g[x], g["sigma_z_mm"], s=4, alpha=0.2, color=colours[sc])
        med = g.groupby(b, observed=True)[[x, "sigma_z_mm"]].median()
        a.plot(med[x], med["sigma_z_mm"], marker="o", ms=3, color=colours[sc], ls=styles.get(st, "-"),
               label=f"{sc}, {st}")
    a.grid(alpha=0.3)
ax[0].set(xlabel="horizontal distance d [m]", ylabel="sigma_z (MAD) [mm]", yscale="log")
ax[1].set(xlabel="incidence angle to the ground plane theta_ground [deg]")
ax[0].legend(fontsize=7)
fig.suptitle("Point noise sigma_z per 0.5 m cell (points) and median per bin (lines), per scanner and station")
fig.tight_layout()
fig.savefig(out / "02_sigma_z.png", dpi=200)
