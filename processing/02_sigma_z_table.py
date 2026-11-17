"""Step 2: table of sigma_z and sigma_Z per scanner and station (setup), slant range bin and incidence bin.

Per bin: number of cells, median and 25 % / 75 % quantile of sigma_z, and the median of the three sigma_Z
estimators (sqrt(n), Taubenheim, sub-cells). Bins with fewer than `min_cells` cells are flagged (indicative = True).
With by_setup, the main table has one line per scanner and station; a pooled table per scanner is a summary.

Input: results/<sok|lip>/01_segments.csv.
Output: results/<sok|lip>/02_sigma_z_table.csv (and 02_sigma_z_table_pooled.csv) and 02_sigma_z.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cells import OUT, cfg, group_keys

SIGMA_Z = ["sigma_Z_sqrt_n_mm", "sigma_Z_taubenheim_mm", "sigma_Z_sub_mm"]


def load():
    df = pd.read_csv(OUT / "01_segments.csv")
    df["r_bin"] = pd.cut(df["r_m"], cfg["distance"], right=False)
    df["theta_bin"] = pd.cut(df["theta_ground_deg"], cfg["incidence"], right=False)
    return df


def table(df, keys):
    g = df.groupby(keys + ["r_bin", "theta_bin"], observed=True)
    t = g["sigma_z_mm"].agg(segments="size", sigma_z_median_mm="median",
                            sigma_z_q25_mm=lambda x: x.quantile(0.25), sigma_z_q75_mm=lambda x: x.quantile(0.75))
    t = t.join(g[SIGMA_Z].median().rename(columns=lambda c: c.replace("_mm", "_median_mm")))
    if "sigma_z_plane_mm" in df:                                             # optional plane fit
        t = t.join(g["sigma_z_plane_mm"].agg(sigma_z_plane_median_mm="median"))
    t["indicative"] = t["segments"] < cfg["min_cells"]
    return t.round(4).reset_index()


def overview(df):
    """sigma_z over r and over theta_ground, one dot per cell; colour per scanner, marker per station."""
    colours = dict(zip(sorted(df["scanner"].unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    markers = dict(zip(sorted(df["setup"].unique()), "os^D"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, x in ((ax[0], "r_m"), (ax[1], "theta_ground_deg")):
        for (sc, st), g in df.groupby(["scanner", "setup"]):
            a.scatter(g[x], g["sigma_z_mm"], s=5, alpha=0.3, color=colours[sc], marker=markers[st],
                      linewidths=0, label=f"{sc}, {st}")
        a.grid(alpha=0.3)
    ax[0].set(xlabel="slant range $r$ [m]", ylabel=r"$\sigma_z$ [mm]", yscale="log")
    ax[1].set(xlabel=r"incidence angle $\theta$ [deg]")
    ax[0].legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "02_sigma_z.png", dpi=200)


if __name__ == "__main__":
    df = load()
    main = table(df, group_keys())
    main.to_csv(OUT / "02_sigma_z_table.csv", index=False)
    print(main.to_string(index=False))
    if cfg["by_setup"]:
        table(df, group_keys(by_setup=False)).to_csv(OUT / "02_sigma_z_table_pooled.csv", index=False)
    overview(df)
