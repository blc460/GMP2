"""Step 3: map the results back onto one point cloud, to see where which sigma_z and sigma_Z is reached.

Usage: python processing/03_map.py Bu_RTC-2   (any part of the file name of one row of datasets.csv)

Every railhead point gets the values of its cell (the same cells as in 01_segments.py): sigma_z_cell_mm,
sigma_z_bin_mm (median sigma_z of the cell's table line, range and angle bin), the three sigma_Z, n_eff, r_m and
theta_ground_deg; with plane_fit also sigma_z_plane_cell_mm and theta_deg. Points in skipped or dropped cells get NaN.

Input: results/<sok|lip>/01_segments.csv and 02_sigma_z_table.csv. Output: results/<sok|lip>/03_map_<dataset>.laz
"""
from pathlib import Path
import sys

import laspy
import numpy as np
import pandas as pd

from cells import OUT, cells, cfg, find_dataset, group_keys, level, read

FIELDS = [("sigma_z_cell_mm", "sigma_z_mm"), ("sigma_z_bin_mm", "sigma_z_median_mm"),
          ("sigma_Z_sqrt_n_mm", "sigma_Z_sqrt_n_mm"), ("sigma_Z_taubenheim_mm", "sigma_Z_taubenheim_mm"),
          ("sigma_Z_sub_mm", "sigma_Z_sub_mm"), ("n_eff", "n_eff"), ("r_m", "r_m"),
          ("theta_ground_deg", "theta_ground_deg")]


def cell_values(name):
    """One row per cell of this dataset, with the median sigma_z of its table bin."""
    seg = pd.read_csv(OUT / "01_segments.csv")
    seg = seg[seg["dataset"] == name].copy()
    seg["r_bin"] = pd.cut(seg["r_m"], cfg["distance"], right=False).astype(str)
    seg["theta_bin"] = pd.cut(seg["theta_ground_deg"], cfg["incidence"], right=False).astype(str)
    table = pd.read_csv(OUT / "02_sigma_z_table.csv")
    keys = group_keys() + ["r_bin", "theta_bin"]
    return seg.merge(table[keys + ["sigma_z_median_mm"]], on=keys, how="left")


if __name__ == "__main__":
    ds = find_dataset(sys.argv[1])
    name = Path(ds["path"]).stem
    las, p, origin = read(ds)
    _, _, cell = cells(level(p, origin)[0])
    points = pd.DataFrame(dict(cell=cell)).merge(cell_values(name), on="cell", how="left")
    fields = FIELDS + ([("sigma_z_plane_cell_mm", "sigma_z_plane_mm"), ("theta_deg", "theta_deg")]
                       if "sigma_z_plane_mm" in points else [])
    for field, column in fields:
        las.add_extra_dim(laspy.ExtraBytesParams(name=field, type=np.float32))
        las[field] = points[column].to_numpy(np.float32)
    target = OUT / f"03_map_{name}.laz"
    las.write(target)
    print(f"{target.name}: {len(points)} points, {points['sigma_z_mm'].notna().mean():.1%} in a cell with sigma_z")
