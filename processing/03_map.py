"""Step 3: map the results back onto one point cloud, to see where which sigma_z is reached.

Usage: python processing/03_map.py Bu_RTC-2   (any part of the file name of one row of datasets.csv)

Every railhead point gets the values of its cell (the same cells as in 01_segments.py): sigma_z_cell_mm
(sigma_z of the cell), sigma_z_bin_mm (median sigma_z of the cell's scanner, distance bin and angle bin),
d_m, theta_deg and theta_ground_deg. Points in cells that were skipped or dropped get NaN.

Input: results/01_segments.csv, results/02_sigma_z_table.csv. Output: results/03_map_<dataset>.laz
"""
from pathlib import Path
import sys

import laspy
import numpy as np
import pandas as pd

from cells import HERE, cfg, cells, level, read

out = HERE / cfg["results"]
datasets = pd.read_csv(HERE / cfg["datasets"], keep_default_na=False)
ds = datasets[datasets["path"].str.contains(sys.argv[1], regex=False)]
if len(ds) != 1:
    sys.exit(f"'{sys.argv[1]}' matches {len(ds)} rows of {cfg['datasets']}, it must match one")
ds = ds.iloc[0]
name = Path(ds["path"]).stem

segments = pd.read_csv(out / "01_segments.csv")
segments = segments[segments["dataset"] == name].copy()
segments["d_bin"] = pd.cut(segments["d_m"], cfg["distance"], right=False).astype(str)
segments["theta_bin"] = pd.cut(segments["theta_deg"], cfg["incidence"], right=False).astype(str)
table = pd.read_csv(out / "02_sigma_z_table.csv")
keys = ["scanner", "d_bin", "theta_bin"] + (["condition"] if cfg["by_condition"] else [])
segments = segments.merge(table[keys + ["sigma_z_median_mm"]], on=keys, how="left")

las, p, origin = read(ds)
_, cell = cells(level(p, origin)[0])
points = pd.DataFrame(dict(cell=cell)).merge(segments, on="cell", how="left")

for field, column in (("sigma_z_cell_mm", "sigma_z_mm"), ("sigma_z_bin_mm", "sigma_z_median_mm"),
                      ("d_m", "d_m"), ("theta_deg", "theta_deg"), ("theta_ground_deg", "theta_ground_deg")):
    las.add_extra_dim(laspy.ExtraBytesParams(name=field, type=np.float32))
    las[field] = points[column].to_numpy(np.float32)
target = out / f"03_map_{name}.laz"
las.write(target)
print(f"{target.name}: {len(points)} railhead points, {points['sigma_z_mm'].notna().mean():.1%} in a cell with sigma_z")
