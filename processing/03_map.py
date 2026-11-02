"""Step 3: map the results back onto one point cloud, to see where which sigma_z is reached.

Usage: python processing/03_map.py Bu_RTC-2   (any part of the file name of one row of datasets.csv)

Every railhead point gets the values of its 0.5 m x 0.5 m cell: sigma_z_cell_mm (sigma_z of the cell),
sigma_z_bin_mm (median sigma_z of the cell's scanner, distance bin and angle bin), d_m and theta_deg.
Points in cells that were skipped (fewer than min_points) get NaN.

Input: results/01_segments.csv, results/02_sigma_z_table.csv. Output: results/03_map_<dataset>.laz
"""
from pathlib import Path
import sys
import tomllib

import laspy
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
with open(HERE / "config.toml", "rb") as f:
    cfg = tomllib.load(f)
out = HERE / cfg["results"]

datasets = pd.read_csv(HERE / cfg["datasets"], keep_default_na=False)
ds = datasets[datasets["path"].str.contains(sys.argv[1], regex=False)]
if len(ds) != 1:
    sys.exit(f"'{sys.argv[1]}' matches {len(ds)} rows of {cfg['datasets']}, it must match one")
path = Path(ds["path"].iloc[0])

cells = pd.read_csv(out / "01_segments.csv")
cells = cells[cells["dataset"] == path.stem].copy()
cells["d_bin"] = pd.cut(cells["d_m"], cfg["distance"], right=False).astype(str)
cells["theta_bin"] = pd.cut(cells["theta_deg"], cfg["incidence"], right=False).astype(str)
table = pd.read_csv(out / "02_sigma_z_table.csv")
keys = ["scanner", "d_bin", "theta_bin"] + (["condition"] if cfg["by_condition"] else [])
cells = cells.merge(table[keys + ["sigma_z_median_mm"]], on=keys, how="left")

las = laspy.read(HERE / path)
mapped = laspy.LasData(las.header)
mapped.points = las.points[np.asarray(las.classification) == cfg["class_sok"]]
ij = np.floor(np.column_stack([mapped.x, mapped.y]) / cfg["cell"]).astype(int)
points = pd.DataFrame(ij, columns=["i", "j"]).merge(cells, on=["i", "j"], how="left")

for name, column in (("sigma_z_cell_mm", "sigma_z_mm"), ("sigma_z_bin_mm", "sigma_z_median_mm"),
                     ("d_m", "d_m"), ("theta_deg", "theta_deg")):
    mapped.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
    mapped[name] = points[column].to_numpy(np.float32)
target = out / f"03_map_{path.stem}.laz"
mapped.write(target)
print(f"{target.name}: {len(points)} railhead points, {points['sigma_z_mm'].notna().mean():.1%} in a cell with sigma_z")
