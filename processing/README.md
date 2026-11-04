# processing

Point noise sigma_z (MAD) of tram rails per scanner, horizontal distance and
incidence angle. Input: classified point clouds (class 1 = railhead).

## Run

From the repository root:

```
python processing/01_segments.py
python processing/02_sigma_z_table.py
python processing/03_map.py Bu_RTC-2
```

Settings are in `config.toml`. `datasets.csv` has one row per point cloud
(path, scanner, condition, scanner origin). The origin is the scan pose (translation)
in the original E57 file of the scan, because LAS/LAZ does not store the scanner position.

## Method

1. Ground plane: one plane through all railhead points (class 1), outliers
   beyond 3 MAD dropped once; points and scanner are rotated so that the plane
   is horizontal.
2. Rails: connected groups of points in plan view, cut into 5 m windows and
   connected pieces; each piece is cut into 0.5 m cells along its main
   direction (PCA). Cells wider than 0.12 m across the rail (switches,
   crossings) and cells with fewer than 30 points are dropped.
3. Per cell: plane z = b0 + b1 x + b2 y, residuals beyond 3 MAD dropped once,
   refit; sigma_z = 1.4826 MAD of the residuals.
4. d = horizontal distance scanner to cell; theta = angle between the scan ray
   and the cell plane normal; theta_ground = angle between the scan ray and
   the ground plane normal.
5. Table: median and quartiles of sigma_z per scanner, distance bin and
   incidence bin (theta_ground).

`cells.py` holds the steps shared by 01 and 03.

## Output (`results/`)

- `01_segments.csv`: one row per cell (with theta_deg and theta_ground_deg)
- `02_sigma_z_table.csv`: the result table
- `02_sigma_z.png`: sigma_z over distance and incidence angle
- `03_map_<dataset>.laz`: railhead points of one cloud (named on the command line) with
  sigma_z_cell_mm, sigma_z_bin_mm, d_m, theta_deg and theta_ground_deg; NaN in cells that
  were skipped or dropped
