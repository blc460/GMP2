# processing

How noisy are the rail heights in a laser scan? For every 0.5 m piece of rail
(a cell) the scripts compute the point noise sigma_z and relate it to the
distance and the incidence angle from the scanner, per scanner and setting.

## Input

- Classified point clouds (LAZ), class 1 = railhead (running surface).
- `datasets.csv`: one row per cloud: `path`, `scanner` (the line in the
  table, e.g. `RTC360` or `FARO 1/2 Q4`), `condition` (dry / wet) and
  `origin_x/y/z`, the scanner position. LAZ does not store it, so it is taken
  from the scan pose in the original E57 file.
- `config.toml`: all settings (cell size, limits, bins).

## Run

From the repository root, in this order:

```
python processing/01_segments.py
python processing/02_sigma_z_table.py
python processing/03_map.py Bu_RTC-2
```

## Steps

1. Level: fit one plane through all railhead points of the cloud and rotate
   cloud and scanner so that this ground plane is horizontal.
2. Find the rails: points closer than `rail_gap` (0.2 m) belong to the same
   rail. Each rail is followed in pieces of `window` (5 m); the direction of
   each piece comes from a principal component analysis (PCA).
3. Cells: cut each piece into cells of `cell` (0.5 m) along the rail.
   Cells with fewer than `min_points` (30) points are skipped.
4. Drop switches: a cell wider than `max_width` (0.12 m) across the rail holds
   more than one railhead (switch, crossing) and is dropped.
5. sigma_z per cell, two variants:
   - A (main result): fit a plane to the heights, drop points beyond 3 MAD
     once, fit again; sigma_z = 1.4826 MAD of the residuals.
   - B (no model): sigma_z_med = 1.4826 MAD of the heights around their
     median.
6. Geometry per cell: horizontal distance d from the scanner, and incidence
   angle theta_ground between the laser beam and the ground plane normal
   (the angle to the cell plane, theta, is stored too).
7. Table: median sigma_z per scanner line in bins of 5 m and 10 degrees.
8. Map: the values are written back onto the railhead points of one cloud.

`cells.py` holds steps 1 to 4, used by 01 and 03.

## Output (`results/`)

- `01_segments.csv`: one row per cell (sigma_z_mm, sigma_z_med_mm, n, d_m,
  theta_deg, theta_ground_deg, position)
- `02_sigma_z_table.csv`: the result table per scanner, distance and angle bin
- `02_sigma_z.png`: sigma_z over distance and over incidence angle
- `03_map_<dataset>.laz`: railhead points with sigma_z per point, to view in
  CloudCompare (NaN where a cell was skipped or dropped)

## Notes

- B is about 20-30 % higher than A: it also contains the slope of the rail
  within the cell (along and across). A is the better measure of scanner noise.
- The plane cannot follow the crown of the railhead, so A is still slightly
  too high (about 0.1-0.2 mm).
- RTC360 stores heights in 1 mm steps; in a few cells the MAD then drops close
  to zero. No correction is applied.
- Bins with fewer than about 10 cells are only indicative.
