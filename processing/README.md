# processing

Point noise sigma_z (MAD) of tram rails per scanner, horizontal distance and
incidence angle. Input: classified point clouds (class 1 = railhead).

## Run

From the repository root:

```
python processing/01_segments.py
python processing/02_sigma_z_table.py
```

Settings are in `config.toml`. `datasets.csv` has one row per point cloud
(path, scanner, condition, scanner origin).

## Method

1. Railhead points (class 1) in 0.5 m x 0.5 m cells in plan view; cells
   with fewer than 30 points are skipped.
2. Per cell: plane z = b0 + b1 x + b2 y, residuals beyond 3 MAD dropped once,
   refit; sigma_z = 1.4826 MAD of the residuals.
3. d = horizontal distance scanner to cell, theta = angle between the scan ray
   and the plane normal.
4. Table: median and quartiles of sigma_z per scanner, distance bin and
   incidence bin.

## Output (`results/`)

- `01_segments.csv`: one row per cell
- `02_sigma_z_table.csv`: the result table
- `02_sigma_z.png`: sigma_z over distance and incidence angle
