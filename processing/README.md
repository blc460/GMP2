# processing

How noisy are the rail heights in a laser scan? For every 0.5 m piece of rail
(a cell) the scripts compute the point noise sigma_z and relate it to the
slant range and the incidence angle from the scanner, per scanner, setting and
station.

## Input

- Classified point clouds (LAZ). `rail_class` in `config.toml` selects the
  points: 1 = railhead (SOK, running surface), 2 = lip.
- `datasets.csv`: one row per cloud: `path`, `scanner` (the line in the
  table, e.g. `RTC360` or `FARO 1/2 Q4`), `setup` (the station: `S1`, `S2`,
  `S3`), `condition` (dry / wet) and
  `origin_x/y/z`, the scanner position. LAZ does not store it, so it is taken
  from the scan pose in the original E57 file.
- `config.toml`: all settings (cell size, limits, bins). `by_setup = true`
  splits the tables by station, `by_condition` by dry / wet.

## Run

From the repository root, in this order:

```
python processing/01_segments.py
python processing/02_sigma_z_table.py
python processing/03_map.py Bu_RTC-2
python processing/04_falloff.py
python processing/05_plots.py
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
5. sigma_z per cell: the median of the levelled heights of the cell, and
   sigma_z = 1.4826 MAD of all heights around it (no model per cell).
   Optional (`plane_fit = true`): fit a plane per cell, drop points beyond
   3 MAD once, fit again; sigma_z_plane = 1.4826 MAD of the residuals.
6. Geometry per cell: slant range r from the scanner to the cell centre, and incidence
   angle theta_ground between the laser beam and the ground plane normal
   (with `plane_fit`, also theta to the cell plane).
7. Table: median sigma_z per scanner and station in bins of 5 m and 10
   degrees; a pooled table per scanner is written as a summary.
8. Map: the values are written back onto the railhead points of one cloud.

`cells.py` holds steps 1 to 4, used by 01 and 03.
Optional: `python processing/plot_cells.py Bu_RTC-2` draws how the cells and the
switch rule work (`cells_explained.png`).

## Falloff with range (04)

`04_falloff.py` shows how density, intensity and noise change with the slant
range r, per scanner and station, in 5 m bins:

- density = points per m2 of the strip; rho_exp = cos(theta_ground) / (r^2 sin(theta_ground) dphi^2)
  is what the scan grid delivers on a horizontal surface (dphi is read from the
  scan grid of each file); return rate = density / rho_exp.
- intensity as stored in the file (RTC360 and FARO use different scales, so
  only the trend is comparable); log-log slope of density and intensity vs r.
- sigma_z, and sigma_r = sigma_z / cos(theta_ground), a rough noise along the
  beam (it assumes all of sigma_z is range noise and grows at grazing angles).

Comparing with a factsheet: the factsheet point spacing at 10 m is r dphi
(e.g. 3 mm at 10 m = 0.3 mrad), which gives the expected density above. The
factsheet range noise is measured along the beam on a flat, usually white
target facing the scanner; here it is a steel rail seen at a flat angle, so
sigma_r is expected to be higher.

## Figures per scanner and station (05)

`05_plots.py` draws three separate plots per scanner and station, and per
scanner pooled over the stations (marker = station), into `figures/`:

- point density in pts/dm2 (points per 10 x 10 cm; right axis in pts/m2) over
  theta_ground and over r: cells, median per bin, dashed = expected from the
  scan grid. The density of a cell is n / (length x width), with length and
  width from 2 % to 98 % of its points along and across the rail (typically
  0.48 m x 0.042 m = 0.020 m2 on the railhead, 0.48 m x 0.016 m = 0.008 m2 on
  the lip), not the nominal 0.5 m cell.
- sigma_z at each cell's r and theta_ground (colour, log scale 0.3 to 3 mm,
  the same in all figures).

## Output (`results/sok/` or `results/lip/`, by `rail_class`)

- `01_segments.csv`: one row per cell (sigma_z_mm, n, r_m, theta_ground_deg,
  density, position; with `plane_fit` also sigma_z_plane_mm and theta_deg)
- `02_sigma_z_table.csv`: the result table per scanner, station, slant range and
  angle bin; `02_sigma_z_table_pooled.csv`: the same without station
- `02_sigma_z.png`: sigma_z over slant range and over incidence angle
- `04_falloff.csv`, `04_falloff_summary.csv`, `04_falloff.png`: falloff with
  range per scanner and station; `04_falloff_pooled.csv` without station
- `figures/05_<scanner>_<station>_<plot>.png`: point density and sigma_z per scanner and
  station, and pooled per scanner
- `03_map_<dataset>.laz`: railhead points with sigma_z per point, to view in
  CloudCompare (NaN where a cell was skipped or dropped)

## Notes

- Stations are reported separately: at the same range and angle they
  differ by a factor of about 1.5 to 3 (station 1, 1.8 m above the rails, is
  the best for both scanners), and they cover different ranges. A table
  pooled over stations mixes this and is only a summary. With three stations
  the cause (scanner height or platform, position) cannot be separated.

- sigma_z uses the heights after the global levelling, without a model per
  cell (chosen for simplicity and transparency). It therefore also contains
  the slope of the rail within the 0.5 m cell (rail inclination 1:40, cant,
  gradient) and is about 20-30 % higher than the optional plane fit, which
  removes this slope (but not the crown of the railhead).
- RTC360 stores heights in 1 mm steps; in a few cells the MAD then drops close
  to zero. No correction is applied.
- Bins with fewer than about 10 cells are only indicative.
