# processing

How noisy are the rail heights in a laser scan, and how precisely is the rail
height known? For every 0.5 m piece of rail (a cell) the scripts compute the
point noise sigma_z and the precision sigma_Z of the cell height, and relate
them to the slant range and the incidence angle from the scanner, per scanner,
setting and station.

| file | content |
| --- | --- |
| `cells.py` | settings, reading, levelling, rails, pieces and cells (shared by all steps) |
| `precision.py` | sigma_z and the three sigma_Z estimators |
| `01_segments.py` ... `06_combine.py` | the steps, run in this order |
| `plot_cells.py` | optional figure explaining the cells |
| `test_precision.py` | check of the sigma_Z estimators on simulated noise |

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
python processing/06_combine.py
```

Optional: `python processing/test_precision.py` checks the sigma_Z estimators.

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
6. sigma_Z per cell, the standard deviation of the cell height Z (median of the
   heights), three ways (`precision.py`):
   - `sigma_Z_sqrt_n = sigma_z / sqrt(n)`: all n points assumed independent.
   - `sigma_Z_taubenheim = sigma_z / sqrt(n_eff)`: effective number of
     measurements n_eff (Taubenheim 1969) in its 2D form on the scan grid
     (ScanColumn x ScanRow, Schmitz et al. 2020, Eq. 2). The autocorrelation
     is estimated from the data after removing a plane per cell (`detrend`),
     because the formula needs a trend-free, stationary series (Schmitz et al.
     2021). Lags are summed up to `max_lag_fraction` (10 %) of the grid extent
     per direction. sigma_z itself is not changed.
   - `sigma_Z_sub`: the cell is cut into sub-cells of `sub_cell` (0.1 m) along
     the rail; the spread of their heights (20 % trimmed mean, so that the 1 mm
     height grid of the RTC360 does not give exact zeros) about a straight line along the rail
     (the gradient is a trend), divided by sqrt(m) (batch means). Needs at least
     `min_sub_cells` sub-cells with `min_sub_points` points.
   Both n_eff and the sub-cells only see correlation inside the cell, so all
   three are lower bounds; errors common to a whole scan (levelling, scanner
   calibration) are not included.
7. Geometry per cell: slant range r from the scanner to the cell centre, and incidence
   angle theta_ground between the laser beam and the ground plane normal
   (with `plane_fit`, also theta to the cell plane).
8. Table: number of cells, median sigma_z and the medians of the three
   sigma_Z per scanner and station in bins of 5 m and 10 degrees; bins with
   fewer than `min_cells` (10) cells are flagged `indicative`; a pooled table
   per scanner is written as a summary.
9. Map: the values are written back onto the railhead points of one cloud.

`cells.py` holds steps 1 to 4, used by 01 and 03.
Optional: `python processing/plot_cells.py Bu_RTC-2` draws how the cells and the
switch rule work (`cells_explained.png`).

## Falloff with range (04)

`04_falloff.py` shows how density, intensity and noise change with the slant
range r, per scanner and station, in 5 m bins:

- density = points per m2 of the strip (mean point spacing 1000 / sqrt(density) in mm); rho_exp = cos(theta_ground) / (r^2 sin(theta_ground) dphi^2)
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

- point density in pts/m2 over theta_ground and over r, right axis = mean
  point spacing in mm = 1000 / sqrt(pts/m2): cells, median per bin, dashed =
  expected from the scan grid. The density of a cell is n / (length x width),
  with length and width from 2 % to 98 % of its points along and across the
  rail (typically 0.48 m x 0.042 m = 0.020 m2 on the railhead, 0.48 m x
  0.016 m = 0.008 m2 on the lip), not the nominal 0.5 m cell. The spacing is a
  mean: at flat angles the points are much farther apart along the beam
  direction than between scan columns.
- sigma_z at each cell's r and theta_ground (colour, log scale 0.3 to 3 mm,
  the same in all figures).
- `sigma_z_r_and_theta`: sigma_z over r and theta_ground, one dot per cell,
  linear axis up to `sigma_z_max`, no median line.
- `sigma_Z_<sqrt_n|taubenheim|sub>_r_and_theta`: sigma_Z over r and
  theta_ground, one dot per cell, with the required values (`requirements`,
  0.67 and 1.2 mm from quality_criteria.tex) as lines.

## Output (`results/sok/` or `results/lip/`, by `rail_class`)

- `01_segments.csv`: one row per cell (sigma_z_mm, sigma_Z_sqrt_n_mm, n_eff,
  sigma_Z_taubenheim_mm, sigma_Z_sub_mm, n_sub, Z_m, n, r_m, theta_ground_deg,
  density, position, rail and piece; with `plane_fit` also sigma_z_plane_mm and theta_deg)
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
- Check of the sigma_Z estimators (`test_precision.py`, simulated cell of
  120 x 12 grid points): without correlation all three match the truth; with
  correlated noise sqrt(n) is far too small, Taubenheim reaches 40 to 75 % and
  the sub-cells 80 to 90 % of the true value. The empirical autocorrelation of
  a short series is biased low, so n_eff is too large: sigma_Z_taubenheim is a
  lower bound.

## References

- Taubenheim, J. (1969). Statistische Auswertung geophysikalischer und
  meteorologischer Daten. Leipzig: Geest & Portig.
- Schmitz, B., Kuhlmann, H., Holst, C. (2020). Investigating the resolution
  capability of terrestrial laser scanners and its impact on the effective
  number of measurements. ISPRS J. Photogramm. Remote Sens. 159, 41-52.
- Schmitz, B., Kuhlmann, H., Holst, C. (2021). Towards the empirical
  determination of correlations in terrestrial laser scanner range
  observations and the comparison of the correlation structure of different
  scanners. ISPRS J. Photogramm. Remote Sens. 182, 228-241.
