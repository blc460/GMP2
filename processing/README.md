# processing

How noisy are the rail heights in a laser scan, how precisely is the height of
a 0.5 m piece of rail (a cell) known, and **how far out, under which angle and
with which scanner is the required precision reached?** For every cell the
scripts compute the point noise sigma_z and the precision sigma_Z of the cell
height, store the measuring geometry (range, incidence angle, beam direction to
the rail, near or far rail) and finally give per scanner the maximum range at
which the requirements (0.67 and 1.2 mm) are met.

| file | content |
| --- | --- |
| `cells.py` | settings, reading, levelling, rails, pieces and cells (shared by all steps) |
| `precision.py` | sigma_z, the sigma_Z estimators (sqrt(n), bins), Allan variance |
| `01_allan.py` ... `08_requirements.py` | the steps, run in this order |
| `plot_cells.py` | optional figure explaining the cells |
| `test_precision.py` | check of the sigma_Z estimators on simulated noise |
| `archive/` | approaches that were tried and dropped (Taubenheim n_eff, 0.1 m sub-cells, per-cell block bootstrap), see `archive/README.md` |

## Input

- Classified point clouds (LAZ). `rail_class` in `config.toml` selects the
  points: 1 = railhead (SOK, running surface), 2 = lip.
- `datasets.csv`: one row per cloud: `path`, `scanner` (the line in the
  table, e.g. `RTC360` or `FARO 1/2 Q4`), `setup` (the station: `S1`, `S2`,
  `S3`), `condition` (dry / wet) and `origin_x/y/z`, the scanner position in
  the coordinates of the cloud. LAZ does not store it, so it is taken from the
  scan pose in the original E57 file (0, 0, 0 for clouds exported in the
  scanner's own system).
- `config.toml`: all settings. `by_setup = true` splits the tables by station,
  `by_condition` by dry / wet.

## Run

From the repository root, in this order:

```
python processing/01_allan.py
python processing/02_cells.py
python processing/03_table.py
python processing/04_map.py Bu_RTC-2
python processing/05_falloff.py
python processing/06_plots.py
python processing/07_combine.py
python processing/08_requirements.py
```

01 must run before 02 (it sets the number of bins); 03 to 08 read the cell
table of 02. Optional: `python processing/test_precision.py` checks the sigma_Z
estimators (`--archive` also runs the dropped Taubenheim and sub-cell ones).

## Cells (`cells.py`, used by 01, 02 and 04)

1. Level: fit one plane through all railhead points of the cloud (least
   squares, points beyond 3 MAD dropped once, refit) and rotate cloud and
   scanner position about the centroid so that this plane is horizontal. One
   rotation per cloud (0.2 to 0.25 deg here); nothing is fitted per cell.
2. Find the rails: points closer than `rail_gap` (0.2 m) belong to the same
   rail. Each rail is followed in pieces of `window` (5 m); the direction of
   each piece comes from a principal component analysis (PCA).
3. Cells: cut each piece into cells of `cell` (0.5 m) along the rail.
   Cells with fewer than `min_points` (30) points are skipped.
4. Drop switches: a cell wider than `max_width` (0.12 m) across the rail holds
   more than one railhead (switch, crossing) and is dropped.

Optional: `python processing/plot_cells.py Bu_RTC-2` draws how the cells and the
switch rule work (`cells_explained.png`).

## 01 Allan deviation and the number of bins

sigma_Z from bins (02) assumes that the bin heights of a cell are independent,
i.e. that a bin is at least as long as the correlation of the noise. The
correlation length is read from the Allan deviation along whole 5 m pieces of
rail (only kept cells, the levelled heights as they are): the piece is cut into
bins of length tau (1 cm to 1 m, `allan_tau`), and

    sigma_A^2(tau) = 1/2 < (mean_{k+1} - mean_k)^2 >     over adjacent bins (Allan 1966)

is pooled over all pieces of a scanner and station, and of a scanner. For
uncorrelated (white) noise sigma_A^2 = sigma_z^2 / n_bin (sigma_z of the cells);
`ratio` = sigma_A / white. ratio ~ 1: the bin means are independent at this
tau; ratio > 1: correlated noise, or rail geometry (gradient, cant, crown; not
removed; a slope b adds b tau / sqrt(2)).

Allan rule, per scanner (all its stations): tau0 = the largest tau up to which
the ratio stays at or below `allan_max_ratio` (1.2), from `allan_min_tau`
(3 cm) on; shorter bins are not used because at flat angles they hold less
than one scan line and the white reference is unreliable. The number of bins
is m = floor(cell / tau0), limited to `min_bins` (4) ... `max_bins` (16), so
the bins are at least tau0 long. First result: RTC360 tau0 = 5 cm, m = 10;
FARO 1/2 Q4 tau0 = 20 cm, m = 4 (bins of 12.5 cm, slightly shorter than tau0
because of the lower limit). Beyond tau0 the ratio rises to 3 to 15 at 0.5 to
1 m (slow noise and above all the rail gradient), which no estimator inside a
cell sees.

Output: `01_allan.csv` (curve per scanner and station; station `all` = per
scanner), `01_bins.csv` (tau0, m, bin length per scanner, read by 02),
`01_allan.png`.

## 02 sigma_z, sigma_Z and geometry per cell

- sigma_z = 1.4826 MAD of the levelled heights about their median: the point
  noise, including the slope (rail inclination 1:40, cant, gradient) and the
  crown of the rail inside the cell (no model per cell, accepted; about 25 to
  65 % higher than the scatter about a plane + crown, which was tested and not
  used). Optional (`plane_fit = true`): sigma_z_plane about a plane per cell.
- sigma_Z = standard deviation of the cell height Z (median of the heights):
  - `sigma_Z_sqrt_n = sigma_z / sqrt(n)`: all n points independent (too small
    for correlated noise).
  - `sigma_Z_bins` (main estimator, batch means): the cell is cut into m bins
    along the rail (m per scanner from 01). Bin height h_k = 20 % trimmed mean
    of the heights in bin k (robust, and not stuck on the 1 mm height grid of
    the RTC360); s = standard deviation of the m bin heights; sigma_Z = s /
    sqrt(m). If a bin has fewer than `min_bin_points` (5) points, the cell uses
    the largest number of bins down to `min_bins` for which all bins have
    enough (longer bins stay independent); `n_bins` records it (0 = no value).
  - `sigma_Z_bins_lo/hi`: 95 % percentile interval from a bootstrap of the m bin
    heights (`bin_boot_runs`, Efron & Tibshirani 1993). On simulated white
    noise it covers the truth in 58 / 74 / 85 % of the cells for m = 4 / 6 / 10:
    with so few bins one cell is a rough estimate, and the interval is a guide
    only. The decisions in 08 use the medians over many cells, whose interval
    is reliable.
  sigma_Z is the random part of the uncertainty of one cell height within one
  scan; errors common to a whole scan (levelling, scanner calibration and
  position) are not included.
- Geometry: slant range r (scanner to cell centroid); incidence angle
  theta_ground between the beam and the vertical; h_m = height of the scanner
  above the cell; `beam_rail_deg` = angle in plan view between the beam and the
  rail axis of the cell, 0 = the scanner looks along the rail, 90 = across it;
  `direction` = its class (`direction`: longitudinal 0-30, oblique 30-60,
  perpendicular 60-90 deg).
- Near / far rail (`rail_side`): the partner of a cell is a kept cell on
  another, parallel rail at `gauge_centres` (1.065 m, metre gauge + head width;
  measured 1.07 m) +- `gauge_tol` across and less than half a cell along, i.e.
  the other rail of the same track (`gauge_m` = measured distance). Of the two,
  the one closer to the scanner in plan view is `near`, the other `far`; cells
  without a partner (single rails, ends of the scan) are `single`.
- Density = n / (length x width), both 2 % to 98 % of the points; rho_exp =
  cos(theta_ground) / (r^2 sin(theta_ground) dphi^2) is what the scan grid
  delivers on a horizontal surface; return_rate = density / rho_exp.

Output: `02_cells.csv`, one row per cell.

## 03 Tables

Number of cells, median and quartiles of sigma_z, medians of sigma_Z_sqrt_n
and sigma_Z_bins per scanner and station in classes of 5 m and 10 deg; for
sigma_z and sigma_Z_bins the 95 % interval of the median from a bootstrap over
the cells of the class (`table_boot_runs`); classes with fewer than
`min_cells` (10) cells are flagged `indicative`. Output: `03_table.csv`,
`03_table_pooled.csv` (without station), `03_sigma_z.png`.

## 04 Map

`04_map.py <dataset>` writes the cell values back onto the railhead points of
one cloud (`04_map_<dataset>.laz`, to view in CloudCompare; NaN where a cell
was skipped or dropped; rail_side 1 = near, 2 = far, 0 = single).

## 05 Falloff with range

How density, intensity and noise change with the slant range r, per scanner
and station, in 5 m classes:

- density = points per m2 of the strip (mean point spacing 1000 / sqrt(density)
  in mm), rho_exp as in 02 (dphi read from the scan grid of each file), return
  rate = density / rho_exp.
- intensity as stored in the file (RTC360 and FARO use different scales, so
  only the trend is comparable); log-log slope of density and intensity vs r.
- sigma_z, and sigma_r = sigma_z / cos(theta_ground), a rough noise along the
  beam (it assumes all of sigma_z is range noise and grows at grazing angles).

Comparing with a factsheet: the factsheet point spacing at 10 m is r dphi (e.g.
3 mm at 10 m = 0.3 mrad), which gives the expected density above. The factsheet
range noise is measured along the beam on a flat, usually white target facing
the scanner; here it is a steel rail seen at a flat angle, so sigma_r is
expected to be higher. Output: `05_falloff.csv`, `05_falloff_summary.csv`,
`05_falloff_pooled.csv`, `05_falloff.png`.

## 06 Figures per scanner and station

Per scanner and station, and per scanner pooled over the stations (marker =
station), into `figures/06_<scanner>_<station>_<plot>.png`:

- point density over theta_ground and over r (right axis = mean point
  spacing); cells, median per class, dashed = expected from the scan grid.
- sigma_z at each cell's r and theta_ground (colour, log scale 0.3 to 3 mm).
- `sigma_z_r_and_theta`: sigma_z over r and theta_ground, one dot per cell.
- `sigma_Z_<sqrt_n|bins>_r_and_theta`: sigma_Z over r and theta_ground, one dot
  per cell, with the requirements (0.67 and 1.2 mm, quality_criteria.tex).

## 07 Combine

`07_cells.csv` (all cells with setting, h, geometry, sigma values, direction,
rail side), `07_r_theta_table.csv` (sigma_z per range x angle class) and
`07_fit.csv` (trial fits of log sigma_z on log r, log 1/cos theta and station;
within one station cos theta = h / r, so angle and station height cannot be
separated).

## 08 Requirements: how far out, under which angle, with which scanner

Per scanner (all stations together), per incidence-angle class (and all angles)
and range class, from sigma_Z_bins:

- cells, median sigma_Z_bins with its 95 % interval (bootstrap over the cells),
  90 % quantile, median sigma_z;
- a class meets a requirement if it has at least `min_cells` (10) cells and the
  upper limit of the 95 % interval of the median is at or below 0.67 / 1.2 mm;
- r_max = the outer edge of the last range class that meets it, walking
  outwards from the scanner over the classes with enough cells and stopping at
  the first that fails. `limited_by` = `sigma_Z` (the requirement limits the
  range) or `data` (every class with enough cells passed; the measured ranges
  limit r_max, the scanner may reach further).

The same is done per rail side (near / far / single) and per beam direction
(longitudinal / oblique / perpendicular). Because sigma_Z_bins leaves out the
errors common to a scan, r_max is an upper limit of the usable range.

Output: `08_requirements_table.csv` (all classes), `08_r_max.csv`,
`08_<scanner>.png` (sigma_Z over r and theta, colour = rail side, marker =
direction, median with its 95 % interval, requirements and r_max).

## Notes

- Stations differ at the same range and angle by a factor of about 1.5 to 3
  (station 1, 1.8 m above the rails, is the best for both scanners), and they
  cover different ranges; 08 pools them per scanner, 03 keeps them apart. With
  three stations the cause (scanner height or platform, position) cannot be
  separated.
- RTC360 stores heights in 1 mm steps; in a few cells the MAD then drops close
  to zero. No correction is applied; 07 leaves cells with sigma_z = 0 out of
  the fits.
- Check of the estimators (`test_precision.py`, simulated cell of 120 x 12 grid
  points, no rail shape; truth = spread of the median height): with correlated
  noise sqrt(n) is far too small (4 to 36 %), the bins reach 45 to 90 % of the
  truth (the longer the correlation compared with a bin, the lower; this is
  why the bin length comes from the Allan rule). Without correlation all are
  about 0.8: they describe the mean, the median scatters about 1.25 times more.

## References

- Allan, D. W. (1966). Statistics of atomic frequency standards. Proc. IEEE
  54(2), 221-230.
- Efron, B., Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
  New York: Chapman & Hall.
- Tried and dropped (see `archive/README.md`): Taubenheim (1969); Schmitz et
  al. (2020, 2021); Kuensch (1989); Politis & Romano (1992).
