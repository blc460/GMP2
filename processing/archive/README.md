# archive: sigma_Z estimators that were tried and dropped

Decided with the supervisors on 2026-11-25. The code is kept in
`taubenheim.py` (not imported by the pipeline) so that the approach can be
reproduced and described in the report. The last pipeline version that used it
is the git tag `taubenheim-tried` (commit fc1437a on `rail-axis-cells`);
`python processing/test_precision.py --archive` still runs it on simulated cells.

## Taubenheim: effective number of measurements

sigma_Z = sigma_z / sqrt(n_eff), with n_eff the effective number of
measurements of Taubenheim (1969) in its 2D form on the scan grid (ScanColumn x
ScanRow, Schmitz et al. 2020, Eq. 2):

    n_eff = n / (1 + sum_{lag != 0} pairs(lag) / n * K(lag))

K = empirical autocorrelation of the residuals after a plane per cell
(trend-free series, Schmitz et al. 2021), computed with an FFT over the
occupied grid nodes, summed up to 10 % of the grid extent per direction.

Results (all scans, railhead, median per cell): n about 200 to 280, n_eff about
73 to 76, so sigma_Z_taubenheim was about 1.9 times sigma_z / sqrt(n):

| scanner | sigma_z [mm] | sqrt(n) [mm] | Taubenheim [mm] | sub-cells [mm] |
| --- | --- | --- | --- | --- |
| FARO 1/1 Q3 | 2.28 | 0.121 | 0.227 | 0.126 |
| FARO 1/2 Q4 | 1.53 | 0.104 | 0.155 | 0.099 |
| RTC360 | 0.99 | 0.063 | 0.114 | 0.075 |

Why it was dropped:

- The empirical autocorrelation of a short series (one 0.5 m cell, a few
  hundred points on an irregular footprint) is biased low, so n_eff is too
  large; in the simulation it reached only 30 to 80 % of the true sigma_Z, the
  less the longer the correlation.
- The result depends on choices without a clear rule: the maximum lag (10 %)
  and the trend model (a plane leaves the crown of the railhead in the data,
  which looks like strong correlation across the rail).
- The scan grid of a cell at grazing angles is sparse and has gaps; the 2D
  formula with pair counts is an approximation there.
- A batch-means estimator (bins along the rail) is simpler, needs no
  autocorrelation model and is checked directly by the Allan deviation.

## Sub-cells of 0.1 m (raw heights)

sigma_Z_sub = spread of the 20 % trimmed means of 0.1 m sub-cells (about 5 per
cell) about a straight line along the rail, divided by sqrt(m). Replaced by
`precision.sigma_Z_bins` (4 bins of 0.125 m of the residuals after plane +
crown), because

- the raw heights contain the cross profile (crown, cant); each sub-cell
  samples it differently at grazing angles, which adds spread;
- with 5 sub-cells and a fitted line it had 3 degrees of freedom: one cell's
  value scattered by a factor 4 to 7 (10 to 90 % quantile);
- together with sigma_z of the raw heights (also too large) its median came
  out almost equal to sigma_z / sqrt(n), so correlation could not be seen.

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
