"""ARCHIVED, not used by the pipeline: the sigma_Z estimators that were tried and dropped (2026-11-25).

Last pipeline version that used them: git tag `taubenheim-tried` (commit fc1437a, branch rail-axis-cells).
See archive/README.md for what they did, the results and why they were dropped. Kept so that the approach can be
reproduced and cited in the report; `python processing/test_precision.py --archive` still runs it on simulated cells.

  taubenheim  sigma_Z = sigma_z / sqrt(n_eff), n_eff = effective number of measurements (Taubenheim 1969) in its
              2D form on the scan grid (Schmitz et al. 2020, Eq. 2), autocorrelation estimated from the residuals
              after a plane per cell (Schmitz et al. 2021).
  sub         0.1 m sub-cells of the raw heights, spread of their 20 % trimmed means about a line along the rail,
              divided by sqrt(m) (m ~ 5). Replaced by precision.sigma_Z_bins (4 bins of the residuals).
"""
import numpy as np
from scipy.stats import trim_mean


# --- Taubenheim ------------------------------------------------------------------------------------------------

def detrend(u, v, z, model="plane"):
    """Residuals of z after removing the rail surface (u along, v across the rail): plane, or plane + crown (v^2)."""
    cols = [np.ones_like(u), u, v] + ([v ** 2] if model == "plane+crown" else [])
    A = np.column_stack(cols)
    return z - A @ np.linalg.lstsq(A, z, rcond=None)[0]


def autocorrelation_2d(e, col, row):
    """Empirical autocorrelation of e on the scan grid (ScanColumn, ScanRow), gaps allowed.
    Returns K and the number of point pairs for all lags (centre = lag 0) and the grid size (nx, ny)."""
    i, j = col - col.min(), row - row.min()
    nx, ny = int(i.max()) + 1, int(j.max()) + 1
    val, cnt = np.zeros((nx, ny)), np.zeros((nx, ny))
    np.add.at(val, (i, j), e)
    np.add.at(cnt, (i, j), 1)
    occ = cnt > 0
    val[occ] /= cnt[occ]                              # two returns on one grid node: their mean
    val[occ] -= val[occ].mean()
    shape = (2 * nx - 1, 2 * ny - 1)

    def corr(a):                                      # sum over all pairs at each lag, via FFT
        f = np.fft.rfft2(a, shape)
        return np.fft.fftshift(np.fft.irfft2(f * np.conj(f), shape), axes=(0, 1))

    pairs = np.rint(corr(occ.astype(float)))
    cov = np.divide(corr(val), pairs, out=np.zeros(shape), where=pairs > 0)
    var = cov[nx - 1, ny - 1]
    return (cov / var if var > 0 else np.zeros(shape)), pairs, (nx, ny)


def n_eff_taubenheim(e, col, row, max_lag_fraction=0.1):
    """Effective number of measurements of the residuals e on the scan grid, clipped to [1, n].

        Var(mean) = sigma^2/n * (1 + sum_{lag != 0} pairs(lag)/n * K(lag))   =>   n_eff = n / (1 + sum ...)

    This is Taubenheim's n_eff = n / (1 + 2 sum_k (n-k)/n K(k)) in 2D; pairs(lag)/n takes the place of (n-k)/n on
    the irregular footprint of a cell. Lags are summed up to max_lag_fraction of the grid extent per direction
    (m = n/10 in Schmitz et al. 2020), at least one grid step."""
    K, pairs, (nx, ny) = autocorrelation_2d(np.asarray(e, float), np.asarray(col, int), np.asarray(row, int))
    mx = max(1, int(round(max_lag_fraction * nx)))
    my = max(1, int(round(max_lag_fraction * ny)))
    win = (slice(nx - 1 - mx, nx + mx), slice(ny - 1 - my, ny + my))
    k, p = K[win], pairs[win].copy()
    nodes = p[mx, my]                                 # number of occupied grid nodes = pairs at lag 0
    p[mx, my] = 0
    n = len(e)
    return float(np.clip(n / (1 + np.sum(p * k) / nodes), 1, n))


# --- sub-cells (batch means) -----------------------------------------------------------------------------------

def sigma_Z_sub(u, z, sub_cell=0.1, min_points=5, min_cells=4, trim=0.2):
    """sigma_Z from the trimmed means (trim per side) of sub-cells of length sub_cell along the rail coordinate u.
    Returns (sigma_Z, m) with m the number of usable sub-cells; NaN if m < min_cells."""
    k = np.floor((u - u.min()) / sub_cell).astype(int)
    centre, height = [], []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() >= min_points:
            centre.append(u[m].mean())
            height.append(trim_mean(z[m], trim))
    m = len(height)
    if m < max(min_cells, 3):
        return np.nan, m
    res = np.asarray(height) - np.polyval(np.polyfit(centre, height, 1), centre)
    return np.sqrt(np.sum(res ** 2) / (m - 2)) / np.sqrt(m), m
