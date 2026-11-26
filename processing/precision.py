"""Point noise sigma_z of a cell and the precision sigma_Z of its height Z (used by 01_allan.py and 02_cells.py).

All values come from the levelled heights z of the cell as they are: no surface model is fitted per cell, so the
slope, cant and crown of the rail inside the cell stay in the numbers (accepted, see README).

sigma_z  : 1.4826 * MAD of the heights about their median (point noise, rail shape in the cell included).
sigma_Z  : standard deviation of the cell height Z (median of the heights). Two estimators:

  sqrt_n  sigma_z / sqrt(n). All n points assumed independent (too small if the noise is correlated).
  bins    The cell is cut into m bins along the rail. Bin height h_k = 20 % trimmed mean of the heights in bin k.
          With s = standard deviation of the m bin heights:
                sigma_Z = s / sqrt(m)                                                    (batch means)
          Correlation shorter than a bin is contained in s; the bins must be at least as long as the correlation,
          so m comes from the Allan deviation (01_allan.py, `allan_rule`). If a bin has fewer than `min_points`
          points, the cell uses fewer bins (m - 1, ... down to `min_bins`): longer bins are still independent.
          95 % interval: bootstrap of the m bin heights (percentile interval of s / sqrt(m) over `runs`
          resamples). With few bins (m <= 5) the interval is coarse and too narrow; it is a rough guide only.

Only correlation inside the cell is seen, so sigma_Z is the random part of the uncertainty of one cell height within
one scan; errors common to a whole scan (levelling, scanner calibration, scanner position) are not included.

Tried and dropped: Taubenheim n_eff (archive/taubenheim.py) and a per-cell block bootstrap of the mean.
"""
import numpy as np
from scipy.stats import trim_mean

MAD_TO_SIGMA = 1.4826


def mad_sigma(x):
    """Robust standard deviation: 1.4826 * median absolute deviation from the median."""
    x = np.asarray(x, float)
    return MAD_TO_SIGMA * np.median(np.abs(x - np.median(x)))


def sigma_Z_sqrt_n(sigma_z, n):
    return sigma_z / np.sqrt(n)


# --- bins (batch means) -----------------------------------------------------------------------------------------

def bin_index(u, m):
    """Index 0 .. m-1 of m equal bins over the extent of u."""
    lo, hi = u.min(), u.max()
    return np.minimum(((u - lo) / (hi - lo + 1e-12) * m).astype(int), m - 1)


def bin_heights(u, z, m, min_points=5, trim=0.2):
    """Trimmed-mean heights of m equal bins along u, or None if a bin has fewer than min_points points."""
    k = bin_index(u, m)
    if np.bincount(k, minlength=m).min() < min_points:
        return None
    return np.array([trim_mean(z[k == j], trim) for j in range(m)])


def sigma_Z_bins(u, z, m=8, min_bins=4, min_points=5, trim=0.2, runs=1000, level=0.95, rng=None):
    """sigma_Z from bins of the heights z along u (batch means, see module doc), with m bins or, if a bin has fewer
    than min_points points, the largest number of bins between min_bins and m for which all bins have enough.
    Returns (sigma_Z, lower, upper, m_used); NaN and m_used = 0 if even min_bins bins are too sparse."""
    for mm in range(m, min_bins - 1, -1):
        h = bin_heights(u, z, mm, min_points, trim)
        if h is not None:
            break
    else:
        return np.nan, np.nan, np.nan, 0
    s = np.std(h, ddof=1) / np.sqrt(mm)
    rng = np.random.default_rng(0) if rng is None else rng
    boot = np.std(h[rng.integers(0, mm, size=(runs, mm))], axis=1, ddof=1) / np.sqrt(mm)
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return s, lo, hi, mm


# --- Allan deviation --------------------------------------------------------------------------------------------

def allan(u, z, taus, var, min_points=3):
    """Allan variance of the heights z along u for each bin length tau: half the mean squared difference of the
    means of adjacent bins (both with at least min_points points). Also the white-noise reference var / n_bin, with
    var the point noise variance (sigma_z^2 of the cells; var(z) of a whole piece would hold the rail geometry).
    Returns arrays (sum of half squared differences, number of differences, sum of white reference) per tau, to be
    pooled over pieces by the caller."""
    e = z - np.median(z)
    a_sum, a_cnt, w_sum = np.zeros(len(taus)), np.zeros(len(taus)), np.zeros(len(taus))
    for i, tau in enumerate(taus):
        j = np.floor((u - u.min()) / tau).astype(int)
        cnt = np.bincount(j)
        mean = np.bincount(j, e) / np.maximum(cnt, 1)
        ok = (cnt[:-1] >= min_points) & (cnt[1:] >= min_points)
        if not ok.any():
            continue
        d = np.diff(mean)[ok]
        n_pair = 2 / (1 / cnt[:-1][ok] + 1 / cnt[1:][ok])        # harmonic mean of the two bin sizes
        a_sum[i] = 0.5 * np.sum(d ** 2)
        a_cnt[i] = len(d)
        w_sum[i] = np.sum(var / n_pair)
    return a_sum, a_cnt, w_sum
