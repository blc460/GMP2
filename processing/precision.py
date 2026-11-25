"""Point noise sigma_z of a cell and the precision sigma_Z of its height Z (used by 01_segments.py and 07_allan.py).

Residuals: the rail surface of the cell is removed first by a least-squares fit
                z = a + b u + c v + d v^2        (u along, v across the rail; d v^2 = crown of the railhead)
           with one rejection of residuals beyond k MAD. The residuals e are the measurement noise; the slope along
           the rail (gradient), the cant / inclination across it and the crown are geometry, not noise.

sigma_z  : 1.4826 * MAD of the residuals e (point noise). The raw value (MAD of the levelled heights about their
           median, rail shape included) is kept as sigma_z_raw.
sigma_Z  : standard deviation of the cell height Z. Three estimators:

  sqrt_n  sigma_z / sqrt(n). All n points assumed independent (lower bound if the noise is correlated).
  bins    The cell is cut into m bins along the rail (m = 4 by default, 0.125 m each). Bin height h_k = trimmed mean of
          the residuals in bin k. Because a and b are already fitted, the h_k scatter about zero with m - 2 degrees
          of freedom:
                s^2 = sum h_k^2 / (m - 2),        sigma_Z = s / sqrt(m)              (batch means)
          Correlation shorter than a bin is contained in s. Assumes the bins are independent, i.e. longer than the
          correlation length: checked with the Allan deviation (07_allan.py). With m = 4 the estimate has only
          2 degrees of freedom; its 95 % interval from the chi-square distribution is returned too.
  boot    Circular block bootstrap along the rail (Kuensch 1989; Politis & Romano 1992): the residuals are cut into
          slices of `slice` along u, blocks of `block` (contiguous slices, any start, wrapping round) are drawn with
          replacement until the cell is full again, and sigma_Z = standard deviation of the mean of the
          bootstrapped residuals. Keeps correlation shorter than the block, misses longer correlation, so it lies
          between sqrt_n and the truth. A point-wise (iid) bootstrap would only reproduce sqrt_n.

Only correlation inside the cell is seen, so all three are lower bounds of the full uncertainty of a rail height;
errors common to a whole scan (levelling, scanner calibration) are not included.

The Taubenheim estimator (n_eff from the 2D autocorrelation) was tried and dropped: archive/taubenheim.py.
"""
import numpy as np
from scipy.stats import chi2, trim_mean

MAD_TO_SIGMA = 1.4826


def mad_sigma(x):
    """Robust standard deviation: 1.4826 * median absolute deviation from the median."""
    x = np.asarray(x, float)
    return MAD_TO_SIGMA * np.median(np.abs(x - np.median(x)))


def sigma_Z_sqrt_n(sigma_z, n):
    return sigma_z / np.sqrt(n)


# --- residuals ----------------------------------------------------------------------------------------------------

def design(u, v, model="plane+crown", u_order=1):
    """Columns of the surface model: 1, u (.. u^u_order), v and, for plane+crown, v^2."""
    cols = [np.ones_like(u)] + [u ** k for k in range(1, u_order + 1)] + [v]
    if model == "plane+crown":
        cols.append(v ** 2)
    elif model != "plane":
        raise ValueError(f"unknown model {model!r}")
    return np.column_stack(cols)


def residuals(u, v, z, model="plane+crown", k=3.0, u_order=1):
    """Residuals of z about the rail surface (least squares; points beyond k MAD dropped once, then refitted).
    All points keep a residual (the dropped ones too), so that noise statistics stay robust but complete."""
    u = u - u.mean()
    v = v - v.mean()
    A = design(u, v, model, u_order)
    b = np.linalg.lstsq(A, z, rcond=None)[0]
    e = z - A @ b
    s = mad_sigma(e)
    if s > 0:
        keep = np.abs(e - np.median(e)) <= k * s
        if keep.sum() > A.shape[1]:
            b = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
            e = z - A @ b
    return e


# --- bins (batch means) -----------------------------------------------------------------------------------------

def bin_index(u, m):
    """Index 0 .. m-1 of m equal bins over the extent of u."""
    lo, hi = u.min(), u.max()
    return np.minimum(((u - lo) / (hi - lo + 1e-12) * m).astype(int), m - 1)


def sigma_Z_bins(u, e, m=4, min_points=5, trim=0.2, level=0.95):
    """sigma_Z from m bins of the residuals e along u (batch means, see module doc).
    Returns (sigma_Z, lower, upper, m_used); NaN if fewer than m bins have min_points points."""
    k = bin_index(u, m)
    h = [trim_mean(e[k == j], trim) for j in range(m) if (k == j).sum() >= min_points]
    if len(h) < m:
        return np.nan, np.nan, np.nan, len(h)
    dof = m - 2
    s = np.sqrt(np.sum(np.square(h)) / dof) / np.sqrt(m)
    a = (1 - level) / 2
    return s, s * np.sqrt(dof / chi2.ppf(1 - a, dof)), s * np.sqrt(dof / chi2.ppf(a, dof)), m


# --- moving-block bootstrap -------------------------------------------------------------------------------------

def sigma_Z_boot(u, e, block=0.0625, slice_=0.01, runs=200, rng=None):
    """Circular block bootstrap of the mean of the residuals e along u (see module doc). Returns sigma_Z.
    The residuals are summed per slice of `slice_`; a block is L = block / slice_ contiguous slices (wrapping round
    the end of the cell, so every slice is used equally often); k = S / L blocks rebuild a cell of S slices.
    The bootstrap variance of a mean is biased low by the factor (1 - L/S) even for white noise; it is corrected."""
    rng = np.random.default_rng(0) if rng is None else rng
    j = np.floor((u - u.min()) / slice_).astype(int)
    S = j.max() + 1
    L = max(1, int(round(block / slice_)))
    if 2 * L > S:
        return np.nan
    sums = np.bincount(j, e, S)
    cnts = np.bincount(j, minlength=S).astype(float)
    wrap = np.r_[np.arange(S), np.arange(L - 1)]               # circular: blocks may run past the end
    bsum = np.convolve(sums[wrap], np.ones(L), "valid")        # sum and count of the S blocks
    bcnt = np.convolve(cnts[wrap], np.ones(L), "valid")
    k = max(2, int(round(S / L)))
    pick = rng.integers(0, S, size=(runs, k))
    means = bsum[pick].sum(axis=1) / np.maximum(bcnt[pick].sum(axis=1), 1)
    return float(np.std(means, ddof=1) * np.sqrt(k * L / S / (1 - L / S)))


# --- Allan deviation --------------------------------------------------------------------------------------------

def allan(u, e, taus, min_points=3):
    """Allan variance of the residuals e along u for each bin length tau: half the mean squared difference of the
    means of adjacent bins (both with at least min_points points). Also the white-noise reference var(e) / n_bin.
    Returns arrays (sum of half squared differences, number of differences, sum of white reference) per tau, to be
    pooled over pieces by the caller."""
    a_sum, a_cnt, w_sum = np.zeros(len(taus)), np.zeros(len(taus)), np.zeros(len(taus))
    var = np.var(e)
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
