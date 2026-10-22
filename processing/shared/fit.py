"""Local plane fits on the railhead band, robust scale estimates and the segment table.

Every quality value and every rail height comes from one least-squares plane per
segment of length seg_len along the rail:

    z = c0 + c1 ds + c2 dq        ds, dq relative to the segment centre and band centre

so c0 is the rail height at station s_i (segment centre) and band centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .geometry import RunningBand, TrackAxis

MAD2SIGMA = 1.4826       # sigma = 1.4826 * MAD for normally distributed values (1 / Phi^-1(0.75))
MAD_EFFICIENCY = 0.3675  # asymptotic efficiency of the MAD relative to the standard deviation (normal)


# --------------------------------------------------------------------------- robust scale
def mad_sigma(x):
    """Robust standard deviation 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(MAD2SIGMA * np.median(np.abs(x - np.median(x))))


def sigma_ci(sigma, dof, alpha=0.05):
    """Two-sided (1 - alpha) confidence interval of a standard deviation (chi-square, dof degrees of freedom)."""
    dof = np.maximum(dof, 1e-9)
    lo = sigma * np.sqrt(dof / stats.chi2.ppf(1 - alpha / 2, dof))
    hi = sigma * np.sqrt(dof / stats.chi2.ppf(alpha / 2, dof))
    return lo, hi


def mad_sigma_ci(sigma_mad, n, alpha=0.05):
    """Approximate CI of 1.4826 MAD from n normal values.

    The MAD has the variance of a standard deviation from MAD_EFFICIENCY * n
    values, so the chi-square interval is used with that effective number of
    degrees of freedom (wider than for the RMS of the same values).
    """
    return sigma_ci(sigma_mad, MAD_EFFICIENCY * n, alpha)


# --------------------------------------------------------------------------- plane fit
@dataclass
class PlaneFit:
    n: int                 # inliers
    n_out: int             # rejected outliers
    coef: np.ndarray       # z = c0 + c1*ds + c2*dq   (ds, dq relative to the window centre)
    sigma: float           # sqrt(r'r / (n - 3)) of the inliers: point noise normal to the surface
    sigma_mad: float       # 1.4826 MAD of the residuals of all points (robust, outliers included)
    q00: float             # cofactor of c0: [(A'A)^-1]_00 of the inliers (~ 1/n)
    resid: np.ndarray
    inlier: np.ndarray

    @property
    def sigma_c0(self):
        """Formal std of the height c0 from the least-squares noise."""
        return self.sigma * np.sqrt(self.q00)

    @property
    def sigma_c0_mad(self):
        """Formal std of the height c0 from the robust (MAD) noise."""
        return self.sigma_mad * np.sqrt(self.q00)


def fit_plane(ds, dq, z, k=3.0, max_iter=10):
    """Least-squares plane with iterative k-sigma outlier rejection (sigma from the MAD).

    The railhead is nearly horizontal, so vertical residuals equal the residuals
    normal to the surface to within cos(slope) ~ 1. A crowned head would show
    up as a pattern of the residuals over dq (block variance factor >> 1).
    """
    A = np.column_stack([np.ones_like(ds), ds, dq])
    inl = np.ones(len(z), bool)
    for _ in range(max_iter):
        if inl.sum() < 6:
            return None
        c, *_ = np.linalg.lstsq(A[inl], z[inl], rcond=None)
        r = z - A @ c
        mad = MAD2SIGMA * np.median(np.abs(r[inl] - np.median(r[inl])))
        new = np.abs(r) <= k * max(mad, 1e-6)
        if np.array_equal(new, inl):
            break
        inl = new
    n = int(inl.sum())
    if n < 6:
        return None
    r_in = r[inl]
    sigma = float(np.sqrt(r_in @ r_in / (n - 3)))
    N_inv = np.linalg.inv(A[inl].T @ A[inl])
    return PlaneFit(n, int(len(z) - n), c, sigma, mad_sigma(r), float(N_inv[0, 0]), r, inl)


def block_variance_factor(ds, resid, n_blocks=4):
    """Diagnostic for correlated residuals (effective n < n).

    Splits the window into n_blocks along s and compares the scatter of the block
    means of the residuals with the value expected for independent points,
    sigma^2 / n_block. A factor clearly > 1 means that sigma/sqrt(n) is too
    optimistic (spatially correlated noise or unmodelled surface shape).
    """
    edges = np.quantile(ds, np.linspace(0, 1, n_blocks + 1))
    k = np.clip(np.digitize(ds, edges[1:-1]), 0, n_blocks - 1)
    means = np.array([resid[k == j].mean() for j in range(n_blocks) if np.sum(k == j) > 2])
    nb = np.array([np.sum(k == j) for j in range(n_blocks) if np.sum(k == j) > 2])
    if len(means) < 3:
        return np.nan
    s2 = resid.var(ddof=3)
    return float(np.mean(means ** 2 * nb) / s2)


# --------------------------------------------------------------------------- segments
@dataclass
class Segment:
    """Band points of one rail segment [s_from, s_to) and their plane fit."""
    rail: str
    s_from: float
    s_to: float
    n_raw: int
    full: bool             # the rail clip covers the whole segment length (no partial end segment)
    width: float           # band width [m]
    q_center: float        # band centre in axis coordinates at the segment centre [m]
    ds: np.ndarray         # station relative to the segment centre [m]
    dq: np.ndarray         # lateral offset relative to the band centre [m]
    pts: np.ndarray        # xyz of the band points
    fit: PlaneFit | None   # None if too few points

    @property
    def s_mid(self):
        return 0.5 * (self.s_from + self.s_to)


def band_segments(pts, s, q, band: RunningBand, seg_len=config.SEG_LEN, min_points=10):
    """Cut the band of one rail into segments on the grid k * seg_len and fit a plane to each.

    The grid is global (multiples of seg_len), so both rails and all scans that
    share the track axis have the same stations s_i = (k + 1/2) seg_len.
    """
    m = band.mask(s, q)
    P, S, Q = pts[m], s[m], band.q_rel(s[m], q[m])
    qc = 0.5 * (band.q_lo + band.q_hi)
    s_lo, s_hi = s.min(), s.max()        # extent of the whole clip, not only of the band
    out = []
    for s0 in np.arange(np.floor(S.min() / seg_len) * seg_len, S.max(), seg_len):
        w = (S >= s0) & (S < s0 + seg_len)
        n_raw = int(w.sum())
        ds, dq = S[w] - (s0 + seg_len / 2), Q[w] - qc
        fit = fit_plane(ds, dq, P[w, 2]) if n_raw >= min_points else None
        out.append(Segment(band.side, float(s0), float(s0 + seg_len), n_raw,
                           bool(s_lo <= s0 and s_hi >= s0 + seg_len), band.q_hi - band.q_lo,
                           float(band.center_q(s0 + seg_len / 2)), ds, dq, P[w], fit))
    return out


def segment_geometry(seg: Segment, origin, axis: TrackAxis):
    """Range, incidence angle and horizontal angle to the track of one fitted segment.

    The beam is the line from the scanner origin to the centroid of the inliers.
    Incidence = angle between beam and the normal of the fitted plane
    (0 deg = perpendicular hit); horizontal angle = angle between the horizontal
    beam direction and the track (0 deg = looking along the rail).
    """
    fit = seg.fit
    c = seg.pts[fit.inlier].mean(axis=0)
    beam = c - np.asarray(origin, float)
    rng = float(np.linalg.norm(beam))
    t = axis.direction_at(None, c[None, :2])[0]
    nrm_xy = np.array([-t[1], t[0]])
    # plane normal in xyz from the slopes along s (c1) and q (c2)
    normal = np.array([-fit.coef[1] * t[0] - fit.coef[2] * nrm_xy[0],
                       -fit.coef[1] * t[1] - fit.coef[2] * nrm_xy[1], 1.0])
    normal /= np.linalg.norm(normal)
    bh = beam[:2] / np.linalg.norm(beam[:2])
    return dict(
        x=float(c[0]), y=float(c[1]), z=float(c[2]),
        range_m=rng, horiz_dist_m=float(np.linalg.norm(beam[:2])), height_diff_m=float(-beam[2]),
        incidence_deg=float(np.degrees(np.arccos(abs(beam @ normal) / rng))),
        horiz_angle_deg=float(np.degrees(np.arccos(min(1.0, abs(bh @ t))))),
    )


def density_metrics(seg: Segment, cell=0.005):
    """Point density of one segment on the band.

    density = inliers per band area (the band is nearly horizontal, so the
    horizontal area equals the surface area to < 0.1 %), mean spacing =
    1 / sqrt(density), completeness = share of occupied cell x cell grid cells.
    """
    fit = seg.fit
    seg_len = seg.s_to - seg.s_from
    ix = np.floor((seg.ds[fit.inlier] + seg_len / 2) / cell).astype(int)
    iy = np.floor((seg.dq[fit.inlier] + seg.width / 2) / cell).astype(int)
    n_cells = int(np.ceil(seg_len / cell) * np.ceil(seg.width / cell))
    dens = fit.n / (seg_len * seg.width * 1e4)
    return dict(density_pts_cm2=dens, mean_spacing_mm=10.0 / np.sqrt(dens),
                completeness=len(set(zip(ix, iy))) / n_cells)


def noise_metrics(seg: Segment):
    """Point-cloud noise of one segment: scatter of the points about the fitted plane.

    sigma_pc = sqrt(r'r / (n - 3)) of the inlier residuals (normal to the nearly
    horizontal railhead) with its 95 % chi-square CI, robust sigma 1.4826 MAD of
    all residuals, outlier share and block variance factor.
    """
    fit = seg.fit
    lo, hi = sigma_ci(fit.sigma, fit.n - 3)
    return dict(
        outlier_share=fit.n_out / seg.n_raw,
        sigma_pc_mm=fit.sigma * 1e3, sigma_pc_ci_lo_mm=lo * 1e3, sigma_pc_ci_hi_mm=hi * 1e3,
        sigma_mad_mm=fit.sigma_mad * 1e3,
        var_factor=block_variance_factor(seg.ds[fit.inlier], fit.resid[fit.inlier]),
    )


def segment_table(preps, metric, seg_len=config.SEG_LEN, keep=lambda seg: True) -> pd.DataFrame:
    """One row per fitted segment of every prepared scan that passes `keep`.

    Columns: scanner, setup, label, rail, s_from, s_to, s_mid, n_raw, n, the
    geometry of `segment_geometry` and the criterion columns of `metric(seg) -> dict`.
    """
    rows = []
    for p in preps:
        for side in ("L", "R"):
            P, s, q = p.rails[side]
            for seg in band_segments(P, s, q, p.bands[side], seg_len):
                if seg.fit is None or not keep(seg):
                    continue
                rows.append(dict(scanner=p.scanner, setup=p.setup, label=p.label, rail=seg.rail,
                                 s_from=seg.s_from, s_to=seg.s_to, s_mid=seg.s_mid, n_raw=seg.n_raw, n=seg.fit.n,
                                 **segment_geometry(seg, p.origin, p.axis), **metric(seg)))
    return pd.DataFrame(rows)
