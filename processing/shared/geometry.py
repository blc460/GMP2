"""Track geometry in the plane: rail split, track axis and stationing, railhead band.

Coordinate conventions
----------------------
Input points are in one levelled frame (x, y, z) with z up, e.g. the registered
frame of REGISTER 360 / SCENE. The scanner origin of each scan is given in the
same frame (translation of the E57 pose).

The track axis defines the stationing: s is the arc length along the axis [m],
q the signed lateral offset [m], positive to the left when looking in +s.
The left rail has q > 0, the right rail q < 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import numpy as np

from . import config


def split_rails(pts, step_deg=0.25):
    """Split one cloud containing both rails into (left, right) along the empty gap.

    The track direction is the horizontal direction whose normal projection shows
    the widest empty gap between the two rails (robust for short sections where a
    PCA direction is poorly defined). Left/right follow the TrackAxis convention
    (direction with positive x component, left = positive lateral offset); the
    caller should still check the sides against the axis actually used.
    """
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    best = (-1.0, None, None)
    for th in np.radians(np.arange(0.0, 180.0, step_deg)):
        d = np.array([np.cos(th), np.sin(th)])
        v = np.sort(xy @ np.array([-d[1], d[0]]))
        lo, hi = np.quantile(v, [0.05, 0.95])     # gap must lie between the rails, not at the border
        inner = (v[:-1] >= lo) & (v[1:] <= hi)
        if not inner.any():
            continue
        g = np.diff(v)[inner]
        k = int(np.argmax(g))
        if g[k] > best[0]:
            best = (g[k], d, 0.5 * (v[:-1][inner][k] + v[1:][inner][k]))
    gap, d, cut = best
    if d is None or gap < 0.2:
        raise RuntimeError(f"no gap between the rails found (widest {gap:.3f} m); clip the rails separately")
    if d[0] < 0:
        d, cut = -d, -cut
    v = xy @ np.array([-d[1], d[0]])
    return pts[v > cut], pts[v <= cut]


# --------------------------------------------------------------------------- track axis
@dataclass
class TrackAxis:
    """Horizontal track axis as polynomial v = p(u) in a rotated frame (u along track).

    Fitted once (e.g. from the densest scan) and reused for every scanner and setup,
    so that all datasets share the same stations s.
    """
    origin: list          # [x0, y0] of the rotated frame
    direction: list       # unit vector of u in the xy plane
    coeffs: list          # polynomial coefficients of v(u), highest power first
    u_min: float
    u_max: float

    # ---- construction
    @classmethod
    def fit(cls, xy, degree=3):
        xy = np.asarray(xy, float)
        o = xy.mean(axis=0)
        _, _, vt = np.linalg.svd(xy - o, full_matrices=False)
        d = vt[0]
        if d[0] < 0:          # make the stationing direction reproducible
            d = -d
        u, v = cls._rot(xy, o, d)
        coeffs = np.polyfit(u, v, degree)
        return cls(o.tolist(), d.tolist(), coeffs.tolist(), float(u.min()), float(u.max()))

    @classmethod
    def from_rails(cls, pts_left, pts_right, degree=3):
        """Axis = mean of the two rail centre lines (each fitted as polynomial)."""
        tmp = cls.fit(np.vstack([pts_left[:, :2], pts_right[:, :2]]), degree=1)
        uL, vL = cls._rot(pts_left[:, :2], np.array(tmp.origin), np.array(tmp.direction))
        uR, vR = cls._rot(pts_right[:, :2], np.array(tmp.origin), np.array(tmp.direction))
        u = np.linspace(max(uL.min(), uR.min()), min(uL.max(), uR.max()), 400)
        vm = 0.5 * (np.polyval(np.polyfit(uL, vL, degree), u) + np.polyval(np.polyfit(uR, vR, degree), u))
        o, d = np.array(tmp.origin), np.array(tmp.direction)
        nrm = np.array([-d[1], d[0]])
        xy = o + np.outer(u, d) + np.outer(vm, nrm)
        return cls.fit(xy, degree)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    # ---- transformation
    @staticmethod
    def _rot(xy, o, d):
        rel = xy - o
        return rel @ d, rel @ np.array([-d[1], d[0]])

    def to_sq(self, xy):
        """Project xy onto the axis: returns station s [m] and lateral offset q [m]."""
        o, d = np.array(self.origin), np.array(self.direction)
        u, v = self._rot(np.asarray(xy, float), o, d)
        p, dp, ddp = np.poly1d(self.coeffs), np.poly1d(self.coeffs).deriv(), np.poly1d(self.coeffs).deriv(2)
        us = u.copy()
        for _ in range(5):  # Newton: foot point of (u, v) on the curve
            g = (us - u) + (p(us) - v) * dp(us)
            h = 1 + dp(us) ** 2 + (p(us) - v) * ddp(us)
            us -= g / h
        q = np.sign(v - p(us)) * np.hypot(u - us, v - p(us))
        grid = np.linspace(min(self.u_min, us.min()) - 1, max(self.u_max, us.max()) + 1, 4000)
        arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(grid), np.diff(p(grid))))])
        s = np.interp(us, grid, arc) - np.interp(self.u_min, grid, arc)
        return s, q

    def to_xy(self, s, q):
        """Inverse of to_sq: plan coordinates of station s [m] and lateral offset q [m]."""
        o, d = np.array(self.origin), np.array(self.direction)
        nrm = np.array([-d[1], d[0]])
        p, dp = np.poly1d(self.coeffs), np.poly1d(self.coeffs).deriv()
        s, q = np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(q, float))
        span = self.u_max - self.u_min
        grid = np.linspace(self.u_min - span - np.abs(s).max(), self.u_max + span + np.abs(s).max(), 20000)
        arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(grid), np.diff(p(grid))))])
        u = np.interp(s + np.interp(self.u_min, grid, arc), arc, grid)
        n_uv = np.column_stack([-dp(u), np.ones_like(u)]) / np.hypot(dp(u), 1)[:, None]   # left normal
        uv = np.column_stack([u, p(u)]) + q[:, None] * n_uv
        return o + np.outer(uv[:, 0], d) + np.outer(uv[:, 1], nrm)

    def direction_at(self, s_query, xy_ref):
        """Unit tangent (xy) of the axis at the stations of reference points xy_ref."""
        o, d = np.array(self.origin), np.array(self.direction)
        u, _ = self._rot(np.asarray(xy_ref, float), o, d)
        dp = np.poly1d(self.coeffs).deriv()(u)
        t = np.outer(np.ones_like(u), d) + np.outer(dp, [-d[1], d[0]])
        return t / np.linalg.norm(t, axis=1, keepdims=True)


# --------------------------------------------------------------------------- running surface
@dataclass
class RunningBand:
    """Strip of one rail used for all fits: trimmed railhead (railhead_band) or running band beside
    the groove (detect_running_band)."""
    side: str              # "L" or "R"
    coeffs: list           # q_rail(s) polynomial (rail centre line of the clip)
    groove_edge: float     # field-side groove edge in q_rel [m]; NaN for railhead clips
    q_lo: float            # band limits in q_rel [m]
    q_hi: float
    profile: dict          # stacked lateral profile, for plotting / documentation

    def q_rel(self, s, q):
        return q - np.polyval(self.coeffs, s)

    def center_q(self, s):
        """Band centre in axis coordinates q at station s."""
        return np.polyval(self.coeffs, s) + 0.5 * (self.q_lo + self.q_hi)

    def mask(self, s, q):
        r = self.q_rel(s, q)
        return (r >= self.q_lo) & (r <= self.q_hi)


def detect_running_band(s, q, z, side, width=0.030, edge_margin=0.005,
                        bin_size=0.002, slice_len=0.25, depth_thresh=0.008,
                        count_frac=0.2, flat_tol=0.002, degree=3):
    """Locate the running surface of a grooved rail in the clipped rail points.

    1. Fit the clip centre line q_rail(s) and compute q_rel = q - q_rail(s).
    2. Reference height per 0.25 m slice = 90th percentile of z (top of rail);
       dz = z - z_ref(s).
    3. Stack all points into a lateral profile (2 mm bins): median dz and count.
       Bins with fewer than 20 % of the median count are treated as empty.
    4. Groove = deepest point of the smoothed profile (5-bin running median),
       or, if it is less than depth_thresh deep, the widest empty run
       (groove bottom occluded). The reference height is not used for this, so
       paving or a guard rail higher than the running surface does not matter.
    5. Walking from the groove towards the field side (away from the track
       centre), the groove edge is the first bin from which the next
       edge_margin + width are flat (unsmoothed profile within flat_tol).
       The band starts edge_margin beyond the groove edge (rounded gauge corner)
       and is `width` wide. Both values come from the rail profile drawing
       (e.g. Ri60N) and must be justified in the report.
    """
    coeffs = np.polyfit(s, q, degree)
    qr = q - np.polyval(coeffs, s)
    edges = np.arange(s.min(), s.max() + slice_len, slice_len)
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)
    top = np.array([np.percentile(z[idx == k], 90) if np.any(idx == k) else np.nan
                    for k in range(len(edges) - 1)])
    ok = np.isfinite(top)
    zref = np.interp(s, 0.5 * (edges[:-1] + edges[1:])[ok], top[ok])
    dz = z - zref

    qb = np.arange(qr.min(), qr.max() + bin_size, bin_size)
    bi = np.clip(np.digitize(qr, qb) - 1, 0, len(qb) - 2)
    cnt = np.bincount(bi, minlength=len(qb) - 1)
    med = np.array([np.median(dz[bi == k]) if cnt[k] else np.nan for k in range(len(qb) - 1)])
    centers = 0.5 * (qb[:-1] + qb[1:])
    empty = cnt < count_frac * np.median(cnt[cnt > 0])
    raw = np.where(empty, np.nan, med)
    sm = _running_median(raw, 5)
    interior = np.zeros(len(sm), bool)
    interior[2:-2] = True                       # the clip borders are not the groove
    k0 = None
    if np.isfinite(sm[interior]).any():
        k = int(np.nanargmin(np.where(interior, sm, np.nan)))
        if np.nanmedian(sm) - sm[k] >= depth_thresh:
            k0 = k
    if k0 is None:                              # groove bottom not seen: widest empty run
        runs = _runs(empty & interior)
        if not runs:
            raise RuntimeError(f"rail {side}: no groove found in the lateral profile; check the clip")
        a, b = max(runs, key=lambda r: r[1] - r[0])
        k0 = (a + b) // 2
    outward = 1 if side == "L" else -1          # field side = away from the track axis
    nb = int(np.ceil((edge_margin + width) / bin_size))
    j = k0
    while True:
        if (j + nb >= len(sm)) if outward > 0 else (j - nb < 0):
            raise RuntimeError(f"rail {side}: no flat running surface of {1e3 * (edge_margin + width):.0f} mm "
                               f"on the field side of the groove; check the clip and the band width")
        w = raw[j:j + nb + 1] if outward > 0 else raw[j - nb:j + 1]
        if np.isfinite(w).all() and np.ptp(w) <= flat_tol:
            break
        j += outward
    if outward > 0:
        g_edge = qb[j]
        lo, hi = g_edge + edge_margin, g_edge + edge_margin + width
    else:
        g_edge = qb[j + 1]
        lo, hi = g_edge - edge_margin - width, g_edge - edge_margin
    profile = dict(q=centers.tolist(), count=cnt.tolist(), median_dz=np.nan_to_num(med, nan=0).tolist())
    return RunningBand(side, coeffs.tolist(), float(g_edge), float(lo), float(hi), profile)


def _running_median(a, k):
    """Running median over k bins, ignoring NaN."""
    h = k // 2
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        w = a[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def _runs(mask):
    """(start, end) index pairs of the contiguous True runs of a boolean array."""
    runs, start = [], None
    for k, g in enumerate(np.append(mask, False)):
        if g and start is None:
            start = k
        elif not g and start is not None:
            runs.append((start, k - 1)); start = None
    return runs


def railhead_band(s, q, z, side, edge_trim=config.EDGE_TRIM, edge_quantile=config.EDGE_QUANTILE,
                  bin_size=0.002, degree=3):
    """Band of a clip cropped to the rail head (default input).

    1. Centre line of the clip q_rail(s) (polynomial), lateral offset q_rel = q - q_rail(s).
    2. Railhead edges = edge_quantile and 1 - edge_quantile quantiles of q_rel
       (robust against a few stray points outside the head).
    3. Band = head minus edge_trim on both sides, so the rounded head corners,
       which are not part of the plane, do not enter the plane fits.

    Returns a RunningBand (groove_edge = NaN) and the measured head width [m].
    The profile (point count and median height per 2 mm bin across the head)
    is kept for the check plot.
    """
    coeffs = np.polyfit(s, q, degree)
    qr = q - np.polyval(coeffs, s)
    lo, hi = np.quantile(qr, [edge_quantile, 1 - edge_quantile])
    if hi - lo - 2 * edge_trim < 0.005:
        raise RuntimeError(f"rail {side}: railhead {1e3 * (hi - lo):.0f} mm wide, less than 5 mm left "
                           f"after trimming {1e3 * edge_trim:.0f} mm per side; check the clip / --edge-trim")
    dz = z - np.polyval(np.polyfit(s, z, degree), s)
    qb = np.arange(qr.min(), qr.max() + bin_size, bin_size)
    bi = np.clip(np.digitize(qr, qb) - 1, 0, len(qb) - 2)
    cnt = np.bincount(bi, minlength=len(qb) - 1)
    med = [float(np.median(dz[bi == k])) if cnt[k] else 0.0 for k in range(len(qb) - 1)]
    profile = dict(q=(0.5 * (qb[:-1] + qb[1:])).tolist(), count=cnt.tolist(), median_dz=med,
                   head_lo=float(lo), head_hi=float(hi))
    band = RunningBand(side, coeffs.tolist(), float("nan"), float(lo + edge_trim), float(hi - edge_trim), profile)
    return band, float(hi - lo)


def station_rails(PL, PR, axis: TrackAxis):
    """Stations (s, q) of both rails; the sides are assigned by the axis (left = q > 0).

    Returns {side: (P, s, q)}.
    """
    sq = {"L": (PL, *axis.to_sq(PL[:, :2])), "R": (PR, *axis.to_sq(PR[:, :2]))}
    if np.median(sq["L"][2]) < 0:
        sq = {"L": sq["R"], "R": sq["L"]}
    if np.median(sq["L"][2]) < 0 or np.median(sq["R"][2]) > 0:
        raise RuntimeError("both rails lie on the same side of the track axis; the axis does not fit this scan")
    return sq
