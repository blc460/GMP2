"""Core functions: rail point clouds -> segment quality metrics and track geometry.

Coordinate conventions
----------------------
Input points are in one levelled frame (x, y, z) with z up, e.g. the registered
frame of REGISTER 360 / SCENE. The scanner origin of each setup is given in the
same frame (translation of the E57 pose).

The track axis defines the stationing: s is the arc length along the axis [m],
q the signed lateral offset [m], positive to the left when looking in +s.
The left rail has q > 0, the right rail q < 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import numpy as np
from scipy import stats

MAD2SIGMA = 1.4826  # sigma = 1.4826 * MAD for normally distributed residuals


# --------------------------------------------------------------------------- I/O
def load_points(path):
    """Read x, y, z from a PLY (binary/ascii) or an ASCII file (CloudCompare export).

    Extra columns (intensity, ...) are ignored. Returns an (n, 3) float64 array.
    """
    path = str(path)
    if path.lower().endswith(".ply"):
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    if path.lower().endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    data = np.loadtxt(path, comments=("#", "//"), delimiter=None, usecols=(0, 1, 2))
    return np.atleast_2d(data).astype(np.float64)


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
    """Running-surface strip of one rail, located relative to the detected groove."""
    side: str              # "L" or "R"
    coeffs: list           # q_rail(s) polynomial (rail centre line of the clip)
    groove_edge: float     # field-side groove edge in q_rel [m]
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
                        count_frac=0.2, degree=3):
    """Locate the running surface of a grooved rail in the clipped rail points.

    1. Fit the clip centre line q_rail(s) and compute q_rel = q - q_rail(s).
    2. Reference height per 0.25 m slice = 90th percentile of z (top of rail);
       dz = z - z_ref(s).
    3. Stack all points into a lateral profile (2 mm bins): median dz and count.
       Groove bins are deep (median dz < -8 mm) or nearly empty (occlusion,
       count < 20 % of the median count).
    4. The running surface lies on the field side of the groove (away from the
       track centre). The band starts edge_margin beyond the groove edge (to skip
       the rounded gauge corner) and is `width` wide. Both values come from the
       rail profile drawing (e.g. Ri60N) and must be justified in the report.
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
    groove = (cnt < count_frac * np.median(cnt[cnt > 0])) | (med < -depth_thresh)
    # interior bins only: the clip borders are not the groove
    groove[:2] = groove[-2:] = False
    if not groove.any():
        raise RuntimeError(f"rail {side}: no groove found in the lateral profile; check the clip")
    # largest contiguous groove run
    runs, start = [], None
    for k, g in enumerate(np.append(groove, False)):
        if g and start is None:
            start = k
        elif not g and start is not None:
            runs.append((start, k - 1)); start = None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    outward = 1.0 if side == "L" else -1.0     # field side = away from the track axis
    if outward > 0:
        g_edge = qb[b + 1]
        lo, hi = g_edge + edge_margin, g_edge + edge_margin + width
    else:
        g_edge = qb[a]
        lo, hi = g_edge - edge_margin - width, g_edge - edge_margin
    profile = dict(q=centers.tolist(), count=cnt.tolist(), median_dz=np.nan_to_num(med, nan=0).tolist())
    return RunningBand(side, coeffs.tolist(), float(g_edge), float(lo), float(hi), profile)


# --------------------------------------------------------------------------- local surface fit
@dataclass
class PlaneFit:
    n: int                 # inliers
    n_out: int             # rejected outliers
    coef: np.ndarray       # z = c0 + c1*ds + c2*dq   (ds, dq relative to the window centre)
    sigma: float           # sqrt(r'r / (n - 3)): estimated point noise normal to the surface
    sigma_mad: float       # 1.4826 * MAD of the residuals (robust cross-check)
    sigma_c0: float        # formal std of c0 = height at the window centre
    resid: np.ndarray
    inlier: np.ndarray


def fit_plane(ds, dq, z, k=3.0, max_iter=10):
    """Least-squares plane with iterative k-sigma outlier rejection (sigma from MAD).

    The running surface is nearly horizontal, so vertical residuals equal the
    residuals normal to the surface to within cos(slope) ~ 1. A crowned running
    surface would show up as a systematic pattern of the residuals over dq;
    check this with `profile_residuals` and, if necessary, add a dq^2 term.
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
    r_in = r[inl]
    sigma = float(np.sqrt(r_in @ r_in / (n - 3)))
    N_inv = np.linalg.inv(A[inl].T @ A[inl])
    return PlaneFit(n, int(len(z) - n), c, sigma, float(mad), float(sigma * np.sqrt(N_inv[0, 0])), r, inl)


def sigma_ci(sigma, dof, alpha=0.05):
    """Two-sided (1-alpha) confidence interval of a standard deviation (chi-square)."""
    lo = sigma * np.sqrt(dof / stats.chi2.ppf(1 - alpha / 2, dof))
    hi = sigma * np.sqrt(dof / stats.chi2.ppf(alpha / 2, dof))
    return lo, hi


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


# --------------------------------------------------------------------------- segment metrics (WP4)
def segment_metrics(pts, s, q, band: RunningBand, origin, axis: TrackAxis,
                    seg_len=0.5, cell=0.005, label=None):
    """Point-cloud quality per rail segment of length seg_len on the running band.

    Returns one dict per segment with: point count, density [pts/cm^2],
    completeness (share of occupied cell x cell grid cells), point noise
    sigma_pc (with 95 % CI) and robust sigma, outlier share, variance factor,
    formal height precision, range, incidence angle and horizontal angle
    between line of sight and track.
    """
    m = band.mask(s, q)
    P, S, Q = pts[m], s[m], band.q_rel(s[m], q[m])
    width = band.q_hi - band.q_lo
    qc = 0.5 * (band.q_lo + band.q_hi)
    origin = np.asarray(origin, float)
    rows = []
    for s0 in np.arange(np.floor(S.min() / seg_len) * seg_len, S.max(), seg_len):
        w = (S >= s0) & (S < s0 + seg_len)
        n_raw = int(w.sum())
        row = dict(label=label, rail=band.side, s_from=s0, s_to=s0 + seg_len, n_raw=n_raw)
        if n_raw < 10:
            rows.append(row); continue
        ds, dq, z = S[w] - (s0 + seg_len / 2), Q[w] - qc, P[w, 2]
        fit = fit_plane(ds, dq, z)
        if fit is None:
            rows.append(row); continue
        # completeness: occupied cells of a cell x cell grid over the band
        ix = np.floor((ds + seg_len / 2) / cell).astype(int)
        iy = np.floor((dq + width / 2) / cell).astype(int)
        n_cells = int(np.ceil(seg_len / cell) * np.ceil(width / cell))
        occupied = len(set(zip(ix[fit.inlier], iy[fit.inlier])))
        # geometry: centroid, beam direction, surface normal
        c = P[w][fit.inlier].mean(axis=0)
        beam = c - origin
        rng = float(np.linalg.norm(beam))
        t = axis.direction_at(None, c[None, :2])[0]
        nrm_xy = np.array([-t[1], t[0]])
        # plane normal in xyz from slopes along s (c1) and q (c2)
        normal = np.array([-fit.coef[1] * t[0] - fit.coef[2] * nrm_xy[0],
                           -fit.coef[1] * t[1] - fit.coef[2] * nrm_xy[1], 1.0])
        normal /= np.linalg.norm(normal)
        inc = float(np.degrees(np.arccos(abs(beam @ normal) / rng)))
        bh = beam[:2] / np.linalg.norm(beam[:2])
        alpha = float(np.degrees(np.arccos(min(1.0, abs(bh @ t)))))  # 0 deg = looking along the rail
        lo, hi = sigma_ci(fit.sigma, fit.n - 3)
        row.update(
            n=fit.n, outlier_share=fit.n_out / n_raw,
            density_pts_cm2=fit.n / (seg_len * width * 1e4),
            mean_spacing_mm=10.0 / np.sqrt(fit.n / (seg_len * width * 1e4)),
            completeness=occupied / n_cells,
            sigma_pc_mm=fit.sigma * 1e3, sigma_pc_ci_lo_mm=lo * 1e3, sigma_pc_ci_hi_mm=hi * 1e3,
            sigma_mad_mm=fit.sigma_mad * 1e3,
            var_factor=block_variance_factor(ds[fit.inlier], fit.resid[fit.inlier]),
            sigma_z_formal_mm=fit.sigma_c0 * 1e3,
            range_m=rng, horiz_dist_m=float(np.linalg.norm(beam[:2])),
            incidence_deg=inc, horiz_angle_deg=alpha,
        )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- rail heights and geometry (WP5)
def rail_heights(pts, s, q, band: RunningBand, stations, half_window=0.25):
    """Height of the running surface at (s_i, band centre) and its formal std.

    A plane is fitted to the band points with |s - s_i| <= half_window; its value
    at the window centre is the rail height z_i (the scanner analogue of the
    levelled point on the rail head). Its formal std follows from the
    adjustment: sigma_c0 ~ sigma_pc / sqrt(n).
    """
    m = band.mask(s, q)
    P, S, Q = pts[m], s[m], band.q_rel(s[m], q[m])
    qc = 0.5 * (band.q_lo + band.q_hi)
    out = np.full((len(stations), 4), np.nan)   # z, sigma_z, n, sigma_pc
    for i, si in enumerate(stations):
        w = np.abs(S - si) <= half_window
        if w.sum() < 10:
            continue
        fit = fit_plane(S[w] - si, Q[w] - qc, P[w, 2])
        if fit is None:
            continue
        out[i] = fit.coef[0], fit.sigma_c0, fit.n, fit.sigma
    return out


def track_geometry(stations, hL, hR, qL_center, qR_center, l=5.0, b0=1.0):
    """Cant, twist and vertical versine with individual variance propagation.

    hL, hR: arrays (z, sigma_z) per station. Heights are assumed uncorrelated
    (see report); for a left/right correlation rho, sigma_vPf grows by sqrt(1+rho).
    Stations must be equally spaced with a spacing that divides l.
    """
    st = np.asarray(stations)
    ds = np.round(st[1] - st[0], 6)
    k = int(round(l / ds))
    zL, sL = hL[:, 0], hL[:, 1]
    zR, sR = hR[:, 0], hR[:, 1]
    b = np.abs(qL_center - qR_center)
    cant = b0 / b * (zR - zL)                       # [m]
    s_cant = b0 / b * np.sqrt(sL ** 2 + sR ** 2)
    H = 0.5 * (zL + zR)
    sH = 0.5 * np.sqrt(sL ** 2 + sR ** 2)
    n = len(st)
    twist = np.full(n, np.nan); s_twist = np.full(n, np.nan)
    vpf = np.full(n, np.nan); s_vpf = np.full(n, np.nan)
    twist[:n - k] = (cant[:n - k] - cant[k:]) / l          # dimensionless (x1000 = per mille)
    s_twist[:n - k] = np.sqrt(s_cant[:n - k] ** 2 + s_cant[k:] ** 2) / l
    i = np.arange(k, n - k)
    vpf[i] = H[i] - 0.5 * (H[i - k] + H[i + k])            # without the constant Pf_Rv
    s_vpf[i] = np.sqrt(sH[i] ** 2 + 0.25 * (sH[i - k] ** 2 + sH[i + k] ** 2))
    return dict(
        s=st, b_m=b,
        zL_mm=zL * 1e3, sigma_zL_mm=sL * 1e3, zR_mm=zR * 1e3, sigma_zR_mm=sR * 1e3,
        cant_mm=cant * 1e3, sigma_cant_mm=s_cant * 1e3,
        twist_permille=twist * 1e3, sigma_twist_permille=s_twist * 1e3,
        vpf_mm=vpf * 1e3, sigma_vpf_mm=s_vpf * 1e3,
        # the monitored versine is a difference of two epochs: sqrt(2) * sigma
        sigma_dvpf_mm=np.sqrt(2) * s_vpf * 1e3,
    )


# attention values (VBZ), per speed class: (T_vPf [mm], T_N [per mille])
ATTENTION = {"<=36": (5.0, 1.5), "36-48": (5.0, 1.0), "48-60": (3.0, 1.0)}


def check_criteria(geo, speed_class="48-60"):
    """Criterion 2 sigma <= T for every cross-section (True/False/NaN)."""
    T_vpf, T_n = ATTENTION[speed_class]
    return dict(
        ok_twist=np.where(np.isfinite(geo["sigma_twist_permille"]), 2 * geo["sigma_twist_permille"] <= T_n, np.nan),
        ok_vpf=np.where(np.isfinite(geo["sigma_dvpf_mm"]), 2 * geo["sigma_dvpf_mm"] <= T_vpf, np.nan),
        sigma_z_req_mm=min(T_vpf / np.sqrt(6), T_n * 1.0 * 5.0 / 4.0),
    )
