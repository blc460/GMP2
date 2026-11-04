"""Shared by 01_segments.py and 03_map.py: read the railhead points, level them, cut them into cells along the rails."""
from pathlib import Path
import tomllib

import laspy
import numpy as np
from scipy import ndimage

HERE = Path(__file__).parent
with open(HERE / "config.toml", "rb") as f:
    cfg = tomllib.load(f)


def mad_sigma(e):
    return 1.4826 * np.median(np.abs(e - np.median(e)))


def plane(p, k):
    """sigma_z, centroid and unit normal of the plane z = b0 + b1 x + b2 y through the points p (n, 3);
    residuals beyond k MAD are dropped once before the final fit, sigma_z is from all residuals."""
    c = p.mean(axis=0)
    q = p - c
    A = np.column_stack([np.ones(len(q)), q[:, 0], q[:, 1]])
    b = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
    e = q[:, 2] - A @ b
    mad = mad_sigma(e)
    if mad > 0:
        keep = np.abs(e - np.median(e)) <= k * mad
        b = np.linalg.lstsq(A[keep], q[keep, 2], rcond=None)[0]
        e = q[:, 2] - A @ b
    normal = np.array([-b[1], -b[2], 1.0])
    return mad_sigma(e), c, normal / np.linalg.norm(normal)


def read(ds):
    """LAS data of the railhead points, their coordinates (n, 3) and the scanner origin (3,)."""
    las = laspy.read(HERE / ds["path"])
    las.points = las.points[np.asarray(las.classification) == cfg["class_sok"]]
    origin = np.array([float(ds[k] or 0) for k in ("origin_x", "origin_y", "origin_z")])
    return las, np.column_stack([las.x, las.y, las.z]), origin


def level(p, origin):
    """Rotate points and origin about the centroid so that the ground plane (one plane through all
    railhead points) is horizontal. Returns the levelled points, origin and the tilt [deg]."""
    _, c, n = plane(p, cfg["outlier_k"])
    axis = np.cross(n, [0, 0, 1])
    angle = np.arcsin(np.linalg.norm(axis))
    if angle == 0:
        return p, origin, 0.0
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K     # Rodrigues
    return (p - c) @ R.T + c, (origin - c) @ R.T + c, np.degrees(angle)


def _labels(xy):
    """Connected groups of points in plan view: points closer than `rail_gap` share a label."""
    ij = np.floor((xy - xy.min(axis=0)) / cfg["raster"]).astype(int)
    occ = np.zeros(ij.max(axis=0) + 1, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    r = max(1, int(round(cfg["rail_gap"] / 2 / cfg["raster"])))
    lab, _ = ndimage.label(ndimage.binary_dilation(occ, iterations=r), np.ones((3, 3)))
    return lab[ij[:, 0], ij[:, 1]]


def _axis(xy):
    """Unit vector of the main direction of points in plan view (PCA)."""
    _, vec = np.linalg.eigh(np.cov(xy.T))
    return vec[:, 1]


def cells(p):
    """Rail id and cell id per point (levelled points p). Rails are connected groups of points; each
    rail is cut into windows of `window` along its main direction, each window again into connected
    pieces, and each piece into slabs of `cell` along its own main direction (PCA); a slab that cuts
    two rails gives one cell per rail."""
    rail = _labels(p[:, :2])
    cell = np.full(len(p), -1)
    n = 0
    for r in np.unique(rail):
        idx = np.flatnonzero(rail == r)
        if len(idx) < 3:
            continue
        s = p[idx, :2] @ _axis(p[idx, :2])
        win = np.floor((s - s.min()) / cfg["window"]).astype(int)
        for w in np.unique(win):
            iw = idx[win == w]
            piece = _labels(p[iw, :2])
            for q in np.unique(piece):
                ip = iw[piece == q]
                if len(ip) < 3:
                    continue
                sp = p[ip, :2] @ _axis(p[ip, :2])
                k = np.floor((sp - sp.min()) / cfg["cell"]).astype(int)
                for kk in np.unique(k):                  # a slab can cut two rails: one cell per rail
                    ik = ip[k == kk]
                    lab = _labels(p[ik, :2])
                    cell[ik] = n + lab - 1
                    n += lab.max()
    return rail, cell


def width(xy):
    """Width of the points across their main direction (2 % to 98 % quantile) [m]."""
    a = _axis(xy)
    t = xy @ np.array([-a[1], a[0]])
    lo, hi = np.quantile(t, [0.02, 0.98])
    return hi - lo
