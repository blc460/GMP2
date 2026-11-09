"""Shared by 01_segments.py and 03_map.py: read the railhead points, level them, cut them into cells along the rails."""
from pathlib import Path
import tomllib

import laspy
import numpy as np
from scipy import ndimage

HERE = Path(__file__).parent
with open(HERE / "config.toml", "rb") as f:
    cfg = tomllib.load(f)
OUT = HERE / cfg["results"] / {1: "sok", 2: "lip"}.get(cfg["rail_class"], f"class{cfg['rail_class']}")


def group_keys(by_setup=None):
    """Columns that make one line of the tables: scanner, setup (station) and condition, as set in config.toml."""
    by_setup = cfg["by_setup"] if by_setup is None else by_setup
    return ["scanner"] + (["setup"] if by_setup else []) + (["condition"] if cfg["by_condition"] else [])


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
    """LAS data of the points of `rail_class`, their coordinates (n, 3) and the scanner origin (3,)."""
    las = laspy.read(HERE / ds["path"])
    las.points = las.points[np.asarray(las.classification) == cfg["rail_class"]]
    origin = np.array([float(ds[k] or 0) for k in ("origin_x", "origin_y", "origin_z")])
    return las, np.column_stack([las.x, las.y, las.z]), origin


def angular_step(las, p, origin):
    """Angular step of the scan grid [rad]: mean of the change of azimuth per ScanColumn and of
    elevation per ScanRow, fitted on the points p (n, 3) seen from the scanner origin."""
    v = p - origin
    col, row = np.asarray(las.ScanColumn, float), np.asarray(las.ScanRow, float)
    o = np.argsort(col)
    az = np.unwrap(np.arctan2(v[o, 1], v[o, 0]))
    el = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))
    return (abs(np.polyfit(col[o], az, 1)[0]) + abs(np.polyfit(row, el, 1)[0])) / 2


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
    """Rail id, piece id and cell id per point (levelled points p). Rails are connected groups of points; each
    rail is cut into windows of `window` along its main direction, each window again into connected
    pieces, and each piece into slabs of `cell` along its own main direction (PCA); a slab that cuts
    two rails gives one cell per rail."""
    rail = _labels(p[:, :2])
    piece_id = np.full(len(p), -1)
    cell = np.full(len(p), -1)
    n = npiece = 0
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
                piece_id[ip] = npiece
                npiece += 1
                sp = p[ip, :2] @ _axis(p[ip, :2])
                k = np.floor((sp - sp.min()) / cfg["cell"]).astype(int)
                for kk in np.unique(k):                  # a slab can cut two rails: one cell per rail
                    ik = ip[k == kk]
                    lab = _labels(p[ik, :2])
                    cell[ik] = n + lab - 1
                    n += lab.max()
    return rail, piece_id, cell


def width(xy):
    """Width of the points across their main direction (2 % to 98 % quantile) [m]."""
    return extent(xy)[1]


def extent(xy):
    """Length along and width across the main direction of the points (2 % to 98 % quantile) [m]."""
    a = _axis(xy)
    q = np.quantile(np.column_stack([xy @ a, xy @ np.array([-a[1], a[0]])]), [0.02, 0.98], axis=0)
    return q[1] - q[0]
