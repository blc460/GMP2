"""Shared by all steps: settings, reading the railhead points, levelling them and cutting them into cells.

The cells are the same as before: rails are connected groups of points, cut into 5 m pieces whose direction comes
from a PCA, and each piece into 0.5 m cells along that direction.
"""
from dataclasses import dataclass
from pathlib import Path
import tomllib

import laspy
import numpy as np
import pandas as pd
from scipy import ndimage

from precision import mad_sigma

HERE = Path(__file__).parent
with open(HERE / "config.toml", "rb") as f:
    cfg = tomllib.load(f)
OUT = HERE / cfg["results"] / {1: "sok", 2: "lip"}.get(cfg["rail_class"], f"class{cfg['rail_class']}")


# --- settings and input --------------------------------------------------------------------------------------

def group_keys(by_setup=None):
    """Columns that make one line of the tables: scanner, setup (station) and condition, as set in config.toml."""
    by_setup = cfg["by_setup"] if by_setup is None else by_setup
    return ["scanner"] + (["setup"] if by_setup else []) + (["condition"] if cfg["by_condition"] else [])


def datasets():
    """Rows of datasets.csv as dictionaries."""
    return pd.read_csv(HERE / cfg["datasets"], keep_default_na=False).to_dict("records")


def find_dataset(part):
    """The one row of datasets.csv whose path contains `part` (e.g. 'Bu_RTC-2')."""
    rows = [d for d in datasets() if part in d["path"]]
    if len(rows) != 1:
        raise SystemExit(f"'{part}' matches {len(rows)} rows of {cfg['datasets']}, it must match one")
    return rows[0]


def read(ds):
    """LAS data of the points of `rail_class`, their coordinates (n, 3) and the scanner origin (3,).
    LAZ does not store the scanner position; it comes from datasets.csv (the scan pose in the E57 file)."""
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


# --- planes and levelling --------------------------------------------------------------------------------------

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


# --- rails, pieces, cells --------------------------------------------------------------------------------------

def main_axis(xy):
    """Unit vector of the main direction of points in plan view (PCA)."""
    return np.linalg.eigh(np.cov(xy.T))[1][:, 1]


def along_across(xy):
    """Plan-view coordinates along and across the main direction of the points, about their mean [m]."""
    a = main_axis(xy)
    d = xy - xy.mean(axis=0)
    return d @ a, d @ np.array([-a[1], a[0]])


def extent(xy):
    """Length along and width across the main direction of the points (2 % to 98 % quantile) [m]."""
    q = np.quantile(np.column_stack(along_across(xy)), [0.02, 0.98], axis=0)
    return q[1] - q[0]


def width(xy):
    """Width of the points across their main direction (2 % to 98 % quantile) [m]."""
    return extent(xy)[1]


def _labels(xy):
    """Connected groups of points in plan view: points closer than `rail_gap` share a label."""
    ij = np.floor((xy - xy.min(axis=0)) / cfg["raster"]).astype(int)
    occ = np.zeros(ij.max(axis=0) + 1, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    r = max(1, int(round(cfg["rail_gap"] / 2 / cfg["raster"])))
    lab, _ = ndimage.label(ndimage.binary_dilation(occ, iterations=r), np.ones((3, 3)))
    return lab[ij[:, 0], ij[:, 1]]


def cells(p):
    """Rail id, piece id and cell id per point (levelled points p; -1 = no cell). Rails are connected groups of
    points; each rail is cut into windows of `window` along its main direction, each window again into connected
    pieces, and each piece into slabs of `cell` along its own main direction (PCA); a slab that cuts two rails
    gives one cell per rail."""
    rail = _labels(p[:, :2])
    piece_id = np.full(len(p), -1)
    cell = np.full(len(p), -1)
    n = npiece = 0
    for r in np.unique(rail):
        idx = np.flatnonzero(rail == r)
        if len(idx) < 3:
            continue
        s = p[idx, :2] @ main_axis(p[idx, :2])
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
                sp = p[ip, :2] @ main_axis(p[ip, :2])
                k = np.floor((sp - sp.min()) / cfg["cell"]).astype(int)
                for kk in np.unique(k):                  # a slab can cut two rails: one cell per rail
                    ik = ip[k == kk]
                    lab = _labels(p[ik, :2])
                    cell[ik] = n + lab - 1
                    n += lab.max()
    return rail, piece_id, cell


@dataclass
class Cell:
    id: int
    idx: np.ndarray          # indices of its points
    status: str              # "keep", "few" (fewer than min_points) or "switch" (wider than max_width)
    length: float = np.nan   # 2 % to 98 % extent along the rail [m]
    width: float = np.nan    # 2 % to 98 % extent across the rail [m]


def iter_cells(p, cell):
    """All cells of the levelled points p with their status (one place for the min_points and switch rules)."""
    order = np.argsort(cell)
    for idx in np.split(order, np.flatnonzero(np.diff(cell[order])) + 1):
        c = int(cell[idx[0]])
        if c < 0:
            continue
        if len(idx) < cfg["min_points"]:
            yield Cell(c, idx, "few")
            continue
        length, wid = extent(p[idx, :2])
        yield Cell(c, idx, "switch" if wid > cfg["max_width"] else "keep", length, wid)
