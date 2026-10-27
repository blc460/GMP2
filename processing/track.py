"""Rail detection in plan view: rail pieces between switches, station s and offset t of every point."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import disk, skeletonize

EIGHT = np.ones((3, 3), bool)


@dataclass
class Rail:
    id: str              # R01, R02, ...
    idx: np.ndarray      # indices into the cloud
    s: np.ndarray        # station along the rail piece [m]
    t: np.ndarray        # offset from the centreline, positive to the left [m]
    line: np.ndarray     # (m, 2) smoothed centreline xy
    s_line: np.ndarray   # (m,) station of the centreline points [m]
    tangent: np.ndarray  # (m, 2) unit tangent of the centreline

    def at(self, s):
        """xy and unit tangent of the centreline at station s."""
        xy = np.array([np.interp(s, self.s_line, self.line[:, i]) for i in (0, 1)])
        tan = np.array([np.interp(s, self.s_line, self.tangent[:, i]) for i in (0, 1)])
        return xy, tan / np.linalg.norm(tan)


def detect_rails(pts, use, rc):
    """Rail pieces and junction points (k, 2) from the plan view of the points `use`.

    The xy of the points are rasterised, small gaps closed, short components dropped and
    small holes filled and
    the rest skeletonised. The skeleton is split at junctions (switches, frogs) into
    branches; each branch longer than `min_length` is one rail piece with a smoothed
    centreline. Every point is assigned to the nearest centreline within `max_offset`.
    """
    cell = rc["cell"]
    r = int(np.ceil(rc["close"] / cell))
    xy = pts[use, :2]
    lo = xy.min(axis=0) - (r + 1) * cell
    ij = ((xy - lo) / cell).astype(int)
    occ = np.zeros(ij.max(axis=0) + r + 2, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_closing(occ, disk(r))
    holes, nh = ndimage.label(ndimage.binary_fill_holes(occ) & ~occ)
    small = np.flatnonzero(ndimage.sum(holes > 0, holes, np.arange(1, nh + 1)) * cell ** 2 < rc["max_hole"]) + 1
    occ |= np.isin(holes, small)                           # holes left between scan lines
    lab, n = ndimage.label(occ, EIGHT)
    long_ = [k + 1 for k, sl in enumerate(ndimage.find_objects(lab))
             if cell * np.hypot(*(b.stop - b.start for b in sl)) >= rc["min_length"]]
    skel = skeletonize(np.isin(lab, long_))

    # prune spurs: short branches with a free end, until none is left
    min_px = rc["min_length"] / cell
    while True:
        cut = ndimage.binary_dilation(_crossings(skel) >= 3, EIGHT) & skel
        lab, n = ndimage.label(skel & ~cut, EIGHT)
        size = ndimage.sum(skel, lab, np.arange(1, n + 1))
        free = np.asarray(ndimage.maximum(skel & (_crossings(skel) <= 1), lab, np.arange(1, n + 1)), bool)
        spur = np.flatnonzero(free & (size < min_px)) + 1
        if not len(spur):
            break
        skel &= ~np.isin(lab, spur)

    jlab, nj = ndimage.label(cut, EIGHT)
    junctions = lo + (np.array(ndimage.center_of_mass(cut, jlab, np.arange(1, nj + 1))).reshape(-1, 2) + 0.5) * cell
    paths = [lo + (_order(pix) + 0.5) * cell for pix in (np.argwhere(lab == k) for k in range(1, n + 1)) if len(pix) > 1]
    paths = [p for p in paths if _length(p) >= rc["min_length"]]
    if len(junctions) and paths:                           # junctions where at least two rail pieces end
        ends = np.array([p[i] for p in paths for i in (0, -1)])
        junctions = junctions[(np.linalg.norm(ends[:, None] - junctions[None], axis=2) <= rc["close"]).sum(axis=0) >= 2]
    paths = _link(paths, junctions, rc)
    w = max(1, int(round(rc["smooth"] / cell)))
    pieces = []
    for path in paths:
        line = ndimage.uniform_filter1d(path, w, axis=0, mode="nearest")
        if line[-1, 0] < line[0, 0]:                       # stations increase with x
            line = line[::-1]
        s_line = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        tan = np.gradient(line, axis=0)
        pieces.append((line, s_line, tan / np.linalg.norm(tan, axis=1, keepdims=True)))

    if not pieces:
        raise ValueError("no rail longer than rails.min_length found")

    # every point to the nearest centreline point
    owner = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(pieces)])
    tree = cKDTree(np.concatenate([p[0] for p in pieces]))
    dist, near = tree.query(pts[:, :2], distance_upper_bound=rc["max_offset"])
    ok = np.isfinite(dist)
    start = np.concatenate([[0], np.cumsum([len(p[0]) for p in pieces])])
    rails = []
    for i, (line, s_line, tan) in enumerate(pieces):
        idx = np.flatnonzero(ok)
        idx = idx[owner[near[idx]] == i]
        k = near[idx] - start[i]
        d = pts[idx, :2] - line[k]
        rails.append(Rail(f"R{i + 1:02d}", idx, s_line[k] + np.einsum("ij,ij->i", d, tan[k]),
                          d[:, 1] * tan[k, 0] - d[:, 0] * tan[k, 1], line, s_line, tan))
    return rails, junctions


def _length(line):
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def _link(paths, junctions, rc):
    """Join rail pieces whose free ends (not at a junction) face each other across at most `link_gap`."""
    cos = np.cos(np.radians(rc["link_angle"]))
    k = max(2, int(round(rc["smooth"] / rc["cell"])))

    def free(p):
        return not len(junctions) or np.linalg.norm(junctions - p, axis=1).min() > rc["close"]

    def outward(p, end):
        v = p[0] - p[min(k, len(p) - 1)] if end == 0 else p[-1] - p[-1 - min(k, len(p) - 1)]
        return v / np.linalg.norm(v)

    while True:
        best = None
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                for ea in (0, -1):
                    for eb in (0, -1):
                        a, b = paths[i][ea], paths[j][eb]
                        gap = np.linalg.norm(b - a)
                        if not 0 < gap <= rc["link_gap"] or not (free(a) and free(b)):
                            continue
                        g = (b - a) / gap
                        if outward(paths[i], ea) @ g >= cos and -outward(paths[j], eb) @ g >= cos                                 and (best is None or gap < best[0]):
                            best = (gap, i, j, ea, eb)
        if best is None:
            return paths
        _, i, j, ea, eb = best
        a = paths[i][::-1] if ea == 0 else paths[i]
        b = paths[j][::-1] if eb == -1 else paths[j]
        paths[i] = np.concatenate([a, b])
        del paths[j]


def _crossings(skel):
    """Number of separate skeleton runs around each pixel (1: end, 2: line, >= 3: junction)."""
    p = np.pad(skel, 1).astype(int)
    ring = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    return sum((ring[k] == 0) & (ring[(k + 1) % 8] == 1) for k in range(8)) * skel


def _order(pix):
    """Pixels of one branch in order along it: the longest shortest path through them."""
    pairs = cKDTree(pix).query_pairs(1.5, output_type="ndarray")
    w = np.linalg.norm(pix[pairs[:, 0]] - pix[pairs[:, 1]], axis=1)
    g = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(len(pix), len(pix)))
    d = dijkstra(g, directed=False, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d), d, -1)))
    d, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    path = [int(np.argmax(np.where(np.isfinite(d), d, -1)))]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return pix[path[::-1]]


def head_strip(s, t, z, sc):
    """Running-surface strip (lo, hi) in t of one segment, railhead-only input.

    The along-track slope is removed and the residual heights binned across t (a high
    quantile per bin). The lowest bin is the groove (or the edge of a railhead-only clip).
    On each side of it, within `search`, the first run of bins within `head_band` of the
    highest bin on that side is railhead or lip; the wider of the two is the railhead.
    Paving higher than the rail farther out does not matter. The strip is that run minus
    `edge_trim` at each edge, at most `width` wide around its centre.
    """
    e = z - np.polyval(np.polyfit(s, z, 1), s)
    b = sc["bin"]
    k = np.floor((t - t.min()) / b).astype(int)
    q = np.full(k.max() + 1, np.nan)
    for i in np.unique(k):
        if (k == i).sum() >= sc["min_bin_points"]:
            q[i] = np.quantile(e[k == i], sc["quantile"])
    if np.isnan(q).all():
        return np.nan, np.nan
    g = int(np.nanargmin(q))
    run = []
    for side in (-1, 1):
        idx = g + side * np.arange(1, int(round(sc["search"] / b)) + 1)
        v = q[idx[(idx >= 0) & (idx < len(q))]]
        if np.isnan(v).all():
            continue
        good = v >= np.nanmax(v) - sc["head_band"]
        pad = np.pad(good, 1)
        good |= np.isnan(v) & pad[:-2] & pad[2:]           # bridge single empty bins
        i0 = int(np.argmax(good))
        i1 = i0 + (int(np.argmin(good[i0:])) if not good[i0:].all() else len(good) - i0)
        if i1 - i0 > len(run):
            run = g + side * np.arange(i0 + 1, i1 + 1)
    if not len(run):
        return np.nan, np.nan
    lo = t.min() + run.min() * b + sc["edge_trim"]
    hi = t.min() + (run.max() + 1) * b - sc["edge_trim"]
    c, half = (lo + hi) / 2, sc["width"] / 2
    return max(lo, c - half), min(hi, c + half)


def class_strip(rail: Rail, cls, cfg, which="sok"):
    """Classified input: the points of the SOK (or lip) class on one rail, minus `edge_trim` at each edge."""
    m = cls[rail.idx] == cfg["input"]["class_sok" if which == "sok" else "class_lip"]
    if m.sum() < 3:
        return m
    lo, hi = np.quantile(rail.t[m], [0.005, 0.995])
    trim = cfg["strip"]["edge_trim"]
    return m & (rail.t >= lo + trim) & (rail.t <= hi - trim)
