"""Reading railhead clips and the prepared scans.

Input: point clouds cropped to the rail heads, i.e. two nearly horizontal,
parallel strips (left and right rail head). One dataset = one scan = one
scanner position, because range and incidence need the scanner origin.

- E57: every scan in the file is one dataset; the points are transformed into
  the file frame and the origin is the translation of the scan pose. Several
  setups may therefore be exported into one E57.
- PLY / ASCII / NPY: one dataset per file; the origin must be given.

Scanner and setup come from the file name `SCANNER_SETUP[_anything].ext`
unless given explicitly (command line or list file).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re

import numpy as np

from trackqc import core
from . import config


@dataclass
class Source:
    path: Path
    scanner: str | None = None
    setup: str | None = None
    origin: np.ndarray | None = None


@dataclass
class Scan:
    label: str             # SCANNER_SETUP, unique per dataset
    scanner: str
    setup: str
    source: str
    pts: np.ndarray        # (n, 3) in the levelled frame
    origin: np.ndarray     # scanner origin in the same frame


@dataclass
class Prepared:
    label: str
    scanner: str
    setup: str
    source: str
    origin: np.ndarray
    axis: core.TrackAxis
    bands: dict            # side -> core.RunningBand (the trimmed railhead strip)
    rails: dict            # side -> (P, s, q)
    head_width: dict       # side -> measured railhead width [m]


# --------------------------------------------------------------------------- sources
def sources_from_list(path) -> list[Source]:
    """CSV with columns path, scanner, setup and optionally origin_x, origin_y, origin_z.

    Relative paths are relative to the list file. Empty cells mean: from the file name / E57 pose.
    """
    path = Path(path)
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            o = [row.get(k, "") or "" for k in ("origin_x", "origin_y", "origin_z")]
            out.append(Source((path.parent / row["path"]).resolve(), row.get("scanner") or None,
                              row.get("setup") or None,
                              np.array(o, float) if all(o) else None))
    return out


def _clean(name):
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(name)).strip("-")


def read_scans(src: Source) -> list[Scan]:
    tokens = Path(src.path).stem.split("_")
    scanner = _clean(src.scanner or tokens[0])
    setup = _clean(src.setup or (tokens[1] if len(tokens) > 1 else Path(src.path).stem))
    if str(src.path).lower().endswith(".e57"):
        import pye57
        e = pye57.E57(str(src.path))
        scans = []
        for i in range(e.scan_count):
            d = e.read_scan(i, ignore_missing_fields=True, transform=True)
            pts = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]).astype(np.float64)
            origin = src.origin if src.origin is not None else np.asarray(e.get_header(i).translation, float)
            su = setup if e.scan_count == 1 else f"{setup}-{i + 1}"
            name = e.get_header(i).node["name"].value() if e.get_header(i).node.isDefined("name") else ""
            print(f"{src.path.name}: scan {i + 1}/{e.scan_count} '{name}' -> {scanner}_{su}, "
                  f"{len(pts)} points, origin {np.round(origin, 3).tolist()}")
            scans.append(Scan(f"{scanner}_{su}", scanner, su, str(src.path), pts, origin))
        return scans
    if src.origin is None:
        raise ValueError(f"{src.path}: the scanner origin is required for non-E57 input")
    return [Scan(f"{scanner}_{setup}", scanner, setup, str(src.path), core.load_points(src.path),
                 np.asarray(src.origin, float))]


# --------------------------------------------------------------------------- railhead band
def railhead_band(s, q, z, side, edge_trim=config.EDGE_TRIM, edge_quantile=config.EDGE_QUANTILE,
                  bin_size=0.002, degree=3):
    """Band of one rail clip that contains only the rail head.

    1. Centre line of the clip q_rail(s) (polynomial), lateral offset q_rel = q - q_rail(s).
    2. Railhead edges = edge_quantile and 1 - edge_quantile quantiles of q_rel
       (robust against a few stray points outside the head).
    3. Band = head minus edge_trim on both sides, so the rounded head corners,
       which are not part of the plane, do not enter the plane fits.

    Returns a core.RunningBand (groove_edge = NaN) and the measured head width.
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
    band = core.RunningBand(side, coeffs.tolist(), float("nan"), float(lo + edge_trim), float(hi - edge_trim), profile)
    return band, float(hi - lo)


def prepare(scan: Scan, edge_trim=config.EDGE_TRIM) -> Prepared:
    """Split into left/right rail, fit the track axis, station the points, find the railhead band."""
    PL, PR = core.split_rails(scan.pts)
    axis = core.TrackAxis.from_rails(PL, PR)
    sq = {"L": (PL, *axis.to_sq(PL[:, :2])), "R": (PR, *axis.to_sq(PR[:, :2]))}
    if np.median(sq["L"][2]) < 0:            # left = positive lateral offset of the axis
        sq = {"L": sq["R"], "R": sq["L"]}
    bands, widths = {}, {}
    for side, (P, s, q) in sq.items():
        bands[side], widths[side] = railhead_band(s, q, P[:, 2], side, edge_trim)
    return Prepared(scan.label, scan.scanner, scan.setup, scan.source, scan.origin, axis, bands, sq, widths)


# --------------------------------------------------------------------------- prepared files
def save(p: Prepared, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{p.label}.npz", origin=p.origin,
                        **{f"{k}{side}": v for side, rail in p.rails.items() for k, v in zip("Psq", rail)})
    meta = dict(label=p.label, scanner=p.scanner, setup=p.setup, source=p.source, origin=p.origin.tolist(),
                head_width_m=p.head_width, axis=asdict(p.axis), bands={k: asdict(b) for k, b in p.bands.items()})
    with open(folder / f"{p.label}.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_prepared(results, scanners=None) -> list[Prepared]:
    """All scans prepared by 00_prepare.py, optionally only those of the given scanners."""
    folder = config.prepared_dir(results)
    out = []
    for js in sorted(folder.glob("*.json")):
        with open(js) as f:
            m = json.load(f)
        if scanners and m["scanner"] not in scanners:
            continue
        a = np.load(js.with_suffix(".npz"))
        out.append(Prepared(m["label"], m["scanner"], m["setup"], m["source"], np.asarray(m["origin"]),
                            core.TrackAxis(**m["axis"]),
                            {k: core.RunningBand(**b) for k, b in m["bands"].items()},
                            {side: (a[f"P{side}"], a[f"s{side}"], a[f"q{side}"]) for side in ("L", "R")},
                            m["head_width_m"]))
    if not out:
        raise SystemExit(f"no prepared scans in {folder}" + (f" for scanner(s) {scanners}" if scanners else "")
                         + "; run 00_prepare.py first")
    return out
