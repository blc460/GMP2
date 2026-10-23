"""Reading point clouds, the prepared scans of step 01, and provenance of the results.

One dataset = one scan = one scanner position, because range and incidence
need the scanner origin.

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
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

from . import config
from .geometry import RunningBand, TrackAxis


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
    origin_from_pose: bool = False   # origin = E57 pose translation (not given explicitly)
    cls: np.ndarray | None = None    # point class (LAS/PLY/ASCII), None if the file has none
    raw: np.ndarray | None = None    # E57: coordinates as stored (scanner frame), for the grid test


@dataclass
class Prepared:
    label: str
    scanner: str
    setup: str
    source: str
    origin: np.ndarray
    axis: TrackAxis
    bands: dict            # side -> RunningBand
    rails: dict            # side -> (P, s, q)
    head_width: dict       # side -> measured railhead width [m] (NaN for band groove)
    band_method: str = "railhead"
    selection: dict | None = None    # class filter, track corridor, origin check of step 01

    @property
    def z_step(self):
        """Storage step of the heights found in step 01 [m] (0: not quantized)."""
        return float((self.selection or {}).get("z_step_m") or 0.0)


# --------------------------------------------------------------------------- point clouds
CLASS_FIELDS = ("classification", "class", "scalar_classification", "scalar_class", "category", "label")


def _class_field(names):
    low = {n.lower(): n for n in names}
    return next((low[c] for c in CLASS_FIELDS if c in low), None)


def read_cloud(path, class_column=None):
    """x, y, z (n, 3) and the point class (n,) or None, from E57, LAS/LAZ, PLY, NPY or ASCII.

    The class is read from a LAS/LAZ `classification`, a PLY property named like
    classification / class / scalar_Classification / label, or the column
    `class_column` (0-based) of an NPY or ASCII file. E57 files have no standard
    class field (CloudCompare does not export it), so E57 always gives None.
    E57 points are transformed with their pose into the file frame.
    """
    path = str(path)
    low = path.lower()
    if low.endswith(".e57"):
        import pye57
        e = pye57.E57(path)
        parts = []
        for i in range(e.scan_count):
            d = e.read_scan(i, ignore_missing_fields=True, transform=True)
            parts.append(np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]))
        return np.vstack(parts).astype(np.float64), None
    if low.endswith((".las", ".laz")):
        try:
            import laspy
        except ImportError as exc:
            raise SystemExit("reading LAS/LAZ needs laspy (and lazrs for LAZ): "
                             "python -m pip install \"laspy[lazrs]\"") from exc
        las = laspy.read(path)
        return np.column_stack([las.x, las.y, las.z]).astype(np.float64), np.asarray(las.classification, int)
    if low.endswith(".ply"):
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        f = _class_field(v.data.dtype.names)
        cls = np.rint(np.asarray(v[f], float)).astype(int) if f else None
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64), cls
    if low.endswith(".npy"):
        a = np.load(path)
    else:
        a = np.atleast_2d(np.loadtxt(path, comments=("#", "//"), delimiter=None if not low.endswith(".csv") else ",",
                                     skiprows=_header_rows(path)))
    cls = np.rint(a[:, class_column]).astype(int) if class_column is not None else None
    return a[:, :3].astype(np.float64), cls


def _header_rows(path):
    """1 if the first line of an ASCII file is a header (not numeric), else 0."""
    with open(path) as f:
        first = f.readline().replace(",", " ").split()
    try:
        [float(x) for x in first]
        return 0
    except ValueError:
        return 1


def load_points(path):
    """x, y, z only (see read_cloud)."""
    return read_cloud(path)[0]


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


def read_scans(src: Source, class_column=None) -> list[Scan]:
    """One Scan per E57 scan (origin = pose translation unless given) or per other file (origin required)."""
    tokens = Path(src.path).stem.split("_")
    scanner = _clean(src.scanner or tokens[0])
    setup = _clean(src.setup or (tokens[1] if len(tokens) > 1 else Path(src.path).stem))
    if str(src.path).lower().endswith(".e57"):
        import pye57
        e = pye57.E57(str(src.path))
        scans = []
        for i in range(e.scan_count):
            h = e.get_header(i)
            d = e.read_scan(i, ignore_missing_fields=True, transform=True)
            pts = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]).astype(np.float64)
            origin = src.origin if src.origin is not None else np.asarray(h.translation, float)
            su = setup if e.scan_count == 1 else f"{setup}-{i + 1}"
            name = h.node["name"].value() if h.node.isDefined("name") else ""
            print(f"{Path(src.path).name}: scan {i + 1}/{e.scan_count} '{name}' -> {scanner}_{su}, "
                  f"{len(pts)} points, origin {np.round(origin, 3).tolist()}"
                  + (" (given)" if src.origin is not None else " (E57 pose)"))
            if _class_field(h.point_fields):
                print(f"  note: field {_class_field(h.point_fields)} is not a standard E57 field and is not read")
            raw = e.read_scan_raw(i)
            raw = np.column_stack([raw["cartesianX"], raw["cartesianY"], raw["cartesianZ"]]).astype(np.float64)
            scans.append(Scan(f"{scanner}_{su}", scanner, su, str(src.path), pts, origin,
                              src.origin is None, None, raw))
        return scans
    if src.origin is None:
        raise ValueError(f"{src.path}: the scanner origin is required for non-E57 input")
    pts, cls = read_cloud(src.path, class_column)
    print(f"{Path(src.path).name} -> {scanner}_{setup}, {len(pts)} points"
          + (f", classes {dict(zip(*np.unique(cls, return_counts=True)))}" if cls is not None else ", no class field"))
    return [Scan(f"{scanner}_{setup}", scanner, setup, str(src.path), pts, np.asarray(src.origin, float),
                 False, cls)]


# --------------------------------------------------------------------------- prepared scans
def save_prepared(p: Prepared, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{p.label}.npz", origin=p.origin,
                        **{f"{k}{side}": v for side, rail in p.rails.items() for k, v in zip("Psq", rail)})
    meta = dict(label=p.label, scanner=p.scanner, setup=p.setup, source=p.source, origin=p.origin.tolist(),
                band_method=p.band_method, selection=p.selection, head_width_m=p.head_width, axis=asdict(p.axis),
                bands={k: asdict(b) for k, b in p.bands.items()})
    with open(folder / f"{p.label}.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_prepared(results, scanners=None) -> list[Prepared]:
    """All scans prepared by 01_prepare.py, optionally only those of the given scanners."""
    folder = config.step_dir(results, "prepare")
    out = []
    for js in sorted(folder.glob("*.json")):
        if js.name in ("axis.json", "prepare_log.json"):
            continue
        with open(js) as f:
            m = json.load(f)
        if scanners and m["scanner"] not in scanners:
            continue
        a = np.load(js.with_suffix(".npz"))
        out.append(Prepared(m["label"], m["scanner"], m["setup"], m["source"], np.asarray(m["origin"]),
                            TrackAxis(**m["axis"]),
                            {k: RunningBand(**b) for k, b in m["bands"].items()},
                            {side: (a[f"P{side}"], a[f"s{side}"], a[f"q{side}"]) for side in ("L", "R")},
                            m["head_width_m"], m.get("band_method", "railhead"), m.get("selection")))
    if not out:
        raise SystemExit(f"no prepared scans in {folder}" + (f" for scanner(s) {scanners}" if scanners else "")
                         + "; run 01_prepare.py first")
    return out


def read_step_csv(results, step, name):
    """Result table of an earlier step, with a clear message if that step has not been run."""
    import pandas as pd
    path = config.step_dir(results, step) / name
    if not path.exists():
        raise SystemExit(f"{path} not found; run {config.STEPS[step]}.py first")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- provenance
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def provenance(script, args) -> dict:
    """What is needed to reproduce a result: script, options, code version, software versions, time."""
    import matplotlib
    import pandas as pd
    import scipy
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.PROCESSING,
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=config.PROCESSING,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return dict(script=Path(script).name, options={k: v for k, v in vars(args).items()},
                git_commit=commit, uncommitted_changes=dirty,
                run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python=sys.version.split()[0], platform=platform.platform(),
                packages=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                              matplotlib=matplotlib.__version__))
