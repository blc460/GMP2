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
    head_width: dict       # side -> measured railhead width [m] (NaN for --band groove)
    band_method: str = "railhead"


# --------------------------------------------------------------------------- point clouds
def load_points(path):
    """Read x, y, z from an E57 (all scans, transformed with their pose), PLY, NPY or ASCII file."""
    path = str(path)
    if path.lower().endswith(".e57"):
        import pye57
        e = pye57.E57(path)
        parts = []
        for i in range(e.scan_count):
            d = e.read_scan(i, ignore_missing_fields=True, transform=True)
            parts.append(np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]))
        return np.vstack(parts).astype(np.float64)
    if path.lower().endswith(".ply"):
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    if path.lower().endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    data = np.loadtxt(path, comments=("#", "//"), delimiter=None, usecols=(0, 1, 2))
    return np.atleast_2d(data).astype(np.float64)


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
            print(f"{Path(src.path).name}: scan {i + 1}/{e.scan_count} '{name}' -> {scanner}_{su}, "
                  f"{len(pts)} points, origin {np.round(origin, 3).tolist()}")
            scans.append(Scan(f"{scanner}_{su}", scanner, su, str(src.path), pts, origin))
        return scans
    if src.origin is None:
        raise ValueError(f"{src.path}: the scanner origin is required for non-E57 input")
    return [Scan(f"{scanner}_{setup}", scanner, setup, str(src.path), load_points(src.path),
                 np.asarray(src.origin, float))]


# --------------------------------------------------------------------------- prepared scans
def save_prepared(p: Prepared, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{p.label}.npz", origin=p.origin,
                        **{f"{k}{side}": v for side, rail in p.rails.items() for k, v in zip("Psq", rail)})
    meta = dict(label=p.label, scanner=p.scanner, setup=p.setup, source=p.source, origin=p.origin.tolist(),
                band_method=p.band_method, head_width_m=p.head_width, axis=asdict(p.axis),
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
                            m["head_width_m"], m.get("band_method", "railhead")))
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
