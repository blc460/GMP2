"""Reading the settings, the dataset list and the point clouds."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np


@dataclass
class Dataset:
    path: Path
    scanner: str
    setup: str
    condition: str
    platform: str
    mode: str
    angular_step_mrad: float        # NaN if unknown (then no return rate)
    origin: np.ndarray | None       # None: from the E57 pose, else (0, 0, 0)

    @property
    def label(self):
        return f"{self.scanner}_{self.setup}_{self.condition}"


def load_config(path):
    path = Path(path).resolve()
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = path
    cfg["datasets"] = (path.parent / cfg["datasets"]).resolve()
    cfg["results"] = (path.parent / cfg["results"]).resolve()
    return cfg


def read_datasets(cfg) -> list[Dataset]:
    """Rows of datasets.csv. Empty mode / origin cells fall back to the config / the scanner pose."""
    path = cfg["datasets"]
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            get = lambda k: (row.get(k) or "").strip()
            o = [get(k) for k in ("origin_x", "origin_y", "origin_z")]
            out.append(Dataset(
                path=(path.parent / get("path")).resolve(),
                scanner=get("scanner"), setup=get("setup"),
                condition=get("condition") or "dry", platform=get("platform") or "tripod",
                mode=get("mode") or cfg["input"]["mode"],
                angular_step_mrad=float(get("angular_step_mrad") or "nan"),
                origin=np.array(o, float) if all(o) else None))
    return out


def read_cloud(ds: Dataset, class_field: str):
    """Points (n, 3), class (n,) or None, the scanner origin (3,) and the storage grid of the heights [m].

    E57: the scan pose is applied, so the points are in the file frame, and the
    origin is the pose translation. For a scan stored in its scanner frame
    (identity pose) this is (0, 0, 0). A CloudCompare export stores a global
    shift as pose instead; then the origin must be given in datasets.csv.
    Other formats: the points are taken as stored and the origin is (0, 0, 0)
    unless given. The height grid is looked for in the stored coordinates (a
    pose with a slight tilt smears it in the transformed ones).
    """
    p = str(ds.path)
    low = p.lower()
    cls = None
    if low.endswith(".e57"):
        import pye57
        e = pye57.E57(p)
        if e.scan_count != 1:
            raise ValueError(f"{p}: {e.scan_count} scans; export one scan per file")
        h = e.get_header(0)
        fields = list(h.point_fields)
        d = e.read_scan(0, ignore_missing_fields=True, transform=True)
        pts = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]])
        f = _match(fields, class_field)
        if f:
            cls = np.rint(np.asarray(e.read_scan_raw(0)[f], float)).astype(int)
        origin = np.asarray(h.translation, float)
    elif low.endswith((".las", ".laz")):
        import laspy
        las = laspy.read(p)
        pts = np.column_stack([las.x, las.y, las.z])
        cls = np.asarray(las.classification, int)
        origin = np.zeros(3)
    elif low.endswith(".ply"):
        from plyfile import PlyData
        v = PlyData.read(p)["vertex"]
        pts = np.column_stack([v["x"], v["y"], v["z"]])
        f = _match(v.data.dtype.names, class_field)
        if f:
            cls = np.rint(np.asarray(v[f], float)).astype(int)
        origin = np.zeros(3)
    else:
        raise ValueError(f"{p}: unsupported format (E57, LAS/LAZ or PLY)")
    if ds.origin is not None:
        origin = ds.origin
    if ds.mode == "classified" and cls is None:
        raise ValueError(f"{p}: mode 'classified' but no field '{class_field}' in the file")
    return np.asarray(pts, float), cls, origin


def _match(names, wanted):
    """Field name that equals `wanted` ignoring case and a 'scalar_' prefix (CloudCompare)."""
    for n in names:
        if n.lower().removeprefix("scalar_") == wanted.lower():
            return n
    return None
