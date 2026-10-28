"""Step 1: point noise sigma_z per 0.5 m x 0.5 m cell of the railhead points of each classified point cloud.

Per cell: least-squares plane z = b0 + b1 x + b2 y, residuals beyond k MAD dropped once, plane
refitted; sigma_z = 1.4826 MAD of all residuals of the final fit. d = horizontal distance from the
scanner to the cell centroid, theta = angle between the scan ray and the plane normal.

Output: results/01_segments.csv
"""
from pathlib import Path
import tomllib

import laspy
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
with open(HERE / "config.toml", "rb") as f:
    cfg = tomllib.load(f)


def mad_sigma(e):
    return 1.4826 * np.median(np.abs(e - np.median(e)))


def plane(p, k):
    """sigma_z, centroid and unit normal of the plane through the points p (n, 3)."""
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


rows = []
for ds in pd.read_csv(HERE / cfg["datasets"], keep_default_na=False).to_dict("records"):
    las = laspy.read(HERE / ds["path"])
    p = np.column_stack([las.x, las.y, las.z])[np.asarray(las.classification) == cfg["class_sok"]]
    origin = np.array([float(ds[k] or 0) for k in ("origin_x", "origin_y", "origin_z")])
    start = len(rows)
    ij = np.floor(p[:, :2] / cfg["cell"]).astype(int)
    _, cell, n = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(cell)
    for idx in np.split(order, np.cumsum(n)[:-1]):
        if len(idx) < cfg["min_points"]:
            continue
        sigma, c, normal = plane(p[idx], cfg["outlier_k"])
        ray = c - origin
        theta = np.degrees(np.arccos(abs(ray @ normal) / np.linalg.norm(ray)))
        rows.append(dict(scanner=ds["scanner"], condition=ds["condition"], x=c[0], y=c[1], n=len(idx),
                         sigma_z_mm=1000 * sigma, d_m=np.hypot(ray[0], ray[1]), theta_deg=theta))
    t = pd.DataFrame(rows[start:])
    print(f"{ds['path']}: {len(p)} railhead points, {len(t)} cells, "
          f"median sigma_z {t['sigma_z_mm'].median():.2f} mm, d {t['d_m'].min():.1f}-{t['d_m'].max():.1f} m, "
          f"theta {t['theta_deg'].min():.0f}-{t['theta_deg'].max():.0f} deg")

out = HERE / cfg["results"]
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "01_segments.csv", index=False)
