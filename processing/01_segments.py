"""Step 1: point noise sigma_z per 0.5 m cell along the rails, for the railhead points of each classified point cloud.

1. Ground plane: one plane through all railhead points (3 MAD rejection, refit); points and scanner
   are rotated so that it is horizontal.
2. Rails: connected groups of points in plan view, cut into windows and pieces; each piece is cut into
   0.5 m cells along its main direction (PCA). Cells wider than `max_width` across the rail (switches,
   crossings) are dropped.
3. Per cell: least-squares plane z = b0 + b1 x + b2 y, residuals beyond k MAD dropped once, plane
   refitted; sigma_z = 1.4826 MAD of all residuals of the final fit. d = horizontal distance from the
   scanner to the cell centroid, theta = angle between the scan ray and the cell plane normal,
   theta_ground = angle between the scan ray and the ground plane normal.
   Variant without plane: sigma_z_med = 1.4826 MAD of the levelled heights about their median (all points).
4. Density per cell: n / (length * width), both 2 % to 98 % of the points along and across the rail;
   expected density on a horizontal surface rho_exp = cos(theta_ground) / (r * dphi)^2 with the slant range r
   and the angular step dphi of the scan grid; return_rate = density / rho_exp; median intensity (as stored).

Output: results/<sok|lip>/01_segments.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cells import HERE, OUT, angular_step, cfg, cells, extent, level, mad_sigma, plane, read, width

rows = []
for ds in pd.read_csv(HERE / cfg["datasets"], keep_default_na=False).to_dict("records"):
    las, p0, origin0 = read(ds)
    dphi = angular_step(las, p0, origin0)
    intensity = np.asarray(las.intensity, float)
    p, origin, tilt = level(p0, origin0)
    rail, _, cell = cells(p)
    start, dropped = len(rows), 0
    order = np.argsort(cell)
    for idx in np.split(order, np.flatnonzero(np.diff(cell[order])) + 1):
        if cell[idx[0]] < 0 or len(idx) < cfg["min_points"]:
            continue
        if width(p[idx, :2]) > cfg["max_width"]:
            dropped += 1
            continue
        sigma, c, normal = plane(p[idx], cfg["outlier_k"])
        ray = c - origin
        r = np.linalg.norm(ray)
        c0 = p0[idx].mean(axis=0)
        length, wid = extent(p[idx, :2])
        cos_g = abs(ray[2]) / r
        rho_exp = cos_g / (r * dphi) ** 2
        rows.append(dict(dataset=Path(ds["path"]).stem, cell=cell[idx[0]], rail=rail[idx[0]],
                         scanner=ds["scanner"], setup=ds["setup"], condition=ds["condition"], x=c0[0], y=c0[1], n=len(idx),
                         sigma_z_mm=1000 * sigma, sigma_z_med_mm=1000 * mad_sigma(p[idx, 2]),
                         d_m=np.hypot(ray[0], ray[1]),
                         theta_deg=np.degrees(np.arccos(abs(ray @ normal) / r)),
                         theta_ground_deg=np.degrees(np.arccos(cos_g)), r_m=r,
                         length_m=length, width_m=wid, points_per_m=len(idx) / length,
                         density_pts_m2=len(idx) / (length * wid), rho_exp_pts_m2=rho_exp,
                         return_rate=len(idx) / (length * wid) / rho_exp, intensity=np.median(intensity[idx]),
                         dphi_mrad=1000 * dphi))
    t = pd.DataFrame(rows[start:])
    print(f"{ds['path']}: ground plane tilt {tilt:.3f} deg, angular step {1000 * dphi:.3f} mrad, {len(p)} points of class {cfg["rail_class"]}, {len(np.unique(rail))} rails, "
          f"{len(t)} cells ({dropped} dropped as switch), median sigma_z {t['sigma_z_mm'].median():.2f} mm (no plane {t['sigma_z_med_mm'].median():.2f} mm), "
          f"d {t['d_m'].min():.1f}-{t['d_m'].max():.1f} m, theta {t['theta_deg'].min():.0f}-{t['theta_deg'].max():.0f} deg")

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "01_segments.csv", index=False)
