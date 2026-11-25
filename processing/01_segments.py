"""Step 1: point noise sigma_z and height precision sigma_Z per 0.5 m cell along the rails.

1. Ground plane: one plane through all railhead points (3 MAD rejection, refit); points and scanner are rotated
   so that it is horizontal.
2. Rails and cells (cells.py): connected groups of points, cut into 5 m pieces and 0.5 m cells along each piece's
   main direction (PCA). Cells with fewer than `min_points` are skipped, cells wider than `max_width` across the
   rail (switches, crossings) are dropped.
3. sigma_z = 1.4826 MAD of the levelled heights about their median (all points of the cell, no model).
4. sigma_Z, the precision of the cell height Z, three ways (see precision.py): sqrt(n), batch means of
   `n_bins` (8) bins along the rail with its 95 % interval, and a moving-block bootstrap along the rail.
5. Geometry: r = slant range scanner to cell centroid, theta_ground = angle between the ray and the ground normal.
   Density = n / (length * width), both 2 % to 98 % of the points; rho_exp = cos(theta_ground) /
   (r^2 sin(theta_ground) dphi^2) is what the scan grid delivers on a horizontal surface; return_rate = ratio.
   Optional (plane_fit): sigma_z_plane = 1.4826 MAD of the residuals of a plane per cell, theta to that plane.

Output: results/<sok|lip>/01_segments.csv, one row per cell
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cells import OUT, along_across, angular_step, cells, cfg, datasets, iter_cells, level, plane, read
from precision import mad_sigma, sigma_Z_boot, sigma_Z_bins, sigma_Z_sqrt_n

RNG = np.random.default_rng(2026)


def heights_and_precision(p):
    """sigma_z and the three sigma_Z of one cell (levelled points p) [mm]."""
    z = p[:, 2]
    u, _ = along_across(p[:, :2])
    sigma_z = 1000 * mad_sigma(z)
    bins, lo, hi, m = sigma_Z_bins(u, z, cfg["n_bins"], cfg["min_bin_points"], cfg["bin_trim"])
    boot = sigma_Z_boot(u, z, cfg["boot_block"], cfg["boot_slice"], cfg["boot_runs"], RNG)
    return dict(Z_m=np.median(z), sigma_z_mm=sigma_z,
                sigma_Z_sqrt_n_mm=sigma_Z_sqrt_n(sigma_z, len(z)),
                sigma_Z_bins_mm=1000 * bins, sigma_Z_bins_lo_mm=1000 * lo, sigma_Z_bins_hi_mm=1000 * hi, n_bins=m,
                sigma_Z_boot_mm=1000 * boot)


def geometry(p, p0, origin, dphi, length, wid):
    """Range, incidence angle and point density of one cell (length and width from iter_cells)."""
    c = p.mean(axis=0)
    ray = c - origin
    r = np.linalg.norm(ray)
    cos_g = abs(ray[2]) / r
    density = len(p) / (length * wid)
    rho_exp = cos_g / (r ** 2 * np.sqrt(1 - cos_g ** 2) * dphi ** 2)
    c0 = p0.mean(axis=0)
    return dict(x=c0[0], y=c0[1], r_m=r, theta_ground_deg=np.degrees(np.arccos(cos_g)), length_m=length,
                width_m=wid, points_per_m=len(p) / length, density_pts_m2=density, rho_exp_pts_m2=rho_exp,
                return_rate=density / rho_exp), ray


def process(ds):
    las, p0, origin0 = read(ds)
    dphi = angular_step(las, p0, origin0)
    p, origin, tilt = level(p0, origin0)
    rail, piece, cell = cells(p)
    intensity = np.asarray(las.intensity, float)
    name = Path(ds["path"]).stem
    rows, dropped = [], 0
    for c in iter_cells(p, cell):
        if c.status == "switch":
            dropped += 1
        if c.status != "keep":
            continue
        i = c.idx
        geo, ray = geometry(p[i], p0[i], origin, dphi, c.length, c.width)
        row_ = dict(dataset=name, cell=c.id, rail=rail[i[0]], piece=piece[i[0]], scanner=ds["scanner"],
                    setup=ds["setup"], condition=ds["condition"], n=len(i), **geo,
                    intensity=np.median(intensity[i]), dphi_mrad=1000 * dphi,
                    **heights_and_precision(p[i]))
        if cfg["plane_fit"]:
            sigma, _, normal = plane(p[i], cfg["outlier_k"])
            row_.update(sigma_z_plane_mm=1000 * sigma,
                        theta_deg=np.degrees(np.arccos(abs(ray @ normal) / geo["r_m"])))
        rows.append(row_)
    t = pd.DataFrame(rows)
    print(f"{name}: tilt {tilt:.3f} deg, step {1000 * dphi:.3f} mrad, {len(p)} points, {len(np.unique(rail))} rails, "
          f"{len(t)} cells ({dropped} dropped as switch), median sigma_z {t['sigma_z_mm'].median():.2f} mm, "
          f"sigma_Z sqrt(n) / bins / boot "
          f"{t['sigma_Z_sqrt_n_mm'].median():.3f} / {t['sigma_Z_bins_mm'].median():.3f} / "
          f"{t['sigma_Z_boot_mm'].median():.3f} mm, "
          f"r {t['r_m'].min():.1f}-{t['r_m'].max():.1f} m")
    return t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([process(ds) for ds in datasets()]).to_csv(OUT / "01_segments.csv", index=False)
