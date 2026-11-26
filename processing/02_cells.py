"""Step 2: point noise sigma_z, height precision sigma_Z and geometry per 0.5 m cell along the rails.

1. Ground plane: one plane through all railhead points (3 MAD rejection, refit); points and scanner are rotated
   so that it is horizontal (cells.level).
2. Rails and cells (cells.py): connected groups of points, cut into 5 m pieces and 0.5 m cells along each piece's
   main direction (PCA). Cells with fewer than `min_points` are skipped, cells wider than `max_width` across the
   rail (switches, crossings) are dropped.
3. sigma_z = 1.4826 MAD of the levelled heights about their median (all points of the cell, no model).
4. sigma_Z (see precision.py): sqrt(n), and from m bins along the rail with its 95 % bootstrap interval. m per
   scanner from the Allan rule (01_bins.csv of step 1); a cell with a sparse bin uses fewer bins (n_bins column).
5. Geometry: r = slant range scanner to cell centroid, theta_ground = angle between the ray and the ground normal,
   h = height of the scanner above the cell. beam_rail_deg = angle in plan view between the ray and the rail axis
   of the cell (0 = the scanner looks along the rail, 90 = across it), direction = its class (`direction`).
   Near / far rail: the partner of a cell is a kept cell on another, parallel rail at `gauge_centres` +-
   `gauge_tol` across and less than half a cell along (the other rail of the same track); of the two, the one
   closer to the scanner in plan view is "near", the other "far"; cells without a partner are "single".
   Density = n / (length * width), both 2 % to 98 % of the points; rho_exp = cos(theta_ground) /
   (r^2 sin(theta_ground) dphi^2) is what the scan grid delivers on a horizontal surface; return_rate = ratio.
   Optional (plane_fit): sigma_z_plane = 1.4826 MAD of the residuals of a plane per cell, theta to that plane.

Output: results/<sok|lip>/02_cells.csv, one row per cell
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from cells import OUT, along_across, angular_step, cells, cfg, datasets, iter_cells, level, main_axis, plane, read
from precision import mad_sigma, sigma_Z_bins, sigma_Z_sqrt_n

RNG = np.random.default_rng(2026)


def bins_per_scanner():
    """Number of bins per scanner from the Allan rule (01_bins.csv); `n_bins` if step 1 has not been run."""
    path = OUT / "01_bins.csv"
    if not path.exists():
        print(f"warning: {path.name} missing (run 01_allan.py first), using n_bins = {cfg['n_bins']} for all scanners")
        return {}
    return dict(pd.read_csv(path)[["scanner", "n_bins"]].itertuples(index=False))


def heights_and_precision(p, m):
    """sigma_z and sigma_Z (sqrt(n) and m bins) of one cell (levelled points p) [mm]."""
    z = p[:, 2]
    u, _ = along_across(p[:, :2])
    sigma_z = 1000 * mad_sigma(z)
    bins, lo, hi, used = sigma_Z_bins(u, z, m, cfg["min_bins"], cfg["min_bin_points"], cfg["bin_trim"],
                                      cfg["bin_boot_runs"], rng=RNG)
    return dict(Z_m=np.median(z), sigma_z_mm=sigma_z, sigma_Z_sqrt_n_mm=sigma_Z_sqrt_n(sigma_z, len(z)),
                sigma_Z_bins_mm=1000 * bins, sigma_Z_bins_lo_mm=1000 * lo, sigma_Z_bins_hi_mm=1000 * hi, n_bins=used)


def geometry(p, p0, origin, dphi, length, wid):
    """Range, incidence angle and point density of one cell (length and width from iter_cells)."""
    c = p.mean(axis=0)
    ray = c - origin
    r = np.linalg.norm(ray)
    cos_g = abs(ray[2]) / r
    density = len(p) / (length * wid)
    rho_exp = cos_g / (r ** 2 * np.sqrt(1 - cos_g ** 2) * dphi ** 2)
    c0 = p0.mean(axis=0)
    axis = main_axis(p[:, :2])
    beam = ray[:2] / np.linalg.norm(ray[:2])
    beam_rail = np.degrees(np.arccos(min(1.0, abs(beam @ axis))))
    return dict(x=c0[0], y=c0[1], xl=c[0], yl=c[1], axis_x=axis[0], axis_y=axis[1], r_m=r,
                r_plan_m=np.linalg.norm(ray[:2]), h_m=-ray[2], theta_ground_deg=np.degrees(np.arccos(cos_g)),
                beam_rail_deg=beam_rail, length_m=length,
                width_m=wid, points_per_m=len(p) / length, density_pts_m2=density, rho_exp_pts_m2=rho_exp,
                return_rate=density / rho_exp), ray


def near_far(t):
    """'near' / 'far' / 'single' per cell of one scan (see module doc)."""
    xy = t[["xl", "yl"]].to_numpy()
    axis = t[["axis_x", "axis_y"]].to_numpy()
    normal = np.column_stack([-axis[:, 1], axis[:, 0]])
    rail = t["rail"].to_numpy()
    dist = t["r_plan_m"].to_numpy()
    out = np.full(len(t), "single", dtype=object)
    partner_gap = np.full(len(t), np.nan)
    tree = cKDTree(xy)
    for i, js in enumerate(tree.query_ball_point(xy, cfg["gauge_centres"] + cfg["gauge_tol"] + cfg["cell"] / 2)):
        best, best_err = None, cfg["gauge_tol"]
        for j in js:
            if rail[j] == rail[i] or abs(axis[i] @ axis[j]) < np.cos(np.radians(10)):
                continue
            d = xy[j] - xy[i]
            across, along = abs(d @ normal[i]), abs(d @ axis[i])
            if along < cfg["cell"] / 2 and abs(across - cfg["gauge_centres"]) <= best_err:
                best, best_err = j, abs(across - cfg["gauge_centres"])
        if best is not None:
            out[i] = "near" if dist[i] < dist[best] else "far"
            partner_gap[i] = abs((xy[best] - xy[i]) @ normal[i])
    return out, partner_gap


def direction_class(deg):
    return pd.cut(deg, cfg["direction"], labels=cfg["direction_names"], include_lowest=True).astype(str)


def process(ds, m):
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
                    **heights_and_precision(p[i], m))
        if cfg["plane_fit"]:
            sigma, _, normal = plane(p[i], cfg["outlier_k"])
            row_.update(sigma_z_plane_mm=1000 * sigma,
                        theta_deg=np.degrees(np.arccos(abs(ray @ normal) / geo["r_m"])))
        rows.append(row_)
    t = pd.DataFrame(rows)
    t["rail_side"], t["gauge_m"] = near_far(t)
    t["direction"] = direction_class(t["beam_rail_deg"])
    print(f"{name}: tilt {tilt:.3f} deg, step {1000 * dphi:.3f} mrad, {len(p)} points, {len(np.unique(rail))} rails, "
          f"{len(t)} cells ({dropped} dropped as switch), median sigma_z {t['sigma_z_mm'].median():.2f} mm, "
          f"sigma_Z sqrt(n) / {m} bins {t['sigma_Z_sqrt_n_mm'].median():.3f} / "
          f"{t['sigma_Z_bins_mm'].median():.3f} mm ({(t['n_bins'] < m).sum()} cells with fewer bins), "
          f"near / far / single {(t['rail_side'] == 'near').sum()} / {(t['rail_side'] == 'far').sum()} / "
          f"{(t['rail_side'] == 'single').sum()}, "
          f"r {t['r_m'].min():.1f}-{t['r_m'].max():.1f} m")
    return t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    bins = bins_per_scanner()
    pd.concat([process(ds, int(bins.get(ds["scanner"], cfg["n_bins"]))) for ds in datasets()]).to_csv(
        OUT / "02_cells.csv", index=False)
