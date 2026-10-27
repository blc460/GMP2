"""Per-segment metrics: density, noise (MAD), rail height, distance and incidence angle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Dataset, read_cloud
from .track import class_strip, detect_rails, head_strip

MAD_TO_SIGMA = 1.4826


def mad_sigma(e):
    return MAD_TO_SIGMA * np.median(np.abs(e - np.median(e)))


def fit_plane(a, b, z):
    """Least-squares plane z = b0 + b1 a + b2 b; returns the coefficients."""
    A = np.column_stack([np.ones_like(a), a, b])
    return np.linalg.lstsq(A, z, rcond=None)[0]


def plane_residuals(coef, a, b, z):
    return z - (coef[0] + coef[1] * a + coef[2] * b)


def segment_table(ds: Dataset, cfg, figures=None):
    """One row per rail, strip and segment of one dataset. Plan view of the rails to `figures` if given."""
    pts, cls, origin = read_cloud(ds, cfg["input"]["class_field"])
    ic = cfg["input"]
    use = np.isin(cls, [ic["class_sok"], ic["class_lip"]]) if ds.mode == "classified" else np.ones(len(pts), bool)
    rails, junctions = detect_rails(pts, use, cfg["rails"])
    if figures is not None:
        from .plots import rail_plan
        rail_plan(pts, rails, junctions, origin, ds.label, figures)
    L = cfg["segments"]["length"]
    strips = ["sok"] + (["lip"] if ds.mode == "classified" and ic["use_lip"] else [])

    rows = []
    for rail in rails:
        s_scanner = float(rail.s_line[np.argmin(np.linalg.norm(rail.line - origin[:2], axis=1))])
        k_all = np.floor(rail.s / L).astype(int)
        for which in strips:
            if ds.mode == "classified":
                m = class_strip(rail, cls, cfg, which)
                width = float(np.ptp(np.quantile(rail.t[m], [0.005, 0.995]))) if m.sum() >= 3 else np.nan
            for seg in np.unique(k_all):
                j = np.flatnonzero(k_all == seg)
                s, t, P = rail.s[j], rail.t[j], pts[rail.idx[j]]
                if ds.mode == "classified":
                    keep = m[j]
                elif len(j) >= 3:
                    lo, hi = head_strip(s, t, P[:, 2], cfg["strip"])
                    keep, width = (t >= lo) & (t <= hi), hi - lo
                    width = width if width > 0 else np.nan
                else:
                    keep, width = np.zeros(len(j), bool), np.nan
                s_mid = (seg + 0.5) * L
                xy, tan = rail.at(s_mid)
                jd = float(np.linalg.norm(junctions - xy, axis=1).min()) if len(junctions) else np.nan
                row = dict(dataset=ds.label, scanner=ds.scanner, setup=ds.setup, condition=ds.condition,
                           platform=ds.platform, rail=rail.id, strip=which, segment=int(seg),
                           s_mid=s_mid, scanner_s_m=s_scanner, strip_width_m=width, junction_dist_m=jd,
                           tan_x=tan[0], tan_y=tan[1])
                row.update(_metrics(P[keep], s[keep], t[keep], tan, seg * L, origin, ds, cfg, width))
                rows.append(row)
    df = pd.DataFrame(rows)
    return _flags(df, cfg)


def _metrics(P, s, t, tan, s_start, origin, ds, cfg, width):
    sc = cfg["segments"]
    L = sc["length"]
    n = len(P)
    out = dict(n=n)
    if n < 3:
        return out
    # plane in the local frame of the segment: a along the rail, b to the left, about the centroid
    c = P.mean(axis=0)
    e_b = np.array([-tan[1], tan[0]])
    a, b = (P[:, :2] - c[:2]) @ tan, (P[:, :2] - c[:2]) @ e_b
    z = P[:, 2]

    # plane fit, reject beyond k * MAD of the first fit, refit once
    coef = fit_plane(a, b, z)
    e = plane_residuals(coef, a, b, z)
    mad = mad_sigma(e)
    keep = np.abs(e - np.median(e)) <= sc["outlier_k"] * mad if mad > 0 else np.ones(n, bool)
    if keep.sum() >= 3:
        coef = fit_plane(a[keep], b[keep], z[keep])
    e = plane_residuals(coef, a, b, z)          # residuals of the final fit, all points
    sigma_z = mad_sigma(e)
    n_used = int(keep.sum())

    # distance and incidence: segment centroid, plane normal in xyz
    v = c - origin
    r = float(np.linalg.norm(v))
    normal = np.append(-coef[1] * tan - coef[2] * e_b, 1.0)
    normal /= np.linalg.norm(normal)
    theta = np.degrees(np.arccos(min(1.0, abs(v @ normal) / r)))
    vp = P - origin
    theta_pts = np.degrees(np.arccos(np.clip(np.abs(vp @ normal) / np.linalg.norm(vp, axis=1), 0, 1)))
    h = float(origin[2] - c[2])

    # density, expected points from the scan resolution, completeness
    density = n / (L * width)
    dphi = ds.angular_step_mrad / 1000
    n_exp = L * width * h / (dphi ** 2 * r ** 3) if np.isfinite(dphi) else np.nan
    na = max(1, int(round(L / sc["cell_along"])))
    nc = max(1, int(round(width / sc["cell_across"])))
    ia = np.clip(((s - s_start) / L * na).astype(int), 0, na - 1)
    ic = np.clip(((t - t.min()) / max(width, 1e-9) * nc).astype(int), 0, nc - 1)
    completeness = len(np.unique(ia * nc + ic)) / (na * nc)
    edges = np.concatenate([[s_start], np.sort(s), [s_start + L]])
    max_gap = float(np.diff(edges).max())

    out.update(n_used=n_used, outlier_share=1 - n_used / n,
               sigma_z_mm=1000 * sigma_z, sigma_zbar_mm=1000 * sigma_z / np.sqrt(n_used),
               z_rail=float(coef[0]), slope=float(coef[1]), x_c=float(c[0]), y_c=float(c[1]),
               density_pts_m2=density, n_expected=n_exp,
               return_rate=n / n_exp if np.isfinite(n_exp) else np.nan,
               completeness=completeness, max_gap_m=max_gap,
               r_m=r, d_m=float(np.hypot(v[0], v[1])), h_m=h,
               theta_deg=theta, theta_min_deg=float(theta_pts.min()), theta_max_deg=float(theta_pts.max()))
    return out


def _flags(df, cfg):
    f = cfg["flags"]
    n_min = cfg["segments"]["min_points"]
    df["flag_few_points"] = df["n_used"].fillna(0) < n_min
    df["flag_outliers"] = df["outlier_share"].fillna(1) > f["max_outlier_share"]
    df["flag_theta_spread"] = (df["theta_max_deg"] - df["theta_min_deg"]).fillna(0) > f["max_theta_spread"]
    df["flag_incomplete"] = df["completeness"].fillna(0) < f["min_completeness"]
    df["flag_junction"] = df["junction_dist_m"] <= cfg["rails"]["junction_buffer"]
    df["flagged"] = df[[c for c in df if c.startswith("flag_")]].any(axis=1)
    return df
