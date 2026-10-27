"""Statistics on the segment table: summaries, operational limits, cant and twist."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = ["scanner", "condition", "strip"]
METRICS = ["sigma_z_mm", "sigma_zbar_mm", "density_pts_m2", "return_rate", "completeness"]


def _stats(g):
    out = {"segments": len(g), "flagged_share": g["flagged"].mean()}
    for m in METRICS:
        x = g[m].dropna()
        q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if len(x) else pd.Series(np.nan, [0.05, 0.25, 0.5, 0.75, 0.95])
        out.update({f"{m}_median": q[0.5], f"{m}_iqr": q[0.75] - q[0.25],
                    f"{m}_q05": q[0.05], f"{m}_q95": q[0.95]})
    return pd.Series(out)


def add_bins(df, cfg):
    df = df.copy()
    df["d_bin"] = pd.cut(df["d_m"], cfg["bins"]["distance"], right=False)
    df["theta_bin"] = pd.cut(df["theta_deg"], cfg["bins"]["incidence"], right=False)
    return df


def summaries(df, cfg):
    """Per scanner, per scanner and setup, per distance bin and per incidence bin."""
    df = add_bins(df, cfg)
    out = {
        "summary_scanner": df.groupby(GROUP).apply(_stats, include_groups=False),
        "summary_setup": df.groupby(GROUP + ["setup", "platform"]).apply(_stats, include_groups=False),
        "summary_distance": df.groupby(GROUP + ["d_bin"], observed=True).apply(_stats, include_groups=False),
        "summary_incidence": df.groupby(GROUP + ["theta_bin"], observed=True).apply(_stats, include_groups=False),
    }
    return {k: v.reset_index() for k, v in out.items()}


def passes(df, cfg):
    """One boolean column per criterion: sigma_zbar <= sigma_req and enough points."""
    df = df.copy()
    ok_n = ~df["flag_few_points"]
    for c in cfg["limits"]["criteria"]:
        df[f"pass_{c['name']}"] = ok_n & (df["sigma_zbar_mm"] <= c["sigma_req_mm"])
    return df


def pass_shares(df, cfg, by):
    """Share of passing segments per bin of `by` ("d_bin" or "theta_bin"), per group and criterion."""
    df = add_bins(passes(df, cfg), cfg)
    rows = []
    for key, g in df.groupby(GROUP + [by], observed=True):
        for c in cfg["limits"]["criteria"]:
            rows.append(dict(zip(GROUP + [by], key), criterion=c["name"], sigma_req_mm=c["sigma_req_mm"],
                             segments=len(g), pass_share=g[f"pass_{c['name']}"].mean()))
    return pd.DataFrame(rows)


def limits(df, cfg):
    """Farthest distance bin and most oblique incidence bin in which at least p_pass of the segments pass.

    The bins are walked from near to far (from steep to oblique); the limit is the
    upper edge of the last bin before the first failing one. If no bin fails, the
    limit is the largest measured value, marked as reached_end.
    """
    p = cfg["limits"]["p_pass"]
    rows = []
    for by, value, name in (("d_bin", "d_m", "d_max_m"), ("theta_bin", "theta_deg", "theta_max_deg")):
        sh = pass_shares(df, cfg, by)
        for key, g in sh.groupby(GROUP + ["criterion"]):
            g = g.sort_values(by)
            ok = (g["pass_share"] >= p).to_numpy()
            n_ok = int(np.argmin(ok)) if not ok.all() else len(ok)
            sub = df[(df[GROUP] == pd.Series(dict(zip(GROUP, key[:3])))).all(axis=1)]
            if n_ok == len(ok):
                lim, end = float(sub[value].max()), True
            elif n_ok == 0:
                lim, end = np.nan, False
            else:
                lim, end = float(g[by].iloc[n_ok - 1].right), False
            rows.append(dict(zip(GROUP + ["criterion"], key), quantity=name, limit=lim, reached_end=end,
                             p_pass=p, sigma_req_mm=g["sigma_req_mm"].iloc[0]))
    return pd.DataFrame(rows)


def geometry(df, cfg):
    """Cant and twist after VBZ at cross-sections every 5 m along each rail, per dataset and strip.

    The partner rail is the segment on another rail whose centroid lies `partner_across`
    away perpendicular to the rail and within `partner_along` along it; its height is
    carried to the cross-section with its along-track slope. Each pair of rails (track)
    is taken once, with the cross-sections on the rail with the lower id. Segments near a
    junction are left out (the partner there is often a diverging rail). Left and right
    are seen in the direction of increasing station of that rail.
    """
    gc = cfg["geometry"]
    L = cfg["segments"]["length"]
    step = int(round(gc["cross_section_spacing"] / L))
    lo, hi = gc["partner_across"]
    rows = []
    for (label, strip), g in df.dropna(subset=["z_rail"]).groupby(["dataset", "strip"]):
        g = g[~g["flag_junction"].astype(bool)]
        for ra in sorted(g["rail"].unique()):
            other = g[g["rail"] > ra]
            cxy = other[["x_c", "y_c"]].to_numpy()
            for _, a in g[(g["rail"] == ra) & (g["segment"] % step == 0)].iterrows():
                ta = np.array([a["tan_x"], a["tan_y"]])
                d = cxy - [a["x_c"], a["y_c"]]
                along, across = d @ ta, d @ [-ta[1], ta[0]]
                ok = (np.abs(across) >= lo) & (np.abs(across) <= hi) & (np.abs(along) <= gc["partner_along"])
                if not ok.any():
                    continue
                j = np.flatnonzero(ok)[np.argmin(np.abs(along[ok]))]
                p = other.iloc[j]
                z_p = p["z_rail"] - p["slope"] * (d[j] @ [p["tan_x"], p["tan_y"]])
                left = across[j] > 0                      # partner to the left of rail a
                z_l, z_r = (z_p, a["z_rail"]) if left else (a["z_rail"], z_p)
                rows.append(dict(dataset=label, strip=strip, track=f"{ra}-{p['rail']}",
                                 rail_left=p["rail"] if left else ra, rail_right=ra if left else p["rail"],
                                 segment=a["segment"], s_m=a["s_mid"], x_m=a["x_c"], y_m=a["y_c"],
                                 rail_distance_m=abs(across[j]), cant_mm=1000 * (z_l - z_r),
                                 sigma_cant_mm=np.hypot(a["sigma_zbar_mm"], p["sigma_zbar_mm"])))
    if not rows:
        return pd.DataFrame()
    out = []
    for (_, strip, _), h in pd.DataFrame(rows).groupby(["dataset", "strip", "track"]):
        h = h.sort_values("s_m")
        b = gc["b_sok"] if strip == "sok" else gc["b_lip"]
        dist = h["s_m"].diff()
        h["twist_permille"] = h["cant_mm"].diff() / dist / b                    # [mm/m] = [per mille]
        h["sigma_twist_permille"] = np.sqrt(h["sigma_cant_mm"] ** 2 + h["sigma_cant_mm"].shift() ** 2) / dist / b
        out.append(h)
    return pd.concat(out, ignore_index=True)
