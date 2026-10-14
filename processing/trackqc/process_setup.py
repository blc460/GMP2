"""Process one dataset (scanner x setup x surface condition).

Input: two clipped rail point clouds (left/right rail head incl. groove, from
CloudCompare), the scanner origin in the same frame, and a shared track axis.

    python -m trackqc.process_setup --left FARO_S1_dry_L.ply --right FARO_S1_dry_R.ply \
        --origin 36.680 3.389 0.002 --axis axis.json --label FARO_S1_dry --out results/

If axis.json does not exist yet it is fitted from this dataset and saved; use the
densest dataset for that and reuse the file for all others, so that the stations
are identical across scanners and setups.

Writes to --out:
  <label>_segments.csv   point-cloud quality per 0.5 m rail segment (WP4)
  <label>_geometry.csv   rail heights, cant, twist, versine + formal std (WP5)
  <label>_profile.png    stacked lateral rail profile with the selected band
  <label>_summary.json   settings and summary statistics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import core


def run(left, right, origin, axis_path, label, out, seg_len=0.5, half_window=0.25,
        spacing=0.5, l=5.0, band_width=0.030, edge_margin=0.005, speed_class="48-60",
        plot=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    PL, PR = core.load_points(left), core.load_points(right)
    axis_path = Path(axis_path)
    if axis_path.exists():
        axis = core.TrackAxis.load(axis_path)
    else:
        axis = core.TrackAxis.from_rails(PL, PR)
        axis.save(axis_path)
        print(f"fitted new track axis -> {axis_path}")

    seg_rows, heights, bands = [], {}, {}
    sq = {}
    for side, P in (("L", PL), ("R", PR)):
        s, q = axis.to_sq(P[:, :2])
        if side == "L" and np.median(q) < 0 or side == "R" and np.median(q) > 0:
            raise RuntimeError(f"rail {side} lies on the wrong side of the axis; swap --left/--right")
        band = core.detect_running_band(s, q, P[:, 2], side, width=band_width, edge_margin=edge_margin)
        bands[side], sq[side] = band, (P, s, q)
        seg_rows += core.segment_metrics(P, s, q, band, origin, axis, seg_len=seg_len, label=label)

    # common stations: multiples of `spacing`, inside both rails
    smin = max(sq[k][1].min() for k in sq) + half_window
    smax = min(sq[k][1].max() for k in sq) - half_window
    stations = np.arange(np.ceil(smin / spacing) * spacing, smax + 1e-9, spacing)
    for side in ("L", "R"):
        P, s, q = sq[side]
        heights[side] = core.rail_heights(P, s, q, bands[side], stations, half_window)

    geo = core.track_geometry(stations, heights["L"][:, :2], heights["R"][:, :2],
                              bands["L"].center_q(stations), bands["R"].center_q(stations), l=l)
    geo.update(nL=heights["L"][:, 2], nR=heights["R"][:, 2],
               sigma_pcL_mm=heights["L"][:, 3] * 1e3, sigma_pcR_mm=heights["R"][:, 3] * 1e3)
    crit = core.check_criteria(geo, speed_class)
    geo.update(ok_twist=crit["ok_twist"], ok_vpf=crit["ok_vpf"])

    seg = pd.DataFrame(seg_rows)
    gdf = pd.DataFrame(geo)
    seg.to_csv(out / f"{label}_segments.csv", index=False)
    gdf.to_csv(out / f"{label}_geometry.csv", index=False)

    sz = np.r_[geo["sigma_zL_mm"], geo["sigma_zR_mm"]]
    summary = dict(
        label=label, origin=list(map(float, origin)), seg_len=seg_len, half_window=half_window,
        spacing=spacing, l=l, band={k: dict(q_lo=b.q_lo, q_hi=b.q_hi, groove_edge=b.groove_edge)
                                     for k, b in bands.items()},
        speed_class=speed_class, sigma_z_req_mm=crit["sigma_z_req_mm"],
        median_sigma_pc_mm=float(seg["sigma_pc_mm"].median()) if "sigma_pc_mm" in seg else None,
        median_sigma_z_formal_mm=float(np.nanmedian(np.r_[geo["sigma_zL_mm"], geo["sigma_zR_mm"]])),
        share_heights_meeting_req=float(np.mean(sz[np.isfinite(sz)] <= crit["sigma_z_req_mm"])),
    )
    with open(out / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if plot:
        plot_profiles(bands, out / f"{label}_profile.png", label)
    return seg, gdf, summary


def plot_profiles(bands, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(9, 5), sharex="col")
    for j, side in enumerate(("L", "R")):
        b = bands[side]
        qmm = np.array(b.profile["q"]) * 1e3
        axs[0, j].plot(qmm, np.array(b.profile["median_dz"]) * 1e3, ".-", ms=3)
        axs[1, j].bar(qmm, b.profile["count"], width=1.8)
        for a in axs[:, j]:
            a.axvspan(b.q_lo * 1e3, b.q_hi * 1e3, color="tab:green", alpha=0.2, label="running band")
            a.axvline(b.groove_edge * 1e3, color="tab:red", lw=1, label="groove edge")
        axs[0, j].set_title(f"{label}: rail {side}")
        axs[0, j].set_ylabel("median dz [mm]")
        axs[1, j].set_ylabel("points per 2 mm bin")
        axs[1, j].set_xlabel("q_rel [mm]")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--origin", nargs=3, type=float, required=True, help="scanner origin x y z")
    ap.add_argument("--axis", required=True, help="track axis JSON (created if missing)")
    ap.add_argument("--label", required=True, help="e.g. FARO_S1_dry")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seg-len", type=float, default=0.5)
    ap.add_argument("--half-window", type=float, default=0.25)
    ap.add_argument("--band-width", type=float, default=0.030)
    ap.add_argument("--edge-margin", type=float, default=0.005)
    ap.add_argument("--speed-class", default="48-60", choices=list(core.ATTENTION))
    a = ap.parse_args()
    _, _, summary = run(a.left, a.right, a.origin, a.axis, a.label, a.out, a.seg_len, a.half_window,
                        band_width=a.band_width, edge_margin=a.edge_margin, speed_class=a.speed_class)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
