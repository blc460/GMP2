"""01 - Prepare the rail point clouds for all following steps.

Two kinds of input are supported:

A) clips cropped to the two rail heads of one track (e.g. track-test.e57):
       python 01_prepare.py ../pointclouds/track-test.e57 --scanner RTC360 --setup E004
B) classified clouds (rail-head class from a segmentation, e.g. track-test-2-classified),
   possibly with several tracks: keep the rail-head class (--keep-class, default 1;
   only if the file has a class field: LAS/LAZ, PLY, ASCII/NPY with --class-column)
   and cut one track out with a corridor along its centre line (--track, --corridor):
       python 01_prepare.py ../pointclouds/track-test-2-classified.e57 --scanner RTC360 --setup E004 \\
           --origin 6.268 3.132 3.140 --track 8.5 8.40 25.5 8.80 --corridor 1.6

Other examples:
    python 01_prepare.py ../pointclouds/FARO_S1_dry.e57 ../pointclouds/RTC360_S1_dry.e57
    python 01_prepare.py --list datasets.csv      # path,scanner,setup[,origin_x,origin_y,origin_z]

Steps per scan:
1. Read the points (E57: every scan is one dataset, transformed with its pose;
   scanner origin = pose translation unless --origin is given).
2. Class filter: keep only points of --keep-class, if the file has a class field.
3. Track corridor: keep only points within --corridor/2 of the --track polyline.
4. Height resolution: if the stored coordinates lie on a grid (RTC360 exports
   of REGISTER 360: 1 mm in the scanner frame), its step q is detected
   (shared/geometry.py: grid_step) or set with --z-step. All plane fits then
   use q / sqrt(12) as lower bound of the robust (MAD) sigma, so a collapsed
   MAD (most points on one height level) is not read as perfect precision.
5. Origin check: the scanner position implied by the point density
   (density ~ h / r^3) must agree in plan with the origin within --origin-tol;
   otherwise stop (an E57 exported from LAS/LAZ carries no scanner pose).
6. Split into left and right rail at the empty gap between them.
7. Track axis = mean of both rail centre lines (cubic). Fitted once from the
   first scan, saved as <results>/01_prepare/axis.json and reused for every
   later scan, so all scans share the same stations s. --refit-axis fits a new one.
8. Stations s along the axis and lateral offsets q for every point.
9. Band used for all fits (--band, default auto: railhead if both rail clips
   are at most 100 mm wide, else groove):
   railhead: clip cropped to the head; head edges = 0.5 % / 99.5 % quantiles
     of the offset from the rail centre line, band = head minus --edge-trim per side;
   groove: clip incl. groove and paving; the groove is the deepest part of the
     stacked lateral profile, the band of --band-width starts --edge-margin beyond
     the groove edge on the field side (see shared/geometry.py).

Writes to <results>/01_prepare/:
  axis.json              track axis shared by all scans
  <label>.npz            points of both rails with s and q
  <label>.json           scanner, setup, origin, selection, axis, band, head width
  <label>_railhead.png   lateral profile with the band: check it for every scan
  prepare_log.json       inputs (with SHA-256), options, code and software versions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared import config, geometry, io, plots, stats


def select(scan: io.Scan, keep_class, track, corridor):
    """Class filter and track corridor; returns the selected points and a record of what was done."""
    P, sel = scan.pts, dict(n_points_file=int(len(scan.pts)))
    if scan.cls is not None and keep_class is not None:
        m = np.isin(scan.cls, keep_class)
        sel.update(class_field=True, keep_class=list(keep_class), n_after_class=int(m.sum()))
        P = P[m]
    else:
        sel.update(class_field=scan.cls is not None, keep_class=None)
        if scan.cls is None and keep_class is not None:
            print(f"  {scan.label}: the file has no class field, all points are used")
    if track:
        m = geometry.corridor_mask(P[:, :2], track, corridor)
        sel.update(track=np.asarray(track, float).reshape(-1, 2).tolist(), corridor_m=corridor,
                   n_after_corridor=int(m.sum()))
        P = P[m]
    if len(P) < 1000:
        raise SystemExit(f"{scan.label}: only {len(P)} points left after the selection; check --keep-class / --track")
    return P, sel


def check_origin(label, P, origin, from_pose, tol):
    est, spread = geometry.estimate_origin(P, origin)
    d = float(np.linalg.norm(est[:2] - origin[:2]))
    print(f"  origin {np.round(origin, 3).tolist()}, density-based estimate {np.round(est, 2).tolist()} "
          f"(plan difference {d:.2f} m, height is biased low on steel)")
    if d > tol:
        msg = (f"{label}: the origin is {d:.1f} m (plan) from the position implied by the point density. "
               + ("The E57 pose is probably not the scanner position (e.g. an E57 exported from LAS/LAZ); "
                  "give the scanner position with --origin." if from_pose else "Check --origin.")
               + " --origin-tol 1e9 skips this check.")
        raise SystemExit(msg)
    return dict(origin=origin.tolist(), origin_source="E57 pose" if from_pose else "given",
                origin_density_estimate=est.tolist(), origin_plan_difference_m=d, density_fit_resid_std=spread)


def prepare_scan(label, scanner, setup, source, P, origin, axis: geometry.TrackAxis, band="auto",
                 edge_trim=config.EDGE_TRIM, band_width=config.GROOVE_BAND_WIDTH,
                 edge_margin=config.GROOVE_EDGE_MARGIN, selection=None) -> io.Prepared:
    PL, PR = geometry.split_rails(P)
    sq = geometry.station_rails(PL, PR, axis)
    widths_clip = {side: geometry.clip_width(s, q) for side, (_, s, q) in sq.items()}
    if band == "auto":
        band = "railhead" if max(widths_clip.values()) <= config.MAX_HEAD_WIDTH else "groove"
        print(f"  band auto: clip widths {', '.join(f'{k} {1e3 * v:.0f} mm' for k, v in widths_clip.items())} "
              f"-> {band}")
    bands, widths = {}, {}
    for side, (Ps, s, q) in sq.items():
        if band == "railhead":
            bands[side], widths[side] = geometry.railhead_band(s, q, Ps[:, 2], side, edge_trim)
        else:
            bands[side] = geometry.detect_running_band(s, q, Ps[:, 2], side, width=band_width,
                                                       edge_margin=edge_margin)
            widths[side] = float("nan")
    selection = dict(selection or {}, clip_width_m=widths_clip)
    return io.Prepared(label, scanner, setup, source, origin, axis, bands, sq, widths, band, selection)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clouds", nargs="*", help="point clouds (E57: every scan is one dataset; LAS/LAZ, PLY, ASCII, NPY)")
    ap.add_argument("--list", help="CSV with path,scanner,setup[,origin_x,origin_y,origin_z]")
    ap.add_argument("--scanner", help="scanner name for all clouds (default: file name up to the first '_')")
    ap.add_argument("--setup", help="setup name (default: second '_' token of the file name)")
    ap.add_argument("--origin", nargs=3, type=float, help="scanner origin x y z (default for E57: scan pose)")
    ap.add_argument("--keep-class", nargs="+", default=["1"],
                    help="point classes to keep if the file has a class field; 'all' keeps every point "
                         "(default: 1 = rail head)")
    ap.add_argument("--class-column", type=int, default=None,
                    help="0-based column of the class in ASCII/NPY files (LAS/LAZ and PLY: found by name)")
    ap.add_argument("--track", nargs="+", type=float, default=None,
                    help="x1 y1 x2 y2 [...]: centre line of the track to keep (plan coordinates, file frame)")
    ap.add_argument("--corridor", type=float, default=1.6, help="width of the corridor along --track [m]")
    ap.add_argument("--z-step", type=float, default=None,
                    help="storage step of the heights [m] (default: detected; 0 = no quantization)")
    ap.add_argument("--origin-tol", type=float, default=2.0,
                    help="largest plan difference between origin and density-based estimate [m] (default: %(default)s)")
    ap.add_argument("--band", choices=("auto", "railhead", "groove"), default="auto",
                    help="clips cropped to the railhead, clips incl. the groove, or decided by the clip width")
    ap.add_argument("--edge-trim", type=float, default=config.EDGE_TRIM,
                    help="band railhead: margin removed at both head edges [m] (default: %(default)s)")
    ap.add_argument("--band-width", type=float, default=config.GROOVE_BAND_WIDTH,
                    help="band groove: width of the running band [m] (default: %(default)s)")
    ap.add_argument("--edge-margin", type=float, default=config.GROOVE_EDGE_MARGIN,
                    help="band groove: gap between groove edge and band [m] (default: %(default)s)")
    ap.add_argument("--refit-axis", action="store_true", help="fit a new track axis from the first scan")
    ap.add_argument("--results", default=str(config.RESULTS), help="results folder (default: %(default)s)")
    a = ap.parse_args()
    if bool(a.clouds) == bool(a.list):
        ap.error("give either clouds or --list")
    if a.track is not None and (len(a.track) < 4 or len(a.track) % 2):
        ap.error("--track needs at least two points: x1 y1 x2 y2 [...]")
    keep = None if [k.lower() for k in a.keep_class] == ["all"] else [int(k) for k in a.keep_class]
    origin = np.array(a.origin) if a.origin else None
    sources = (io.sources_from_list(a.list) if a.list
               else [io.Source(Path(c), a.scanner, a.setup, origin) for c in a.clouds])

    out = config.step_dir(a.results, "prepare")
    out.mkdir(parents=True, exist_ok=True)
    axis_path = out / "axis.json"
    axis = None if a.refit_axis or not axis_path.exists() else geometry.TrackAxis.load(axis_path)
    if axis is not None:
        print(f"reusing the track axis {axis_path} (--refit-axis to fit a new one)")
    labels, inputs = [], []
    for src in sources:
        inputs.append(dict(path=str(src.path), sha256=io.sha256(src.path)))
        for scan in io.read_scans(src, a.class_column):
            if scan.label in labels:
                raise SystemExit(f"label {scan.label} occurs twice; give distinct scanner/setup names")
            labels.append(scan.label)
            P, sel = select(scan, keep, a.track, a.corridor)
            if a.z_step is not None:
                sel.update(z_step_m=a.z_step, z_step_source="--z-step")
            else:
                step = geometry.grid_step(scan.raw if scan.raw is not None else scan.pts)
                sel.update(z_step_m=step, z_step_source="detected in the stored coordinates")
            print(f"  height resolution: " + (f"stored on a {1e3 * sel['z_step_m']:.2f} mm grid -> sigma floor "
                                              f"{1e3 * sel['z_step_m'] / 12 ** 0.5:.2f} mm in the plane fits"
                                              if sel["z_step_m"] else "no storage grid found"))
            sel.update(check_origin(scan.label, P, scan.origin, scan.origin_from_pose, a.origin_tol))
            if axis is None:
                axis = geometry.TrackAxis.from_rails(*geometry.split_rails(P))
                axis.save(axis_path)
                print(f"fitted the track axis from {scan.label} -> {axis_path}")
            p = prepare_scan(scan.label, scan.scanner, scan.setup, scan.source, P, scan.origin, axis, a.band,
                             a.edge_trim, a.band_width, a.edge_margin, sel)
            io.save_prepared(p, out)
            plots.plot_railhead_check(p, out / f"{p.label}_railhead.png")
            for side in ("L", "R"):
                Ps, s, _ = p.rails[side]
                rng = np.linalg.norm(Ps - p.origin, axis=1)
                b = p.bands[side]
                head = f"head {1e3 * p.head_width[side]:.0f} mm, " if np.isfinite(p.head_width[side]) else ""
                print(f"  rail {side}: {len(Ps)} points, s {s.min():.2f}..{s.max():.2f} m, "
                      f"range {rng.min():.1f}..{rng.max():.1f} m, {head}band {1e3 * (b.q_hi - b.q_lo):.0f} mm "
                      f"({p.band_method})")
                if p.band_method == "railhead" and p.head_width[side] > config.MAX_HEAD_WIDTH:
                    print(f"  WARNING rail {side}: {1e3 * p.head_width[side]:.0f} mm is wider than a railhead "
                          f"(> {1e3 * config.MAX_HEAD_WIDTH:.0f} mm); use --band groove or crop the clip")
    stats.write_json(out / "prepare_log.json", dict(scans=labels, inputs=inputs, provenance=io.provenance(__file__, a)))
    print(f"{len(labels)} scan(s) -> {out}; check the *_railhead.png plots")


if __name__ == "__main__":
    main()
