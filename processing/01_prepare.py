"""01 - Prepare the rail clips for all following steps.

Input: point clouds cropped to the two rail heads (two nearly horizontal,
parallel strips), in one levelled frame (z up). Every scan becomes one
dataset SCANNER_SETUP:

    # scanner and setup from the file name SCANNER_SETUP[_anything].e57
    python 01_prepare.py ../pointclouds/FARO_S1_dry.e57 ../pointclouds/RTC360_S1_dry.e57
    # scanner and setup given explicitly (all scans of the files)
    python 01_prepare.py ../pointclouds/track-test.e57 --scanner RTC360 --setup E004
    # list file: path,scanner,setup[,origin_x,origin_y,origin_z]  (origin needed for PLY/ASCII/NPY)
    python 01_prepare.py --list datasets.csv

Steps per scan:
1. Split into left and right rail at the empty gap between them.
2. Track axis = mean of both rail centre lines (cubic). The axis is fitted
   once, from the first scan, saved as <results>/01_prepare/axis.json and
   reused for every later scan, so all scans share the same stations s
   (needed to compare scans in step 06). --refit-axis fits it anew.
3. Stations s along the axis and lateral offsets q for every point.
4. Band used for all fits:
   --band railhead (default, clips cropped to the head): head edges =
     0.5 % / 99.5 % quantiles of the lateral offset from the rail centre line,
     band = head minus --edge-trim on both sides (rounded head corners);
   --band groove (clips incl. groove): running band of --band-width beyond
     the groove edge plus --edge-margin (see shared/geometry.py).

Writes to <results>/01_prepare/:
  axis.json              track axis shared by all scans
  <label>.npz            points of both rails with s and q
  <label>.json           scanner, setup, origin, axis, band, head width
  <label>_railhead.png   lateral profile with head edges and band: check it for every scan
  prepare_log.json       inputs (with SHA-256), options, code and software versions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared import config, geometry, io, plots, stats


def prepare_scan(scan: io.Scan, axis: geometry.TrackAxis, band="railhead", edge_trim=config.EDGE_TRIM,
                 band_width=config.GROOVE_BAND_WIDTH, edge_margin=config.GROOVE_EDGE_MARGIN) -> io.Prepared:
    PL, PR = geometry.split_rails(scan.pts)
    sq = geometry.station_rails(PL, PR, axis)
    bands, widths = {}, {}
    for side, (P, s, q) in sq.items():
        if band == "railhead":
            bands[side], widths[side] = geometry.railhead_band(s, q, P[:, 2], side, edge_trim)
        else:
            bands[side] = geometry.detect_running_band(s, q, P[:, 2], side, width=band_width,
                                                       edge_margin=edge_margin)
            widths[side] = float("nan")
    return io.Prepared(scan.label, scan.scanner, scan.setup, scan.source, scan.origin, axis, bands, sq, widths,
                       band)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clouds", nargs="*", help="rail clips (E57: every scan is one dataset)")
    ap.add_argument("--list", help="CSV with path,scanner,setup[,origin_x,origin_y,origin_z]")
    ap.add_argument("--scanner", help="scanner name for all clouds (default: file name up to the first '_')")
    ap.add_argument("--setup", help="setup name (default: second '_' token of the file name)")
    ap.add_argument("--origin", nargs=3, type=float, help="scanner origin x y z (default for E57: scan pose)")
    ap.add_argument("--band", choices=("railhead", "groove"), default="railhead",
                    help="clips cropped to the railhead, or clips incl. the groove (default: %(default)s)")
    ap.add_argument("--edge-trim", type=float, default=config.EDGE_TRIM,
                    help="--band railhead: margin removed at both head edges [m] (default: %(default)s)")
    ap.add_argument("--band-width", type=float, default=config.GROOVE_BAND_WIDTH,
                    help="--band groove: width of the running band [m] (default: %(default)s)")
    ap.add_argument("--edge-margin", type=float, default=config.GROOVE_EDGE_MARGIN,
                    help="--band groove: gap between groove edge and band [m] (default: %(default)s)")
    ap.add_argument("--refit-axis", action="store_true", help="fit a new track axis from the first scan")
    ap.add_argument("--results", default=str(config.RESULTS), help="results folder (default: %(default)s)")
    a = ap.parse_args()
    if bool(a.clouds) == bool(a.list):
        ap.error("give either clouds or --list")
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
        for scan in io.read_scans(src):
            if scan.label in labels:
                raise SystemExit(f"label {scan.label} occurs twice; give distinct scanner/setup names")
            labels.append(scan.label)
            if axis is None:
                axis = geometry.TrackAxis.from_rails(*geometry.split_rails(scan.pts))
                axis.save(axis_path)
                print(f"fitted the track axis from {scan.label} -> {axis_path}")
            p = prepare_scan(scan, axis, a.band, a.edge_trim, a.band_width, a.edge_margin)
            io.save_prepared(p, out)
            plots.plot_railhead_check(p, out / f"{p.label}_railhead.png")
            for side in ("L", "R"):
                P, s, _ = p.rails[side]
                rng = np.linalg.norm(P - p.origin, axis=1)
                b = p.bands[side]
                print(f"  rail {side}: {len(P)} points, s {s.min():.2f}..{s.max():.2f} m, "
                      f"range {rng.min():.1f}..{rng.max():.1f} m, head {1e3 * p.head_width[side]:.0f} mm, "
                      f"band {1e3 * (b.q_hi - b.q_lo):.0f} mm")
                if p.head_width[side] > config.MAX_HEAD_WIDTH:
                    print(f"  WARNING rail {side}: {1e3 * p.head_width[side]:.0f} mm is wider than a railhead "
                          f"(> {1e3 * config.MAX_HEAD_WIDTH:.0f} mm); is the clip cropped to the railhead?")
    stats.write_json(out / "prepare_log.json", dict(scans=labels, inputs=inputs, provenance=io.provenance(__file__, a)))
    print(f"{len(labels)} scan(s) -> {out}; check the *_railhead.png plots")


if __name__ == "__main__":
    main()
