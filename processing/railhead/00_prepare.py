"""00 - Prepare the railhead clips for all following scripts.

Input: point clouds cropped to the two rail heads (two nearly horizontal,
parallel strips). Every scan becomes one dataset SCANNER_SETUP:

    # scanner and setup from the file name SCANNER_SETUP[_anything].e57
    python railhead/00_prepare.py ../pointclouds/FARO_S1_dry.e57 ../pointclouds/RTC360_S1_dry.e57
    # scanner given explicitly (all scans of the files)
    python railhead/00_prepare.py ../pointclouds/track-test.e57 --scanner RTC360
    # list file: path,scanner,setup[,origin_x,origin_y,origin_z]  (origin needed for PLY/ASCII)
    python railhead/00_prepare.py --list datasets.csv

Steps per scan: split into left and right rail at the gap between them, track
axis from both rails, stations s along the axis, railhead band = measured
head width minus --edge-trim per side (see shared/data.py: railhead_band).

Writes to <results>/00_prepare/:
  <label>.npz            points of both rails with s and q
  <label>.json           scanner, setup, origin, axis, band, head width
  <label>_railhead.png   lateral profile with head edges and band: check it for every scan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared import config, data, plots

MAX_HEAD_WIDTH = 0.100   # [m]; wider clips contain more than the railhead (groove, paving)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clouds", nargs="*", help="railhead clips (E57: every scan is one dataset)")
    ap.add_argument("--list", help="CSV with path,scanner,setup[,origin_x,origin_y,origin_z]")
    ap.add_argument("--scanner", help="scanner name for all clouds (default: file name up to the first '_')")
    ap.add_argument("--setup", help="setup name (default: second '_' token of the file name)")
    ap.add_argument("--origin", nargs=3, type=float, help="scanner origin x y z (default for E57: scan pose)")
    ap.add_argument("--edge-trim", type=float, default=config.EDGE_TRIM,
                    help="margin removed at both railhead edges [m] (default: %(default)s)")
    ap.add_argument("--results", default=str(config.RESULTS), help="results folder (default: %(default)s)")
    a = ap.parse_args()
    if bool(a.clouds) == bool(a.list):
        ap.error("give either clouds or --list")
    origin = np.array(a.origin) if a.origin else None
    sources = (data.sources_from_list(a.list) if a.list
               else [data.Source(Path(c), a.scanner, a.setup, origin) for c in a.clouds])

    out = config.prepared_dir(a.results)
    labels = []
    for src in sources:
        for scan in data.read_scans(src):
            if scan.label in labels:
                raise SystemExit(f"label {scan.label} occurs twice; give distinct scanner/setup names")
            labels.append(scan.label)
            p = data.prepare(scan, a.edge_trim)
            data.save(p, out)
            plots.plot_railhead_check(p, out / f"{p.label}_railhead.png")
            for side in ("L", "R"):
                P, s, _ = p.rails[side]
                rng = np.linalg.norm(P - p.origin, axis=1)
                b = p.bands[side]
                print(f"  rail {side}: {len(P)} points, s {s.min():.2f}..{s.max():.2f} m, "
                      f"range {rng.min():.1f}..{rng.max():.1f} m, head {1e3 * p.head_width[side]:.0f} mm, "
                      f"band {1e3 * (b.q_hi - b.q_lo):.0f} mm")
                if p.head_width[side] > MAX_HEAD_WIDTH:
                    print(f"  WARNING rail {side}: {1e3 * p.head_width[side]:.0f} mm is wider than a railhead "
                          f"(> {1e3 * MAX_HEAD_WIDTH:.0f} mm); is the clip cropped to the railhead?")
    print(f"{len(labels)} scan(s) -> {out}; check the *_railhead.png plots")


if __name__ == "__main__":
    main()
