"""Default settings, result folders and the command line options shared by the analysis scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

PROCESSING = Path(__file__).resolve().parents[2]
RESULTS = PROCESSING / "results" / "railhead"     # <RESULTS>/00_prepare, 01_point_density, ...
PREPARED = "00_prepare"

SEG_LEN = 0.5          # segment length along the rail [m] (WP4)
EDGE_TRIM = 0.005      # band = railhead minus this margin on both sides (rounded head corners) [m]
EDGE_QUANTILE = 0.005  # railhead edges = this / 1 - this quantile of the lateral offsets
RANGE_BIN = 1.0        # bin width of the tables and plots [m]
INCIDENCE_BIN = 5.0    # [deg]


def prepared_dir(results) -> Path:
    return Path(results) / PREPARED


def add_analysis_args(ap: argparse.ArgumentParser):
    """Options of every analysis script (01, 02, ...) that reads the prepared scans."""
    ap.add_argument("--results", default=str(RESULTS),
                    help=f"results folder; prepared scans are read from <results>/{PREPARED}/ (default: %(default)s)")
    ap.add_argument("--scanner", nargs="+", default=None, help="only these scanners (default: all prepared scans)")
    ap.add_argument("--seg-len", type=float, default=SEG_LEN, help="segment length along the rail [m]")
    ap.add_argument("--range-bin", type=float, default=RANGE_BIN, help="bin width for range [m]")
    ap.add_argument("--incidence-bin", type=float, default=INCIDENCE_BIN, help="bin width for incidence [deg]")
