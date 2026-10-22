"""Default settings, result folders and the command line options shared by the numbered scripts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

PROCESSING = Path(__file__).resolve().parents[1]
RESULTS = PROCESSING / "results"      # <RESULTS>/01_prepare, 02_point_density, ...

STEPS = dict(prepare="01_prepare", density="02_point_density", noise="03_point_noise",
             heights="04_rail_heights", geometry="05_track_geometry", compare="06_compare_scans",
             limits="07_operational_limits")

# ---- railhead extraction (01)
EDGE_TRIM = 0.005      # band = railhead minus this margin on both sides (rounded head corners) [m]
EDGE_QUANTILE = 0.005  # railhead edges = this / 1 - this quantile of the lateral offsets
MAX_HEAD_WIDTH = 0.100  # [m]; a wider "railhead" means the clip is not cropped to the head
GROOVE_BAND_WIDTH = 0.030   # --band groove: running band width beyond the groove [m]
GROOVE_EDGE_MARGIN = 0.005  # --band groove: margin between groove edge and band [m]

# ---- segments and heights (02-04)
SEG_LEN = 0.5          # segment length along the rail = station spacing of the rail heights [m] (WP4/WP5)
MIN_POINTS = 30        # minimum plane inliers per segment for noise and height
RANGE_BIN = 1.0        # bin width of the tables and plots [m]
INCIDENCE_BIN = 5.0    # [deg]
HEIGHT_WINDOW = 4.0    # neighbourhood of the leave-one-out polynomial along the rail [m] (04)
HEIGHT_DEGREE = 2      # degree of that polynomial (2 = constant vertical curvature)

# ---- track geometry (05) and quality criteria (report, section quality criteria)
B0 = 1.0               # reference distance of the cant [m]
TWIST_BASE = 5.0       # l [m]
VERSINE_BASE = 10.0    # chord length of the vertical versine [m] (neighbours at +-5 m)
# attention values of the VBZ per speed class: (T_vPf [mm], T_N [per mille])
ATTENTION = {"<=36": (5.0, 1.5), "36-48": (5.0, 1.0), "48-60": (3.0, 1.0)}
SPEED_CLASS = "48-60"
# criterion K sigma <= T: K = 2 (interpretations 1 and 3), K = k_0.975 + k_0.95 = 3.605 (interpretation 2)
K_FACTORS = {"K2": 2.0, "K3.6": 3.605}


def sigma_z_requirement(speed_class=SPEED_CLASS, K=2.0, b=1.0, l=TWIST_BASE, b0=B0):
    """Largest permissible std of one rail height [mm] (report, eqs. sigma_versine and sigma_twist).

    versine change of two epochs: sigma_z <= T_vPf / (K sqrt(3/2))
    twist of one epoch:           sigma_z <= T_N b l / (2 K b0)
    """
    T_vpf, T_n = ATTENTION[speed_class]
    return min(T_vpf / (K * 1.5 ** 0.5), T_n * b * l / (2 * K * b0))


def requirements(speed_class=SPEED_CLASS):
    """{key: (sigma_z,req [mm] rounded down to 0.01 mm as in the report, plot label)} for every K."""
    out = {}
    for key, K in K_FACTORS.items():
        v = math.floor(sigma_z_requirement(speed_class, K) * 100) / 100
        out[key] = (v, f"σ_z,req = {v:.2f} mm (K = {K:g}, {speed_class} km/h)")
    return out


def step_dir(results, step) -> Path:
    return Path(results) / STEPS[step]


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--results", default=str(RESULTS),
                    help="results folder; every step writes to <results>/<step>/ (default: %(default)s)")
    ap.add_argument("--scanner", nargs="+", default=None, help="only these scanners (default: all prepared scans)")


def add_segment_args(ap: argparse.ArgumentParser, min_points=True):
    """Options of the steps that cut the prepared railheads into segments (02, 03, 04)."""
    add_common_args(ap)
    ap.add_argument("--seg-len", type=float, default=SEG_LEN, help="segment length along the rail [m]")
    ap.add_argument("--range-bin", type=float, default=RANGE_BIN, help="bin width for range [m]")
    ap.add_argument("--incidence-bin", type=float, default=INCIDENCE_BIN, help="bin width for incidence [deg]")
    if min_points:
        ap.add_argument("--min-points", type=int, default=MIN_POINTS, help="minimum plane inliers per segment")
