"""All quality criteria for one dataset, from one loading of the cloud.

    python -m trackqc.quality --cloud ../pointclouds/track-test.e57 [--out results]

Same options as the single criteria (python -m trackqc.quality.density -h).
"""
from __future__ import annotations

import argparse

from . import common, density, noise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_dataset_args(ap)
    ap.add_argument("--cell", type=float, default=0.005, help="grid cell for completeness [m]")
    ap.add_argument("--min-points", type=int, default=30, help="minimum inliers per segment (noise)")
    a = ap.parse_args()
    ds = common.load_dataset(a)
    density.evaluate(ds, a.cell)
    noise.evaluate(ds, a.min_points)


if __name__ == "__main__":
    main()
