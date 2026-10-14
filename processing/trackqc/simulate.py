"""Synthetic scans of a grooved track, to test and validate the pipeline.

The simulated values (noise model, sampling, drop-outs) are illustrative and
NOT properties of the FARO or RTC360. They only serve to show that the
estimators are unbiased and that the formal precision matches the Monte-Carlo
scatter when the assumptions hold (independent noise). That check belongs in
the methods chapter as a validation of the processing.

    python -m trackqc.simulate --out demo/            # one setup + pipeline run
    python -m trackqc.simulate --out demo/ --mc 30     # Monte-Carlo validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# grooved-rail cross-section relative to the groove centre, outward positive [m]
GUARD = (-0.032, -0.018)
GROOVE = (-0.018, 0.018)
HEAD = (0.018, 0.073)
CLIP = (-0.050, 0.100)
Y_GROOVE = 0.505          # |y| of the groove centre -> running-band centres ~1.07 m apart


def true_heights(x):
    """Known rail heights [m]: twist ramp + a 4 mm settlement trough of both rails."""
    cant = 0.002 + 0.0004 * x              # cant ramp -> constant twist of -0.4 per mille
    trough = -0.004 * np.exp(-0.5 * ((x - 15.0) / 2.5) ** 2)
    base = 0.01 * x / 30.0                  # longitudinal gradient (cancels in vPf)
    zL = base + trough - cant / 2
    zR = base + trough + cant / 2
    return zL, zR


def write_ply(path, xyz):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(xyz, dtype="<f8").tobytes())


def simulate_rail(side, origin, rng, x_range=(0.0, 30.0), dphi=0.6e-3, sigma0=0.0008,
                  k_r=0.00003, cell=0.002, p_outlier=0.01):
    """Points on one rail clip: Poisson sampling with density cos(theta)/(r dphi)^2,
    return probability depending on incidence (steel), noise along the beam."""
    out = 1.0 if side == "L" else -1.0
    xs = np.arange(*x_range, cell) + cell / 2
    yr = np.arange(*CLIP, cell) + cell / 2
    X, YR = np.meshgrid(xs, yr, indexing="ij")
    X, YR = X.ravel(), YR.ravel()
    zL, zR = true_heights(X)
    zr = zL if side == "L" else zR
    in_groove = (YR > GROOVE[0]) & (YR < GROOVE[1])
    on_steel = ((YR >= HEAD[0]) & (YR <= HEAD[1])) | ((YR >= GUARD[0]) & (YR <= GUARD[1]))
    Z = zr + np.where(in_groove, -0.045, np.where(on_steel, 0.0, -0.003))
    Y = out * (Y_GROOVE + YR)
    P = np.column_stack([X, Y, Z])
    beam = P - origin
    r = np.linalg.norm(beam, axis=1)
    cos_t = np.abs(beam[:, 2]) / r                      # surface normal ~ vertical
    lam = cos_t / (r * dphi) ** 2 * cell ** 2
    lam *= np.where(on_steel, np.sqrt(cos_t) * 3, 1.0)  # weak, incidence-dependent steel returns
    lam *= np.where(in_groove, 0.15, 1.0)               # groove bottom mostly occluded
    n = rng.poisson(np.minimum(lam, 50))
    idx = np.repeat(np.arange(len(P)), n)
    pts = P[idx] + np.column_stack([rng.uniform(-cell / 2, cell / 2, (len(idx), 2)), np.zeros(len(idx))])
    pts[:, 2] = Z[idx]
    u = (pts - origin) / np.linalg.norm(pts - origin, axis=1, keepdims=True)
    sig = sigma0 + k_r * r[idx]
    sig = np.where(on_steel[idx], sig, sig * 2.5)       # rough asphalt
    eps = rng.normal(0, 1, len(idx)) * sig / np.maximum(cos_t[idx], 0.05) ** 0.5
    pts += u * eps[:, None]
    bad = rng.random(len(idx)) < p_outlier
    pts[bad, 2] += rng.normal(0, 0.01, bad.sum())
    return pts


def make_dataset(folder, label, origin, seed):
    rng = np.random.default_rng(seed)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for side in ("L", "R"):
        p = folder / f"{label}_{side}.ply"
        write_ply(p, simulate_rail(side, np.asarray(origin, float), rng))
        paths[side] = p
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--mc", type=int, default=0, help="number of Monte-Carlo repetitions")
    a = ap.parse_args()
    from .process_setup import run
    from .compare import compare
    from . import core

    out = Path(a.out)
    setups = {"S1": (12.0, -4.0, 1.5), "S2": (20.0, 6.0, 1.5)}
    axis = out / "axis.json"
    if axis.exists():
        axis.unlink()
    for i, (name, o) in enumerate(setups.items()):
        p = make_dataset(out / "clips", f"SIM_{name}_dry", o, seed=100 + i)
        _, geo, summ = run(p["L"], p["R"], o, axis, f"SIM_{name}_dry", out / "results")
        zL, zR = true_heights(geo["s"].to_numpy())
        print(f"{name}: median sigma_pc {summ['median_sigma_pc_mm']:.2f} mm, "
              f"median formal sigma_z {summ['median_sigma_z_formal_mm']:.3f} mm, "
              f"mean height error L {np.nanmean(geo['zL_mm'] - zL * 1e3):+.3f} mm")
    print(json.dumps(compare(out / "results/SIM_S1_dry_geometry.csv", out / "results/SIM_S2_dry_geometry.csv"), indent=2))

    if a.mc:
        errs, formal = [], []
        for k in range(a.mc):
            p = make_dataset(out / "mc", "MC_S1_dry", setups["S1"], seed=1000 + k)
            _, geo, _ = run(p["L"], p["R"], setups["S1"], axis, "MC_S1_dry", out / "mc_results", plot=False)
            zL, _ = true_heights(geo["s"].to_numpy())
            errs.append(geo["zL_mm"].to_numpy() - zL * 1e3)
            formal.append(geo["sigma_zL_mm"].to_numpy())
        errs, formal = np.array(errs), np.array(formal)
        emp = np.nanstd(errs, axis=0, ddof=1)
        ratio = emp / np.nanmean(formal, axis=0)
        print(f"Monte Carlo ({a.mc} runs): bias of z_L median {np.nanmedian(np.nanmean(errs, 0)):+.4f} mm, "
              f"empirical/formal sigma_z: median {np.nanmedian(ratio):.2f} "
              f"(5-95 %: {np.nanpercentile(ratio, 5):.2f}-{np.nanpercentile(ratio, 95):.2f})")


if __name__ == "__main__":
    main()
