"""Check of the sigma_Z estimators on simulated cells with known correlation (not part of the pipeline).

Usage: python processing/test_precision.py [--archive]
A cell of nx x ny scan-grid points (0.5 m along, 4 cm across) gets a railhead shape (slope along the rail, crown
across it; geometry, not noise) plus white noise plus correlated noise (white noise smoothed with a Gaussian of
`corr` grid steps along and across). The true standard deviation of the cell height (mean of the noise) comes from
many repetitions; the estimators should come close to it (sqrt(n) only without correlation).
--archive also runs the dropped estimators (Taubenheim n_eff, 0.1 m sub-cells of the raw heights).
"""
import sys

import numpy as np
from scipy import ndimage

from precision import mad_sigma, residuals, sigma_Z_boot, sigma_Z_bins

rng = np.random.default_rng(1)


def simulate(nx, ny, corr, coloured):
    z = rng.normal(size=(nx, ny))
    if corr > 0:
        c = ndimage.gaussian_filter(rng.normal(size=(nx, ny)), corr, mode="wrap")
        z = np.sqrt(1 - coloured) * z + np.sqrt(coloured) * c / c.std()
    return z.ravel()


def check(nx=120, ny=12, corr=0.0, coloured=0.0, runs=300, archive=False):
    col, row = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    col, row = col.ravel(), row.ravel()
    u = col / nx * 0.5
    v = (row / ny - 0.5) * 0.04
    shape = 2.0 * u + 1.5 * (v / 0.02) ** 2                      # slope and crown, in units of the noise sigma
    out = {k: [] for k in ("mean", "raw", "sqrt_n", "bins", "boot", "taub", "sub")}
    if archive:
        sys.path.insert(0, "archive")
        from taubenheim import detrend, n_eff_taubenheim, sigma_Z_sub
    for _ in range(runs):
        noise = simulate(nx, ny, corr, coloured)
        z = shape + noise
        out["mean"].append(noise.mean())
        e = residuals(u, v, z)
        out["raw"].append(mad_sigma(z) / np.sqrt(len(z)))
        out["sqrt_n"].append(mad_sigma(e) / np.sqrt(len(z)))
        out["bins"].append(sigma_Z_bins(u, e)[0])
        out["boot"].append(sigma_Z_boot(u, e, rng=rng))
        if archive:
            out["taub"].append(mad_sigma(z) / np.sqrt(n_eff_taubenheim(detrend(u, v, z), col, row)))
            out["sub"].append(sigma_Z_sub(u, z)[0])
    true = np.std(out["mean"])
    line = (f"corr {corr:3.0f} steps, coloured {coloured:3.0%}: true {true:.4f} | "
            + "  ".join(f"{k} {np.nanmedian(out[k]) / true:4.2f}" for k in ("raw", "sqrt_n", "bins", "boot")))
    if archive:
        line += "  " + "  ".join(f"{k} {np.nanmedian(out[k]) / true:4.2f}" for k in ("taub", "sub"))
    bins = np.array(out["bins"]) / true
    print(line + f"   (bins 10-90 %: {np.nanquantile(bins, 0.1):.2f}-{np.nanquantile(bins, 0.9):.2f})")


if __name__ == "__main__":
    print("estimate / true (median over runs); raw = MAD of the raw heights / sqrt(n)")
    for corr, coloured in [(0, 0), (1, 0.5), (2, 0.8), (4, 0.8), (8, 0.8)]:
        check(corr=corr, coloured=coloured, archive="--archive" in sys.argv)
