"""Check of the sigma_Z estimators on simulated cells with known correlation (not part of the pipeline).

Usage: python processing/test_precision.py
A cell of nx x ny scan-grid points (0.5 m along, 4 cm across) gets white noise plus correlated noise (white noise
smoothed with a Gaussian of `corr` grid steps). The true standard deviation of the cell mean comes from many
repetitions; each estimator should come close to it (sqrt(n) only without correlation).
"""
import numpy as np
from scipy import ndimage

from precision import detrend, n_eff_taubenheim, sigma_Z_sub

rng = np.random.default_rng(1)


def simulate(nx=120, ny=12, corr=0.0, coloured=0.0):
    white = rng.normal(size=(nx, ny))
    if corr > 0:
        c = ndimage.gaussian_filter(rng.normal(size=(nx, ny)), corr, mode="wrap")
        white = np.sqrt(1 - coloured) * white + np.sqrt(coloured) * c / c.std()
    return white.ravel()


def check(nx=120, ny=12, corr=0.0, coloured=0.0, runs=300):
    col, row = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    col, row = col.ravel(), row.ravel()
    u = col / nx * 0.5
    v = (row / ny - 0.5) * 0.04
    means, taub, sub = [], [], []
    for _ in range(runs):
        z = simulate(nx, ny, corr, coloured)
        means.append(z.mean())
        e = detrend(u, v, z)
        taub.append(z.std() / np.sqrt(n_eff_taubenheim(e, col, row)))
        sub.append(sigma_Z_sub(u, z)[0])
    n = nx * ny
    print(f"corr {corr:4.1f} steps, coloured {coloured:.0%}: true {np.std(means):.4f}  "
          f"sqrt(n) {1 / np.sqrt(n):.4f}  Taubenheim {np.median(taub):.4f}  sub-cells {np.median(sub):.4f}")


if __name__ == "__main__":
    for corr, coloured in [(0, 0), (1, 0.5), (2, 0.8), (4, 0.8)]:
        check(corr=corr, coloured=coloured)
