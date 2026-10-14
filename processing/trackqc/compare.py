"""Empirical check of the formal precision with two independent datasets.

Two scans of the same track section (other setup, other scanner, or a repeated
scan) are treated as two epochs with zero true deformation. Then
  dN_i   = N_i(A)   - N_i(B)        (offsets and tilts cancel)
  dvPf_i = vPf_i(A) - vPf_i(B)      (= the monitored versine under H0: no change)
should scatter with the formal std sqrt(sigma_A^2 + sigma_B^2). If the empirical
std is larger, the excess is the part sigma_sys that does not average out
(correlated noise, incidence-dependent range errors, surface wear ...).

    python -m trackqc.compare results/FARO_S1_dry_geometry.csv results/RTC_S1_dry_geometry.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import core


# Neighbouring stations share no points (disjoint height windows), but stations
# l apart share rail heights: for equal sigma_z the correlation of the differences
# is -1/2 at lag l for the twist, and -2/3 (lag l) and +1/6 (lag 2l) for the
# versine. Bartlett's approximation for the variance of the sample variance then
# gives the effective number of observations n_eff = n / (1 + 2 sum rho_h^2).
N_EFF_FACTOR = {"twist_permille": 1 + 2 * 0.5 ** 2, "vpf_mm": 1 + 2 * ((2 / 3) ** 2 + (1 / 6) ** 2)}


def std_ci(std, n_eff, alpha=0.05):
    """Chi-square confidence interval of a standard deviation with n_eff - 1 dof."""
    return core.sigma_ci(std, max(n_eff - 1, 1), alpha)


def compare(path_a, path_b, l=5.0, speed_class="48-60", b0=1.0):
    A, B = pd.read_csv(path_a), pd.read_csv(path_b)
    m = A.merge(B, on="s", suffixes=("_a", "_b"))
    T_vpf, T_n = core.ATTENTION[speed_class]
    res = {}
    for name, val, sig, T in (("twist_permille", "twist_permille", "sigma_twist_permille", T_n),
                              ("vpf_mm", "vpf_mm", "sigma_vpf_mm", T_vpf)):
        d = (m[f"{val}_a"] - m[f"{val}_b"]).to_numpy()
        formal = np.sqrt(m[f"{sig}_a"] ** 2 + m[f"{sig}_b"] ** 2).to_numpy()
        ok = np.isfinite(d) & np.isfinite(formal)
        d, formal = d[ok], formal[ok]
        if len(d) < 5:
            res[name] = dict(n=int(len(d)), note="too few common stations")
            continue
        emp = d.std(ddof=1)
        rob = core.MAD2SIGMA * np.median(np.abs(d - np.median(d)))
        n_eff = len(d) / N_EFF_FACTOR[name]
        lo, hi = std_ci(emp, n_eff)
        res[name] = dict(
            n=int(len(d)), n_eff=float(n_eff), mean=float(d.mean()), std_empirical=float(emp), std_robust=float(rob),
            std_ci95=[float(lo), float(hi)], std_formal_rms=float(np.sqrt(np.mean(formal ** 2))),
            ratio_empirical_formal=float(emp / np.sqrt(np.mean(formal ** 2))),
            # share of differences above the attention value = empirical false-alarm rate
            share_above_T=float(np.mean(np.abs(d) > T)), T=T,
        )
    # rail height std implied by the empirical scatter (equal sigma_z, uncorrelated):
    # Var(dvPf) = 2 * 3/4 sigma_z^2 ;  Var(dN) = 2 * 4 b0^2 / (b^2 l^2) sigma_z^2
    b = float(np.nanmedian(m["b_m_a"]))
    if "std_empirical" in res.get("vpf_mm", {}):
        res["sigma_z_from_vpf_mm"] = res["vpf_mm"]["std_empirical"] / np.sqrt(1.5)
    if "std_empirical" in res.get("twist_permille", {}):
        res["sigma_z_from_twist_mm"] = res["twist_permille"]["std_empirical"] * b * l / (2 * np.sqrt(2) * b0)
    res["sigma_z_formal_rms_mm"] = float(np.sqrt(np.nanmean(np.r_[
        m["sigma_zL_mm_a"], m["sigma_zR_mm_a"], m["sigma_zL_mm_b"], m["sigma_zR_mm_b"]] ** 2)))
    for key in ("sigma_z_from_vpf_mm", "sigma_z_from_twist_mm"):
        if key in res:
            excess = res[key] ** 2 - res["sigma_z_formal_rms_mm"] ** 2
            res[key.replace("from", "sys_from")] = float(np.sqrt(excess)) if excess > 0 else 0.0
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("geometry_a")
    ap.add_argument("geometry_b")
    ap.add_argument("--speed-class", default="48-60", choices=list(core.ATTENTION))
    a = ap.parse_args()
    print(json.dumps(compare(a.geometry_a, a.geometry_b, speed_class=a.speed_class), indent=2))


if __name__ == "__main__":
    main()
