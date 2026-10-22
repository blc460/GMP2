"""06 - Empirical precision from two scans of the same track section (other setup, other scanner or repeated scan).

    python 06_compare_scans.py [--scanner FARO RTC360] [--pairs RTC360_S1:RTC360_S2 ...]

Two scans of the same section are two epochs with zero true deformation. All
scans share the stations of the track axis of step 01, so the parameters of
step 05 can be differenced station by station for every pair of scans
(default: all pairs with at least --min-common common stations):

    dz     = z(A) - z(B) per rail     (the mean is a height offset: levelling, registration; removed)
    dü     = ü(A) - ü(B)              (common tilt about the track axis remains)
    dN     = N(A) - N(B)              (offsets and tilts cancel)
    dvPf   = vPf(A) - vPf(B)          (= the monitored versine change under H0: no change)

Per quantity: robust std 1.4826 MAD (primary) and std with 95 % CI, the
formal std from the station-wise and the empirical sigma_z of step 04, and
the sigma_z implied by the observed scatter (equal, uncorrelated sigma_z):

    Var(dz) = 2 sigma_z^2              Var(dü) = 4 (b0/b)^2 sigma_z^2
    Var(dN) = 8 b0^2 / (b^2 l^2) sigma_z^2      Var(dvPf) = 3/2 sigma_z^2

sigma_sys = sqrt(sigma_z,implied^2 - RMS(sigma_z,station)^2) is the part that
does not average out within one scan (incidence- and range-dependent errors,
surface effects). Unlike step 04 it also contains errors that vary smoothly
along the track, and it does not contain the rail's own unevenness.
Neighbouring twists and versines share heights (stations l apart), so the
degrees of freedom are reduced with Bartlett's approximation
(n_eff = n / 1.5 for the twist, n / 1.94 for the versine).

Writes to <results>/06_compare_scans/:
  compare_pairs.csv          one row per pair and quantity
  compare_fit.json           the same as nested values, settings, provenance
  <A>__<B>_differences.png   differences along the track
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from shared import config, fit, io, plots, stats

# effective number of observations n / factor (correlation of neighbours at lag l and 2 l, see docstring)
N_EFF_FACTOR = {"dz": 1.0, "dcant": 1.0, "dtwist": 1 + 2 * 0.5 ** 2, "dvpf": 1 + 2 * ((2 / 3) ** 2 + (1 / 6) ** 2)}


def implied_sigma_z(name, std, b, l=config.TWIST_BASE, b0=config.B0):
    return {"dz": std / np.sqrt(2), "dcant": std * b / (2 * b0), "dtwist": std * b * l / (2 * np.sqrt(2) * b0),
            "dvpf": std / np.sqrt(1.5)}[name]


def compare_pair(A, B, min_common):
    m = A.merge(B, on="s", suffixes=("_a", "_b"))
    b = float(np.nanmedian(0.5 * (m["b_m_a"] + m["b_m_b"])))
    station_rms = float(np.sqrt(np.nanmean(np.r_[m["sigma_zL_station_mm_a"], m["sigma_zR_station_mm_a"],
                                                 m["sigma_zL_station_mm_b"], m["sigma_zR_station_mm_b"]] ** 2)))
    quantities = [("dz", f"z{r}_mm", f"sigma_z{r}_{{}}_mm", r) for r in ("L", "R")]
    quantities += [("dcant", "cant_mm", "sigma_cant_{}_mm", ""), ("dtwist", "twist_permille",
                   "sigma_twist_{}_permille", ""), ("dvpf", "vpf_mm", "sigma_vpf_{}_mm", "")]
    rows, diffs = [], {}
    for name, val, sig, rail in quantities:
        d = (m[f"{val}_a"] - m[f"{val}_b"]).to_numpy()
        ok = np.isfinite(d)
        key = f"{name}{rail}"
        row = dict(quantity=key, n=int(ok.sum()))
        if ok.sum() < min_common:
            rows.append(row | dict(note=f"fewer than {min_common} common values"))
            continue
        d, s = d[ok], m["s"].to_numpy()[ok]
        offset = float(d.mean()) if name == "dz" else 0.0
        d = d - offset
        diffs[key] = (s, d)
        n_eff = len(d) / N_EFF_FACTOR[name]
        rob = fit.mad_sigma(d)
        # heights: std about the removed mean; others: RMS, since the true difference is 0
        std = float(d.std(ddof=1)) if name == "dz" else float(np.sqrt(np.mean(d ** 2)))
        lo, hi = fit.mad_sigma_ci(rob, n_eff)
        formal = {src: float(np.sqrt(np.mean(m[sig.format(src) + "_a"].to_numpy()[ok] ** 2
                                             + m[sig.format(src) + "_b"].to_numpy()[ok] ** 2)))
                  for src in ("station", "emp")}
        sz = implied_sigma_z(name, rob, b)
        rows.append(row | dict(n_eff=n_eff, mean_removed=offset, mean=float(d.mean()),
                               std_mad=rob, std_mad_ci_lo=float(lo), std_mad_ci_hi=float(hi), std_rms=std,
                               std_formal_station=formal["station"], std_formal_emp=formal["emp"],
                               ratio_mad_formal_station=rob / formal["station"],
                               sigma_z_implied_mm=sz, sigma_z_station_rms_mm=station_rms,
                               sigma_sys_mm=float(np.sqrt(max(sz ** 2 - station_rms ** 2, 0.0)))))
    return rows, diffs, b


def plot_differences(diffs, path, title):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(diffs), 1, figsize=(10, 1.9 * len(diffs) + 0.8), sharex=True, squeeze=False)
    for ax, (k, (s, d)) in zip(axs[:, 0], diffs.items()):
        ax.plot(s, d, ".-", ms=4, lw=1, color=plots.COLORS[0])
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_ylabel(k + (" [‰]" if "twist" in k else " [mm]"))
        plots._style(ax)
    axs[-1, 0].set_xlabel("station along the track s [m]")
    axs[0, 0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(ap)
    ap.add_argument("--pairs", nargs="+", default=None, help="pairs LABEL_A:LABEL_B (default: all pairs)")
    ap.add_argument("--min-common", type=int, default=5, help="minimum number of common stations per quantity")
    a = ap.parse_args()
    folder = config.step_dir(a.results, "geometry")
    files = sorted(folder.glob("*_geometry.csv"))
    if not files:
        raise SystemExit(f"no *_geometry.csv in {folder}; run 05_track_geometry.py first")
    geo = {f.name[:-len("_geometry.csv")]: pd.read_csv(f) for f in files}
    if a.scanner:
        geo = {k: g for k, g in geo.items() if g["scanner"].iloc[0] in a.scanner}
    pairs = [tuple(p.split(":")) for p in a.pairs] if a.pairs else list(combinations(sorted(geo), 2))
    out = config.step_dir(a.results, "compare")
    out.mkdir(parents=True, exist_ok=True)
    if not pairs:
        msg = (f"only {len(geo)} scan(s) ({', '.join(geo)}); step 06 needs two scans of the same section "
               "(another setup, scanner or a repeated scan). Nothing compared.")
        stats.write_json(out / "compare_fit.json", dict(note=msg, provenance=io.provenance(__file__, a)))
        print(msg)
        return
    rows, res = [], {}
    for A, B in pairs:
        r, diffs, b = compare_pair(geo[A], geo[B], a.min_common)
        for x in r:
            x.update(scan_a=A, scan_b=B, b_m=b)
        rows += r
        res[f"{A}:{B}"] = r
        if diffs:
            plot_differences(diffs, out / f"{A}__{B}_differences.png", f"{A} minus {B}")
    t = pd.DataFrame(rows)
    t = t[["scan_a", "scan_b"] + [c for c in t.columns if c not in ("scan_a", "scan_b")]]
    t.to_csv(out / "compare_pairs.csv", index=False)
    stats.write_json(out / "compare_fit.json", dict(settings=dict(min_common=a.min_common, n_eff_factor=N_EFF_FACTOR),
                                                     provenance=io.provenance(__file__, a), pairs=res))
    print(f"compare: {len(pairs)} pair(s) -> {out}")


if __name__ == "__main__":
    main()
