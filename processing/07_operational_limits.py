"""07 - Operational limits: up to which distance and angle of incidence does a scanner meet sigma_z,req?

    python 07_operational_limits.py [--scanner FARO RTC360] [--sigma-sys 0.3] [--speed-class 48-60]

Input: station-wise rail-height precision sigma_z,station of step 04 (1.4826
MAD of the plane residuals x sqrt(q00) x sqrt(VF), so it already contains point
noise, point density and correlation). Per scanner two models are fitted by
least squares,

    ln sigma_z,station = a + b ln r                 (r = distance to the scanner)
    ln sigma_z,station = a + b ln cos(theta)        (theta = angle of incidence)

and the predicted precision of a rail height is

    sigma_z(x) = sqrt( exp(a + b f(x))^2 + sigma_sys^2 ).

sigma_sys is the error that does not average out within a station. Default:
sigma_sys of step 04 per scanner (empirical minus station-wise; an upper bound
because it contains the rail's own unevenness); --sigma-sys sets it, e.g. to
the value of step 06. The limit r_max (theta_max) is the largest value for
which sigma_z(x) <= sigma_z,req, for K = 2 and K = 3.605 (step 04).

Uncertainty: standard errors clustered by setup (stations of one setup share
instrument, atmosphere, registration). The 95 % interval of the limit comes
from resampling whole setups (>= 3 setups) or, with fewer setups, from
resampling stations, which ignores the setup effects and is therefore too
narrow (flagged). Limits outside the observed distance / incidence range are
extrapolations (flagged). With one instrument height, r and theta are almost
collinear (cos theta ~ h / r), so the two limits describe the same effect.

Writes to <results>/07_operational_limits/:
  limits_summary.csv      one row per scanner, variable (range | incidence) and requirement
  limits_fit.json         model coefficients, limits, settings, provenance
  limits_<scanner>.png    sigma_z vs distance and vs incidence with fit and limits
"""
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pandas as pd

from shared import config, io, plots, stats

GRID = {"range": np.linspace(0.5, 100.0, 2000), "incidence": np.linspace(0.0, 89.5, 1791)}
XCOL = {"range": "range_m", "incidence": "incidence_deg"}


def regressor(by, x):
    x = np.asarray(x, float)
    return np.log(x) if by == "range" else np.log(np.cos(np.radians(x)))


def ols(f, y, groups):
    """y = a + b f; SEs clustered by setup (>= 2 setups, statsmodels), else plain OLS SEs."""
    A = np.column_stack([np.ones_like(f), f])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    n = len(y)
    s2 = r @ r / (n - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    se_type = "ols"
    if len(np.unique(groups)) >= 2 and importlib.util.find_spec("statsmodels"):
        import statsmodels.api as sm
        m = sm.OLS(y, A).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
        cov, se_type = m.cov_params(), "cluster(setup)"
    return dict(a=float(c[0]), b=float(c[1]), se_a=float(np.sqrt(cov[0, 0])), se_b=float(np.sqrt(cov[1, 1])),
                rmse_log=float(np.sqrt(s2)), n=int(n), se_type=se_type)


def limit(coef, by, req, sigma_sys):
    x = GRID[by]
    total = np.sqrt(np.exp(coef["a"] + coef["b"] * regressor(by, x)) ** 2 + sigma_sys ** 2)
    ok = total <= req
    if not ok[0]:
        return np.nan
    return float(x[-1] if ok.all() else x[np.argmin(ok) - 1])


def bootstrap(d, by, reqs, sigma_sys, n_boot, seed):
    rng = np.random.default_rng(seed)
    setups = d["setup"].unique()
    by_setup = len(setups) >= 3
    res = {k: [] for k in reqs}
    for _ in range(n_boot):
        if by_setup:
            db = pd.concat([d[d["setup"] == s] for s in rng.choice(setups, len(setups))])
        else:
            db = d.iloc[rng.integers(0, len(d), len(d))]
        f = regressor(by, db[XCOL[by]])
        A = np.column_stack([np.ones_like(f), f])
        c, *_ = np.linalg.lstsq(A, np.log(db["sigma_z_station_mm"].to_numpy()), rcond=None)
        for k, (v, _) in reqs.items():
            res[k].append(limit(dict(a=c[0], b=c[1]), by, v, sigma_sys))
    return {k: np.array(v) for k, v in res.items()}, "setups" if by_setup else "stations (too narrow: < 3 setups)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(ap)
    ap.add_argument("--sigma-sys", type=float, default=None,
                    help="sigma_sys [mm] for all scanners (default: per scanner from step 04)")
    ap.add_argument("--speed-class", default=config.SPEED_CLASS, choices=list(config.ATTENTION))
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap repetitions for the CI of the limits")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    seg = io.read_step_csv(a.results, "heights", "heights_segments.csv")
    summ = io.read_step_csv(a.results, "heights", "heights_summary.csv").set_index("scanner")
    if a.scanner:
        seg = seg[seg["scanner"].isin(a.scanner)]
    out = config.step_dir(a.results, "limits")
    out.mkdir(parents=True, exist_ok=True)
    reqs = config.requirements(a.speed_class)

    rows, fits = [], {}
    for sc, d in seg.groupby("scanner"):
        d = d[np.isfinite(d["sigma_z_station_mm"]) & (d["sigma_z_station_mm"] > 0)]
        if len(d) < 5:
            print(f"{sc}: fewer than 5 stations, skipped")
            continue
        if a.sigma_sys is not None:
            sys_mm, sys_src = a.sigma_sys, "--sigma-sys"
        else:
            sys_mm = float(summ.loc[sc, "sigma_sys_mm"]) if np.isfinite(summ.loc[sc, "sigma_sys_mm"]) else 0.0
            sys_src = "04_rail_heights (upper bound)"
        fits[sc] = dict(n_setups=int(d["setup"].nunique()), n_stations=len(d), sigma_sys_mm=sys_mm,
                        sigma_sys_source=sys_src, corr_range_incidence=float(np.corrcoef(d["range_m"],
                                                                                          d["incidence_deg"])[0, 1]))
        curves = {}
        for by in ("range", "incidence"):
            coef = ols(regressor(by, d[XCOL[by]]), np.log(d["sigma_z_station_mm"].to_numpy()), d["setup"].to_numpy())
            boot, boot_src = bootstrap(d, by, reqs, sys_mm, a.n_boot, a.seed)
            x = GRID[by]
            model = np.exp(coef["a"] + coef["b"] * regressor(by, x))
            curves[by] = dict(x=x, model=model, total=np.sqrt(model ** 2 + sys_mm ** 2), limits={})
            lo_obs, hi_obs = float(d[XCOL[by]].min()), float(d[XCOL[by]].max())
            fits[sc][by] = dict(model=("ln sigma_z = a + b ln r" if by == "range" else "ln sigma_z = a + b ln cos(theta)"),
                                observed=[lo_obs, hi_obs], **coef, limits={})
            for k, (v, _) in reqs.items():
                lim = limit(coef, by, v, sys_mm)
                bk = boot[k][np.isfinite(boot[k])]
                ci = np.percentile(bk, [2.5, 97.5]).tolist() if len(bk) > 10 else [np.nan, np.nan]
                at_grid_end = np.isfinite(lim) and lim >= GRID[by][-1]
                curves[by]["limits"][k] = lim
                r = dict(scanner=sc, by=by, requirement=k, sigma_z_req_mm=v, limit=lim, limit_ci_lo=ci[0],
                         limit_ci_hi=ci[1], ci_from=boot_src, at_grid_end=bool(at_grid_end),
                         extrapolated=bool(np.isfinite(lim) and (lim > hi_obs or lim < lo_obs)),
                         observed_min=lo_obs, observed_max=hi_obs, slope_b=coef["b"], se_b=coef["se_b"],
                         se_type=coef["se_type"], sigma_sys_mm=sys_mm, sigma_sys_source=sys_src,
                         n_setups=fits[sc]["n_setups"], n_stations=len(d))
                fits[sc][by]["limits"][k] = r
                rows.append(r)
        plots.plot_limits(d, curves, reqs, out / f"limits_{sc}.png", sc)
    t = pd.DataFrame(rows)
    t.to_csv(out / "limits_summary.csv", index=False)
    stats.write_json(out / "limits_fit.json", dict(
        settings=dict(speed_class=a.speed_class, requirements_mm={k: v for k, (v, _) in reqs.items()},
                      n_boot=a.n_boot, seed=a.seed, grid={k: [float(v[0]), float(v[-1])] for k, v in GRID.items()}),
        provenance=io.provenance(__file__, a), scanners=fits))
    for r in rows:
        unit = "m" if r["by"] == "range" else "deg"
        flag = " (extrapolated)" if r["extrapolated"] else ""
        flag += " (end of grid: requirement met everywhere)" if r["at_grid_end"] else ""
        print(f"{r['scanner']} {r['by']:9s} {r['requirement']}: limit {r['limit']:.1f} {unit} "
              f"[{r['limit_ci_lo']:.1f}, {r['limit_ci_hi']:.1f}]{flag}")
    print(f"limits -> {out}")


if __name__ == "__main__":
    main()
