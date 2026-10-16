# From point clouds to the values that must meet the thresholds

Starter code: `trackqc/` (Python ≥ 3.10; packages in `requirements.txt` at the repository root, see section 4 for the venv).
Tested on simulated scans only; the real Bucheggplatz clouds were not available here.

## 1. Which value is actually tested

The VBZ attention values apply to the twist *N* (5 m base) and the vertical versine Δ*vPf* (10 m base).
Both are linear combinations of **rail heights z at cross-sections**. The point cloud therefore has to deliver
one number per rail and cross-section, its standard deviation σ_z, and from that

| quantity | from | criterion (48 < V_R ≤ 60 km/h) |
|---|---|---|
| rail height z_L,i, z_R,i | plane fit to the running surface, window ±0.25 m around station s_i | σ_z ≤ 1.22 mm |
| cant ü_i = b0/b (z_R − z_L) | two heights | – |
| twist N_i = (ü_i − ü_i+k)/l | four heights | 2σ_N ≤ 1.0 ‰ |
| versine vPf_i = H_i − (H_i−k + H_i+k)/2 | six heights | 2·√2·σ_vPf ≤ 3 mm |

The code computes all of them with **individual** variance propagation (each height has its own σ_z,
which depends on distance and incidence), so the report's formulas with equal σ_z are the special case.

## 2. Processing chain

| step | what | method | output / uncertainty contribution |
|---|---|---|---|
| 0 | Export | E57 per setup; scanner origin = E57 pose translation (`scripts/e57_check_frame.py`). Work in the registered, levelled frame so all scans share stations. | `setups.csv`: label, origin x y z |
| 1 | Rail clipping (manual, as in the concept) | CloudCompare: one clip per rail = rail head incl. groove + ~2 cm margin. Save as PLY. Name `SCANNER_SETUP_CONDITION_L/R.ply`. | Clip boundaries do not affect results, the band is found automatically (step 3). |
| 2 | Track axis and stationing | Fit once (densest scan) the mean of both rail centre lines as cubic polynomial; project every point to (s, q). Reuse `axis.json` for every dataset. | Identical cross-sections across scanners, setups and conditions. Axis errors only shift points along/across the flat running surface → no height error. |
| 3 | Running-surface band | Stacked lateral profile (2 mm bins): the groove is the deepest (or empty) part. Groove edge = first bin on the field side from which the next 35 mm are flat (±2 mm), so paving or a guard rail higher than the head does not matter. Band starts 5 mm beyond the groove edge (gauge corner), 30 mm wide. | **Take width and margin from the VBZ rail profile drawing (e.g. Ri60N)** and justify them; check `*_profile.png` for every dataset. |
| 4 | Segment quality (WP4) | Per 0.5 m segment and rail: plane z = c0 + c1·ds + c2·dq by least squares with 3σ(MAD) outlier rejection. | n, density, completeness (5 mm grid), σ_pc with 95 % χ² CI, robust σ, outlier share, range, incidence angle, horizontal angle to the track. |
| 5 | Rail heights (WP5) | Same plane fit in a window ±0.25 m around each station (every 0.5 m); z_i = c0 at the band centre. | Formal σ_z = σ_pc·√[(AᵀA)⁻¹]₀₀ ≈ σ_pc/√n. |
| 6 | Cant, twist, versine | Formulas above, l = 5 m, b = measured distance between band centres (~1.08 m in the simulation), b0 = 1 m. | σ_ü, σ_N, σ_vPf, criterion flags per cross-section. |
| 7 | Empirical check | Two independent scans of the same section (other setup or scanner, or a repeated scan) = two epochs with zero deformation. dN and dvPf are computed; offsets and tilts cancel. | Empirical σ vs formal σ, implied σ_z, **σ_sys** (excess), empirical false-alarm rate. |
| 8 | Operational limits | Per scanner and condition: log σ_pc and log density regressed on log r and horizontal angle, SEs clustered by setup; predicted σ_z(r) = √(VF·σ_pc²/(ρA) + σ_sys²). | r_max where σ_z(r) = σ_z,req, with a setup-bootstrap CI. |

## 3. Points your professor will check

1. **σ_pc/√n is a lower bound.** It assumes independent points. Neighbouring laser points overlap at grazing
   incidence (footprint ≫ point spacing), and incidence-dependent range errors do not average out.
   Two safeguards are built in: the block variance factor `var_factor` per segment (≫ 1 means correlated
   residuals) and step 7, which measures the precision empirically. Report σ_z from step 7, not only the formal value.
   This is exactly the σ_sys term of the quality-criteria section, so step 7 is how you estimate it.
2. **Distance and incidence angle are not independent.** The running surface is horizontal, so its incidence angle
   is θ = arctan(d_h/h) for instrument height h. With a fixed tripod height, r and θ are almost collinear
   and their effects cannot be separated by regression. What your setups really vary besides distance is the
   **horizontal angle α between line of sight and track** (column `horiz_angle_deg`). Either say this
   explicitly or vary the instrument height. Expected density ∝ cos θ / r² = h / r³, so a fitted slope of
   about −3 for log density vs log r is a physical plausibility check (the simulation gives −3.4).
3. **Segments are not independent samples.** Segments of one setup share the instrument, atmosphere and
   registration. Use setup-clustered standard errors (done in `limits.py`) and say how many setups each
   estimate rests on. With 5 to 7 setups the CI of r_max is wide; that is a result, not a flaw.
4. **Uncertainty of σ itself.** Every σ comes with a χ² confidence interval; compare 2σ with T using the upper
   bound if you want a conservative statement.
5. **Neighbouring versines/twists are correlated** (they share heights). `compare.py` corrects the degrees of
   freedom (Bartlett: n_eff = n/1.5 for twist, n/1.94 for versine).
6. **Model check.** A crowned running surface would bias σ_pc upwards. Plot residuals against dq for a few
   segments; if there is a curvature, add a dq² term in `fit_plane`.
7. **Validation of the code.** `python -m trackqc.simulate --mc 25` simulates scans with known heights:
   height bias +0.001 mm and empirical/formal σ_z = 1.07 (5–95 %: 0.82–1.43) for 25 runs, i.e. the estimator
   is unbiased and the formal precision is right when the noise is independent. Put this in the methods chapter.
   The simulated noise and sampling are illustrative, not FARO or RTC360 values.
8. **Reference.** If you levelled the rail heads (Messkörner on the groove head), compare those heights with
   z from the cloud. That is the only check that also catches a common bias, e.g. a height offset of the
   running surface relative to the groove head.

## 4. How to run

Set up a virtual environment once, in the repository root (Windows cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS/Linux use `source .venv/bin/activate`. Activate the venv in every new terminal; `.venv/` is not committed.
After installing a new package, add it to `requirements.txt` so the environment can be rebuilt.

```bash
cd processing
# one dataset (repeat for every scanner x setup x condition; first call creates axis.json)
python -m trackqc.process_setup --left clips/FARO_S1_dry_L.ply --right clips/FARO_S1_dry_R.ply \
    --origin 36.680 3.389 0.002 --axis axis.json --label FARO_S1_dry --out results/
# alternatively one clip with both rails (E57/PLY/ASCII), split at the gap between the rails;
# for E57 the origin defaults to the scan pose translation (check it, override with --origin)
python -m trackqc.process_setup --cloud ../pointclouds/track-test.e57 \
    --axis axis_test.json --label TLS_E004_test --out results/
# empirical precision from two scans of the same section
python -m trackqc.compare results/FARO_S1_dry_geometry.csv results/FARO_S2_dry_geometry.csv
# operational limits for all datasets in results/
python -m trackqc.limits results/ --sigma-sys 0.3 --req 1.22
# test and validation on simulated scans
python -m trackqc.simulate --out demo/ --mc 25
```

Labels must be `SCANNER_SETUP_CONDITION` (no further underscores); `limits.py` splits on them.

### Quality criteria vs range and incidence (`trackqc/quality/`)

One script per criterion, same inputs as `process_setup` (one clip with both rails, or `--left/--right`;
origin from the E57 pose unless `--origin` is given). Reuse an existing `--axis` to get the same stations
as other datasets; without it the axis is fitted from the cloud and saved to `<out>/<label>/axis.json`.

```bash
python -m trackqc.quality.density --cloud ../pointclouds/track-test.e57 --out results/
python -m trackqc.quality.noise   --cloud ../pointclouds/track-test.e57 --out results/
python -m trackqc.quality         --cloud ../pointclouds/track-test.e57 --out results/   # both, one load
```

```
trackqc/quality/
  common.py     loading, running band, segments, range/incidence, bin tables, fits, plots
  density.py    criterion: point density, spacing, completeness; model density = k cos(inc) / r^2
  noise.py      criterion: sigma_pc about a plane per segment, pooled sigma per bin, fits
  __main__.py   all criteria from one loading of the cloud
results/<label>/
  band_profile.png                  check the running band first
  axis.json                         track axis (if not given with --axis)
  density.csv  noise.csv            one row per 0.5 m segment: metric, range_m, incidence_deg, ...
  density_bins.csv  noise_bins.csv  per range bin, per incidence bin, per range x incidence cell
  density_fit.json  noise_fit.json  fits against range and incidence, settings
  density.png  noise.png            metric vs range and vs incidence
```

A new criterion is one more module: a per-segment metric function (as `core.density_metrics`) passed to
`common.segment_table`, then `common.bin_table` and `common.write_criterion`.
On a horizontal surface cos(incidence) = h / range, so range and incidence are nearly collinear
(`corr_range_incidence` in the fit files); the bins by range and by incidence describe the same trend.

## 5. Choices you need to make and justify

- Segment length 0.5 m (WP4) and height window ±0.25 m (WP5): trade-off between n and the spatial resolution
  of a "point" height. The VBZ levelled points are single points, so a short window is closer to the standard.
- Band width and edge margin from the rail profile (step 3).
- Speed class of the test section (default 48–60 km/h → σ_z,req = 1.22 mm).
- Which scan pairs you use in step 7 (same scanner/two setups isolates geometry; two scanners/same setup isolates the instrument).
