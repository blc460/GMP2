# Processing: from point clouds to quality values and track geometry

Two parts, both Python ≥ 3.10 (packages in `requirements.txt` at the repository root, see [Setup](#setup)):

| folder | input | what it delivers |
|---|---|---|
| `railhead/` | clouds cropped to the **rail heads** (two parallel strips) | point density and point-cloud noise **per scanner**, vs measurement distance and angle of incidence |
| `trackqc/` | clips of the rail head **incl. groove** | rail heights, cant, twist, versine with variance propagation; empirical check; operational limits |

`railhead/` reuses the numerical building blocks of `trackqc/core.py` (rail split, track axis, plane fit,
segment geometry, density and noise per segment).

## Setup

Once, in the repository root (Windows cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS/Linux use `source .venv/bin/activate`. Activate the venv in every new terminal; `.venv/` is not committed.
After installing a new package, add it to `requirements.txt` so the environment can be rebuilt.
All commands below are run from `processing/`. Results go to `processing/results/` (not committed).

---

## Part A: point quality on the railhead (`railhead/`)

### Input

Point clouds cropped to the rail heads, i.e. in theory two nearly horizontal, parallel planes (left and
right rail head). Classification of the track parts (later, by Luca's algorithm) is not needed and not
used yet. Points must be in one levelled frame (z up), e.g. the registered frame of REGISTER 360 / SCENE.

One dataset = one scan = one scanner position, because range and incidence need the scanner origin.

- **E57:** every scan in the file is one dataset; points are transformed with the scan pose into the file
  frame and the origin is the pose translation. Several setups may therefore be exported into one E57;
  they get the labels `SCANNER_SETUP-1`, `SCANNER_SETUP-2`, …
- **PLY / ASCII / NPY:** one dataset per file; the origin must be given (`--origin` or list file).

Scanner and setup come from the file name `SCANNER_SETUP[_anything].ext` (e.g. `RTC360_S3_dry.e57`),
or from `--scanner/--setup`, or from a list file:

```csv
path,scanner,setup,origin_x,origin_y,origin_z
clips/FARO_S1.ply,FARO,S1,36.680,3.389,0.002
clips/RTC360_all.e57,RTC360,,,,
```

Relative paths are relative to the list file; empty cells mean "from the file name / E57 pose".

### Run

```bash
python railhead/00_prepare.py ../pointclouds/FARO_S1_dry.e57 ../pointclouds/RTC360_S1_dry.e57
python railhead/00_prepare.py ../pointclouds/track-test.e57 --scanner RTC360 --setup E004
python railhead/00_prepare.py --list datasets.csv

python railhead/01_point_density.py                          # all prepared scans
python railhead/02_point_noise.py --scanner FARO RTC360      # only these scanners
```

`00_prepare.py` runs once per scan; 01 and 02 read all prepared scans in `results/railhead/00_prepare/`
and evaluate them per scanner. `-h` on any script shows its options and method.

### Scripts

| script | method |
|---|---|
| `00_prepare.py` | Split the cloud into left/right rail at the gap between them; track axis = mean of both rail centre lines (cubic); stations s and lateral offset q for every point. Railhead band per rail: centre line q_rail(s), head edges = 0.5 % / 99.5 % quantiles of q − q_rail, band = head minus `--edge-trim` (5 mm) on both sides, so the rounded head corners do not enter the plane fits. Warns if a "railhead" is wider than 100 mm (clip not cropped). |
| `01_point_density.py` | Per 0.5 m segment (only segments the clip covers completely): inlier points per cm² of the band, mean spacing 1/√density, completeness (occupied 5 mm cells). Per scanner: model density = k·cos θ / r² → effective angular step δ = 1/√k (compare with the resolution setting); power laws ln ρ vs ln r (expected slope −3 for one instrument height) and vs ln cos θ. |
| `02_point_noise.py` | Per 0.5 m segment (≥ 30 inliers): plane fit with 3σ (MAD) outlier rejection, σ_pc = noise normal to the railhead with 95 % χ² CI, robust σ (MAD), outlier share, block variance factor. σ_range_equiv = σ_pc / cos θ = range noise if all noise were along the beam (compare with the manufacturer's spec; an upper bound at grazing incidence). Pooled σ per scanner, bin, rail and setup. Fits: σ_pc as power law of r and of cos θ (exponent 1 for pure beam noise), σ_range_equiv linear in r and in θ. |
| `shared/` | `config.py` defaults and common options, `data.py` reading, railhead band, prepared files, `segments.py` segment table, `stats.py` bin tables, fits, result files, `plots.py` plots |

Range = distance scanner origin → centroid of the segment inliers; incidence θ = angle between that beam
and the normal of the fitted plane (0° = perpendicular); `horiz_angle_deg` = angle between beam and track in plan view.

### Options

| option | scripts | default | meaning |
|---|---|---|---|
| `--scanner`, `--setup`, `--origin` | 00 | file name / E57 pose | dataset names and scanner origin |
| `--list` | 00 | – | list file instead of cloud paths |
| `--edge-trim` | 00 | 0.005 m | margin removed at both railhead edges |
| `--results` | all | `processing/results/railhead` | results folder |
| `--scanner` | 01, 02 | all | evaluate only these scanners |
| `--seg-len` | 01, 02 | 0.5 m | segment length along the rail (WP4) |
| `--range-bin`, `--incidence-bin` | 01, 02 | 1 m, 5° | bin widths of tables and plots |
| `--cell` | 01 | 0.005 m | grid cell for completeness |
| `--min-points` | 02 | 30 | minimum inliers per segment |

### Output

```
results/railhead/
  00_prepare/
    <label>.npz, <label>.json   points of both rails with s, q; scanner, setup, origin, axis, band, head width
    <label>_railhead.png        lateral profile with head edges and band: check it for every scan
  01_point_density/
    density_segments.csv        one row per segment: density, spacing, completeness, range, incidence, ...
    density_bins.csv            per scanner and range bin / incidence bin / range x incidence cell
    density_summary.csv         one row per scanner (table for the report)
    density_fit.json            models and fits per scanner, settings
    density.png                 density vs distance and vs incidence, one colour per scanner
  02_point_noise/
    noise_segments.csv, noise_bins.csv, noise_summary.csv, noise_fit.json
    noise.png                   sigma_pc vs distance and vs incidence
    noise_range_equiv.png       sigma_pc / cos(incidence) vs distance and vs incidence
```

Plots: dots = segments, line with band = bin median and interquartile range, dashed = fit; one colour
and marker per scanner (at most 8 scanners per plot, otherwise filter with `--scanner`).

### Adding a criterion

One more script `03_<criterion>.py` with the same pattern as 01/02:
`data.load_prepared` → `segments.segment_table(preps, metric)` with a per-segment `metric(seg) -> dict`
→ `stats.bin_table` → `stats.write_results` → `plots.plot_vs_range_incidence`.

### Validation

Synthetic railheads (two scanners with angular step 0.6 / 0.4 mrad and range noise 0.5 / 1.0 mm, two
setups each) give angular step 0.60 / 0.40 mrad, σ_range_equiv 0.49 / 0.99 mm and a log-log density
slope of −2.99. The test cloud `pointclouds/track-test.e57` is not cropped to the railhead (head width
≈ 250 mm is reported); it only shows that the scripts run.

### Interpretation

- **Distance and incidence are not independent.** On a horizontal railhead cos θ = h / r for a height
  difference h, so with one instrument height r and θ are almost collinear (`corr_range_incidence` in the
  `*_fit.json`, ≈ 0.95 on the test cloud). The two plot panels show the same trend twice, not two separate
  effects. To separate them, vary the instrument height.
- **σ_pc falls with distance** because less of the range noise is normal to the surface at grazing
  incidence; use σ_range_equiv to compare scanners with their specification.
- **Segments are not independent samples.** Segments of one setup share instrument, atmosphere and
  registration; the standard errors of the fits treat them as independent. Report the number of setups
  (`n_setups` in every table) with each estimate.
- **var_factor ≫ 1** means correlated residuals or surface shape (e.g. a crowned head); then σ/√n is too optimistic.

---

## Part B: rail heights and track geometry (`trackqc/`)

Tested on simulated scans; works on clips of the rail head **incl. groove** (the running band is found
from the groove).

### Which value is actually tested

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

### Processing chain

| step | what | method | output / uncertainty contribution |
|---|---|---|---|
| 0 | Export | E57 per setup; scanner origin = E57 pose translation. Work in the registered, levelled frame so all scans share stations. | origin x y z per setup |
| 1 | Rail clipping (manual, as in the concept) | CloudCompare: one clip per rail = rail head incl. groove + ~2 cm margin. Save as PLY. Name `SCANNER_SETUP_CONDITION_L/R.ply`. | Clip boundaries do not affect results, the band is found automatically (step 3). |
| 2 | Track axis and stationing | Fit once (densest scan) the mean of both rail centre lines as cubic polynomial; project every point to (s, q). Reuse `axis.json` for every dataset. | Identical cross-sections across scanners, setups and conditions. Axis errors only shift points along/across the flat running surface → no height error. |
| 3 | Running-surface band | Stacked lateral profile (2 mm bins): the groove is the deepest (or empty) part. Groove edge = first bin on the field side from which the next 35 mm are flat (±2 mm), so paving or a guard rail higher than the head does not matter. Band starts 5 mm beyond the groove edge (gauge corner), 30 mm wide. | **Take width and margin from the VBZ rail profile drawing (e.g. Ri60N)** and justify them; check `*_profile.png` for every dataset. |
| 4 | Segment quality (WP4) | Per 0.5 m segment and rail: plane z = c0 + c1·ds + c2·dq by least squares with 3σ(MAD) outlier rejection. | n, density, completeness (5 mm grid), σ_pc with 95 % χ² CI, robust σ, outlier share, range, incidence angle, horizontal angle to the track. |
| 5 | Rail heights (WP5) | Same plane fit in a window ±0.25 m around each station (every 0.5 m); z_i = c0 at the band centre. | Formal σ_z = σ_pc·√[(AᵀA)⁻¹]₀₀ ≈ σ_pc/√n. |
| 6 | Cant, twist, versine | Formulas above, l = 5 m, b = measured distance between band centres (~1.08 m in the simulation), b0 = 1 m. | σ_ü, σ_N, σ_vPf, criterion flags per cross-section. |
| 7 | Empirical check | Two independent scans of the same section (other setup or scanner, or a repeated scan) = two epochs with zero deformation. dN and dvPf are computed; offsets and tilts cancel. | Empirical σ vs formal σ, implied σ_z, **σ_sys** (excess), empirical false-alarm rate. |
| 8 | Operational limits | Per scanner and condition: log σ_pc and log density regressed on log r and horizontal angle, SEs clustered by setup; predicted σ_z(r) = √(VF·σ_pc²/(ρA) + σ_sys²). | r_max where σ_z(r) = σ_z,req, with a setup-bootstrap CI. |

### Run

```bash
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

### Points your professor will check

1. **σ_pc/√n is a lower bound.** It assumes independent points. Neighbouring laser points overlap at grazing
   incidence (footprint ≫ point spacing), and incidence-dependent range errors do not average out.
   Two safeguards are built in: the block variance factor `var_factor` per segment (≫ 1 means correlated
   residuals) and step 7, which measures the precision empirically. Report σ_z from step 7, not only the formal value.
   This is exactly the σ_sys term of the quality-criteria section of the concept, so step 7 is how you estimate it.
2. **Distance and incidence angle are not independent** (see Part A). What your setups really vary besides
   distance is the **horizontal angle α between line of sight and track** (column `horiz_angle_deg`).
   Either say this explicitly or vary the instrument height. Expected density ∝ cos θ / r² = h / r³, so a
   fitted slope of about −3 for log density vs log r is a physical plausibility check.
3. **Segments are not independent samples.** Use setup-clustered standard errors (done in `limits.py`) and
   say how many setups each estimate rests on. With 5 to 7 setups the CI of r_max is wide; that is a result, not a flaw.
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

---

## Choices you need to make and justify

- Segment length 0.5 m (WP4) and height window ±0.25 m (WP5): trade-off between n and the spatial resolution
  of a "point" height. The VBZ levelled points are single points, so a short window is closer to the standard.
- Railhead edge trim (Part A, 5 mm) and band width / edge margin (Part B, step 3) from the rail profile drawing.
- Bin widths for range and incidence in the tables and plots (1 m, 5°).
- Speed class of the test section (default 48–60 km/h → σ_z,req = 1.22 mm).
- Which scan pairs you use in step 7 (same scanner/two setups isolates geometry; two scanners/same setup isolates the instrument).
