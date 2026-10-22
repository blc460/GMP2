# Processing: from rail point clouds to track geometry and its precision

One pipeline of numbered steps. Each step reads the results of the earlier steps from
`processing/results/<step>/` and writes its own folder; the reused functions are in `shared/`.
Every result file records the options, the git commit, the software versions and the time of the run
(`provenance` in the `*_fit.json` / `*_log.json`), and step 01 the SHA-256 of every input cloud.

| step | question it answers | main output |
|---|---|---|
| `00_simulate.py` | Does the code estimate heights, σ_z and twist correctly when the truth is known? | `results_sim/validation.json` |
| `01_prepare.py` | Which points belong to which rail, where along the track (station s), which band is fitted? | prepared scans, shared track axis |
| `02_point_density.py` | Point density and completeness per 0.5 m segment vs distance and incidence (WP4) | `density_*` |
| `03_point_noise.py` | Point-cloud noise σ_pc per segment vs distance and incidence (WP4) | `noise_*` |
| `04_rail_heights.py` | Rail height per station and its standard deviation **σ_z (1.4826 MAD)** vs distance and incidence (WP5) | `heights_*`, `sigma_z_vs_distance.png`, `sigma_z_vs_incidence.png` |
| `05_track_geometry.py` | Cant, twist, vertical versine with variance propagation; criterion K σ ≤ T (WP5) | `<label>_geometry.csv/.png` |
| `06_compare_scans.py` | Empirical precision from two scans of the same section; σ_sys | `compare_pairs.csv` |
| `07_operational_limits.py` | Up to which distance / angle of incidence is σ_z ≤ σ_z,req? | `limits_summary.csv`, `limits_<scanner>.png` |

## Setup

Once, in the repository root (Windows cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS/Linux use `source .venv/bin/activate`. All commands below are run from `processing/`;
`-h` on any script shows its options and the method.

## Run

```bash
python 00_simulate.py --mc 30                                            # validation (optional, ~5 min)
python 01_prepare.py ../pointclouds/track-test.e57 --scanner RTC360 --setup E004
python 02_point_density.py
python 03_point_noise.py
python 04_rail_heights.py
python 05_track_geometry.py
python 06_compare_scans.py        # needs two scans of the same section
python 07_operational_limits.py
```

Several datasets: give all clouds to `01_prepare.py` (or a list file, see its `-h`), then run 02–07 once.
02–07 evaluate all prepared scans per scanner; `--scanner FARO RTC360` restricts them.

### Input (step 01)

Point clouds cropped to the **two rail heads** (default, `--band railhead`) or clips of the rail head incl.
groove (`--band groove`), in one levelled frame (z up), e.g. the registered frame of REGISTER 360 / SCENE.
One dataset = one scan = one scanner position, because range and incidence need the scanner origin.

- **E57:** every scan in the file is one dataset; points are transformed with the scan pose and the origin is
  the pose translation. Several setups in one E57 get the labels `SCANNER_SETUP-1`, `SCANNER_SETUP-2`, …
- **PLY / ASCII / NPY:** one dataset per file; the origin must be given (`--origin` or list file).

Scanner and setup come from the file name `SCANNER_SETUP[_anything].ext`, `--scanner/--setup` or a list file
`path,scanner,setup,origin_x,origin_y,origin_z`.

The track axis is fitted once (first scan) and saved as `results/01_prepare/axis.json`; every later scan
reuses it, so all scans have **the same stations** and can be differenced in step 06. `--refit-axis`
replaces it (e.g. for another test section).

## Method

### Segments and stations (02–04)

Each rail band is cut into segments of 0.5 m on the global grid k · 0.5 m; the station s_i is the segment
centre. Per segment one least-squares plane z = c0 + c1·ds + c2·dq (ds, dq relative to segment and band
centre) with iterative 3σ outlier rejection (σ from the MAD). Distance r = scanner origin → centroid of the
inliers; angle of incidence θ = angle between that beam and the plane normal (0° = perpendicular);
`horiz_angle_deg` = angle between beam and track in plan view.

| step | per segment | per scanner |
|---|---|---|
| 02 | inliers per cm² of band, mean spacing, completeness (5 mm cells) | model density = k cos θ / r² → effective angular step 1/√k; ln ρ vs ln r (−3 expected for one instrument height) and vs ln cos θ |
| 03 | σ_pc (inlier RMS, 95 % χ² CI), robust σ = 1.4826 MAD, outlier share, block variance factor VF, σ_range_equiv = σ_pc / cos θ | pooled σ, power laws in r and cos θ, σ_range_equiv linear in r and θ |

### σ_z of a rail height (04)

z_i = c0 is the rail height at the station and band centre. σ_z is estimated in two independent ways,
**both with the MAD** (robust against the stray points that remain on steel at grazing incidence):

| symbol (column) | definition | meaning |
|---|---|---|
| σ_z,MAD (`sigma_z_mad_mm`) | 1.4826 · MAD(plane residuals of all points) · √q00, q00 = [(AᵀA)⁻¹]₀₀ ≈ 1/n | point noise propagated into c0 |
| σ_z,station (`sigma_z_station_mm`) | σ_z,MAD · √max(VF, 1) | the same, corrected for correlated residuals; **lower bound** |
| σ_z,formal (`sigma_z_formal_mm`) | σ_pc · √q00 (least squares) | for comparison with the MAD value |
| σ_z,emp (`sigma_z_emp_mm`) | 1.4826 · MAD(e) over a group of stations, e_i = (z_i − ẑ_i)/√(1 + h_i) | **empirical**; upper bound |

ẑ_i is the leave-one-out prediction of a quadratic through the heights within ±2 m (`--window 4`,
`--degree 2`), h_i its variance factor, so Var(e_i) = σ_z² for equal σ_z. The 95 % CI of σ_z,emp uses the
χ² interval with 0.37 n degrees of freedom (asymptotic efficiency of the MAD, 37 %); RMS(e) with its χ² CI
is given as cross-check. σ_z,emp is computed per scanner, rail, setup, distance bin (1 m) and incidence bin
(5°), and for the cross level L − R (σ_cross/√2, scanner tilt cancels).
σ_sys = √(σ_z,emp² − RMS(σ_z,station)²) is what the point noise does not explain.

σ_z,emp contains the rail's own short-wave unevenness (welds, joints, wear), so it is an upper bound for the
scanner. Errors that are constant or vary smoothly along the track (levelling, vertical angle, range offset,
registration) are absorbed by the polynomial and appear only in step 06 or against a reference.

Plots: `sigma_z_vs_distance.png` and `sigma_z_vs_incidence.png` show σ_z,station per station (dots),
σ_z,emp per bin with 95 % CI (line and band) and the requirements σ_z,req for K = 2 and K = 3.605.

### Track geometry (05)

From the heights of both rails at the same station (report, section *Quality criteria*):

| parameter | formula | variance (uncorrelated heights) |
|---|---|---|
| cant | ü_i = b0/b (z_R,i − z_L,i), b0 = 1 m, b = distance of the band centres (measured; `--b 1` as in the report) | σ_ü² = (b0/b)² (σ_zL² + σ_zR²) |
| twist (base l = 5 m) | N_i = (ü_i − ü_i+k)/l in mm/m = ‰ | σ_N² = (σ_ü,i² + σ_ü,i+k²)/l² |
| vertical versine (chord 10 m) | vPf_i = H_i − (H_i−m + H_i+m)/2, H = (z_L + z_R)/2 | σ_vPf² = σ_H,i² + (σ_H,i−m² + σ_H,i+m²)/4; σ_ΔvPf = √2 σ_vPf |

Each σ is propagated twice: with σ_z,station (lower bound) and with σ_z,emp (upper bound). With equal σ_z
this reduces to the report's σ_N = 2 b0 σ_z/(b l) and σ_ΔvPf = √(3/2) σ_z. Per station the criterion
K σ ≤ T is checked for K = 2 (interpretations 1 and 3) and K = 3.605 (interpretation 2) with the attention
values of `--speed-class` (default 48–60 km/h: T_vPf = 3 mm, T_N = 1 ‰, so σ_z,req = 1.22 / 0.67 mm).
The design versine Pf_Rv is not subtracted; it cancels in the monitored change.

### Two scans (06) and operational limits (07)

06 differences heights, cant, twist and versine of every pair of scans station by station (height offset
removed), estimates their std with 1.4826 MAD (Bartlett-corrected degrees of freedom for twist and versine)
and the implied σ_z and σ_sys. 07 fits ln σ_z,station = a + b ln r and = a + b ln cos θ per scanner,
predicts σ_z = √(exp(…)² + σ_sys²) (σ_sys from 04 or `--sigma-sys`, e.g. from 06) and reports the largest
r and θ with σ_z ≤ σ_z,req, with a bootstrap interval (over setups if ≥ 3, otherwise over stations and
flagged as too narrow) and a flag when the limit is an extrapolation.

## Results on the test track (`track-test.e57`)

RTC360, one setup on a bridge about 3.1 m above the rails, 12 m of track, distance 7–10 m, angle of
incidence 45–63°. All values from `results/` (run of this commit).

| quantity | value | file |
|---|---|---|
| point density (median, 5–95 %) | 13.3 pts/cm² (6.2–21.2), effective angular step 0.28 mrad | `02_point_density/density_summary.csv` |
| point noise σ_pc (pooled) | 0.44 mm | `03_point_noise/noise_summary.csv` |
| σ_z,station (median) | 0.019 mm | `04_rail_heights/heights_summary.csv` |
| σ_z,emp = 1.4826 MAD(e) | 0.079 mm (95 % CI 0.058–0.120, n = 43); RMS(e) 0.099 mm | same |
| σ_sys | 0.075 mm (σ_z,emp / σ_z,station = 3.4) | same |
| cant | −1.2 … +1.0 mm, b = 1.064 m | `05_track_geometry/RTC360_E004_geometry.csv` |
| twist (13 values) | −0.44 … +0.22 ‰, σ_N = 0.008 ‰ (station) / 0.031 ‰ (empirical) | same |
| vertical versine (3 values) | −1.7 … −1.3 mm, σ_ΔvPf = 0.10 mm (empirical) | same |
| criterion | met at every station for K = 2 and K = 3.605, with either σ_z | `05_track_geometry/geometry_summary.csv` |
| limits (extrapolated far beyond 7–10 m) | r_max ≈ 37 m (K = 2), 29 m (K = 3.605) | `07_operational_limits/limits_summary.csv` |

What this does and does not show: σ_z is about 15 times smaller than required, so a single RTC360 scan at
7–10 m meets the criterion with a wide margin. The limits of step 07 come from one setup with 3 m of distance
range and are an extrapolation of a steep fit (density slope −3.5); they must not be reported as operational
limits before the campaign with 5–7 setups. σ_z,emp is three times σ_z,station: the point noise does not
explain the station-to-station scatter, which is either the rail's unevenness or an error that changes along
the track; 06 with a second scan separates the two.

## Validation (`00_simulate.py`)

Simulated railheads (30 m, two setups at 1.5 m height, angular step 0.6 mrad, range noise along the beam,
1 % gross outliers, known heights with a 4 mm settlement trough and a twist of −0.4 ‰; illustrative values,
not a real scanner). `python 00_simulate.py --mc 30`:

| check | result |
|---|---|
| height bias (two scans) | −0.001 / +0.001 mm (SE 0.004 / 0.003 mm) |
| twist | −0.398 / −0.400 ‰ (true −0.4 ‰), RMSE 0.011 / 0.012 ‰ |
| Monte Carlo, 30 runs, 103 stations: median height bias | −0.0002 mm |
| std(true height error) / σ_z,station (5 / 50 / 95 %) | 0.75 / 0.95 / 1.22 |
| std(true height error) / σ_z,MAD | 0.77 / 0.98 / 1.24 |
| σ_z,emp / 1.4826 MAD(true height error) per run | 0.84 / 1.01 / 1.22 |

Both MAD-based σ_z estimates are unbiased when their assumptions hold (independent noise, smooth rail); the
spread of the ratios is the sampling uncertainty of a σ from about 20–40 values. Put this in the methods
chapter as the validation of the processing.

## Interpretation and choices to justify in the report

- **Distance and incidence are not independent.** On a horizontal railhead cos θ = h / r, so with one
  instrument height r and θ are almost collinear (`corr_range_incidence` in the `*_fit.json`). The two plots
  show the same trend twice. To separate the effects, vary the instrument height or the horizontal angle.
- **σ_pc falls with distance** at grazing incidence because less of the range noise is normal to the
  surface; σ_range_equiv = σ_pc / cos θ is the value to compare with the specification.
- **Stations are not independent samples.** Stations of one setup share instrument, atmosphere and
  registration. Report the number of setups (`n_setups`) with every estimate; cluster standard errors (07)
  are only meaningful with about 5 or more setups.
- **σ_z,station is a lower bound, σ_z,emp an upper bound** (see 04). Use σ_z,emp, or σ_sys from 06, when the
  criterion is checked.
- Choices: segment length 0.5 m (= station spacing and height window), edge trim 5 mm, window ±2 m and
  degree 2 of the leave-one-out polynomial, bin widths 1 m / 5°, speed class 48–60 km/h, b measured.
