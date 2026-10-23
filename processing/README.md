# Processing: from rail point clouds to track geometry and its precision

One pipeline of numbered steps. Each step reads the results of the earlier steps from
`<results>/<step>/` and writes its own folder; the functions used by several steps are in `shared/`.
Every result file records the options, the git commit, the software versions and the time of the run
(`provenance` in the `*_fit.json` / `*_log.json`), and step 01 the SHA-256 of every input cloud.

| step | question it answers | main output |
|---|---|---|
| `00_simulate.py` | Does the code estimate heights, σ_z and twist correctly when the truth is known? | `results_sim/validation.json` |
| `01_prepare.py` | Which points belong to the track and to which rail, where along the track (station s), which strip of the rail is fitted? | prepared scans, shared track axis |
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
`-h` on any script shows its options and the method. Reading LAS/LAZ needs `laspy[lazrs]` in addition.

## Two kinds of input

| | A: railhead clips | B: classified clouds |
|---|---|---|
| example | `track-test.e57` (railheads of one track, cropped by hand) | `track-test-2-classified.e57` (segmentation of a junction) |
| what is in it | only the two rail heads of the test track | rail-head class of all tracks, here incl. groove and ~10 cm of paving per rail |
| class filter | none (no class field) | `--keep-class 1` (default) if the file has a class field: LAS/LAZ `classification`, PLY scalar field (`scalar_Classification`, `class`, …), ASCII/NPY column `--class-column` |
| one track | the clip is one track | `--track x1 y1 x2 y2 [...] --corridor 1.6`: centre line of the test track |
| band | `railhead` (chosen automatically: clips ≤ 100 mm wide) | `groove` (chosen automatically: wider clips) |
| scanner origin | E57 pose | `--origin x y z` (an E57 exported from LAZ has no scanner pose) |

**Note on `track-test-2-classified.e57`:** this E57 has no class field (only x, y, z, intensity). The class was
lost when it was exported from the LAZ. It already contains only the rail-head class, so all its points are used.
Its E57 pose (−15.95, 6.16, 0) is not the scanner position; step 01 detects this. The scanner position used is
that of setup E004 from `track-test.e57`; the point density of both clouds puts the scanner within
0.2–0.3 m of it in plan.

## Run

```bash
python 00_simulate.py --mc 30                        # validation (optional, ~5 min)

# A: railhead clips
python 01_prepare.py ../pointclouds/track-test.e57 --scanner RTC360 --setup E004 --results results/track-test

# B: classified cloud, one track out of the junction
python 01_prepare.py ../pointclouds/track-test-2-classified.e57 --scanner RTC360 --setup E004 \
    --origin 6.268 3.132 3.140 --track 8.5 8.40 25.5 8.80 --corridor 1.6 --results results/track-test-2-classified

# then for each results folder
python 02_point_density.py      --results results/track-test-2-classified
python 03_point_noise.py        --results results/track-test-2-classified
python 04_rail_heights.py       --results results/track-test-2-classified
python 05_track_geometry.py     --results results/track-test-2-classified
python 06_compare_scans.py      --results results/track-test-2-classified    # needs two scans of the section
python 07_operational_limits.py --results results/track-test-2-classified
```

Several scans of one section (setups, scanners, wet/dry): give all of them to one `01_prepare.py` call (or a list
file `path,scanner,setup,origin_x,origin_y,origin_z`, see `-h`) with one `--results` folder, then run 02–07 once.
They then share the track axis and the stations, which step 06 needs. `--scanner FARO RTC360` restricts 02–07.

## What each step does

### 01 Prepare (`01_prepare.py`, `shared/io.py`, `shared/geometry.py`)

1. **Read** the cloud. E57: every scan in the file is one dataset; the points are transformed with the scan pose
   into the file frame and the scanner origin is the pose translation, unless `--origin` is given. PLY, LAS/LAZ,
   ASCII, NPY: one dataset per file, `--origin` required. Scanner and setup names come from the file name
   `SCANNER_SETUP_….ext` or `--scanner/--setup`; the dataset label is `SCANNER_SETUP`.
2. **Class filter.** If the file has a class field, only the points of `--keep-class` (default 1 = rail head) are
   kept. Without a class field all points are used and a note is printed.
3. **Track corridor.** With `--track`, only points within `--corridor`/2 of the polyline (in plan) and between its
   end points are kept. This cuts one track out of a junction.
4. **Origin check.** A scanner with a constant angular step puts density ∝ cos θ / r² = h / r³ points per area
   on a horizontal surface. The local point density (neighbours within 15 mm) of 20 000 random points is fitted
   with ln ρ = c + ln h − 3 ln r over the scanner position. If the given origin is more than `--origin-tol`
   (2 m) away in plan, the step stops, because every distance and incidence angle would be wrong. The estimated
   height is biased (weak steel returns at grazing incidence), so it is printed but not used.
5. **Split the rails.** The cloud is projected onto every horizontal direction (0.25° steps); the direction with
   the widest empty gap between the 5 % and 95 % quantiles is the track direction, and the gap separates left and
   right rail (it must be ≥ 0.2 m).
6. **Track axis.** For each rail a cubic polynomial of the lateral position along the track; the axis is the mean
   of both. It is fitted from the first scan, saved as `axis.json` and reused for all later scans in the same
   results folder, so they all share the same stations.
7. **Stationing.** Every point gets its station s (arc length along the axis, Newton iteration for the foot point)
   and its lateral offset q (positive to the left). Left rail = q > 0.
8. **Band.** The strip of each rail that all later fits use.
   - `railhead` (clips ≤ 100 mm wide): rail centre line q_rail(s) as cubic polynomial; head edges = 0.5 % and
     99.5 % quantiles of q − q_rail; band = head minus 5 mm (`--edge-trim`) on both sides, so the rounded head
     corners are not fitted.
   - `groove` (wider clips): all points are stacked into a lateral profile (2 mm bins, height relative to the
     90th percentile of each 0.25 m slice). The groove is the deepest point of the smoothed profile (or the widest
     empty run if its bottom is not seen). Walking from the groove towards the field side, the groove edge is the
     first bin from which the next 35 mm are flat (±2 mm). The band starts 5 mm (`--edge-margin`) beyond the
     groove edge and is 30 mm wide (`--band-width`), i.e. on the running surface of the head. Take both values
     from the rail profile drawing (e.g. Ri60N) and justify them in the report.
   `<label>_railhead.png` shows the lateral profile, the point count per bin and the band: check it for every scan.

### 02 Point density (`02_point_density.py`)

The band of each rail is cut into **segments of 0.5 m** on the global grid k · 0.5 m (so both rails and all scans
have the same segments; station s_i = segment centre). For every segment (`shared/fit.py`):

- **Plane fit** z = c0 + c1·ds + c2·dq (ds, dq relative to segment centre and band centre) by least squares,
  repeated with points beyond 3σ removed, σ = 1.4826 MAD of the residuals, until the inliers do not change.
- **Distance** r = scanner origin → centroid of the inliers; **angle of incidence** θ = angle between that beam and
  the normal of the fitted plane (0° = perpendicular); horizontal angle between beam and track.
- **Density** = inliers per cm² of band, mean spacing = 1/√density, **completeness** = share of occupied 5 mm cells.
  Only segments that the clip covers completely are used.

Per scanner: the model density = k cos θ / r² gives the effective angular step δ = 1/√k (compare with the
resolution setting); ln ρ is regressed on ln r (slope −3 expected for one instrument height) and on ln cos θ.
Tables per 1 m distance bin, 5° incidence bin and their combination (`density_bins.csv`).

### 03 Point noise (`03_point_noise.py`)

Same segments and plane fits (≥ 30 inliers). Per segment: **σ_pc** = √(Σr²/(n−3)) of the inlier residuals = noise
normal to the rail head, with its 95 % χ² interval; robust σ = 1.4826 MAD of all residuals; outlier share; block
variance factor VF (the segment is split into 4 blocks along the rail; VF = observed scatter of the block means /
the scatter expected for independent points; VF ≫ 1 means correlated residuals or an unmodelled shape such as a
crowned head). σ_range_equiv = σ_pc / cos θ is the range noise that would produce σ_pc if all noise were along the
beam (compare with the specification). Per scanner and bin the pooled σ, power laws of σ_pc in r and cos θ.

### 04 Rail heights and σ_z (`04_rail_heights.py`)

The rail height of station i is **z_i = c0** of the segment plane: the height at the station and band centre, the
scanner analogue of a levelled point on the rail head. σ_z is estimated in two independent ways, both with the MAD:

| symbol (column) | definition | meaning |
|---|---|---|
| σ_z,MAD (`sigma_z_mad_mm`) | 1.4826 · MAD(plane residuals of all points) · √q00, q00 = [(AᵀA)⁻¹]₀₀ ≈ 1/n | point noise propagated into c0 |
| σ_z,station (`sigma_z_station_mm`) | σ_z,MAD · √max(VF, 1) | the same, corrected for correlated residuals; **lower bound** |
| σ_z,formal (`sigma_z_formal_mm`) | σ_pc · √q00 (least squares) | for comparison |
| σ_z,emp (`sigma_z_emp_mm`) | 1.4826 · MAD(e) over a group of stations | **empirical**; upper bound |

For σ_z,emp a quadratic in s is fitted to the heights within ±2 m (`--window 4`, `--degree 2`) without station i
and predicts ẑ_i. With equal σ_z, Var(z_i − ẑ_i) = σ_z² (1 + h_i), h_i = [(BᵀB)⁻¹]₀₀ of that fit, so
e_i = (z_i − ẑ_i)/√(1 + h_i) is one sample of the height error. σ_z,emp = 1.4826 · MAD(e) per scanner, rail, setup,
1 m distance bin and 5° incidence bin, with a 95 % interval (χ² with 0.37 n degrees of freedom, the efficiency of
the MAD); RMS(e) is given as cross-check. The same for the cross level L − R (scanner tilt cancels; σ_cross/√2).
σ_sys = √(σ_z,emp² − RMS(σ_z,station)²) is what the point noise does not explain (input of 07).

σ_z,emp contains the rail's own short-wave unevenness (welds, joints, wear), so it is an upper bound for the
scanner. Errors that are constant or vary smoothly along the track (levelling, vertical angle, range offset,
registration) are absorbed by the quadratic and only show up in 06 or against a reference.

Plots: `sigma_z_vs_distance.png` and `sigma_z_vs_incidence.png`: dots = σ_z,station of every height, line with
band = σ_z,emp per bin with 95 % interval, horizontal lines = σ_z,req for K = 2 and K = 3.605.
`<label>_height_residuals.png` shows e along the track.

### 05 Track geometry (`05_track_geometry.py`)

Both rails side by side on the station grid (missing stations stay empty, nothing is interpolated):

| parameter | formula | variance (uncorrelated heights) |
|---|---|---|
| cant | ü_i = b0/b (z_R,i − z_L,i), b0 = 1 m, b = distance of the band centres (measured; `--b 1` as in the report) | σ_ü² = (b0/b)² (σ_zL² + σ_zR²) |
| twist (base l = 5 m) | N_i = (ü_i − ü_i+10)/l in mm/m = ‰ | σ_N² = (σ_ü,i² + σ_ü,i+10²)/l² |
| vertical versine (chord 10 m) | vPf_i = H_i − (H_i−10 + H_i+10)/2, H = (z_L + z_R)/2 | σ_vPf² = σ_H,i² + (σ_H,i−10² + σ_H,i+10²)/4; σ_ΔvPf = √2 σ_vPf |

(indices in stations of 0.5 m). Each σ is propagated twice: with σ_z,station (lower bound) and with σ_z,emp (upper
bound). With equal σ_z this reduces to the report's σ_N = 2 b0 σ_z/(b l) and σ_ΔvPf = √(3/2) σ_z. Per station the
criterion K σ ≤ T is checked for K = 2 (interpretations 1 and 3) and K = 3.605 (interpretation 2) with the attention
values of `--speed-class` (default 48–60 km/h: T_vPf = 3 mm, T_N = 1 ‰, σ_z,req = 1.22 / 0.67 mm), and the measured
twist is compared with T_N. The design versine Pf_Rv is not subtracted; it cancels in the monitored change.

### 06 Compare scans (`06_compare_scans.py`)

For every pair of scans in the same results folder: differences of heights (mean offset removed: levelling,
registration), cant, twist and versine station by station. Their std (1.4826 MAD, and RMS) with 95 % interval
(Bartlett-corrected degrees of freedom for twist and versine, whose neighbours share heights), the formal std from
04, the σ_z implied by the scatter (Var(dz) = 2σ_z², Var(dvPf) = 3/2 σ_z², …) and σ_sys. Needs two scans.

### 07 Operational limits (`07_operational_limits.py`)

Per scanner: ln σ_z,station = a + b ln r and ln σ_z,station = a + b ln cos θ by least squares (SEs clustered by
setup with ≥ 2 setups); predicted σ_z = √(exp(a + b f)² + σ_sys²) with σ_sys from 04 (or `--sigma-sys`, e.g. from
06). r_max and θ_max = largest value with σ_z ≤ σ_z,req, with a bootstrap interval (over setups if ≥ 3, otherwise
over stations, flagged as too narrow) and a flag when the limit lies outside the observed range (extrapolation).

### 00 Validation (`00_simulate.py`)

Simulated railheads with known heights (30 m, two setups at 1.5 m height, angular step 0.6 mrad, range noise along
the beam, 1 % gross outliers, settlement trough of 4 mm, twist −0.4 ‰; illustrative values, not a real scanner),
run through 01–07 and compared with the truth. `python 00_simulate.py --mc 30`:

| check | result |
|---|---|
| height bias (two scans) | −0.001 / +0.001 mm (SE 0.004 / 0.003 mm) |
| twist | −0.398 / −0.400 ‰ (true −0.4 ‰), RMSE 0.011 / 0.012 ‰ |
| Monte Carlo, 30 runs, 103 stations: median height bias | −0.0002 mm |
| std(true height error) / σ_z,station (5 / 50 / 95 %) | 0.75 / 0.95 / 1.22 |
| std(true height error) / σ_z,MAD | 0.77 / 0.98 / 1.24 |
| σ_z,emp / 1.4826 MAD(true height error) per run | 0.84 / 1.01 / 1.22 |

Both MAD-based σ_z estimates are unbiased when their assumptions hold (independent noise, smooth rail).

## Results on the test clouds

RTC360, setup E004 on a bridge about 5 m above the rails (origin 6.268, 3.132, 3.140). Values from
`results/track-test/` and `results/track-test-2-classified/` (run of this commit).

| quantity | A: `track-test` (railhead clips) | B: `track-test-2-classified` (track at y ≈ 8–9 m, x 8.5–25.5 m) |
|---|---|---|
| band | railhead, 15 / 23 mm | groove, 30 mm beyond the groove |
| stations with both rails | 23 (12 m) | 31 (16.5 m) |
| distance / incidence | 7–10 m / 45–63° | 8–20 m / 51–76° |
| point density (median) | 13.3 pts/cm² | 2.4 pts/cm² |
| σ_pc pooled | 0.44 mm | 0.54 mm |
| σ_z,station (median) | 0.019 mm | 0.044 mm (95 % of stations < 0.22 mm) |
| σ_z,emp = 1.4826 MAD(e) | 0.079 mm (95 % CI 0.058–0.120, n = 43) | 0.196 mm (0.151–0.277, n = 60) |
| σ_sys | 0.075 mm | 0.163 mm |
| twist | −0.44 … +0.22 ‰ (13 values) | −0.28 … +0.77 ‰ (21 values) |
| criterion K σ ≤ T | met at every station, K = 2 and 3.605, either σ_z | same |
| r_max (extrapolated, 1 setup) | 37 m (K = 2), 29 m (K = 3.605) | 60 m, 44 m |

Points to keep in mind:

- σ_z stays clearly below σ_z,req = 0.67 mm up to 20 m in B. The limits of 07 come from one setup, are
  extrapolations, and must not be reported before the campaign with 5–7 setups.
- σ_z,emp is 3–4 times σ_z,station: the point noise does not explain the scatter of neighbouring heights; it is the
  rail's unevenness or an error that changes along the track. A second scan (06) separates the two.
- The heights in B look stored with a resolution of about 1 mm (horizontal bands in a cross-section; one station
  has σ_z,station = 0.001 mm). With σ_pc ≈ 0.5 mm this distorts σ_pc and the MAD of single segments. Ask for the
  classified LAZ with the class field and a coordinate scale of 0.1 mm or finer.
- In A the right rail head is crowned (about 1.5 mm over the band); the plane does not model this.

## Interpretation and choices to justify in the report

- **Distance and incidence are not independent.** On a horizontal railhead cos θ = h / r, so with one instrument
  height r and θ are almost collinear (`corr_range_incidence` in the `*_fit.json`: 0.89 in A, 0.96 in B). The two
  σ_z plots show the same trend twice. Vary the instrument height or the horizontal angle to separate them.
- **σ_pc falls with distance** at grazing incidence because less of the range noise is normal to the surface;
  σ_range_equiv = σ_pc / cos θ is the value to compare with the specification.
- **Stations are not independent samples.** Stations of one setup share instrument, atmosphere and registration.
  Report the number of setups (`n_setups`) with every estimate.
- **σ_z,station is a lower bound, σ_z,emp an upper bound.** Use σ_z,emp, or σ_sys from 06, for the criterion.
- Choices: segment length 0.5 m (= station spacing and height window), edge trim 5 mm, groove band 30 mm with
  5 mm margin, window ±2 m and degree 2 of the leave-one-out polynomial, bin widths 1 m / 5°, speed class
  48–60 km/h, b measured, test-track corridor 1.6 m.
