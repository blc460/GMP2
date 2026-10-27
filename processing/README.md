# processing

Point cloud quality of tram rails per 0.5 m segment: point density, point noise
sigma_z (MAD), rail height precision, distance and incidence angle, checked
against the quality criteria.

## Run

From the repository root:

```
python -m processing
```

Needs Python 3.11+ and the packages in `requirements.txt`.

## Input

- `config.toml`: all settings (segment length, strip width, trim, n_min, bins,
  criteria, p_pass, input mode).
- `datasets.csv`: one row per point cloud (path, scanner, setup, condition,
  platform). Leave the origin empty unless the scanner is not at the file's
  origin or E57 pose.

## Method

1. Find the rails in plan view (several tracks, switches and loose clipping
   are fine): rasterise the points (classified input: SOK and lip only), close
   small gaps, skeletonise, split the skeleton at junctions into rail pieces
   (R01, R02, ...; pieces facing each other across a gap are joined, pieces
   shorter than 2 m dropped). Every point gets station s and offset t from the
   nearest centreline. Cut each rail into 0.5 m segments; segments within 1 m
   of a junction are flagged.
2. Railhead-only input: per segment, the strip is the railhead next to the
   groove (the lowest band across the rail): the wider of the two raised runs
   beside it, minus the edge trim, at most 40 mm wide. Classified input: the
   SOK (or lip) class.
3. Fit a plane z = b0 + b1 a + b2 b per segment (a along, b across the rail),
   remove outliers (3 MAD), refit. sigma_z = 1.4826 MAD of the residuals;
   rail height precision = sigma_z / sqrt(n).
4. A segment passes if sigma_z / sqrt(n) <= sigma_req and n >= n_min.
5. Limit = farthest distance / most oblique incidence bin where at least
   p_pass of the segments pass.
6. Cant and twist every 5 m as in VBZ section 5.3; the partner rail is the
   segment 0.8-1.3 m across (not near a junction), each pair once.

## Output (`results/`)

- `segments.csv`: one row per segment
- `summary_*.csv`, `pass_share_*.csv`, `limits.csv`, `cant_twist.csv`
- `figures/`: six plots (png and pdf), and `fig0_rails_<dataset>.png`: detected
  rails and junctions
- `config_used.toml`, `run_info.json`: settings and versions of the run

sigma_z / sqrt(n) ignores correlated and systematic errors, so it is optimistic.
