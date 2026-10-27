"""Command line: python -m processing [segments | analyse | run] [--config path]

segments  read every dataset and write the segment table (results/segments.csv)
analyse   summaries, limits, cant/twist and figures from an existing segment table
run       both (default)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from . import analysis, plots
from .io import load_config, read_datasets
from .segments import segment_table

DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


def run_segments(cfg):
    out = cfg["results"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tables, inputs = [], []
    for ds in read_datasets(cfg):
        print(f"{ds.label}: {ds.path.name} ({ds.mode})")
        t = segment_table(ds, cfg, out / "figures")
        print(f"  {t['rail'].nunique()} rails, {len(t)} segments, {t['flagged'].mean():.0%} flagged, "
              f"median sigma_z {t['sigma_z_mm'].median():.2f} mm, "
              f"d {t['d_m'].min():.1f}-{t['d_m'].max():.1f} m, theta {t['theta_deg'].min():.0f}-{t['theta_deg'].max():.0f} deg")
        tables.append(t)
        inputs.append(dict(path=str(ds.path), sha256=_sha256(ds.path)))
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out / "segments.csv", index=False)
    _provenance(cfg, "segments", inputs)
    return df


def run_analyse(cfg):
    out = cfg["results"]
    df = pd.read_csv(out / "segments.csv")
    for name, table in analysis.summaries(df, cfg).items():
        table.to_csv(out / f"{name}.csv", index=False)
    for by in ("d_bin", "theta_bin"):
        analysis.pass_shares(df, cfg, by).to_csv(out / f"pass_share_{by.split('_')[0]}.csv", index=False)
    lim = analysis.limits(df, cfg)
    lim.to_csv(out / "limits.csv", index=False)
    analysis.geometry(df, cfg).to_csv(out / "cant_twist.csv", index=False)
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    plots.make_all(df, lim, cfg, fig)
    _provenance(cfg, "analyse", [])
    print(f"results in {out}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _provenance(cfg, step, inputs):
    """Config copy and run_info.json: what is needed to reproduce the results."""
    out = cfg["results"]
    shutil.copy(cfg["_path"], out / "config_used.toml")
    shutil.copy(cfg["datasets"], out / "datasets_used.csv")
    here = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    import matplotlib
    info_path = out / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info[step] = dict(run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit=commit,
                      uncommitted_changes=dirty, python=sys.version.split()[0], platform=platform.platform(),
                      packages=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    if inputs:
        info[step]["inputs"] = inputs
    info_path.write_text(json.dumps(info, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="processing", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["segments", "analyse", "run"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="settings file (default: %(default)s)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.command in ("segments", "run"):
        run_segments(cfg)
    if a.command in ("analyse", "run"):
        run_analyse(cfg)


if __name__ == "__main__":
    main()
