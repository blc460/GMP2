"""Shared resources of the numbered railhead scripts (00_prepare.py, 01_..., 02_...).

    config.py    default settings, result folders, common command line options
    data.py      reading the clips (one dataset per scan), railhead band, prepared files
    segments.py  segments along each rail with plane fit, range and incidence angle
    stats.py     bin tables and fits per scanner, result files
    plots.py     metric vs range and vs incidence per scanner, railhead check plot

The numerical building blocks (plane fit, rail split, track axis, segment
geometry, density and noise per segment) are reused from `trackqc.core`.
"""
import sys
from pathlib import Path

# make `trackqc` (in processing/) importable when a numbered script is run directly
_PROCESSING = Path(__file__).resolve().parents[2]
if str(_PROCESSING) not in sys.path:
    sys.path.insert(0, str(_PROCESSING))
