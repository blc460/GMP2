"""Segments along each railhead with plane fit, range and incidence angle."""
from __future__ import annotations

import pandas as pd

from trackqc import core
from . import config


def segment_table(preps, metric, seg_len=config.SEG_LEN, keep=lambda seg: True) -> pd.DataFrame:
    """One row per fitted segment of every prepared scan that passes `keep`.

    Each segment (seg_len along the rail, full band width) gets a least-squares
    plane with 3 sigma (MAD) outlier rejection (`core.fit_plane`). Range =
    distance origin -> centroid of the inliers, incidence = angle between that
    beam and the plane normal (0 deg = perpendicular), horizontal angle =
    angle between beam and track in plan view (`core.segment_geometry`).
    `metric(seg) -> dict` adds the criterion columns.
    """
    rows = []
    for p in preps:
        for side in ("L", "R"):
            P, s, q = p.rails[side]
            for seg in core.band_segments(P, s, q, p.bands[side], seg_len):
                if seg.fit is None or not keep(seg):
                    continue
                rows.append(dict(scanner=p.scanner, setup=p.setup, label=p.label, rail=seg.rail,
                                 s_from=seg.s_from, s_to=seg.s_to, n_raw=seg.n_raw, n=seg.fit.n,
                                 **core.segment_geometry(seg, p.origin, p.axis), **metric(seg)))
    return pd.DataFrame(rows)
