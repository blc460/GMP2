"""Functions shared by the numbered processing scripts (00_simulate.py, 01_prepare.py, ...).

    config.py    default settings, requirements (VBZ attention values), result folders, common options
    io.py        reading point clouds, prepared scans of step 01, provenance of every result
    geometry.py  rail split, track axis and stationing, railhead band (or running band beside the groove)
    fit.py       plane fit per segment, robust scale (1.4826 MAD) and its CI, segment table
    stats.py     bin tables, linear fits, coverage, result files
    plots.py     all plots
"""
