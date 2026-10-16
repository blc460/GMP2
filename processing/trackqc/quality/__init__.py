"""Point-cloud quality criteria on the rail running surface, as a function of
measurement distance and angle of incidence.

One module per criterion, each usable on its own:

    python -m trackqc.quality.density --cloud ../pointclouds/track-test.e57
    python -m trackqc.quality.noise   --cloud ../pointclouds/track-test.e57

or all criteria from one loading of the cloud:

    python -m trackqc.quality --cloud ../pointclouds/track-test.e57

Shared steps (loading, track axis, running band, segments, range and incidence)
are in `common.py`; a new criterion only needs a metric function per segment
and a call to `common.write_criterion`.
"""
