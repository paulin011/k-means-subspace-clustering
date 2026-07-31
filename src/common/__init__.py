"""Shared library imported by every role directory under `src/`.

`cluster_io` (sampling/IO + the model.pt/assignments.pt schema) and `worldmap`
(HEALPix geometry, cluster coloring, map rendering) are used across clustering
and analysis alike, so they live here rather than in either one.

Scripts reach this package by putting `src/` on `sys.path` themselves:

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.cluster_io import load_tokens

which keeps the plain `python3 src/<role>/<script>.py` invocation (run from the
repo root) working with no package machinery or PYTHONPATH.
"""
