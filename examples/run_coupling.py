"""Orthogonal frequency coupling as an instrument for issue #3.

Orthogonal random features (Yu et al. 2016) reduce the VARIANCE of the
kernel estimate without changing anything else — same d, same bytes,
same decode path, only a different draw of W. That makes them a probe
as much as an optimisation: if a scene's decode error is dominated by
Monte-Carlo variance, coupling should remove a large slice of it; if
the error is coherent, coupling cannot touch it.

Running this on a sparse and a dense capture separates the two, which
is what issue #3 needed and what doubling `d` could only hint at.
Shrinkage (`holo/denoise.py`, issue #1) is measured alongside, because
the interaction between them is the informative part: where both work
on the same error they compete, and where they attack different error
they add.

Usage: python -m examples.run_coupling [data/scan-tucson.spz ...]

Run it as a MODULE, not by path — from a worktree, `python
examples/run_coupling.py` puts sys.path[0] at examples/ and silently
imports the shared checkout instead of the tree you are testing.
"""

import os
import sys
import time

import numpy as np

from holo import budget
from holo.capture import (
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    mass_mode,
    slice_grid,
)
from holo.demokit import Table, banner
from holo.denoise import percentile_threshold, shrink

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = [os.path.join(ROOT, "data", "scan-tucson.spz"),
            os.path.join(ROOT, "data", "train.splat")]
PCT = 25


def _slices(scene, box):
    w = scene.amp[:, 0]
    return [("top-down", slice_grid((0, box[0]), (0, box[2]), "y",
                                    mass_mode(scene.mu[:, 1], w, box[1]))),
            ("side", slice_grid((0, box[2]), (0, box[1]), "x",
                                mass_mode(scene.mu[:, 0], w, box[0])))]


def measure(path):
    scene, smax, box = build_scene(path, verbose=False)
    slices = _slices(scene, box)
    out = {}
    for coupling in ("iid", "orthogonal"):
        books = band_codebooks(np.random.default_rng(42), coupling=coupling)
        bundles, members = encode_bands(scene, smax, books, verbose=False)
        truth = {n: exact_slice(pts, scene, members)
                 for n, (pts, _) in slices}

        def err(bs, truth=truth, books=books):
            return [float(np.linalg.norm(decode_slice(pts, bs, books)[:, 0]
                                         - truth[n][:, 0])
                          / np.linalg.norm(truth[n][:, 0]))
                    for n, (pts, _) in slices]

        out[coupling] = err(bundles)
        out[coupling + "+shrink"] = err(
            {b: {k: shrink(v, percentile_threshold(v, PCT))
                 for k, v in cells.items()} for b, cells in bundles.items()})
    return len(scene.mu), out


def main(paths):
    t0 = time.time()
    for path in paths:
        n, r = measure(path)
        banner("%s — %s splats" % (os.path.basename(path), f"{n:,}"))
        t = Table(("configuration", 20), ("top-down", 10, ".4f"),
                  ("side", 8, ".4f"), indent="  ")
        t.header()
        for key in ("iid", "iid+shrink", "orthogonal", "orthogonal+shrink"):
            t.row(key, r[key][0], r[key][1])
        a, b = r["iid"], r["orthogonal"]
        gain = lambda x, y, i: 100 * (x[i] - y[i]) / x[i]   # noqa: E731
        print("    coupling gain        %+.1f%% / %+.1f%%"
              % (gain(a, b, 0), gain(a, b, 1)))
        print("    shrink gain (iid)    %+.1f%% / %+.1f%%"
              % (gain(a, r["iid+shrink"], 0), gain(a, r["iid+shrink"], 1)))
        print("    shrink gain (orth)   %+.1f%% / %+.1f%%"
              % (gain(b, r["orthogonal+shrink"], 0),
                 gain(b, r["orthogonal+shrink"], 1)))
    print("\ntotal %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    with budget.heavy_run(6.0, "coupling", "--force-memory" in sys.argv):
        main([a for a in sys.argv[1:] if not a.startswith("--")]
             or DEFAULTS)
