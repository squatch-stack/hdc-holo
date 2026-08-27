"""Threshold calibration: where do real restatements score vs noise?

Two populations against the built index:
  signal — every current claim's rendered statement, scored against
           the best chunk from its own cite files (a restatement the
           index is supposed to find);
  noise  — the same statements with their CHARACTERS deterministically
           scrambled (seeded), scored against their best chunk
           anywhere. Character scrambling, not word shuffling: trigram
           profiles largely ignore word order (the documented
           dispatch-lane property), so shuffled words keep almost the
           same trigram multiset and score as signal — measured here
           before this docstring said so.

The advice line places the threshold at the noise p95 with the
observed gap; docs/facts.md records the measured values.
"""

import os

import numpy as np

from . import index as indexmod
from .check import load_config
from .registry import load_registry

__all__ = ["main"]


def main(root):
    config = load_config(root)
    claims = load_registry(os.path.join(root, "claims", "registry.jsonl"))
    prof = indexmod.profiler()
    meta, mat = indexmod.load_index(root)
    files = [c["file"] for c in meta["chunks"]]

    rng = np.random.default_rng(0)
    signal, noise = [], []
    for claim in claims:
        if claim.status != "current" or not claim.statement:
            continue
        probe = claim.statement.replace("{value}", str(claim.value))
        q = prof.unit_profile(probe)
        scores = np.real(mat.conj() @ q)
        cite_rows = [i for i, f in enumerate(files) if f in claim.cites]
        if cite_rows:
            signal.append((claim.id, float(scores[cite_rows].max())))
        chars = np.array(list(probe.lower().replace(" ", "")))
        rng.shuffle(chars)
        qn = prof.unit_profile("".join(chars))
        noise.append(float(np.real(mat.conj() @ qn).max()))

    if not signal:
        print("no current claims with cites — nothing to calibrate")
        return 2
    sig = np.array([s for _, s in signal])
    noi = np.array(noise)
    print("signal (best own-cite chunk per claim):")
    print("  min %.3f   median %.3f   max %.3f" %
          (sig.min(), np.median(sig), sig.max()))
    for cid, s in sorted(signal, key=lambda t: t[1])[:3]:
        print("    weakest: %-28s %.3f" % (cid, s))
    print("noise (shuffled statements, best chunk anywhere):")
    print("  median %.3f   p95 %.3f   max %.3f" %
          (np.median(noi), np.percentile(noi, 95), noi.max()))
    p95 = float(np.percentile(noi, 95))
    current = config.get("fuzzy_threshold", 0.18)
    print("threshold advice: noise p95 = %.3f; config fuzzy_threshold "
          "= %.2f (%s)" %
          (p95, current,
           "ok" if current >= p95 else "RAISE — below the noise floor"))
    return 0
