#!/usr/bin/env python3
"""What a richer object vocabulary costs, and ReplicaCAD's object counts.

Two measurements the ARKitScenes audit raised but could not answer.

The first is a second real object-count distribution, from ReplicaCAD's
scene configs -- 91 apartment layouts whose object instances are listed
as JSON, about a megabyte in total. It is furniture-scale like
ARKitScenes but has 106 distinct object templates rather than 19.

The second is controlled, and it corrects a conditional this project
wrote on 2026-09-13: that the memory advantage "exists only where the
object vocabulary is open and dense". Holding the object count fixed
and varying only the number of classes shows the opposite. A richer
vocabulary costs dimension, because the capacity law predicts the noise
a readout carries but not the decision that noise has to survive, and
that decision is between more alternatives when the vocabulary is
larger.

    python results/semantic_diagnostics/vocabulary_cost.py
"""
import argparse
import json
import os
import statistics
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

API = "https://huggingface.co/api/datasets/ai-habitat/ReplicaCAD_dataset"
FILES = "https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset/resolve/main"


def replicacad(cache):
    """Scene-config object instances. Ungated; about 1 MB of JSON."""
    os.makedirs(cache, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as f:
        meta = json.load(f)
    scenes = [s["rfilename"] for s in meta["siblings"]
              if "scene_instance" in s["rfilename"]]

    def one(rf):
        dst = os.path.join(cache, os.path.basename(rf))
        if not os.path.exists(dst):
            try:
                with urllib.request.urlopen(f"{FILES}/{rf}", timeout=60) as f:
                    payload = f.read()
                with open(dst, "wb") as out:
                    out.write(payload)
            except OSError:
                return None
        return dst

    with ThreadPoolExecutor(max_workers=12) as ex:
        return [p for p in ex.map(one, scenes) if p]


def scene_objects(paths, dtype):
    templates, raw = set(), []
    for p in paths:
        with open(p) as handle:
            spec = json.load(handle)
        objs = [o for o in spec.get("object_instances", []) if o.get("translation")]
        if len(objs) < 4:
            continue
        raw.append(objs)
        templates |= {o.get("template_name", "?") for o in objs}
    templates = sorted(templates)
    out = []
    for objs in raw:
        pos = np.array([o["translation"] for o in objs], dtype=float)
        lo = pos.min(axis=0)
        span = float((pos.max(axis=0) - lo).max()) or 1.0
        a = np.empty(len(objs), dtype=dtype)
        a["position"] = ((pos - lo) / span * 0.9 + 0.05).astype(np.float32)
        # ReplicaCAD carries the extent in the object asset, not the scene
        # config. A nominal extent is honest here because the query is at a
        # position and never reads the extent; it costs the same bytes in
        # both stores either way.
        a["extent"] = np.float32(0.01)
        a["label"] = [templates.index(o.get("template_name", "?")) for o in objs]
        out.append(a)
    return out, templates


def smallest_dimension(sm, n, classes, seeds, target, grid):
    """The smallest d whose mean accuracy over `seeds` reaches `target`."""
    for d in grid:
        scores = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            a = np.empty(n, dtype=sm.OBJECT)
            a["position"] = rng.uniform(0.05, 0.95, (n, 3)).astype(np.float32)
            a["extent"] = np.float32(0.01)
            a["label"] = rng.integers(0, classes, n)
            scores.append(sm.measure(a, d, classes=classes)["four_bit"]["accuracy"])
        if statistics.mean(scores) >= target:
            return d
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/replicacad-scenes")
    ap.add_argument("--objects", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    from bench import semantic_memory as sm

    scans, templates = scene_objects(replicacad(args.cache), sm.OBJECT)
    counts = sorted(len(s) for s in scans)
    n = len(counts)
    print(f"ReplicaCAD: {n} scenes, {len(templates)} object templates")
    print(f"objects per scene: min {counts[0]} median {counts[n // 2]} "
          f"max {counts[-1]}; >= 64: {sum(1 for c in counts if c >= 64)}")
    print()
    print(f"Smallest d reaching 90% over {args.seeds} seeds at "
          f"{args.objects} objects:")
    print(f"{'classes':>8} {'d*':>7} {'sigma at d*':>12} {'cost vs k=2':>12}")
    grid = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    base = None
    for k in (2, 4, 8, 19, 48, 106):
        d = smallest_dimension(sm, args.objects, k, args.seeds, 0.90, grid)
        base = base or d
        sigma = (args.objects / (2 * d)) ** 0.5 if d else float("nan")
        ratio = f"{d / base:.1f}x" if d else "n/a"
        print(f"{k:>8} {d if d else '>8192':>7} {sigma:>12.3f} {ratio:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
