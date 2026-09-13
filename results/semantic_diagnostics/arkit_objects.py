#!/usr/bin/env python3
"""Object counts and the byte comparison on real ARKitScenes annotations.

Fetches only the 3D object-detection annotation JSONs -- a few kilobytes
each -- because the semantic-memory experiment's unit is an object, not
geometry. The full 3DOD download is 623 GB and none of it is needed to
answer the question this asks.

    python results/semantic_diagnostics/arkit_objects.py --scans 600

Licence: the annotations are Apple's (ARKitScenes LICENSE, a conditional
commercial grant). They are fetched at run time and deliberately NOT
vendored into this repository.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = ("https://docs-assets.developer.apple.com/ml-research/datasets"
        "/arkitscenes/v1/raw")
SPLITS = ("https://raw.githubusercontent.com/apple/ARKitScenes/main"
          "/threedod/3dod_train_val_splits.csv")
# The repository's own downloader lists these as having no 3DOD assets.
MISSING = {"47334522", "47334523", "42897421", "45261582", "47333152",
           "47333155"}


def fetch(cache, scans):
    os.makedirs(cache, exist_ok=True)
    with urllib.request.urlopen(SPLITS, timeout=60) as f:
        rows = [r for r in csv.DictReader(f.read().decode().splitlines())
                if r["video_id"] not in MISSING][:scans]

    def one(r):
        vid = r["video_id"]
        split = "Training" if r["fold"] == "Training" else "Validation"
        dst = os.path.join(cache, vid + ".json")
        if os.path.exists(dst):
            return True
        try:
            url = f"{BASE}/{split}/{vid}/{vid}_3dod_annotation.json"
            with urllib.request.urlopen(url, timeout=60) as f:
                payload = f.read()
            with open(dst, "wb") as out:
                out.write(payload)
            return True
        except OSError:
            return False

    with ThreadPoolExecutor(max_workers=16) as ex:
        return sum(ex.map(one, rows))


def objects(path, classes, dtype):
    """One scan's boxes, normalized into a unit cube.

    `segments.obb` is the field the released files actually carry;
    `obbAligned` sits beside it. Both stores receive this one array, so
    the normalization cannot favour either.
    """
    rows = []
    with open(path) as handle:
        annotation = json.load(handle)
    for o in annotation.get("data", []):
        box = o.get("segments", {}).get("obb") or {}
        centre, sizes = box.get("centroid"), box.get("axesLengths")
        if centre and sizes:
            rows.append((o.get("label", "?"), np.array(centre, float),
                         np.array(sizes, float)))
    if not rows:
        return None
    pos = np.stack([r[1] for r in rows])
    ext = np.stack([r[2] for r in rows])
    lo = pos.min(axis=0)
    span = float((pos.max(axis=0) - lo).max()) or 1.0
    out = np.empty(len(rows), dtype=dtype)
    out["position"] = ((pos - lo) / span * 0.9 + 0.05).astype(np.float32)
    out["extent"] = (ext / span / 2).astype(np.float32)
    out["label"] = [classes.index(r[0]) for r in rows]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=int, default=600)
    ap.add_argument("--cache", default="/tmp/arkit-annotations")
    ap.add_argument("--dims", default="32,64,128,256")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    from bench import semantic_memory as sm

    got = fetch(args.cache, args.scans)
    files = sorted(f for f in os.listdir(args.cache) if f.endswith(".json"))
    classes = set()
    for name in files:
        with open(os.path.join(args.cache, name)) as handle:
            for o in json.load(handle).get("data", []):
                classes.add(o.get("label", "?"))
    classes = sorted(classes)
    scans = [s for s in (objects(os.path.join(args.cache, f), classes,
                                 sm.OBJECT) for f in files) if s is not None]
    counts = sorted(len(s) for s in scans)
    n = len(counts)
    print(f"fetched {got} annotation files; {n} carry boxes; "
          f"{len(classes)} classes")
    print(f"objects per scan: min {counts[0]} p25 {counts[n // 4]} "
          f"median {counts[n // 2]} p75 {counts[3 * n // 4]} "
          f"p95 {counts[int(n * 0.95)]} max {counts[-1]}")
    for t in (32, 64, 128):
        print(f"  scans with >= {t:3d} objects: "
              f"{sum(1 for c in counts if c >= t)} / {n}")
    print()
    head = (f"{'d':>5} {'4bit acc':>9} {'4bit B':>7} {'table B':>9} "
            f"{'ratio':>7} {'wins@90%':>10} {'c64 B':>7} {'c64 wins':>9}")
    print(head)
    for d in (int(x) for x in args.dims.split(",")):
        rows = [sm.measure(o, d, classes=len(classes)) for o in scans]
        acc = statistics.mean(r["four_bit"]["accuracy"] for r in rows) * 100
        four = rows[0]["four_bit"]["bytes"]
        c64 = rows[0]["complex64"]["bytes"]
        table = statistics.median(r["table"]["bytes"] for r in rows)
        wins = sum(1 for r in rows
                   if r["four_bit"]["bytes"] < r["table"]["bytes"]
                   and r["four_bit"]["accuracy"] >= 0.9)
        cwin = sum(1 for r in rows
                   if r["complex64"]["bytes"] < r["table"]["bytes"]
                   and r["complex64"]["accuracy"] >= 0.9)
        print(f"{d:>5} {acc:>8.1f}% {four:>7} {int(table):>9} "
              f"{table / four:>6.2f}x {wins:>7}/{n} {c64:>7} {cwin:>6}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
