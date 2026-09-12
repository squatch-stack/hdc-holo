"""Position bundle payload bytes against SPZ using recorded sweep counts.

Prior art abstracts: arXiv:2508.10227 (EntropyGS), arXiv:2502.19457,
arXiv:2512.07197 (compression surveys). This is Squatch Stack arithmetic,
not a reproduction of their codecs or rate-distortion definitions.
SPZ's 22 B/splat comes from results/baseline_table.md. MB means 1,000,000 B.
Payload only: one vector per cell, excluding headers, codebooks, attributes,
and replication state. HG-8 and the D2 knee coincide in bytes, not error.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

BANDS = ("xfine", "fine", "mid", "coarse")
RATES = {"complex64": 65536, "hg8": 16384, "knee": 16384}


def required(record, key, context):
    """Fetch a required field without guessing missing counts."""
    if key not in record:
        raise ValueError("%s: missing key %s" % (context, key))
    return record[key]


def count(record, key, context, minimum=0):
    """Require a JSON integer count in its valid range."""
    value = required(record, key, context)
    if type(value) is not int or value < minimum:
        raise ValueError("%s: invalid count %s" % (context, key))
    return value


def compute_rows(sweep):
    """Use splats_encoded and cells_per_band; verify the cells total."""
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("sweep must be a nonempty list")
    rows = []
    seen = set()
    for index, source in enumerate(sweep):
        context = "row %d" % index
        if not isinstance(source, dict):
            raise ValueError("%s must be an object" % context)
        scene = required(source, "scene", context)
        if not isinstance(scene, str) or not scene or scene in seen:
            raise ValueError("%s: invalid or duplicate scene" % context)
        seen.add(scene)
        splats = count(source, "splats_encoded", scene, minimum=1)
        bands = required(source, "cells_per_band", scene)
        if not isinstance(bands, dict) or set(bands) - set(BANDS):
            raise ValueError("%s: invalid cells_per_band" % scene)
        cells = sum(count(bands, band, scene + '.cells_per_band') for band in BANDS)
        if cells != count(source, "cells", scene, minimum=1):
            raise ValueError("%s: cells disagrees with cells_per_band" % scene)
        row = {"capture": scene, "splats": splats, "cells": cells,
               "cells_per_band": dict(bands), "spz_bytes": splats * 22}
        for name, rate in RATES.items():
            row[name + "_bytes"] = cells * rate
            row[name + "_ratio"] = cells * rate / row["spz_bytes"]
        rows.append(row)
    return rows


def read_sweep(path):
    """Read and validate a sweep JSON file."""
    return compute_rows(json.loads(Path(path).read_text()))


def synthetic_sweep(seed=7):
    """Produce seeded count fixtures, without requiring capture assets."""
    rng = random.Random(seed)
    rows = []
    for index in range(3):
        bands = {band: rng.randint(1, 100) for band in BANDS}
        rows.append({"scene": "synthetic-%d" % index,
                     "splats_encoded": rng.randint(1000, 10000),
                     "cells_per_band": bands, "cells": sum(bands.values())})
    return rows


def markdown(rows):
    """Render one row per capture and the unweighted median knee ratio."""
    lines = [
        "| capture | splats | cells | SPZ MB | complex64 MB | HG-8 MB | "
        "4/4 knee MB | knee/SPZ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = row["capture"].replace("|", "&#124;").replace("\n", " ")
        name = name.replace("\r", " ")
        sizes = [row[key + "_bytes"] / 1_000_000
                 for key in ("spz", "complex64", "hg8", "knee")]
        lines.append(
            "| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2fx |"
            % (name, row["splats"], row["cells"], *sizes, row["knee_ratio"])
        )
    lines.extend(["", "Median ratio at the knee: %.2fx SPZ."
                  % statistics.median(row["knee_ratio"] for row in rows)])
    return "\n".join(lines)


def main(argv=None):
    """Print markdown, optionally writing exact byte counts and ratios as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sweep", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, help="output path for computed rows")
    args = parser.parse_args(argv)
    try:
        rows = (compute_rows(synthetic_sweep(args.seed)) if args.synthetic
                else read_sweep(args.sweep))
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(markdown(rows))


if __name__ == "__main__":
    main()
