"""Object semantic memory: preregistration written before implementation.

Gate 1: what_is_at accuracy >= 90% of the exact table, with strictly fewer
serialized bytes; report the first tested crossover N, or explicitly call
this a "queryable by algebra" wedge and not a "less memory" wedge.
Gate 2: measured accuracy knee within a factor of 2 of N_pred = 2*d/R,
where sigma = sqrt(N*R/(2*d)) reaches 1. Define the knee BEFORE measurement
as the first tested N with accuracy <= 50%, using four balanced classes,
complex64 codewords and object-centre queries. Report censored knees.
Gate 3: at target N=512, d=256, R=1, frequency-encoded continuous payloads
must separate matching/nonmatching scores by >= 2 sigma of the measured
nonmatching crosstalk floor; otherwise choose discrete codewords.

Fixed design: seed 17, four classes, dimensions 128/256/512, powers of two
N=8..2048. Continuous proxies are seeded unit 32-D embeddings encoded with
unit-bandwidth Gaussian frequencies, not CLIP or a claim about CLIP.
Definitions and thresholds are ours. No cited arXiv definitions are used.
Queries concern labels at supplied object centres; extent reconstruction,
perception and free-space detection are outside this experiment. where_is
requires an external candidate position set, whose bytes are query input,
not retained scene memory. Latencies exclude encoding and include decoding
serialized stores and regenerating fixed codebooks/frequencies each call.
"""

import argparse
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from bench import labelled_scene
from holo.attribute_field import AttributeSplatField
from holo.capture import load_scene_file
from holo.fhrr import FHRR
from holo.phase import pack_polar, unpack
from holo.record import RecordSpace

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBJECT = np.dtype([("position", "<f4", (3,)), ("extent", "<f4", (3,)),
                   ("label", "<u2")])
# magic, version, d, seed, class count, field count, sigma, encoding, precision
HEADER = struct.Struct("<2sBIIHHfBB")
TABLE_HEADER = struct.Struct("<2sBI")
SEED = 17
CLASSES = 4
SIGMA = 0.01


@dataclass(frozen=True)
class Store:
    blob: bytes

    def serialize(self):
        return self.blob


def object_scene(n, rng, classes=CLASSES, extent_range=(0.005, 0.01)):
    """Separated 3-D object boxes, balanced labels, randomized assignment."""
    objects = np.empty(n, dtype=OBJECT)
    side = int(np.ceil(n ** (1 / 3)))
    grid = np.indices((side, side, side)).reshape(3, -1).T[:n]
    objects["position"] = grid * 0.1 + rng.uniform(-0.005, 0.005, (n, 3))
    objects["extent"] = rng.uniform(*extent_range, size=(n, 3))
    objects["label"] = rng.permutation(np.arange(n) % classes)
    return objects


def capture_objects(path, n, rng):
    """Occupied 3-cm voxels are region proxies, never labelled splats.

    Sample regions once; every selected region represents all splats in its
    voxel. Labels are assigned, not inferred. Extents are voxel half-widths.
    """
    positions = load_scene_file(path)[0]
    cells = np.unique(np.floor(positions / 0.03).astype(np.int32), axis=0)
    if n > len(cells):
        raise ValueError("not enough occupied capture regions")
    chosen = cells[rng.choice(len(cells), n, replace=False)]
    objects = np.empty(n, dtype=OBJECT)
    objects["position"] = (chosen + 0.5) * 0.03
    objects["extent"] = 0.015
    objects["label"] = rng.permutation(np.arange(n) % CLASSES)
    return objects


def payloads(space, classes, fields=1, continuous=False):
    if continuous:
        rng = np.random.default_rng(space.seed + 101)
        embeddings = rng.normal(size=(classes, 32))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        frequencies = rng.normal(size=(32, space.dim))
        return np.exp(1j * (embeddings @ frequencies)).astype(np.complex64)
    if fields == 1:
        return np.stack([space.label_vector(str(k)) for k in range(classes)])
    records = RecordSpace(space)
    return np.stack([records.encode({f"role{r}": f"value{r}:{k}"
                                    for r in range(fields)})
                     for k in range(classes)])


def encode_hologram(objects, d, sigma, space, fields=1, continuous=False,
                    classes=CLASSES):
    if space.dim != d:
        raise ValueError("space dimension must equal d")
    field = AttributeSplatField(space, np.eye(3) * sigma ** 2)
    codes = payloads(space, classes, fields, continuous)
    for obj in objects:
        field.add_splat(obj["position"], codes[int(obj["label"])])
    header = HEADER.pack(b"SM", 1, d, space.seed, classes, fields,
                         sigma, continuous, 0)
    store = Store(header + field.S.astype("<c8").tobytes())
    return store, len(store.serialize())


def encode_table(objects):
    objects = np.asarray(objects, dtype=OBJECT)
    store = Store(TABLE_HEADER.pack(b"ST", 1, len(objects)) + objects.tobytes())
    return store, len(store.serialize())


def decode_hologram(store):
    _, _, d, seed, classes, fields, sigma, continuous, precision = (
        HEADER.unpack_from(store.blob))
    space = FHRR(d, seed=seed)
    field = AttributeSplatField(space, np.eye(3) * sigma ** 2)
    data = store.blob[HEADER.size:]
    field.S = unpack(data) if precision else np.frombuffer(data, dtype="<c8")
    return field, payloads(space, classes, fields, continuous), fields


def four_bit(store):
    field, _, _ = decode_hologram(store)
    values = list(HEADER.unpack_from(store.blob))
    values[-1] = 4
    return Store(HEADER.pack(*values) + pack_polar(field.S, bits=4))


def table_objects(store):
    return np.frombuffer(store.blob, dtype=OBJECT, offset=TABLE_HEADER.size)


def query_scores(store, points):
    field, codes, fields = decode_hologram(store)
    if fields > 1:
        records = RecordSpace(field.space)
        codes = np.stack([FHRR.bind(records.roles.get("role0"),
                                   records.fillers.get(f"value0:{k}"))
                          for k in range(len(codes))])
    channels = np.stack([field.where_query(code) for code in codes])
    return field.eval_positions(channels, points)


def query_what_is_at(store, points):
    """Closed-set centre query; table uses exact nearest-centre lookup."""
    points = np.asarray(points, dtype=np.float32)
    if store.blob[:2] == b"ST":
        objects = table_objects(store)
        labels = []
        for start in range(0, len(points), 128):
            delta = points[start:start + 128, None] - objects["position"]
            nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
            labels.extend(objects["label"][nearest])
        return np.asarray(labels, dtype=np.uint16)
    return np.argmax(query_scores(store, points), axis=1).astype(np.uint16)


def query_where_is(store, label, points=None):
    """Filter supplied candidate centres; no hidden position list in SM.

    Table can additionally enumerate exact stored positions without a grid.
    SM returns approximate class membership, not exact continuous peaks.
    """
    if store.blob[:2] == b"ST" and points is None:
        objects = table_objects(store)
        return objects["position"][objects["label"] == label]
    if points is None:
        raise ValueError("hologram where_is needs external candidate positions")
    return np.asarray(points)[query_what_is_at(store, points) == label]


def predicted_knee(d, fields=1):
    return 2 * d / fields


def timed_query(store, points):
    query_what_is_at(store, points)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        labels = query_what_is_at(store, points)
        samples.append((time.perf_counter() - start) * 1e6 / len(points))
    return labels, float(np.median(samples))


def measure(objects, d, classes=CLASSES):
    holo, _ = encode_hologram(objects, d, SIGMA, FHRR(d, seed=SEED),
                              classes=classes)
    table, _ = encode_table(objects)
    row = {"n": len(objects), "d": d,
           "sigma": float(np.sqrt(len(objects) / (2 * d)))}
    for name, store in (("table", table), ("complex64", holo),
                        ("four_bit", four_bit(holo))):
        labels, latency = timed_query(store, objects["position"])
        row[name] = {"accuracy": float(np.mean(labels == objects["label"])),
                     "bytes": len(store.serialize()), "us_query": latency}
    return row


def sweep(n_values, d_values, capture=None):
    rows = []
    for n in n_values:
        rng = np.random.default_rng(SEED)
        objects = (capture_objects(capture, n, rng) if capture else
                   object_scene(n, rng))
        for d in d_values:
            row = measure(objects, d)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def encoding_fork(capture=None):
    rng = np.random.default_rng(SEED)
    objects = (capture_objects(capture, 512, rng) if capture else
               object_scene(512, rng))
    store, _ = encode_hologram(objects, 256, SIGMA, FHRR(256, seed=SEED),
                              continuous=True)
    scores = query_scores(store, objects["position"])
    mask = np.arange(CLASSES)[None, :] == objects["label"][:, None]
    matching, nonmatching = scores[mask], scores[~mask]
    floor = float(nonmatching.std(ddof=1))
    separation = float(matching.mean() - nonmatching.mean())
    return {"n": 512, "d": 256, "matching_mean": float(matching.mean()),
            "nonmatching_mean": float(nonmatching.mean()),
            "floor": floor, "separation": separation,
            "separation_sigma": separation / floor,
            "selected": "frequency" if separation >= 2 * floor else "codewords"}


def summarize(rows):
    crossovers = {}
    knees = {}
    for mode in ("complex64", "four_bit"):
        eligible = [r for r in rows
                    if r[mode]["accuracy"] >= 0.9 * r["table"]["accuracy"]
                    and r[mode]["bytes"] < r["table"]["bytes"]]
        crossovers[mode] = min((r["n"] for r in eligible), default=None)
    for d in sorted({r["d"] for r in rows}):
        failed = [r["n"] for r in rows
                  if r["d"] == d and r["complex64"]["accuracy"] <= 0.5]
        knee = min(failed, default=None)
        predicted = predicted_knee(d)
        knees[d] = {"measured": knee, "predicted": predicted,
                    "ratio": knee / predicted if knee else None}
    return {"crossovers": crossovers, "knees": knees}


def plot_rows(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d in sorted({r["d"] for r in rows}):
        selected = [r for r in rows if r["d"] == d]
        for mode, style in (("complex64", "-"), ("four_bit", "--")):
            axes[0].plot([r["n"] for r in selected],
                         [r[mode]["accuracy"] for r in selected],
                         style, marker=".", label=f"d={d} {mode}")
            axes[1].plot([r["n"] for r in selected],
                         [r[mode]["bytes"] for r in selected], style)
    selected = [r for r in rows if r["d"] == rows[0]["d"]]
    axes[0].axhline(0.9, color="black", linestyle=":", label="90% gate")
    axes[1].plot([r["n"] for r in selected],
                 [r["table"]["bytes"] for r in selected], "k", label="table")
    for ax, ylabel in zip(axes, ("Centre-label accuracy", "Serialized bytes")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Objects N")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    axes[1].set_yscale("log")
    fig.suptitle("Assigned labels: capacity and storage (not perception)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--capture", type=Path)
    source.add_argument("--scene", type=Path)
    parser.add_argument("--label-map", type=Path)
    parser.add_argument("--objects", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    if args.label_map and not args.scene:
        parser.error("--label-map requires --scene")
    if args.scene:
        if args.figure:
            parser.error("--figure is only supported by the assigned-label sweep")
        try:
            labelled_scene.run_scene(args.scene, args.label_map, args.dims)
        except ValueError as exc:
            parser.error(str(exc))
        return
    rows = sweep(args.objects, args.dims, args.capture)
    print(json.dumps({"summary": summarize(rows),
                      "encoding_fork": encoding_fork(args.capture)}), flush=True)
    if args.figure:
        plot_rows(rows, args.figure)


if __name__ == "__main__":
    main()
