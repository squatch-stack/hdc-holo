"""Capture replication: state size versus contributions, without correspondence.

Merging two disjoint partitions is bit-identical to a single encode because
bundling is addition when partitions own entire cells (the CI fixture). This
is a mechanism check, not a finding. Arbitrary within-cell partitions change
float32 reduction order and are only algebraically identical.

Rules and experimental definitions here are ours. Live shard payload is flat
in contributions; Loro history is not. No history is trimmed in this study.
Codebooks are regenerated from the shared seed, never transmitted as splats.
"""

import argparse
import json
import struct
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from bench.change_detection import drift
from holo.capture import (
    BANDS,
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
)
from holo.crdt import HAVE_LORO, HoloReplica, unpack_bundle
from holo.fhrr import FHRR
from holo.locality import locality_report
from holo.spectral import SplatScene

if HAVE_LORO:
    from loro import LoroDoc


class BundleSpace(FHRR):
    """Keep capture channels in the existing tagged multichannel wire format."""

    def __init__(self, dim, channels):
        super().__init__(dim=dim, seed=0)
        self.channels = channels

    def zeros(self):
        return np.zeros((self.channels, self.dim), dtype=np.complex64)


def _bands(cell):
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError("cell must be finite and positive")
    return [(name, cap, cell) for name, cap, _ in BANDS]


def _subset(scene, idx):
    return SplatScene(scene.mu[idx], scene.cov[idx], scene.amp[idx])


def _boundary(scene, axis):
    # Integrated Gaussian mass, including covariance volume, not peak alpha.
    mass = scene.amp[:, 0] * np.sqrt(np.linalg.det(scene.cov.astype(float)))
    order = np.argsort(scene.mu[:, axis], kind="stable")
    rank = np.searchsorted(np.cumsum(mass[order]), mass.sum() / 2)
    return float(scene.mu[order[min(rank, scene.n - 1)], axis])


def split_scene(scene, axis=0, overlap=0.0):
    """Half-open mass-median cut; overlap is full width in normalized units."""
    if axis not in (0, 1, 2) or not 0 <= overlap <= 1 or scene.n < 2:
        raise ValueError("invalid split")
    cut = _boundary(scene, axis)
    x = scene.mu[:, axis]
    return (np.flatnonzero(x < cut + overlap / 2),
            np.flatnonzero(x >= cut - overlap / 2))


def encode_half(scene, idx, books, dim, cell):
    sub = _subset(scene, idx)
    smax = np.sqrt(np.linalg.eigvalsh(sub.cov.astype(float))[:, -1])
    # Float32 covariance roundoff must not put a clamped splat above its cap.
    smax = np.minimum(smax, BANDS[-1][1])
    return encode_bands(sub, smax, books, bands=_bands(cell), dim=dim,
                        verbose=False)


def _flat(bundles):
    return {name + "/" + ",".join(map(str, key)): value
            for name, cells in bundles.items() for key, value in cells.items()}


def _nested(values):
    result = {name: {} for name, _, _ in BANDS}
    for key, value in values.items():
        name, coords = key.split("/")
        result[name][tuple(map(int, coords.split(",")))] = value
    return result


def _empty(dim, channels, peer, codec="raw"):
    if not HAVE_LORO:
        raise RuntimeError("capture replication requires the optional loro dependency")
    doc = LoroDoc()
    doc.peer_id = int(peer)
    return HoloReplica(BundleSpace(dim, channels), doc=doc, codec=codec)


def replicate(bundles, peer, codec="raw"):
    values = _flat(bundles)
    if not values:
        raise ValueError("cannot infer dimensions from empty bundles")
    channels, dim = next(iter(values.values())).shape
    replica = _empty(dim, channels, peer, codec)
    for key, value in values.items():
        replica.add(key, value)
    replica.flush()
    return replica


def merge_rules(replicas, container):
    """Read only the first replica's received shards, without consulting peers."""
    replica = replicas if isinstance(replicas, HoloReplica) else replicas[0]
    if container not in replica.containers():
        raise KeyError(container)
    stored = replica.doc.get_map("bundles")
    stored_keys = stored.keys()
    keys = sorted((k for k in stored_keys if k.rsplit("::", 1)[0] == container),
                  key=lambda k: int(k.rsplit("::", 1)[1]))
    shards = [np.atleast_2d(unpack_bundle(stored.get(k).value)) for k in keys]
    total = replica.merged(container)
    return {"sum": total, "mean": total / len(shards), "owner": shards[0]}


def _exchange(a, b):
    va, vb = a.version(), b.version()
    da, db = a.updates_since(vb), b.updates_since(va)
    rows = []
    for target, delta in ((b, da), (a, db)):
        frame = struct.pack(">I", len(delta)) + delta
        start = perf_counter()
        target.apply(frame[4:])
        rows.append({"frame_bytes": len(frame), "apply_s": perf_counter() - start})
    return rows


def state_growth(bundles, splats, codec="raw"):
    """Successive flush epochs of one region, retaining full document history."""
    replica = replicate(bundles, 1, codec)
    rows = []
    for k in range(1, 17):
        if k > 1:
            for key, value in _flat(bundles).items():
                replica.add(key, value)
            replica.flush()
        if k in (1, 2, 4, 8, 16):
            m = replica.doc.get_map("bundles")
            stored_keys = m.keys()
            rows.append({"k": k, "containers": len(replica.containers()),
                         "live_bytes": sum(len(m.get(key).value)
                                           for key in stored_keys),
                         "snapshot_bytes": len(replica.snapshot()),
                         "splat_bytes": k * splats * 22})
    return rows


def _score(truth, recon, mask):
    if not mask.any():
        return None
    # Alpha field is the mass fidelity metric; RGB is still encoded/transmitted.
    return asdict(locality_report(truth[mask, 0], recon[mask, 0]))


def _fidelity(scene, dim, cell, overlap, drift_kw, rng, codec, rules):
    books = band_codebooks(np.random.default_rng(0), dim=dim)
    ia, ib = split_scene(scene, overlap=overlap)
    whole, members = encode_half(scene, np.arange(scene.n), books, dim, cell)
    half_a, ma = encode_half(scene, ia, books, dim, cell)
    half_b, mb = encode_half(scene, ib, books, dim, cell)
    cut = _boundary(scene, 0)
    # Stratify by region so a small overlap never disappears by random chance.
    inside = np.abs(scene.mu[:, 0] - cut) < overlap / 2
    ids = np.concatenate([rng.choice(np.flatnonzero(mask), 128, replace=True)
                          for mask in (inside, ~inside) if mask.any()])
    spread = np.linalg.cholesky(scene.cov[ids].astype(float))
    points = (scene.mu[ids] + 0.15 * np.einsum(
        "nij,nj->ni", spread, rng.normal(size=(len(ids), 3)))).astype(np.float32)
    masks = {"inside": np.abs(points[:, 0] - cut) < overlap / 2,
             "outside": np.abs(points[:, 0] - cut) >= overlap / 2}
    truth = exact_slice(points, scene, members, bands=_bands(cell))
    rows = []
    decoded = decode_slice(points, whole, books, bands=_bands(cell))
    for region, mask in masks.items():
        rows.append({"rule": "whole", "drift": None, "region": region,
                     "score": _score(truth, decoded, mask)})
    own_errors = []
    for name, idx, bundles, membership, mask in (
        ("half_a", ia, half_a, ma, points[:, 0] < cut + overlap / 2),
        ("half_b", ib, half_b, mb, points[:, 0] >= cut - overlap / 2),
    ):
        own_truth = exact_slice(points, _subset(scene, idx), membership,
                                bands=_bands(cell))
        rec = decode_slice(points, bundles, books, bands=_bands(cell))
        score = _score(own_truth, rec, mask)
        own_errors.append(score["rel_l2"])
        rows.append({"rule": name, "drift": None, "region": "own",
                     "score": score})
    drift_seed = int(rng.integers(2**31))
    # Pristine overlap isolates double counting; the required ladder includes
    # amplitude noise and exact splits even at zero position drift.
    ladder = [(None, half_b)]
    for sigma in sorted({0, 0.05, 0.1, 0.2, drift_kw[0]}):
        moved = drift(_subset(scene, ib), np.random.default_rng(drift_seed),
                      sigma, drift_kw[1], drift_kw[2])
        bundles, _ = encode_half(moved, np.arange(moved.n), books, dim, cell)
        ladder.append((sigma, bundles))
    for sigma, bundles in ladder:
        a, b = replicate(half_a, 1, codec), replicate(bundles, 2, codec)
        _exchange(a, b)
        merged = {rule: {} for rule in rules}
        for container in a.containers():
            local = merge_rules(a, container)
            for rule in rules:
                merged[rule][container] = local[rule]
        for rule in rules:
            rec = decode_slice(points, _nested(merged[rule]), books,
                               bands=_bands(cell))
            for region, mask in masks.items():
                rows.append({"rule": rule, "drift": sigma, "region": region,
                             "score": _score(truth, rec, mask)})
    baseline = max(own_errors)
    pristine = [r for r in rows if r["drift"] is None and r["region"] == "inside"
                and r["rule"] in rules and r["score"] is not None]
    best = min(pristine, key=lambda r: r["score"]["rel_l2"]) if pristine else None
    return rows, {"best_rule": best["rule"] if best else None,
                  "best_error": best["score"]["rel_l2"] if best else None,
                  "worse_half_error": baseline,
                  "ratio": best["score"]["rel_l2"] / baseline if best else None}, books


def _timed_apply(bundles, codec, repeats=5):
    source = replicate(bundles, 1, codec)
    channels, dim = next(iter(_flat(bundles).values())).shape
    samples = []
    for _ in range(repeats):
        target = _empty(dim, channels, 2, codec)
        delta = source.updates_since(target.version())
        start = perf_counter()
        target.apply(delta)
        samples.append(perf_counter() - start)
    return float(np.median(samples)), samples


def _transport(scene, books, dim, cell, overlap):
    ia, ib = split_scene(scene, overlap=overlap)
    sweep = []
    encoded = []
    for size in sorted({cell, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1}):
        a, ma = encode_half(scene, ia, books, dim, size)
        b, mb = encode_half(scene, ib, books, dim, size)
        encoded.append((size, a, b, ma, mb))
    fixed_count = min([16] + [len(_flat(a)) for _, a, _, _, _ in encoded])
    for size, a, b, ma, _mb in encoded:
        count = len(_flat(a)) + len(_flat(b))
        splats = len(ia) + len(ib)
        # Fixed dirty count isolates occupancy from per-container work.
        keys = sorted(_flat(a))
        select = [keys[i] for i in np.linspace(0, len(keys) - 1,
                                             fixed_count, dtype=int)]
        small = _nested({key: _flat(a)[key] for key in select})
        memberships = _flat(ma)
        occupancy = sum(len(memberships[key]) for key in select) / fixed_count
        for codec in ("raw", "hg8"):
            peer_a, peer_b = replicate(a, 1, codec), replicate(b, 2, codec)
            frames = _exchange(peer_a, peer_b)
            seconds, samples = _timed_apply(small, codec)
            wire = sum(row["frame_bytes"] for row in frames)
            sweep.append({"cell": size, "codec": codec,
                          "dirty_containers": count, "splats": splats,
                          "mean_splats_per_dirty_cell": splats / count,
                          "frame_bytes": wire, "spz_bytes": 22 * splats,
                          "crossover_occupancy": wire / count / 22,
                          "apply_s": sum(row["apply_s"] for row in frames),
                          "directions": frames, "fixed_dirty": fixed_count,
                          "fixed_occupancy": occupancy,
                          "fixed_apply_s": seconds, "fixed_samples_s": samples})
    fits = {}
    for codec in ("raw", "hg8"):
        rows = [r for r in sweep if r["codec"] == codec]
        x = np.array([r["fixed_occupancy"] for r in rows])
        y = np.array([r["fixed_apply_s"] for r in rows])
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        fits[codec] = {"slope": float(slope), "intercept": float(intercept),
                       "occupancy_span": float(x.max() / x.min()),
                       "fixed_dirty": fixed_count}
    # State region: one actual cell, with its actual original splat count.
    _, a, _, ma, _ = encoded[0]
    key = sorted(_flat(a))[len(_flat(a)) // 2]
    region = _nested({key: _flat(a)[key]})
    growth = {codec: state_growth(region, len(_flat(ma)[key]), codec)
              for codec in ("raw", "hg8")}
    return sweep, fits, growth


def merge_study(path, dim=2048, cell=0.125, overlap=0.2,
                drift_kw=(0, 0.2, 0.1), rng=None,
                rules=("sum", "mean", "owner"), codec="raw"):
    if dim < 16:
        raise ValueError("dim must be at least 16")
    if not HAVE_LORO:
        raise RuntimeError("capture replication requires the optional loro dependency")
    rng = np.random.default_rng(0) if rng is None else rng
    scene = path if isinstance(path, SplatScene) else build_scene(
        path, crop_quantile=1.0, crop_margin=1.0, verbose=False)[0]
    fidelity, criterion, books = _fidelity(
        scene, dim, cell, overlap, drift_kw, rng, codec, rules)
    transport, fits, growth = _transport(scene, books, dim, cell, overlap)
    return {"source": "synthetic" if isinstance(path, SplatScene) else str(path),
            "dim": dim, "cell": cell, "overlap": overlap, "codec": codec,
            "splats": scene.n, "fidelity": fidelity, "overlap_criterion": criterion,
            "transport": transport, "time_fits": fits, "state_growth": growth}


def synthetic_scene(seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.15, 0.85, (32, 3)).astype(np.float32)
    cov = np.tile(np.eye(3, dtype=np.float32) * 0.015**2, (32, 1, 1))
    return SplatScene(mu, cov, np.ones((32, 1), dtype=np.float32))


def run_synthetic(dim=2048, seed=0):
    return merge_study(synthetic_scene(seed), dim=dim, rng=np.random.default_rng(seed))


def _figure(result, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for rule in ("sum", "mean", "owner"):
        rows = [r for r in result["fidelity"] if r["rule"] == rule
                and r["region"] == "inside" and r["drift"] is not None]
        axes[0].plot([r["drift"] for r in rows],
                     [r["score"]["rel_l2"] for r in rows], "o-", label=rule)
    axes[0].set(xlabel="Position drift", ylabel="Overlap relative L2")
    for codec in ("raw", "hg8"):
        rows = [r for r in result["transport"] if r["codec"] == codec]
        axes[1].loglog([r["mean_splats_per_dirty_cell"] for r in rows],
                       [r["frame_bytes"] / r["spz_bytes"] for r in rows],
                       "o-", label=codec)
        axes[2].loglog([r["fixed_occupancy"] for r in rows],
                       [r["fixed_apply_s"] * 1000 for r in rows], "o-", label=codec)
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set(xlabel="Splats / dirty cell", ylabel="Wire / SPZ-equivalent bytes")
    axes[2].set(xlabel="Splats / cell (fixed dirty count)", ylabel="Apply ms")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("capture", nargs="?")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--cell", type=float, default=0.125)
    parser.add_argument("--drift", default="0,0.2,0.1")
    parser.add_argument("--codec", choices=("raw", "hg8"), default="raw")
    parser.add_argument("--figure")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic and not args.capture:
        parser.error("provide CAPTURE or --synthetic")
    drift_kw = tuple(map(float, args.drift.split(",")))
    if len(drift_kw) != 3:
        parser.error("--drift requires position,amplitude,split")
    result = merge_study(synthetic_scene() if args.synthetic else args.capture,
                         dim=args.dim, cell=args.cell, overlap=args.overlap,
                         drift_kw=drift_kw, codec=args.codec)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if args.figure:
        _figure(result, args.figure)
    print(json.dumps({"overlap": result["overlap_criterion"],
                      "time": result["time_fits"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
