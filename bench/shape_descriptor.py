"""Compare four shape descriptors under an explicit class hypothesis.

Prior art: Osada et al., Shape Distributions (2002), abstract/summary read at
https://gfx.cs.princeton.edu/gfx/pubs/Osada_2002_SD/index.php . D2 samples
surface-point distances there; our adaptation samples alpha-weighted splat
centres with replacement, including self pairs, in units of the supplied cube.
These are our definitions, not a reproduction of their implementation.

Inputs to descriptors are already in a subject-scaled unit cube. core_box
lifts the axis-aligned median/L-infinity quantile rule from blob_test.py.
Translation and uniform scale cancel, but this frame is NOT yaw equivariant.
Horizontal radial mass and D2 are yaw invariant in a fixed centred frame;
finite sampled radial_power is only approximately yaw invariant. A finite yaw
search only guarantees alignment for rotations on its lattice. No stronger
invariance or corpus classification claim is made.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from bench.place_recognition import (
    build_scene_fixed,
    correlate,
    fingerprint,
    radial_power,
    translation_grid,
    yaw_scene,
)
from holo.capture import ALPHA_MIN, load_scene_file, weighted_quantile
from holo.spectral import SplatScene, sample_frequencies

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KINDS = ("radial", "spectral", "whitened", "d2")
CAVEAT = (
    "Hypothesis only. Axis-aligned core framing is not yaw equivariant; "
    "finite spectral radial bins and finite yaw searches are approximate. "
    "D2 samples alpha-weighted centres, not mesh surfaces."
)


def _mass(scene):
    mass = np.asarray(scene.amp[:, 0], dtype=np.float64)
    if not len(mass) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("alpha mass must be nonempty, finite and nonnegative")
    if mass.sum() <= 0:
        raise ValueError("alpha mass must have positive total")
    return mass / mass.sum()


def _box(pos, mass, margin, share):
    if margin <= 0 or not np.isfinite(margin) or not 0 < share <= 1:
        raise ValueError("positive finite margin and share in (0, 1] required")
    if not len(pos) or not np.isfinite(pos).all():
        raise ValueError("no finite splats above alpha floor")
    center = np.array([weighted_quantile(pos[:, i], mass, 0.5) for i in range(3)])
    radius = float(weighted_quantile(np.abs(pos - center).max(axis=1), mass, share))
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("subject cube has zero or invalid extent")
    return center - margin * radius, float(2 * margin * radius), radius


def core_box(path, margin=3.0, share=0.5) -> tuple:
    """Return (lower corner, extent, half-mass L-infinity radius)."""
    pos, _, rgba, _ = load_scene_file(path)
    keep = rgba[:, 3] >= ALPHA_MIN
    return _box(pos[keep], rgba[keep, 3], margin, share)


def normalise_scene(scene):
    """Apply the same core frame and centre crop to an in-memory fixture."""
    keep = scene.amp[:, 0] >= ALPHA_MIN
    pos, cov, amp = scene.mu[keep], scene.cov[keep], scene.amp[keep]
    lo, extent, _ = _box(pos, amp[:, 0], 3.0, 0.5)
    inside = np.all((pos >= lo) & (pos <= lo + extent), axis=1)
    return SplatScene((pos[inside] - lo) / extent,
                      cov[inside] / extent**2, amp[inside].copy())


def radial_profile(scene, n_shells=20, horizontal=True) -> np.ndarray:
    """Alpha shares in [0, 1] radius bins (default width 0.05 cube units)."""
    if n_shells < 1:
        raise ValueError("n_shells must be positive")
    delta = np.asarray(scene.mu, dtype=np.float64) - 0.5
    radius = np.linalg.norm(delta[:, [0, 2]] if horizontal else delta, axis=1)
    hist, _ = np.histogram(radius, np.linspace(0, 1, n_shells + 1),
                           weights=_mass(scene))
    return hist


def spectral_radial(scene, freqs, sigma_box, n_bins=32) -> np.ndarray:
    """Mean fingerprint power in horizontal/vertical frequency bins."""
    return radial_power(fingerprint(scene, freqs, sigma_box), freqs, n_bins)


def whitened_similarity(fp_a, yaw_fps_b, freqs, grid) -> float:
    """Maximum PHAT correlation over supplied yaw candidates."""
    if not len(yaw_fps_b):
        raise ValueError("at least one yaw candidate required")
    return max(correlate(fp_a, b, freqs, grid, whiten=1)[0] for b in yaw_fps_b)


def d2_histogram(scene, rng, n_pairs=200_000, n_bins=64) -> np.ndarray:
    """Mass-weighted centre distances / unit cube extent, bins [0, sqrt(3)]."""
    if n_pairs < 1 or n_bins < 1:
        raise ValueError("n_pairs and n_bins must be positive")
    pairs = rng.choice(len(scene.mu), size=(n_pairs, 2), p=_mass(scene))
    distances = np.linalg.norm(scene.mu[pairs[:, 0]] - scene.mu[pairs[:, 1]], axis=1)
    hist, _ = np.histogram(distances, np.linspace(0, np.sqrt(3), n_bins + 1))
    if hist.sum() != n_pairs:
        raise ValueError("D2 input must fit a unit cube")
    return hist / n_pairs


def similarity(desc_a, desc_b, kind) -> float:
    """Cosine similarity for radial, spectral and D2 descriptors."""
    if kind not in ("radial", "spectral", "d2", "a", "b", "d"):
        raise ValueError("use whitened_similarity for the phase descriptor")
    a, b = np.asarray(desc_a).ravel(), np.asarray(desc_b).ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / norm, -1, 1)) if norm else 0.0


def class_report(sims, labels) -> dict:
    """Ordered off-diagonal means; LOO ties choose first input index.

    Singleton classes remain in accuracy's denominator. Missing within or
    between pairs yield None, making small/degenerate reports valid JSON.
    """
    sims, labels = np.asarray(sims, dtype=float), np.asarray(labels)
    n = len(labels)
    if n < 2 or sims.shape != (n, n) or not np.isfinite(sims).all():
        raise ValueError("need a finite square matrix and at least two labels")
    off = ~np.eye(n, dtype=bool)
    same = labels[:, None] == labels[None, :]
    within, between = sims[off & same], sims[off & ~same]
    scores = sims.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbors = scores.argmax(axis=1)
    return {"within_mean": float(within.mean()) if within.size else None,
            "between_mean": float(between.mean()) if between.size else None,
            "loo_accuracy": float(np.mean(labels[neighbors] == labels)),
            "neighbors": neighbors.tolist(), "within_pairs": int(within.size),
            "between_pairs": int(between.size), "n": n}


def synthetic_shapes(rng, kinds=("sphere", "rod", "disc"), per_kind=6) -> list:
    """Independent filled ellipsoids, random scale, yaw pose and position.

    Pose is yaw about y, not arbitrary SO(3): horizontal descriptors retain
    vertical information. Each object has independent samples and aspect jitter.
    Returned scenes are in world units; normalise_scene supplies the core cube.
    """
    axes = {"sphere": [1, 1, 1], "rod": [1, 0.12, 0.12],
            "disc": [1, 0.12, 1]}
    shapes = []
    for kind in kinds:
        for _ in range(per_kind):
            points = rng.normal(size=(96, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            points *= rng.random((96, 1)) ** (1 / 3)
            points *= np.asarray(axes[kind]) * rng.uniform(0.95, 1.05, 3)
            scene = SplatScene(points + 0.5,
                               np.tile(np.eye(3) * 0.008**2, (96, 1, 1)),
                               rng.uniform(0.6, 1, (96, 1)))
            scene = yaw_scene(scene, rng.uniform(0, 2 * np.pi))
            scale = rng.uniform(0.5, 4)
            shapes.append((kind, SplatScene(
                scene.mu * scale + rng.uniform(-5, 5, 3),
                scene.cov * scale**2, scene.amp)))
    return shapes


def _matrices(scenes, freqs, args):
    angles = np.arange(args.yaws) * 2 * np.pi / args.yaws
    fps = [fingerprint(s, freqs, args.sigma) for s in scenes]
    yaws = [[fp, *[fingerprint(yaw_scene(s, t), freqs, args.sigma)
                    for t in angles[1:]]] for s, fp in zip(scenes, fps)]
    descriptors = {
        "radial": [radial_profile(s) for s in scenes],
        "spectral": [radial_power(fp, freqs) for fp in fps],
        "d2": [d2_histogram(s, np.random.default_rng(args.seed + i), args.pairs)
               for i, s in enumerate(scenes)],
    }
    matrices = {kind: np.array([[similarity(a, b, kind) for b in desc]
                               for a in desc]) for kind, desc in descriptors.items()}
    grid = (translation_grid(args.grid) if args.grid > 1
            else np.zeros((1, 3), dtype=np.float32))
    matrices["whitened"] = np.array([
        [whitened_similarity(a, b, freqs, grid) for b in yaws] for a in fps])
    return matrices


def _inputs(args, parser, rng):
    if args.synthetic:
        shapes = synthetic_shapes(rng, per_kind=args.per_kind)
        return ([normalise_scene(s) for _, s in shapes],
                [f"{label}-{i}" for i, (label, _) in enumerate(shapes)],
                [label for label, _ in shapes])
    if not args.paths or args.labels is None:
        parser.error("captures require scene paths and --labels")
    hypothesis = json.loads(args.labels.read_text())
    names = [Path(p).name for p in args.paths]
    if not isinstance(hypothesis, dict) or any(n not in hypothesis for n in names):
        parser.error("--labels must map every scene filename to a class")
    labels = [hypothesis[n] for n in names]
    if any(not isinstance(label, str) or not label for label in labels):
        parser.error("class labels must be nonempty strings")
    scenes = [build_scene_fixed(p, *core_box(p)[:2])[0] for p in args.paths]
    return scenes, names, labels


def _figure(matrices, names, path):
    fig, axs = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, kind in zip(axs.flat, KINDS):
        im = ax.imshow(matrices[kind], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(kind)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6)
    fig.colorbar(im, ax=axs, shrink=0.65, label="Similarity")
    fig.suptitle("Shape descriptor comparison — class hypothesis, not a claim")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Print hypothesis, matrices and retrieval reports; save reproducible JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--yaws", type=int, default=16)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-kind", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=200_000)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    if min(args.dim, args.yaws, args.grid, args.per_kind, args.pairs) < 1:
        parser.error("counts must be positive")
    if args.sigma <= 0 or not np.isfinite(args.sigma):
        parser.error("--sigma must be positive and finite")
    if args.synthetic and (args.paths or args.labels):
        parser.error("--synthetic cannot be combined with captures or --labels")
    rng = np.random.default_rng(args.seed)
    scenes, names, labels = _inputs(args, parser, rng)
    if len(scenes) < 2:
        parser.error("at least two scenes required")
    hypothesis = dict(zip(names, labels))
    print("The hypothesis under test:", json.dumps(hypothesis, sort_keys=True))
    print(CAVEAT, flush=True)
    freqs = sample_frequencies(args.dim, 3, 1 / args.sigma, rng)
    matrices = _matrices(scenes, freqs, args)
    reports = {k: class_report(matrices[k], labels) for k in KINDS}
    result = {"hypothesis": hypothesis, "names": names, "labels": labels,
              "matrices": {k: matrices[k].tolist() for k in KINDS},
              "reports": reports, "caveat": CAVEAT,
              "settings": {k: getattr(args, k) for k in
                           ("seed", "dim", "yaws", "grid", "sigma", "pairs",
                            "synthetic", "per_kind")}}
    for kind in KINDS:
        print(kind, np.array2string(matrices[kind], precision=3, threshold=np.inf))
        print(json.dumps(reports[kind]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.figure:
        _figure(matrices, names, args.figure)
    return result


if __name__ == "__main__":
    main()
