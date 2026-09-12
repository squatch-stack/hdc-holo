"""Squatch Stack capture resonator experiment (research only).

arXiv:2208.12880 abstract read 2026-09-12: scenes are sums of bound
identity/pose factors. Our 3-D coarse-cell ranking and refinement definitions
are ours, not a reproduction of that paper's hierarchical network.
FTO note supplied by the brief: US 12,014,263; research only.
Spectral positions use exp(-i w.t). RMS scaling retains identity envelopes;
phase-only projection discards them. Loads are candidate products / dimension,
per refinement cell, with total refinement work also reported. A load below
10 is not a guarantee for correlated spectral dictionaries.

Kanerva's algebra here uses conjugate (phasor) inverses. Positions differ
between captures, so the prototype form is the honest analogy; the algebraic
cleanup is a diagnostic, not evidence of cross-capture discovery.
"""

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from bench.place_recognition import (
    build_scene_fixed,
    crop_box,
    fingerprint,
    translation_grid,
)
from holo.fhrr import FHRR
from holo.resonator import deflate, factorize_all, grid_codebook
from holo.spectral import (
    SplatScene,
    decode_field_phasor,
    random_scene,
    sample_frequencies,
    translate_bundle,
)


def _rms(v):
    return np.sqrt(np.mean(np.abs(v) ** 2, axis=-1, keepdims=True))


def _center(scene, freqs, sigma):
    centroid = np.average(scene.mu, axis=0, weights=scene.amp[:, 0])
    code = translate_bundle(fingerprint(scene, freqs, sigma)[None], freqs, -centroid)[0]
    return code, centroid


def object_codeword(path, lo, extent, freqs, sigma_units):
    """Encode a crop in its parent's cube, then remove its alpha centroid."""
    scene, _, _ = build_scene_fixed(path, lo, extent)
    return _center(scene, freqs, sigma_units / extent)


def position_codebooks(freqs, values):
    """Accept a shared axis or three separate axes, using the spectral sign."""
    axes = [values] * 3 if np.asarray(values).ndim == 1 else values
    return [grid_codebook(freqs[:, i], axis, sign=-1) for i, axis in enumerate(axes)]


def _cells(result, books, top_r):
    scores = [
        np.abs(m.conj() @ v) / m.shape[1]
        for m, v in zip(books[1:], result.estimates[1:])
    ]
    choices = [np.argsort(-s, kind="stable")[:top_r] for s in scores]
    candidates = list(itertools.product(*choices))
    candidates.sort(key=lambda c: -sum(float(s[i]) for s, i in zip(scores, c)))
    return candidates[:top_r]


def _refine(signal, books, freqs, cells, settings):
    values, coarse, kwargs = settings
    attempts = []
    for cell in cells:
        axes = [np.linspace(i / coarse, (i + 1) / coarse, values) for i in cell]
        fine_books = [books[0], *position_codebooks(freqs, axes)]
        result = factorize_all(signal, fine_books, 1, **kwargs)[0]
        binding = FHRR.bind(*(m[i] for m, i in zip(fine_books, result.indices)))
        amplitude = float(np.vdot(binding, signal).real / len(signal))
        point = [float(a[i]) for a, i in zip(axes, result.indices[1:])]
        attempts.append((result, fine_books, amplitude, point))
    best = max(attempts, key=lambda a: (a[0].converged, a[2]))
    return (*best, sum(a[0].n_iters for a in attempts))


def what_is_where(
    S,
    identity,
    freqs,
    max_objects,
    values=32,
    coarse=8,
    top_r=4,
    *,
    coarse_sigma=None,
    **resonator_kw,
):
    """Factorize each residual coarsely and refine its top-r joint cells.

    extent is an optional keyword (default 1); scene offsets are relative to
    the parent cube's lower corner. Only factorize_all's iters/score_floor are
    supported. Failed attempts are returned and terminate deflation.
    """
    identity_phase_only = resonator_kw.pop("identity_phase_only", False)
    sigma = resonator_kw.pop("sigma", None)
    extent = resonator_kw.pop("extent", 1.0)
    if min(values, coarse) < 2 or not 1 <= top_r <= coarse or max_objects < 1:
        raise ValueError("invalid grid counts or object count")
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("extent must be positive and finite")
    identity = np.asarray(identity, np.complex64)
    if identity.ndim != 2 or identity.shape[1] != len(freqs):
        raise ValueError("identity must be a nonempty (K, d) matrix")
    norms = _rms(identity)
    if not len(identity) or np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("identities must have finite nonzero energy")
    residual = np.asarray(S, np.complex64) / float(norms.max())
    codes = FHRR.normalize(identity) if identity_phase_only else identity / norms
    books = [codes, *position_codebooks(freqs, (np.arange(coarse) + 0.5) / coarse)]
    loads = {
        "coarse": len(codes) * coarse**3 / len(freqs),
        "fine_per_cell": len(codes) * values**3 / len(freqs),
        "fine_total": top_r * len(codes) * values**3 / len(freqs),
    }
    coarse_filter = np.ones(len(freqs), dtype=np.float32)
    if coarse_sigma is not None:
        if (
            sigma is None
            or not np.isfinite(sigma)
            or sigma <= 0
            or not np.isfinite(coarse_sigma)
            or coarse_sigma < sigma
        ):
            raise ValueError("coarse_sigma must be finite and >= native sigma > 0")
        coarse_filter = np.exp(
            -0.5 * (coarse_sigma**2 - sigma**2) * np.sum(freqs**2, axis=1)
        )
        coarse_filter[coarse_filter < 1e-4] = 0
        coarse_codes = identity * coarse_filter
        coarse_codes = (
            FHRR.normalize(coarse_codes)
            if identity_phase_only
            else coarse_codes / _rms(coarse_codes)
        )
        coarse_books = [coarse_codes, *books[1:]]
    else:
        coarse_books = books
    found = []
    for _ in range(max_objects):
        coarse_signal = residual if coarse_sigma is None else residual * coarse_filter
        first = factorize_all(coarse_signal, coarse_books, 1, **resonator_kw)[0]
        cells = _cells(first, books, top_r)
        result, fine_books, amplitude, point, fine_iters = _refine(
            residual, books, freqs, cells, (values, coarse, resonator_kw)
        )
        found.append(
            {
                "k_hat": result.indices[0],
                "t_hat": point,
                "t_hat_scene": (np.asarray(point) * extent).tolist(),
                "score": result.scores[0],
                "amplitude": amplitude,
                "converged": result.converged,
                "n_iters": first.n_iters + fine_iters,
                "winning_iters": result.n_iters,
                "stage_loads": loads,
                "coarse_converged": first.converged,
                "coarse_cells": [list(map(int, c)) for c in cells],
                "coarse_sigma": coarse_sigma,
            }
        )
        if not result.converged:
            break
        residual = deflate(residual, result, fine_books)
    return found


def _correlate_batch(codes, signal, freqs, grid, whiten):
    """Multi-channel form of correlate, sharing its expensive coarse decode.

    Same PHAT floor, bounds and three local refinements as correlate. All
    spectral evaluation dispatches through holo.decode_field_phasor.
    """
    codes = np.asarray(codes)
    signal = np.asarray(signal)
    grid = np.asarray(grid, np.float32)
    if (
        codes.ndim != 2
        or codes.shape[1:] != signal.shape
        or signal.shape != (len(freqs),)
    ):
        raise ValueError("fingerprints must match the frequency count")
    if grid.ndim != 2 or grid.shape[1] != freqs.shape[1] or not len(grid):
        raise ValueError("grid must be nonempty (G, spatial dimensions)")
    if not 0 <= whiten <= 1:
        raise ValueError("whiten must lie in [0, 1]")
    products = codes.conj() * signal
    norms = np.linalg.norm(codes, axis=1) * np.linalg.norm(signal)
    if whiten:
        magnitudes = np.abs(products)
        floors = 1e-6 * magnitudes.max(axis=1, keepdims=True)
        divisor = np.maximum(magnitudes, floors)
        products = np.divide(
            products, divisor**whiten, out=np.zeros_like(products), where=divisor > 0
        )
        norms = np.abs(products).sum(axis=1)
    products = products.astype(np.complex64)
    values = decode_field_phasor(products, freqs, grid)
    axes = [np.unique(grid[:, i]) for i in range(grid.shape[1])]
    initial_step = np.array([np.max(np.diff(x)) if len(x) > 1 else 0.0 for x in axes])
    lower, upper = grid.min(axis=0), grid.max(axis=0)
    results = []
    for k, norm in enumerate(norms):
        if norm == 0:
            results.append((0.0, grid[0].copy()))
            continue
        idx = int(np.argmax(values[:, k]))
        peak, point = float(values[idx, k]), grid[idx].copy()
        step = initial_step.copy()
        for _ in range(3):
            local_axes = [
                np.unique(np.clip(p + np.linspace(-h, h, 5), lo, hi))
                for p, h, lo, hi in zip(point, step, lower, upper)
            ]
            local = np.stack(np.meshgrid(*local_axes, indexing="ij"), -1)
            local = local.reshape(-1, grid.shape[1]).astype(np.float32)
            local_values = decode_field_phasor(products[k : k + 1], freqs, local)[:, 0]
            idx = int(np.argmax(local_values))
            if local_values[idx] > peak:
                peak, point = float(local_values[idx]), local[idx].copy()
            step /= 2
        results.append((float(np.clip(peak * len(freqs) / float(norm), -1, 1)), point))
    return results


def one_shot_baseline(S, identity, freqs, grid, whiten=1.0, band_floor=0.0, sigma=None):
    """Independent PHAT peaks with phase-only reverse identity cleanup.

    Mask blur-envelope components before correlation. Scores are normalized
    over the retained band by correlate; a low score flags a weak or missed
    peak, not proof of absence. No truth is used by the search.
    """
    if not 0 <= band_floor < 1:
        raise ValueError("band_floor must lie in [0, 1)")
    mask = np.ones(len(freqs), dtype=bool)
    if band_floor:
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive finite sigma required for band limiting")
        mask = np.exp(-0.5 * sigma**2 * np.sum(freqs**2, axis=1)) >= band_floor
        if not mask.any():
            raise ValueError("band contains no frequencies")
    signal = np.asarray(S) * mask
    codes = np.asarray(identity) * mask
    results = []
    for k, (score, point) in enumerate(
        _correlate_batch(codes, signal, freqs, grid, whiten)
    ):
        aligned = translate_bundle(signal[None], freqs, -point)[0]
        reverse = np.real(FHRR.normalize(codes).conj() @ FHRR.normalize(aligned))
        results.append(
            {
                "k_hat": k,
                "t_hat": point.tolist(),
                "score": score,
                "what_at": int(np.argmax(reverse)),
                "weak_peak": score < 0.1,
                "band_floor": band_floor,
                "retained_frequencies": int(mask.sum()),
            }
        )
    return results


def prototype_search(S, instances, freqs, grid, sigma, foreign=None):
    """Search centered training spectra; errors need held-out truth afterward.

    The prototype is the phase-only sum of training instances. The first
    instance is the single control; foreign is an independent training set.
    error_box is null until an evaluator supplies the held-out target position.
    """
    instances = np.asarray(instances)
    if instances.ndim != 2 or not len(instances):
        raise ValueError("instances must be a nonempty (N, d) matrix")
    queries = {"prototype": prototype(instances), "single": instances[0]}
    if foreign is not None:
        queries["foreign"] = prototype(foreign)
    result = {}
    rows = one_shot_baseline(S, list(queries.values()), freqs, grid, sigma=sigma)
    for name, row in zip(queries, rows):
        result[name] = {
            "score": row["score"],
            "t_hat": row["t_hat"],
            "error_box": None,
            "weak_peak": row["weak_peak"],
        }
    return result


def _prototype_errors(result, target):
    for row in result.values():
        row["error_box"] = float(np.linalg.norm(np.asarray(row["t_hat"]) - target))
    return result


def prototype(codewords):
    """Phase projection of the superposed training instances only."""
    return FHRR.normalize(np.asarray(codewords).sum(axis=0))


def analogy(S_target, S_source, c_source, identity, freqs, grid, **resonator_kw):
    """Return prototype factorization and position-confounded Kanerva cleanup."""
    query = FHRR.bind(
        FHRR.normalize(S_source),
        FHRR.bind(FHRR.normalize(S_target), FHRR.normalize(c_source).conj()).conj(),
    )
    scores = np.real(FHRR.normalize(np.asarray(identity)).conj() @ query) / len(query)
    return {
        "prototype": what_is_where(S_target, identity, freqs, 1, **resonator_kw),
        "baseline": one_shot_baseline(S_target, identity, freqs, grid),
        "kanerva_k": int(np.argmax(scores)),
        "kanerva_scores": scores.tolist(),
        "caveat": "Different capture positions confound Kanerva cleanup.",
    }


def composite(bundles, shifts, freqs):
    """Superpose independently shifted one-channel spectra."""
    if len(bundles) != len(shifts) or not len(bundles):
        raise ValueError("one shift is required for each nonempty bundle")
    return sum(
        translate_bundle(np.asarray(b).reshape(1, -1), freqs, t)[0]
        for b, t in zip(bundles, shifts)
    )


def synthetic_fixture(dim=4096, seed=0, objects=3, extras=0):
    """Distinct 40-80-splat clusters and diffuse 20%-mass background."""
    rng = np.random.default_rng(seed)
    # Recognition blur at an eighth of the object size (objects span
    # 0.08-0.2 of the cube); at box/40 every cluster is the same blob and
    # identities are indistinguishable (results/place_recognition.md).
    sigma = 0.012
    freqs = sample_frequencies(dim, 3, 1 / sigma, rng)
    crops, codes, truth = [], [], []
    for k in range(objects + extras):
        scene = random_scene(
            int(rng.integers(40, 81)), 3, rng, scale_range=(0.002, 0.006)
        )
        stretch = rng.uniform(0.08, 0.2, 3)
        mu = (scene.mu - 0.5) * stretch
        mu -= np.average(mu, axis=0, weights=scene.amp[:, 0])
        center = rng.uniform(0.18, 0.82, 3)
        scene = SplatScene((mu + center).astype(np.float32), scene.cov, scene.amp)
        code, center = _center(scene, freqs, sigma)
        crops.append(scene)
        codes.append(code)
        if k < objects:
            truth.append({"k": k, "t": center.tolist()})
    background = random_scene(100, 3, rng, scale_range=(0.02, 0.04))
    # Integrated Gaussian mass, not peak alpha, defines the background fraction.
    mass = sum(
        np.sum(c.amp[:, 0] * np.sqrt(np.linalg.det(c.cov))) for c in crops[:objects]
    )
    bg_mass = np.sum(background.amp[:, 0] * np.sqrt(np.linalg.det(background.cov)))
    background.amp *= 0.25 * mass / bg_mass
    parts = [*crops[:objects], background]
    parent = SplatScene(
        *(np.concatenate([getattr(c, a) for c in parts]) for a in ("mu", "cov", "amp"))
    )
    return {
        "S": fingerprint(parent, freqs, sigma),
        "identity": np.array(codes),
        "freqs": freqs,
        "truth": truth,
        "crops": crops,
        "sigma": sigma,
        "parent": parent,
        "background": background,
    }


def _metrics(found, baseline, truth, step, sigma, coarse=8):
    """Identity coverage and spatial coverage are independent of convergence.

    Each rate counts truths matched by any selected prediction. Joint rate
    additionally requires identity and position in the same prediction.
    Convergence is reported separately and never interpreted as correctness.
    """
    tolerance = max(step, sigma / 2)
    metrics = {
        "position_tolerance_box": tolerance,
        "fine_step_box": step,
        "sigma_box": sigma,
        "tolerance_rule": "max(fine_step_box, sigma_box / 2)",
    }
    for name, predictions in (
        ("resonator", found),
        ("baseline", sorted(baseline, key=lambda r: -r["score"])[: len(truth)]),
    ):
        identity_hits, position_hits, joint_hits = 0, 0, 0
        for item in truth:
            identity_hits += any(r["k_hat"] == item["k"] for r in predictions)
            position_hits += any(
                np.linalg.norm(np.array(r["t_hat"]) - item["t"]) <= tolerance
                for r in predictions
            )
            joint_hits += any(
                r["k_hat"] == item["k"]
                and np.linalg.norm(np.array(r["t_hat"]) - item["t"]) <= tolerance
                for r in predictions
            )
        metrics[name + "_identity_rate"] = identity_hits / len(truth)
        metrics[name + "_position_rate"] = position_hits / len(truth)
        metrics[name + "_joint_rate"] = joint_hits / len(truth)
    errors = []
    for row in found:
        target = next((t["t"] for t in truth if t["k"] == row["k_hat"]), None)
        error = (
            float(np.linalg.norm(np.array(row["t_hat"]) - target))
            if target is not None
            else None
        )
        row.update(
            {
                "truth": target,
                "error_box": error,
                "error_scene": error,
                "error_coarse_cells": error * coarse if error is not None else None,
                "error_fine_steps": error / step if error is not None else None,
            }
        )
        if error is not None:
            errors.append(error / step)
    metrics.update(
        {
            "converged_rate": float(np.mean([r["converged"] for r in found])),
            "coarse_converged_rate": float(
                np.mean([r["coarse_converged"] for r in found])
            ),
            "mean_error_fine_steps": float(np.mean(errors)) if errors else None,
            "mean_iterations": float(np.mean([r["n_iters"] for r in found])),
            "stage_loads": found[0]["stage_loads"],
        }
    )
    return metrics


def _query(
    fixture,
    signal=None,
    phase=False,
    values=16,
    coarse=8,
    top_r=4,
    baseline_size=32,
    *,
    band_floor=0.0,
    **options,
):
    coarse_sigma = options.pop("coarse_sigma", None)
    baselines = options.pop("baselines", None)
    if options:
        raise TypeError(f"unknown query options: {sorted(options)}")
    signal = fixture["S"] if signal is None else signal
    found = what_is_where(
        signal,
        fixture["identity"],
        fixture["freqs"],
        len(fixture["truth"]),
        values=values,
        coarse=coarse,
        top_r=top_r,
        identity_phase_only=phase,
        sigma=fixture["sigma"],
        coarse_sigma=coarse_sigma,
    )
    grid = translation_grid(baseline_size, 0.5) + 0.5
    baseline = (
        one_shot_baseline(
            signal,
            fixture["identity"],
            fixture["freqs"],
            grid,
            band_floor=band_floor,
            sigma=fixture["sigma"],
        )
        if baselines is None
        else baselines
    )
    return {
        "found": found,
        "baseline_queries": baseline,
        **_metrics(
            found,
            baseline,
            fixture["truth"],
            1 / (coarse * (values - 1)),
            fixture["sigma"],
            coarse,
        ),
    }


def _jittered_instances(base, freqs, sigma, jitter, rng):
    size = float(np.ptp(base.mu, axis=0).max())
    training = []
    for _ in range(2):
        variant = SplatScene(
            base.mu + rng.normal(0, jitter * size, base.mu.shape),
            base.cov,
            base.amp * rng.uniform(0.8, 1.2, base.amp.shape),
        )
        training.append(_center(variant, freqs, sigma)[0])
    return training


def _prototype_experiment(fixture, seed, grid_size=32):
    rng = np.random.default_rng(seed)
    foreign_scene = synthetic_fixture(len(fixture["freqs"]), seed + 1, 1)["crops"][0]
    grid = translation_grid(grid_size, 0.5) + 0.5
    rows = []
    for jitter in (0.05, 0.1, 0.2):
        for item in fixture["truth"]:
            training = _jittered_instances(
                fixture["crops"][item["k"]],
                fixture["freqs"],
                fixture["sigma"],
                jitter,
                rng,
            )
            foreign = _jittered_instances(
                foreign_scene, fixture["freqs"], fixture["sigma"], jitter, rng
            )
            searches = prototype_search(
                fixture["S"],
                training,
                fixture["freqs"],
                grid,
                fixture["sigma"],
                foreign,
            )
            rows.append(
                {
                    "k": item["k"],
                    "jitter": jitter,
                    **_prototype_errors(searches, item["t"]),
                }
            )
    return rows


def _figure(result, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for item in result["truth"]:
        axes[0].scatter(*item["t"][:2], marker="x", color="black")
        axes[0].annotate(str(item["k"]), item["t"][:2])
    plotted_objects = len(result["truth"])
    plotted_phase = result["rows"][0]["phase_only"]
    for row in result["rows"]:
        if (
            row["objects"] == plotted_objects
            and row["distractors"] == 0
            and row["phase_only"] == plotted_phase
        ):
            for item in row["found"]:
                axes[0].scatter(
                    *item["t_hat"][:2],
                    marker="o",
                    facecolors="none",
                    edgecolors="tab:blue" if item["converged"] else "red",
                )
                axes[0].annotate(
                    f"pred {item['k_hat']}",
                    item["t_hat"][:2],
                    xytext=(4, -12),
                    textcoords="offset points",
                    fontsize=8,
                )
    for phase in (False, True):
        rows = [
            r
            for r in result["rows"]
            if r["objects"] == plotted_objects and r["phase_only"] == phase
        ]
        axes[1].plot(
            [r["distractors"] for r in rows],
            [r["resonator_joint_rate"] for r in rows],
            "o-",
            label="resonator phase" if phase else "resonator raw",
        )
        axes[1].plot(
            [r["distractors"] for r in rows],
            [r["baseline_joint_rate"] for r in rows],
            "x--",
            label="baseline (phase run)" if phase else "baseline (raw run)",
        )
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="x (box)",
        ylabel="y (box)",
        title="Truth ×; estimates ○ (red: failed)",
    )
    axes[1].set(
        xlabel="Distractor parents",
        ylabel="Joint identity + position rate",
        ylim=(-0.05, 1.05),
    )
    axes[1].legend(fontsize=7)
    proto_rows = result.get("prototype", [])
    if isinstance(proto_rows, list) and proto_rows:
        for name in ("prototype", "single", "foreign"):
            jitter = sorted({r["jitter"] for r in proto_rows})
            scores = [
                np.mean([r[name]["score"] for r in proto_rows if r["jitter"] == j])
                for j in jitter
            ]
            axes[2].plot(jitter, scores, "o-", label=name)
        axes[2].legend()
    axes[2].set(
        xlabel="Jitter / object size",
        ylabel="Mean PHAT score",
        title="Held-out objects; two training instances",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_synthetic(
    dim=4096,
    seed=0,
    figure=None,
    values=16,
    coarse=8,
    top_r=4,
    *,
    grid_size=32,
    band_floor=0.0,
    **options,
):
    """Report failures as well as successes; no truth enters the solver."""
    coarse_sigma = options.pop("coarse_sigma", None)
    if options:
        raise TypeError(f"unknown synthetic options: {sorted(options)}")
    started = perf_counter()
    rows = []
    for objects in (1, 2, 3):
        fixture = synthetic_fixture(dim, seed, objects, extras=4)
        rng = np.random.default_rng(seed + 100)
        distractor = synthetic_fixture(dim, seed + 100, 1)
        # Re-encode the unrelated parent on the SAME frequencies.
        other = fingerprint(distractor["parent"], fixture["freqs"], fixture["sigma"])
        additions = [
            composite([other], [rng.uniform(-0.3, 0.3, 3)], fixture["freqs"])
            for _ in range(8)
        ]
        for count in (0, 1, 2, 4, 8):
            signal = fixture["S"] + sum(additions[:count])
            baselines = {
                floor: one_shot_baseline(
                    signal,
                    fixture["identity"],
                    fixture["freqs"],
                    translation_grid(grid_size, 0.5) + 0.5,
                    band_floor=floor,
                    sigma=fixture["sigma"],
                )
                for floor in sorted({0.0, 0.3, band_floor})
            }
            for phase in (False, True):
                row = {
                    "objects": objects,
                    "distractors": count,
                    "phase_only": phase,
                    **_query(
                        fixture,
                        signal,
                        phase,
                        values,
                        coarse,
                        top_r,
                        grid_size,
                        coarse_sigma=coarse_sigma,
                        baselines=baselines[0.0],
                    ),
                }
                row["band_limited"] = {
                    str(floor): {
                        "queries": baseline,
                        **_metrics(
                            row["found"],
                            baseline,
                            fixture["truth"],
                            1 / (coarse * (values - 1)),
                            fixture["sigma"],
                            coarse,
                        ),
                    }
                    for floor, baseline in baselines.items()
                    if floor
                }
                rows.append(row)
    first_two = fixture["identity"][:2]
    bg = fingerprint(fixture["background"], fixture["freqs"], fixture["sigma"])
    parents = [
        fingerprint(c, fixture["freqs"], fixture["sigma"]) + bg / 2
        for c in fixture["crops"][:2]
    ]
    shifts = np.array([[0, 0, 0], [0.1, 0, 0]])
    combined = composite(parents, shifts, fixture["freqs"])
    composite_truth = [
        {"k": t["k"], "t": (np.array(t["t"]) + shift).tolist()}
        for t, shift in zip(fixture["truth"][:2], shifts)
    ]
    cliff = synthetic_fixture(1024, seed, 6)
    result = {
        "settings": {
            "dim": dim,
            "seed": seed,
            "values": values,
            "coarse": coarse,
            "top_r": top_r,
            "sigma_box": fixture["sigma"],
            "coarse_sigma": coarse_sigma,
            "band_floor": band_floor,
            "position_tolerance_box": max(
                1 / (coarse * (values - 1)), fixture["sigma"] / 2
            ),
            "tolerance_rule": "max(fine_step_box, sigma_box / 2)",
            "baseline_size": grid_size,
            "baseline_whiten": 1.0,
            "identity_modes": ["raw", "phase_only"],
        },
        "truth": fixture["truth"],
        "rows": rows,
        "prototype": _prototype_experiment(fixture, seed + 200, grid_size),
        "composite_truth": composite_truth,
        "composite": what_is_where(
            combined,
            first_two,
            fixture["freqs"],
            2,
            values=values,
            coarse=coarse,
            top_r=top_r,
        ),
        "capacity_cliff": _query(cliff),
    }
    result["composite_metrics"] = _metrics(
        result["composite"],
        one_shot_baseline(
            combined,
            first_two,
            fixture["freqs"],
            translation_grid(grid_size, 0.5) + 0.5,
        ),
        composite_truth,
        1 / (coarse * (values - 1)),
        fixture["sigma"],
        coarse,
    )
    result["coarse_sweep"] = []
    for coarse_count in (8, 16):
        for blur in (None, 1 / 32, 1 / 16, 1 / 8):
            for phase in (False, True):
                found = what_is_where(
                    fixture["S"],
                    fixture["identity"],
                    fixture["freqs"],
                    3,
                    values=values,
                    coarse=coarse_count,
                    top_r=top_r,
                    identity_phase_only=phase,
                    sigma=fixture["sigma"],
                    coarse_sigma=blur,
                )
                result["coarse_sweep"].append(
                    {
                        "coarse": coarse_count,
                        "coarse_sigma": blur,
                        "phase_only": phase,
                        **_metrics(
                            found,
                            next(
                                r["baseline_queries"]
                                for r in rows
                                if r["objects"] == 3 and r["distractors"] == 0
                            ),
                            fixture["truth"],
                            1 / (coarse_count * (values - 1)),
                            fixture["sigma"],
                            coarse_count,
                        ),
                        "found": found,
                    }
                )
    gram = fixture["identity"] / np.linalg.norm(fixture["identity"], axis=1)[:, None]
    phase = FHRR.normalize(fixture["identity"]) / np.sqrt(dim)
    off_diagonal = ~np.eye(len(gram), dtype=bool)
    result["identity_gram"] = {
        "raw_max": float(np.abs(gram @ gram.conj().T)[off_diagonal].max()),
        "phase_max": float(np.abs(phase @ phase.conj().T)[off_diagonal].max()),
    }
    one = composite(
        fixture["identity"][:1], [fixture["truth"][0]["t"]], fixture["freqs"]
    )
    result["grid_control"] = {
        str(size): one_shot_baseline(
            one,
            fixture["identity"][:1],
            fixture["freqs"],
            translation_grid(size, 0.5) + 0.5,
        )[0]
        for size in (16, 32)
    }
    if figure is not None:
        _figure(result, figure)
    result["elapsed_seconds"] = perf_counter() - started
    return result


def _capture(args):
    lo, extent = crop_box(args.parent)
    scene, _, _ = build_scene_fixed(args.parent, lo, extent)
    # Infer object size in physical units if no explicit blur was supplied.
    crop, _, _ = build_scene_fixed(args.crop[0], lo, extent)
    sigma = args.sigma_units or float(np.ptp(crop.mu, axis=0).max() * extent / 8)
    rng = np.random.default_rng(args.seed)
    freqs = sample_frequencies(args.dim, 3, extent / sigma, rng)
    pairs = [object_codeword(p, lo, extent, freqs, sigma) for p in args.crop]
    external = [
        _external_parent(p, c, freqs, extent, sigma)[1] for p, c in args.candidate
    ]
    identity = np.array([*[c for c, _ in pairs], *external])
    signal = fingerprint(scene, freqs, sigma / extent)
    truth = [{"k": k, "t": t.tolist()} for k, (_, t) in enumerate(pairs)]
    others = [_external_parent(p, p, freqs, extent, sigma)[0] for p in args.distractor]
    additions = (
        [
            composite([others[i % len(others)]], [rng.uniform(-0.3, 0.3, 3)], freqs)
            for i in range(8)
        ]
        if others
        else []
    )
    rows = []
    for count in (0, 1, 2, 4, 8) if others else (0,):
        current = signal + sum(additions[:count])
        found = what_is_where(
            current,
            identity,
            freqs,
            len(truth),
            values=args.values,
            coarse=args.coarse,
            top_r=args.top_r,
            identity_phase_only=args.identity_phase_only,
            extent=extent,
            sigma=sigma / extent,
            coarse_sigma=args.coarse_sigma,
        )
        baseline = one_shot_baseline(
            current,
            identity,
            freqs,
            translation_grid(args.grid, 0.5) + 0.5,
            band_floor=args.band_floor,
            sigma=sigma / extent,
        )
        rows.append(
            {
                "objects": len(truth),
                "identity_count": len(identity),
                "distractors": count,
                "phase_only": args.identity_phase_only,
                "found": found,
                "baseline_queries": baseline,
                **_metrics(
                    found,
                    baseline,
                    truth,
                    1 / (args.coarse * (args.values - 1)),
                    sigma / extent,
                    args.coarse,
                ),
            }
        )
    for row in rows:
        for item in row["found"]:
            error = item["error_box"]
            item["error_scene"] = error * extent if error is not None else None
            item["error_coarse_cells"] = (
                error * args.coarse if error is not None else None
            )
    result = {
        "truth": truth,
        "rows": rows,
        "extent": extent,
        "settings": {
            "sigma_units": sigma,
            "values": args.values,
            "coarse": args.coarse,
            "top_r": args.top_r,
            "baseline_size": args.grid,
            "band_floor": args.band_floor,
            "sigma_box": sigma / extent,
        },
    }
    result.update(_capture_association(args, signal, identity, freqs, extent, sigma))
    return result


def _external_parent(parent, crop, freqs, extent, sigma):
    lo, own_extent = crop_box(parent)
    scene, _, _ = build_scene_fixed(parent, lo, own_extent)
    obj, _, _ = build_scene_fixed(crop, lo, own_extent)
    ratio = own_extent / extent
    scaled = [SplatScene(s.mu * ratio, s.cov * ratio**2, s.amp) for s in (scene, obj)]
    return (
        fingerprint(scaled[0], freqs, sigma / extent),
        _center(scaled[1], freqs, sigma / extent)[0],
    )


def _capture_association(args, signal, identity, freqs, extent, sigma):
    if args.instances:
        lo, _ = crop_box(args.parent)
        training = [
            object_codeword(p, lo, extent, freqs, sigma)[0] for p in args.instances
        ]
        searches = prototype_search(
            signal,
            training,
            freqs,
            translation_grid(args.grid, 0.5) + 0.5,
            sigma / extent,
        )
        target = object_codeword(args.crop[0], lo, extent, freqs, sigma)[1]
        return {
            "prototype": _prototype_errors(searches, target),
            "composite": {"status": "not run: source parent required"},
        }
    if len(args.instance) != 2:
        return {
            "prototype": {"status": "not run: supply two --instance PARENT CROP"},
            "composite": {"status": "not run: source parent required"},
        }
    sources = [_external_parent(p, c, freqs, extent, sigma) for p, c in args.instance]
    proto = prototype([c for _, c in sources]) * float(_rms(identity[0])[0])
    dictionary = np.vstack([proto, identity[1:]])
    association = analogy(
        signal,
        sources[0][0],
        identity[0],
        dictionary,
        freqs,
        translation_grid(args.grid, 0.5) + 0.5,
        values=args.values,
        coarse=args.coarse,
        top_r=args.top_r,
        identity_phase_only=args.identity_phase_only,
        extent=extent,
    )
    association["correlation"] = prototype_search(
        signal,
        [c for _, c in sources],
        freqs,
        translation_grid(args.grid, 0.5) + 0.5,
        sigma / extent,
    )
    lo, _ = crop_box(args.parent)
    target = object_codeword(args.crop[0], lo, extent, freqs, sigma)[1]
    _prototype_errors(association["correlation"], target)
    combined = composite([signal, sources[0][0]], [[0, 0, 0], args.shift], freqs)
    recovered = what_is_where(
        combined,
        np.vstack([identity[0], sources[0][1]]),
        freqs,
        2,
        values=args.values,
        coarse=args.coarse,
        top_r=args.top_r,
        extent=extent,
        identity_phase_only=args.identity_phase_only,
    )
    return {
        "prototype": association,
        "composite": recovered,
        "association_labels": [
            "prototype of two sources",
            *(Path(p).name for p in args.crop[1:]),
            *(Path(c).name for _, c in args.candidate),
        ],
        "composite_labels": [Path(args.crop[0]).name, Path(args.instance[0][1]).name],
    }


def comparison_trials(dim=4096, seed=0, trials=30):
    """Two objects, four unused identities; independent seeded parents.

    Eight samples per local axis and an eight-point baseline coarse axis
    keep this CPU regression small. Joint correctness uses max(fine step, sigma / 2).
    """
    rows = [
        _query(synthetic_fixture(dim, seed + i, 2, extras=4), values=8, baseline_size=8)
        for i in range(trials)
    ]
    return {
        "trials": trials,
        "seed": seed,
        "values": 8,
        "resonator_joint_rate": float(
            np.mean([r["resonator_joint_rate"] for r in rows])
        ),
        "baseline_joint_rate": float(np.mean([r["baseline_joint_rate"] for r in rows])),
        "per_trial": [
            [r["resonator_joint_rate"], r["baseline_joint_rate"]] for r in rows
        ],
    }


def _validate_args(parser, args):
    if not args.synthetic and not args.trials and (not args.parent or not args.crop):
        parser.error("supply --synthetic or PARENT with at least one --crop")
    if args.sigma_units is not None and (
        not np.isfinite(args.sigma_units) or args.sigma_units <= 0
    ):
        parser.error("--sigma-units must be finite and positive")
    if min(args.dim, args.values, args.coarse, args.grid) < 2 or args.trials < 0:
        parser.error("dimension/grid sizes must be >= 2 and trials nonnegative")
    if not 1 <= args.top_r <= args.coarse or len(args.instance) not in (0, 2):
        parser.error("top-r must be in [1, coarse]; supply zero or two instances")
    if not 0 <= args.band_floor < 1:
        parser.error("--band-floor must lie in [0, 1)")
    if args.instances and args.instance:
        parser.error("choose --instances or --instance pairs")
    if args.prototype and not (args.synthetic or args.instances or args.instance):
        parser.error("--prototype requires two training instances")
    if args.synthetic and (
        args.instances
        or args.parent
        or args.crop
        or args.instance
        or args.distractor
        or args.candidate
    ):
        parser.error("synthetic fixtures and capture inputs are alternatives")


def main(argv=None):
    """Run on a seeded fixture or a parent with crops in its original frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("parent", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument(
        "--instance", nargs=2, action="append", default=[], metavar=("PARENT", "CROP")
    )
    parser.add_argument("--instances", nargs=2, metavar=("FIRST", "SECOND"))
    parser.add_argument("--frame-of", choices=["parent"], default="parent")
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="search two training instances by phase-only correlation",
    )
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--band-floor", type=float, default=0.0)
    parser.add_argument(
        "--coarse-sigma", type=float, help="stage-1 blur in normalized box units"
    )
    parser.add_argument("--shift", type=float, nargs=3, default=[0.1, 0, 0])
    parser.add_argument("--crop", action="append", default=[])
    parser.add_argument("--distractor", action="append", default=[])
    parser.add_argument(
        "--candidate", nargs=2, action="append", default=[], metavar=("PARENT", "CROP")
    )
    parser.add_argument("--sigma-units", type=float)
    parser.add_argument("--values", type=int, default=32)
    parser.add_argument("--coarse", type=int, default=8)
    parser.add_argument("--top-r", type=int, default=4)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--identity-phase-only", action="store_true")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trials:
        result = comparison_trials(args.dim, args.seed, args.trials)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return result
    result = (
        run_synthetic(
            args.dim,
            args.seed,
            args.figure,
            args.values,
            args.coarse,
            args.top_r,
            grid_size=args.grid,
            band_floor=args.band_floor,
            coarse_sigma=args.coarse_sigma,
        )
        if args.synthetic
        else _capture(args)
    )
    if args.figure and not args.synthetic:
        _figure(result, args.figure)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
