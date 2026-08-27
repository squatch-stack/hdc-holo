"""Experiments: Gaussian splats as complex64 hypervector bundles.

1. 2D RGB splat scene decoded from a single bundle at increasing dimension d.
2. Capacity: relative error vs d for N = 100 / 1,000 / 10,000 splats (3D),
   for both the anisotropic spectral encoder and the classic FHRR phasor
   encoder -- both should follow the sqrt(N/d) crosstalk law.
3. Bundle algebra: translating the whole scene with one elementwise multiply.

Outputs PNGs into results/ and prints a summary.
"""

import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hdc_splat import (
    SplatScene, random_scene, sample_frequencies, spectral_bundle,
    phasor_bundle, decode_field, decode_field_phasor, translate_bundle,
    eval_scene_exact,
)

# ---- reference palette (validated: slots 1-3 pass all-pairs, light mode) ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # blue, orange, aqua
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

MIN_SCALE, MAX_SCALE = 0.02, 0.045
SIGMA_RHO = 1.3 / MIN_SCALE          # cover the narrowest splat's spectrum


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)


def rel_err(approx, exact):
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


def render_grid(res):
    xs = np.linspace(0.0, 1.0, res, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def fmt_bytes(nbytes):
    return f"{nbytes / 1024:.0f} KB" if nbytes < 1 << 20 else f"{nbytes / (1 << 20):.1f} MB"


# ---------------------------------------------------------------------------
# Experiment 1: 2D RGB reconstruction vs dimension
# ---------------------------------------------------------------------------

def experiment_recon(rng):
    print("== Experiment 1: 2D reconstruction vs d ==")
    n, res = 128, 160
    d_values = [1024, 4096, 16384, 65536]
    scene = random_scene(n, dim=2, rng=rng, scale_range=(MIN_SCALE, MAX_SCALE),
                         channels=3, amp_range=(0.15, 1.0))
    freqs_max = sample_frequencies(max(d_values), 2, SIGMA_RHO, rng)
    bundle_max = spectral_bundle(scene, freqs_max)

    pts = render_grid(res)
    exact = eval_scene_exact(scene, pts)
    recons, errs = [], []
    for d in d_values:
        t0 = time.time()
        approx = decode_field(bundle_max[:, :d], freqs_max[:d], SIGMA_RHO, pts,
                              chunk=512)
        e = rel_err(approx, exact)
        recons.append(approx)
        errs.append(e)
        print(f"  d={d:6d}  rel err {e:6.3f}  bundle {fmt_bytes(3 * d * 8)}"
              f"  decode {time.time() - t0:5.1f}s")

    splat_bytes = scene.n * (2 + 3 + 3) * 4     # mu + sym cov + rgb, float32
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4))
    fig.patch.set_facecolor(PAGE)
    panels = [("ground truth (mixture eval)", exact, None)] + [
        (f"d = {d:,}", r, e) for d, r, e in zip(d_values, recons, errs)]
    for ax, (title, field, e) in zip(axes, panels):
        img = np.clip(field, 0, 1).reshape(res, res, 3)
        ax.imshow(img, origin="lower", extent=(0, 1, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#d8d7d2"); side.set_linewidth(0.75)
        ax.set_title(title, fontsize=10, color=INK, pad=6)
        if e is not None:
            ax.set_xlabel(f"rel. error {100 * e:.1f}%", fontsize=9, color=INK2)
    fig.suptitle(f"{n} anisotropic RGB splats decoded from one complex64 bundle "
                 f"(explicit splats: {fmt_bytes(splat_bytes)})",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "recon_2d.png"), dpi=150,
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    return scene, freqs_max, bundle_max, exact, recons[-1], res, errs


# ---------------------------------------------------------------------------
# Experiment 2: capacity -- error vs d for several N, both encoders
# ---------------------------------------------------------------------------

def experiment_capacity(rng):
    """Crosstalk noise vs d. Metric: RMS decode error per unit mean splat
    amplitude -- *absolute* noise, since it is what follows sqrt(N/d); the
    field's own norm grows with N once splats overlap densely, which would
    make a relative-error plot saturate. Averaged over codebook seeds."""
    print("== Experiment 2: capacity curves (3D) ==")
    n_values = [100, 1000, 10000]
    d_values = [512, 2048, 8192, 32768]
    d_max = max(d_values)
    iso_sigma = 0.03
    n_seeds = 3

    results = {}   # (encoder, n) -> [noise per d]
    for n in n_values:
        # constant splat density: box grows with N, so crosstalk between
        # overlapping splats stays a fixed fraction and sqrt(N) is isolated
        box = (n / n_values[0]) ** (1.0 / 3.0)
        scene = random_scene(n, dim=3, rng=rng, box=box,
                             scale_range=(MIN_SCALE, MAX_SCALE), channels=1)
        near = (scene.mu[rng.integers(0, n, 1024)]
                + 0.03 * rng.standard_normal((1024, 3))).astype(np.float32)
        uniform = rng.uniform(0, box, size=(1024, 3)).astype(np.float32)
        pts = np.concatenate([near, uniform])
        mean_amp = float(scene.amp.mean())

        exact_aniso = eval_scene_exact(scene, pts)
        exact_iso = eval_scene_exact(scene, pts, iso_sigma=iso_sigma)

        sq = {(enc, d): [] for enc in ("spectral", "phasor") for d in d_values}
        for seed in range(n_seeds):
            book_rng = np.random.default_rng(1000 + seed)
            freqs_s = sample_frequencies(d_max, 3, SIGMA_RHO, book_rng)
            freqs_p = sample_frequencies(d_max, 3, 1.0 / iso_sigma, book_rng)
            b_s = spectral_bundle(scene, freqs_s)
            b_p = phasor_bundle(scene, freqs_p)
            for enc, bundle, freqs, exact in (
                    ("spectral", b_s, freqs_s, exact_aniso),
                    ("phasor", b_p, freqs_p, exact_iso)):
                for d in d_values:
                    if enc == "spectral":
                        approx = decode_field(bundle[:, :d], freqs[:d],
                                              SIGMA_RHO, pts)
                    else:
                        approx = decode_field_phasor(bundle[:, :d], freqs[:d], pts)
                    rms = float(np.sqrt(np.mean((approx - exact) ** 2)))
                    sq[(enc, d)].append((rms / mean_amp) ** 2)

        for enc in ("spectral", "phasor"):
            errs = [float(np.sqrt(np.mean(sq[(enc, d)]))) for d in d_values]
            results[(enc, n)] = errs
            print(f"  N={n:5d} {enc:8s}  " +
                  "  ".join(f"d={d}:{e:.3f}" for d, e in zip(d_values, errs)))

    # fitted power-law exponent in d, pooled per encoder
    slopes = {}
    for enc in ("spectral", "phasor"):
        pts_fit = [(np.log(d), np.log(results[(enc, n)][i]))
                   for n in n_values for i, d in enumerate(d_values)]
        x = np.array([p[0] for p in pts_fit])
        y = np.array([p[1] for p in pts_fit])
        x_c = x - x.mean()
        slopes[enc] = float((x_c @ (y - y.mean())) / (x_c @ x_c))
    print(f"  fitted error ~ d^b: spectral b={slopes['spectral']:.2f}, "
          f"phasor b={slopes['phasor']:.2f}  (theory -0.50)")

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig.patch.set_facecolor(PAGE)
    style_axes(ax)
    for ci, n in enumerate(n_values):
        for enc, ls in (("spectral", "-"), ("phasor", "--")):
            errs = results[(enc, n)]
            ax.plot(d_values, errs, ls, color=SERIES[ci], linewidth=1.8,
                    marker="o", markersize=5.5, markerfacecolor=SERIES[ci],
                    markeredgecolor=SURFACE, markeredgewidth=1)
        ax.annotate(f"N = {n:,}", xy=(d_values[-1], results[("spectral", n)][-1]),
                    xytext=(8, 0), textcoords="offset points", fontsize=9.5,
                    color=SERIES[ci], va="center")
    guide_d = np.array(d_values, dtype=float)
    for n in n_values:
        ax.plot(guide_d, np.sqrt(n / (2 * guide_d)), ":", color=GRID,
                linewidth=1.2, zorder=1)
    ax.annotate("guides: phasor theory √(N∕2d)",
                xy=(guide_d[0], np.sqrt(n_values[0] / (2 * guide_d[0]))),
                xytext=(10, -14), textcoords="offset points",
                fontsize=9.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(d_values)
    ax.set_xticklabels([f"{d:,}" for d in d_values])
    ax.set_xlim(d_values[0] * 0.85, d_values[-1] * 2.1)
    ax.set_xlabel("hypervector dimension d (complex64 components)",
                  fontsize=10, color=INK2)
    ax.set_ylabel("crosstalk noise (RMS error ∕ mean splat amplitude)",
                  fontsize=10, color=INK2)
    fig.suptitle("Crosstalk of superposed splats follows √(N∕d)",
                 fontsize=12.5, color=INK, y=0.985)
    ax.set_title("3D scenes at constant splat density (box side ∝ N^⅓), "
                 f"averaged over {n_seeds} codebook seeds",
                 fontsize=9.5, color=INK2, pad=8)
    handles = [Line2D([], [], color=SERIES[i], linewidth=1.8, marker="o",
                      markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1,
                      label=f"N = {n:,} splats") for i, n in enumerate(n_values)]
    handles += [Line2D([], [], color=INK2, linewidth=1.8, linestyle=ls, label=lab)
                for ls, lab in (("-", "spectral (anisotropic)"),
                                ("--", "phasor FHRR (shared kernel)"))]
    ax.legend(handles=handles, fontsize=9, frameon=False, labelcolor=INK2,
              loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "capacity_curve.png"), dpi=150,
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    return results, slopes


# ---------------------------------------------------------------------------
# Experiment 3: whole-scene translation by binding
# ---------------------------------------------------------------------------

def experiment_translation(scene, freqs, bundle, recon, res):
    print("== Experiment 3: translation by binding ==")
    t = np.array([0.18, -0.12], dtype=np.float32)
    pts = render_grid(res)

    shifted_bundle = translate_bundle(bundle, freqs, t)
    t0 = time.time()
    recon_shifted = decode_field(shifted_bundle, freqs, SIGMA_RHO, pts, chunk=512)
    dt = time.time() - t0
    shifted_scene = SplatScene(mu=scene.mu + t, cov=scene.cov, amp=scene.amp)
    exact_shifted = eval_scene_exact(shifted_scene, pts)
    e = rel_err(recon_shifted, exact_shifted)
    print(f"  rel err after translation {e:.3f} (decode {dt:.1f}s); "
          f"bind cost: one elementwise multiply of {bundle.shape[1]:,} complex64")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    fig.patch.set_facecolor(PAGE)
    panels = [
        ("decoded scene  S   (d = 65,536)", np.clip(recon, 0, 1)),
        ("decoded  S ⊙ e^(−i W·t)", np.clip(recon_shifted, 0, 1)),
        ("ground truth, splats moved by t", np.clip(exact_shifted, 0, 1)),
    ]
    for ax, (title, field) in zip(axes, panels):
        ax.imshow(field.reshape(res, res, 3), origin="lower", extent=(0, 1, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#d8d7d2"); side.set_linewidth(0.75)
        ax.set_title(title, fontsize=10, color=INK, pad=6)
    axes[1].set_xlabel(f"one complex multiply on the bundle — rel. error {100 * e:.1f}%",
                       fontsize=9, color=INK2)
    fig.suptitle("Translating every splat at once = binding the bundle with one phasor",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "translation.png"), dpi=150,
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    return e


def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    scene, freqs, bundle, exact, recon_best, res, recon_errs = experiment_recon(rng)
    results, slopes = experiment_capacity(rng)
    trans_err = experiment_translation(scene, freqs, bundle, recon_best, res)
    print(f"\nAll experiments done in {time.time() - t0:.0f}s. "
          f"Figures in {RESULTS}/")


if __name__ == "__main__":
    main()
