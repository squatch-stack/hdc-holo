"""Mixture-of-Gaussians frequency codebook vs a single-sigma codebook.

The spectral encoder's importance weights are 1/rho(w_j). A single
rho = N(0, sigma_rho^2 I) must cover the *narrowest* splat's spectrum, so
on a scene with a spread of splat scales the *widest* splats -- whose
spectra live at low frequencies where that rho is thin relative to their
energy -- pay a large Monte-Carlo variance penalty. A codebook drawn from
an equal-weight mixture of Gaussians spanning the scale range keeps
1/rho bounded over every splat's spectral band, at the small fixed cost
of splitting the codebook M ways.

Relation to hdc/spatial.py's MultiBandSplatField: bands quantize
covariance into discrete classes, one codebook and one bundle per class,
with each splat's covariance snapped to its band's. The mixture codebook
here keeps ONE codebook and ONE bundle and preserves every splat's exact
anisotropic covariance (it lives in the spectrum's magnitude envelope);
the mixture is over *sampling scales*, purely a variance-reduction
device. The two compose: bands for coarse classes, a mixture rho within
each band.

Scene: 3D, N = 999 anisotropic splats, base scales log-uniform in
[0.02, 0.10] (a 5x spread). Metric: RMS decode error per unit mean splat
amplitude (the crosstalk-noise metric of examples/run_prototype.py), averaged over
codebook seeds. The dotted floor is the phasor-encoder theory sqrt(N/2d),
the best any shared-codebook superposition can do here.

Empirical finding (see results/mog_penalty.png): the single-sigma curve
does not just sit 16-33x above the phasor floor -- it flattens at large d,
because its importance weights are heavy-tailed: variance is finite only
while Sigma_splat - I/(2 sigma_rho^2) stays positive definite, and the
narrowest splats sit close to that boundary, so the Monte-Carlo average
converges erratically. The mixture codebook restores clean ~1/sqrt(d)
scaling and cuts noise 3-10x, landing ~2.4-3.2x over the floor (the
M-way codebook split costs at most sqrt(M) = 2x of that).

Outputs results/mog_penalty.png and a summary table.
"""

import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hdc_splat import (
    SplatScene,
    decode_field,
    eval_scene_exact,
    random_rotations_3d,
    sample_frequencies,
    spectral_bundle,
)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

S_MIN, S_MAX = 0.02, 0.10
N_SPLATS = 999
D_VALUES = [512, 2048, 8192, 32768]
N_SEEDS = 3
SINGLE_RHO = 1.3 / S_MIN
MIX_RHO = list(1.3 / np.geomspace(S_MIN, S_MAX, 4))   # [65, 38, 22, 13]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)


def multiscale_scene(n, rng):
    """Anisotropic 3D splats whose base scale spans S_MIN..S_MAX
    log-uniformly; returns the scene and each splat's base scale."""
    mu = rng.uniform(0.08, 0.92, size=(n, 3)).astype(np.float32)
    base = np.exp(rng.uniform(np.log(S_MIN), np.log(S_MAX), size=n))
    axes = (base[:, None] * rng.uniform(0.7, 1.3, size=(n, 3))).astype(np.float32)
    rots = random_rotations_3d(n, rng)
    cov = np.einsum("nij,nj,nkj->nik", rots, axes**2, rots).astype(np.float32)
    amp = rng.uniform(0.3, 1.0, size=(n, 1)).astype(np.float32)
    return SplatScene(mu=mu, cov=cov, amp=amp), base.astype(np.float32)


def subset(scene, idx):
    return SplatScene(mu=scene.mu[idx], cov=scene.cov[idx], amp=scene.amp[idx])


def noise_of(scene, rho, freqs, pts, exact, mean_amp):
    bundle = spectral_bundle(scene, freqs)
    errs = []
    for d in D_VALUES:
        approx = decode_field(bundle[:, :d], freqs[:d], rho, pts)
        errs.append(np.mean((approx - exact) ** 2) / mean_amp**2)
    return errs  # squared, for seed-averaging


def _figure(codebooks, curves, d_ter, floor, t0, ter_names, ter_noise):
    """The penalty curves and the ternary-scene comparison."""
    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.8, 4.9), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(PAGE)
    for ax in (ax1, ax2):
        style_axes(ax)

    guide_d = np.array(D_VALUES, dtype=float)
    ax1.plot(guide_d, floor, ":", color=MUTED, linewidth=1.4, zorder=1)
    ax1.annotate("phasor theory √(N∕2d)", xy=(guide_d[1], floor[1]),
                 xytext=(6, -13), textcoords="offset points",
                 fontsize=9, color=MUTED)
    for ci, (name, _) in enumerate(codebooks):
        ax1.plot(D_VALUES, curves[name], "-", color=SERIES[ci], linewidth=1.8,
                 marker="o", markersize=5.5, markerfacecolor=SERIES[ci],
                 markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        ax1.annotate(name, xy=(D_VALUES[-1], curves[name][-1]),
                     xytext=(8, 0), textcoords="offset points",
                     fontsize=9.5, color=SERIES[ci], va="center")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(D_VALUES)
    ax1.set_xticklabels([f"{d:,}" for d in D_VALUES])
    ax1.set_xlim(D_VALUES[0] * 0.85, D_VALUES[-1] * 2.6)
    ax1.set_xlabel("hypervector dimension d", fontsize=10, color=INK2)
    ax1.set_ylabel("crosstalk noise (RMS error ∕ mean amplitude)",
                   fontsize=10, color=INK2)
    ax1.set_title("Multi-scale scene: one σ_ρ vs a 4-scale mixture codebook",
                  fontsize=11, color=INK, pad=10)
    ax1.legend(fontsize=9, frameon=False, labelcolor=INK2, loc="lower left")

    x = np.arange(3)
    for ci, (name, _) in enumerate(codebooks):
        vals = ter_noise[name]
        bars = ax2.bar(x + (ci - 0.5) * 0.38, vals, width=0.36,
                       color=SERIES[ci], edgecolor=SURFACE, linewidth=1,
                       label=name)
        for b, v in zip(bars, vals):
            ax2.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8.5, color=INK2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"narrow\n{ter_names[0]}", f"mid\n{ter_names[1]}",
                         f"wide\n{ter_names[2]}"], fontsize=9)
    ax2.set_ylabel(f"noise contributed (d = {d_ter:,})", fontsize=10, color=INK2)
    ax2.set_title("Who pays the penalty: noise by splat scale",
                  fontsize=11, color=INK, pad=10)
    ax2.legend(fontsize=9, frameon=False, labelcolor=INK2, loc="upper left")
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Mixture-of-Gaussians codebook shrinks the spectral "
                 "encoder's importance-sampling penalty",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout()
    path = os.path.join(RESULTS, "mog_penalty.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    print(f"saved {path}  ({time.time() - t0:.0f}s)")




def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    scene, base_scale = multiscale_scene(N_SPLATS, rng)
    mean_amp = float(scene.amp.mean())
    near = (scene.mu[rng.integers(0, N_SPLATS, 1024)]
            + 0.03 * rng.standard_normal((1024, 3))).astype(np.float32)
    pts = np.concatenate([near, rng.uniform(0, 1, (1024, 3)).astype(np.float32)])
    exact = eval_scene_exact(scene, pts)

    order = np.argsort(base_scale)
    terciles = np.array_split(order, 3)
    ter_names = [f"σ {base_scale[t].min():.3f}–{base_scale[t].max():.3f}"
                 for t in terciles]
    ter_exact = [eval_scene_exact(subset(scene, t), pts) for t in terciles]

    codebooks = [("single ρ", SINGLE_RHO), ("mixture ρ", MIX_RHO)]
    curves = {name: np.zeros(len(D_VALUES)) for name, _ in codebooks}
    ter_noise = {name: np.zeros(3) for name, _ in codebooks}
    d_ter = 8192

    for seed in range(N_SEEDS):
        book_rng = np.random.default_rng(1000 + seed)
        for name, rho in codebooks:
            freqs = sample_frequencies(max(D_VALUES), 3, rho, book_rng)
            curves[name] += noise_of(scene, rho, freqs, pts, exact, mean_amp)
            for i, t in enumerate(terciles):
                b = spectral_bundle(subset(scene, t), freqs[:d_ter])
                approx = decode_field(b, freqs[:d_ter], rho, pts)
                ter_noise[name][i] += np.mean(
                    (approx - ter_exact[i]) ** 2) / mean_amp**2

    for name in curves:
        curves[name] = np.sqrt(curves[name] / N_SEEDS)
        ter_noise[name] = np.sqrt(ter_noise[name] / N_SEEDS)

    floor = np.sqrt(N_SPLATS / (2 * np.array(D_VALUES, dtype=float)))
    print(f"splat scales {S_MIN}-{S_MAX} (5x), N={N_SPLATS}, "
          f"{N_SEEDS} codebook seeds")
    print(f"{'d':>7} {'single ρ':>10} {'mixture ρ':>10} {'floor √(N∕2d)':>14}"
          f" {'penalty single':>15} {'penalty mix':>12}")
    for i, d in enumerate(D_VALUES):
        print(f"{d:>7} {curves['single ρ'][i]:>10.3f} "
              f"{curves['mixture ρ'][i]:>10.3f} {floor[i]:>14.3f} "
              f"{curves['single ρ'][i]/floor[i]:>14.2f}x "
              f"{curves['mixture ρ'][i]/floor[i]:>11.2f}x")
    print(f"noise by splat-scale tercile at d={d_ter}:")
    for i, tn in enumerate(ter_names):
        print(f"  {tn:>18}: single {ter_noise['single ρ'][i]:.3f}  "
              f"mixture {ter_noise['mixture ρ'][i]:.3f}")

    _figure(codebooks, curves, d_ter, floor, t0, ter_names, ter_noise)

if __name__ == "__main__":
    main()
