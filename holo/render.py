"""Closed-form ray rendering: images straight out of a bundle.

The bundle stores the scene's Fourier transform sampled at random
frequencies — sum_k alpha_k e^{i w_j . mu_k} IS that transform at w_j.
The projection-slice theorem then says an orthographic projection of the
scene lives on the slice w . v = 0 of the spectrum. Integrating the
holographic field estimator along a ray o + t v does exactly that, in
closed form:

    integral over [0, T] of e^{i w.(o + t v)} dt
        = e^{i w.o} (e^{i (w.v) T} - 1) / (i (w.v))

So a whole VIEW is just another bundle: fold the per-frequency slice
factor F_j = (e^{i a_j T} - 1)/(i a_j), a_j = w_j . v (with F_j -> T as
a_j -> 0), into conj(S) once per view direction, and every pixel is
again a single inner product with an ordinary point codeword on the
image plane. No ray marching, no depth sampling, no sorting.

This is emission/X-ray tomography, not alpha compositing: line integrals
are linear, occlusion is not, and superposition cannot express the
ordered product of transmittances. What you get is a true parallel
projection of the density field.

Noise budget: |F_j| <= min(T, 2/|a_j|), so frequencies steep along the
view direction are suppressed — only the ~perpendicular slice of the
spectrum carries signal, shrinking the effective dimension. Renders are
noisier than point queries at the same d; spend dimension accordingly.
"""

import numpy as np

from .field import GaussianSplatField


def _camera_basis(view):
    v = np.asarray(view, dtype=np.float64)
    v = v / np.linalg.norm(v)
    up = np.array([0.0, 0.0, 1.0])
    if abs(v @ up) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    u1 = np.cross(up, v)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    return v.astype(np.float32), u1.astype(np.float32), u2.astype(np.float32)


def view_bundle(S, W, view, t_extent):
    """Fold the slice factor for direction `view` into the bundle:
    the returned vector renders this view by plain point queries."""
    v, _, _ = _camera_basis(view)
    a = (W @ v).astype(np.float64)
    T = float(t_extent)
    F = np.where(np.abs(a) < 1e-9 / T, T,
                 (np.exp(1j * a * T) - 1.0) / (1j * np.where(a == 0, 1, a)))
    return (np.conj(S) * F).astype(np.complex64)


def render_orthographic(S, W, view, center, half_width, res, t_extent,
                        chunk=2048):
    """Parallel-project the holographic field onto a res x res image
    plane perpendicular to `view`, integrating rays of length t_extent
    centered on the plane through `center`.

    S may be a single bundle (d,) -> (res, res) image, or c stacked
    channel bundles (c, d) -> (res, res, c): color rides along as extra
    columns in the same per-pixel inner product."""
    S = np.atleast_2d(S)                                   # (c, d)
    multi = S.shape[0] > 1
    v, u1, u2 = _camera_basis(view)
    Sv = np.stack([view_bundle(s, W, view, t_extent) for s in S], axis=1)
    xs = np.linspace(-half_width, half_width, res, dtype=np.float32)
    px, py = np.meshgrid(xs, xs)
    origins = (np.asarray(center, dtype=np.float32)
               + px.reshape(-1, 1) * u1 + py.reshape(-1, 1) * u2
               - (t_extent / 2) * v)
    # readout computes Re(E @ conj(.))/d; Sv already IS conj(S)*F, so
    # hand it conj(Sv) to cancel — every pixel is then one backend call
    from .accel import readout
    out = readout(origins, W, np.conj(Sv.T), chunk=chunk)
    img = out.reshape(res, res, -1)
    return img if multi else img[..., 0]


def exact_projection(splats, sigma_inv, view, center, half_width, res):
    """Analytic full-line integral of each splat: for isotropic-enough
    rays the integral of a Gaussian along a line is a 1-D Gaussian
    integral with closed form. Ground truth for the renderer."""
    v, u1, u2 = _camera_basis(view)
    sigma2 = 1.0 / sigma_inv[0, 0]        # isotropic Sigma assumed
    xs = np.linspace(-half_width, half_width, res)
    px, py = np.meshgrid(xs, xs)
    origins = (np.asarray(center, dtype=np.float64)
               + px.reshape(-1, 1) * u1.astype(np.float64)
               + py.reshape(-1, 1) * u2.astype(np.float64))
    out = np.zeros(len(origins))
    for mu, alpha in splats:
        delta = origins - mu
        along = delta @ v.astype(np.float64)
        perp2 = (delta ** 2).sum(axis=1) - along ** 2
        out += alpha * np.sqrt(2 * np.pi * sigma2) \
            * np.exp(-perp2 / (2 * sigma2))
    return out.reshape(res, res)


def trefoil_points(n):
    """n points along a trefoil knot, scaled into the unit cube, with
    the curve parameter (for coloring along the curve)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    knot = np.stack([np.sin(t) + 2 * np.sin(2 * t),
                     np.cos(t) - 2 * np.cos(2 * t),
                     -np.sin(3 * t)], axis=1)
    return 0.5 + knot / (2 * np.abs(knot).max()) * 0.72, t


def _trefoil_scene(dim, sigma, n_splats, seed=0):
    """Splats strung along a trefoil knot — projections differ strongly
    with view angle, which is the point of a renderer."""
    field = GaussianSplatField(dim, np.eye(3) * sigma ** 2, seed=seed)
    knot, _ = trefoil_points(n_splats)
    for mu in knot:
        field.add_splat(mu, 1.0)
    return field


def demo(dim=4096, seed=0, save_png=True):
    d = max(dim, 16384)   # projections need headroom (see module docstring)
    sigma = 0.025
    print(f"== Closed-form ray rendering (d={d}) ==")
    field = _trefoil_scene(d, sigma, 240, seed=seed)
    center, half, T = np.array([0.5, 0.5, 0.5]), 0.45, 1.6
    res = 160

    views = {"view (1,0,0)": [1, 0, 0],
             "view (1,1,0)": [1, 1, 0],
             "view (1,1,1)": [1, 1, 1]}
    rows = []
    print(f"  {'view':>12} {'RMSE':>7} {'rel peak':>9}")
    for name, view in views.items():
        truth = exact_projection(field.splats, field.sigma_inv, view,
                                 center, half, res)
        holo = render_orthographic(field.S, field.W, view, center, half,
                                   res, T)
        rmse = float(np.sqrt(np.mean((holo - truth) ** 2)))
        rows.append((name, truth, holo))
        print(f"  {name:>12} {rmse:>7.4f} {rmse/truth.max():>9.1%}")
    print(f"  240-splat trefoil knot; each view = fold one slice factor "
          f"into the bundle, then {res*res:,} inner products — "
          "no ray marching, no depth samples, no sort")

    if not save_png:
        print()
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print()
        return
    import os
    os.makedirs("out", exist_ok=True)

    vmax = max(t.max() for _, t, _ in rows)
    fig, axes = plt.subplots(2, len(rows), figsize=(3.9 * len(rows), 8))
    for col, (name, truth, holo) in enumerate(rows):
        for row_i, img in enumerate((truth, holo)):
            ax = axes[row_i, col]
            ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, col].set_title(f"analytic projection, {name}", fontsize=10)
        axes[1, col].set_title(f"rendered from hologram, {name}", fontsize=10)
    fig.suptitle("X-ray renders of a 3-D scene straight from ONE complex "
                 f"vector (d={d}): the ray integral has a closed form",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("out/ray_render.png", dpi=110)
    plt.close(fig)

    # rotating view, truth | hologram side by side
    from PIL import Image
    frames = []
    res_g = 108
    n_frames = 36
    for i in range(n_frames):
        ang = 2 * np.pi * i / n_frames
        view = [np.cos(ang), np.sin(ang), 0.35]
        truth = exact_projection(field.splats, field.sigma_inv, view,
                                 center, half, res_g)
        holo = render_orthographic(field.S, field.W, view, center, half,
                                   res_g, T)
        pair = np.concatenate([truth, np.full((res_g, 2), vmax), holo],
                              axis=1)
        rgb = (cm.magma(np.clip(pair / vmax, 0, 1))[:, :, :3] * 255) \
            .astype(np.uint8)
        frames.append(Image.fromarray(rgb[::-1]))
    frames[0].save("out/ray_render.gif", save_all=True,
                   append_images=frames[1:], duration=120, loop=0)
    print("  saved out/ray_render.png, out/ray_render.gif "
          "(left: analytic, right: hologram)")
    print()


__all__ = [
    "exact_projection",
    "render_orthographic",
    "trefoil_points",
    "view_bundle",
]
