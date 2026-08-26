"""Color payloads: RGB as three amplitude channels on ONE frequency basis.

A splat's color is not a symbol (that's attribute_field.py) but a
continuous amplitude — so give the scene three bundles sharing a single
frequency matrix W:

    S_c = sum_k alpha_k * rgb_k[c] * e^{i W mu_k},   c in {R, G, B}

Point queries return RGB as one (n, d) x (d, 3) product — the phasor
matrix E is computed once and reused across channels, so color costs
almost nothing over grayscale. The same holds everywhere downstream:
render.render_orthographic accepts the (3, d) stack and returns a color
image; fit.HoloRegressor fits all three channels against one shared
design-matrix factorization (extra right-hand sides, not extra solves);
and ReplicatedColorScene replicates each channel's cell bundles through
the same Loro machinery as any other container.
"""

import colorsys

import numpy as np

from .crdt import ReplicatedSplatScene
from .field import GaussianSplatField


class ColorSplatField:
    def __init__(self, dim, sigma, seed=0):
        proto = GaussianSplatField(dim, sigma, seed=seed)
        self.dim, self.W = dim, proto.W
        self.sigma_inv = proto.sigma_inv
        self.S = np.zeros((3, dim), dtype=np.complex64)
        self.splats = []   # (mu, rgb, alpha), ground truth only

    def add_splat(self, mu, rgb, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.float32)
        pos = np.exp(1j * (self.W @ mu)).astype(np.complex64)
        self.S += (alpha * rgb)[:, None] * pos[None, :]
        self.splats.append((mu, rgb, float(alpha)))

    def eval_rgb(self, points, chunk=8192):
        from .accel import readout
        return readout(points, self.W, self.S, chunk=chunk)

    def exact_rgb(self, points):
        points = np.asarray(points, dtype=np.float64)
        out = np.zeros((len(points), 3))
        for mu, rgb, alpha in self.splats:
            delta = points - mu
            k = np.exp(-0.5 * np.einsum(
                "ij,jk,ik->i", delta, self.sigma_inv, delta))
            out += (alpha * k)[:, None] * rgb[None, :]
        return out

    def channel_splats(self, c):
        """(mu, weight) pairs for one channel — the form
        render.exact_projection expects for ground-truth projections."""
        return [(mu, float(a * rgb[c])) for mu, rgb, a in self.splats]


class ReplicatedColorScene:
    """Three replicated scalar scenes (R, G, B) on one HoloReplica/doc.
    Every channel shares the same W (drawn from the space seed), so all
    peers agree on the basis with no coordination."""

    def __init__(self, replica, sigma, cell_size=0.25, name="cscene"):
        self.scenes = [ReplicatedSplatScene(replica, sigma, cell_size,
                                            name=f"{name}-{c}")
                       for c in "rgb"]

    def add_splat(self, mu, rgb, alpha=1.0):
        for scene, weight in zip(self.scenes, rgb):
            scene.add_splat(mu, alpha * float(weight))

    def eval_rgb(self, points):
        return np.stack([s.eval(points) for s in self.scenes], axis=-1)


def _load_photo_rgb(res=128):
    import os

    import matplotlib
    from PIL import Image
    path = os.path.join(os.path.dirname(matplotlib.__file__),
                        "mpl-data", "sample_data", "grace_hopper.jpg")
    img = Image.open(path).convert("RGB").resize((res, res))
    return np.asarray(img, dtype=np.float32) / 255.0


def demo(dim=4096, seed=0, save_png=True):
    from .render import (exact_projection, render_orthographic,
                         trefoil_points)
    d = max(dim, 16384)
    sigma = 0.025
    print(f"== Color payloads: RGB amplitude channels (d={d}) ==")

    # -- rainbow trefoil: hue runs along the curve ----------------------
    field = ColorSplatField(d, np.eye(3) * sigma ** 2, seed=seed)
    knot, t = trefoil_points(240)
    for mu, ti in zip(knot, t):
        field.add_splat(mu, colorsys.hsv_to_rgb(ti / (2 * np.pi), 1.0, 1.0))
    center, half, T, res = np.array([0.5, 0.5, 0.5]), 0.45, 1.6, 150

    views = {"(1,0,0)": [1, 0, 0], "(1,1,0)": [1, 1, 0], "(1,1,1)": [1, 1, 1]}
    rows = []
    print(f"  {'view':>9} {'RMSE':>7} {'rel peak':>9}")
    for name, view in views.items():
        truth = np.stack([exact_projection(field.channel_splats(c),
                                           field.sigma_inv, view, center,
                                           half, res)
                          for c in range(3)], axis=-1)
        holo = render_orthographic(field.S, field.W, view, center, half,
                                   res, T)
        rmse = float(np.sqrt(np.mean((holo - truth) ** 2)))
        rows.append((name, truth, holo))
        print(f"  {name:>9} {rmse:>7.4f} {rmse/truth.max():>9.1%}")
    print(f"  color renders reuse the pixel phasors across channels: "
          f"one (n x d) @ (d x 3) product per view, 3 x {8*d//1024}KB scene")

    # -- color photograph: 3 channels, ONE factorization ----------------
    from .fit import FrequencyBands, HoloRegressor, _psnr
    img = _load_photo_rgb(128)
    xs = np.linspace(0, 1, 128)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    Y = img.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    train = rng.choice(len(P), size=len(P) * 3 // 4, replace=False)
    test = np.setdiff1d(np.arange(len(P)), train)
    bands = FrequencyBands(np.array([1, 2, 4, 9]) * 16384 // 16,
                           [0.12, 0.04, 0.016, 0.008], ndim=2, seed=seed)
    reg = HoloRegressor(bands).fit(P[train], Y[train], lam=1e-2)
    pred = reg.eval(P)
    print(f"  color photo fit (D={bands.D}, one solve, 3 RHS): "
          f"train PSNR {_psnr(pred[train], Y[train]):.1f}, "
          f"test PSNR {_psnr(pred[test], Y[test]):.1f}")

    if not save_png:
        print()
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print()
        return
    import os
    os.makedirs("out", exist_ok=True)

    vmax = max(tr.max() for _, tr, _ in rows)
    fig, axes = plt.subplots(2, len(rows), figsize=(3.9 * len(rows), 8))
    for col, (name, truth, holo) in enumerate(rows):
        for row_i, im in enumerate((truth, holo)):
            ax = axes[row_i, col]
            ax.imshow(np.clip(im / vmax, 0, 1), origin="lower")
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, col].set_title(f"analytic, view {name}", fontsize=10)
        axes[1, col].set_title(f"hologram, view {name}", fontsize=10)
    fig.suptitle("Rainbow trefoil: RGB rides three channel bundles on one "
                 f"frequency basis (d={d})", fontsize=11)
    fig.tight_layout()
    fig.savefig("out/color_knot.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    for ax, (title, im) in zip(axes, [
            ("original photo", img),
            (f"color hologram, D={bands.D}", np.clip(pred.reshape(128, 128, 3),
                                                     0, 1))]):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle("A color photograph as three complex vectors "
                 "(one shared ridge factorization)", fontsize=11)
    fig.savefig("out/color_photo.png", dpi=110)
    plt.close(fig)

    # rotating color GIF: analytic | hologram
    from PIL import Image
    frames, res_g = [], 100
    for i in range(30):
        ang = 2 * np.pi * i / 30
        view = [np.cos(ang), np.sin(ang), 0.35]
        truth = np.stack([exact_projection(field.channel_splats(c),
                                           field.sigma_inv, view, center,
                                           half, res_g)
                          for c in range(3)], axis=-1)
        holo = render_orthographic(field.S, field.W, view, center, half,
                                   res_g, T)
        divider = np.full((res_g, 2, 3), vmax)
        pair = np.concatenate([truth, divider, np.clip(holo, 0, None)],
                              axis=1)
        frames.append(Image.fromarray(
            (np.clip(pair / vmax, 0, 1) * 255).astype(np.uint8)[::-1]))
    frames[0].save("out/color_knot.gif", save_all=True,
                   append_images=frames[1:], duration=140, loop=0)
    print("  saved out/color_knot.png, out/color_photo.png, "
          "out/color_knot.gif")
    print()


def _turntable_scene(dim, seed):
    """A multi-object colored scene: rainbow trefoil, tilted cyan ring,
    warm helix, pastel stars — ~520 splats, one (3, d) hologram."""
    from .render import trefoil_points
    field = ColorSplatField(dim, np.eye(3) * 0.02 ** 2, seed=seed)
    knot, t = trefoil_points(240)
    for mu, ti in zip(knot, t):
        field.add_splat(mu, colorsys.hsv_to_rgb(ti / (2 * np.pi), 1, 1), 0.9)
    a = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = np.stack([0.5 + 0.38 * np.cos(a),
                     0.5 + 0.38 * np.sin(a) * np.cos(0.9),
                     0.5 + 0.38 * np.sin(a) * np.sin(0.9)], axis=1)
    for mu, ai in zip(ring, a):
        field.add_splat(mu, colorsys.hsv_to_rgb(
            0.5 + 0.1 * np.sin(ai), 0.8, 1.0), 0.7)
    h = np.linspace(0, 4 * np.pi, 100)
    helix = np.stack([0.5 + 0.10 * np.cos(h), 0.5 + 0.10 * np.sin(h),
                      np.linspace(0.15, 0.85, 100)], axis=1)
    for mu, hi in zip(helix, h):
        field.add_splat(mu, colorsys.hsv_to_rgb(
            0.02 + 0.08 * hi / (4 * np.pi), 0.9, 1.0), 0.8)
    rng = np.random.default_rng(seed + 40)
    for _ in range(60):
        field.add_splat(rng.uniform(0.12, 0.88, 3),
                        colorsys.hsv_to_rgb(rng.uniform(0, 1), 0.35, 1.0),
                        0.35)
    return field


def demo_turntable(dim=4096, seed=0, save_png=True):
    """Turntable: a whole multi-object colored scene orbited from one
    hologram. Each frame folds a fresh slice factor into conj(S) and
    reads out — the scene never exists as geometry at render time."""
    from .render import exact_projection, render_orthographic
    import time
    d = max(dim, 32768)
    print(f"== Turntable: orbiting a scene stored as 3 x {d} complex64 "
          f"({3 * 8 * d // 1024}KB) ==")
    field = _turntable_scene(d, seed)
    center, half, T, res = np.array([0.5, 0.5, 0.5]), 0.48, 1.7, 200
    n_frames = 72

    def view_at(i):
        az = 2 * np.pi * i / n_frames
        el = 0.25 + 0.12 * np.sin(2 * az)
        return [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el),
                np.sin(el)]

    # fidelity metrics on three sampled frames (analytic truth is O(N*px))
    vmax = None
    for i in (0, n_frames // 3, 2 * n_frames // 3):
        view = view_at(i)
        truth = np.stack([exact_projection(field.channel_splats(c),
                                           field.sigma_inv, view, center,
                                           half, res) for c in range(3)],
                         axis=-1)
        holo = render_orthographic(field.S, field.W, view, center, half,
                                   res, T, chunk=2048)
        rmse = float(np.sqrt(np.mean((holo - truth) ** 2)))
        vmax = max(vmax or 0.0, float(truth.max()))
        print(f"  frame {i:>2}: RMSE {rmse:.4f} ({rmse/truth.max():.1%} "
              f"of peak)")

    t0 = time.time()
    frames = [np.clip(render_orthographic(field.S, field.W, view_at(i),
                                          center, half, res, T,
                                          chunk=2048) / vmax, 0, 1)
              for i in range(n_frames)]
    per = (time.time() - t0) / n_frames
    print(f"  {n_frames} frames at {res}x{res}: {per*1000:.0f} ms/frame "
          f"({len(field.splats)} splats, backend-dispatched readout)")

    if not save_png:
        print()
        return
    from PIL import Image
    import os
    os.makedirs("out", exist_ok=True)
    imgs = [Image.fromarray((f * 255).astype(np.uint8)[::-1])
            for f in frames]
    imgs[0].save("out/turntable.gif", save_all=True,
                 append_images=imgs[1:], duration=90, loop=0)
    sheet_idx = range(0, n_frames, n_frames // 8)
    sheet = np.concatenate([frames[i] for i in sheet_idx], axis=1)
    Image.fromarray((sheet * 255).astype(np.uint8)[::-1]) \
        .save("out/turntable.png")
    print("  saved out/turntable.gif, out/turntable.png")
    print()
