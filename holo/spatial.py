"""Per-splat covariance and spatial chunking for holographic splat fields.

MultiBandSplatField — covariance classes as frequency bands. One shared
frequency matrix W bakes in one shared Sigma (see field.py); the fix is
several bands, each with its own W drawn from N(0, Sigma_b^{-1}) and its
own bundle. A splat lands in the band matching its covariance; a query
is one inner product per band. This is quantized per-splat covariance —
the continuum limit (every splat its own Sigma) would need per-splat
frequencies, i.e. per-primitive storage again.

ChunkedSplatField — one bundle per occupied grid cell instead of one
global bundle. A Gaussian kernel has ~zero value a few sigma out, but in
a single global bundle every distant splat still contributes full-power
*noise* to every query. Chunking makes distant noise exactly zero: a
query consults only cells whose box lies within `reach` of the point, so
crosstalk comes from ~N_local splats instead of N_total, and updates or
syncs touch one cell. The trade is honest: at EQUAL total bytes a single
giant-d bundle is statistically stronger, but it makes every query pay
O(total) bandwidth and every update rewrite the whole hologram. Chunking
buys locality — of compute, of mutation, and of replication (each cell
is a natural CRDT sync unit; see crdt.py).
"""

import numpy as np

from .field import GaussianSplatField


class MultiBandSplatField:
    def __init__(self, dim, sigmas, seed=0):
        self.bands = [GaussianSplatField(dim, s, seed=seed + 101 * i)
                      for i, s in enumerate(sigmas)]

    def add_splat(self, mu, alpha=1.0, band=0):
        self.bands[band].add_splat(mu, alpha)

    def eval(self, points):
        return sum(b.eval(points) for b in self.bands)

    def exact(self, points):
        return sum(b.exact(points) for b in self.bands)


class ChunkedSplatField:
    def __init__(self, dim, sigma, cell_size, reach=3.0, seed=0):
        proto = GaussianSplatField(dim, sigma, seed=seed)
        self.dim, self.W = dim, proto.W
        self.sigma_inv = proto.sigma_inv
        self.k = proto.k
        self.cell_size = cell_size
        # consult a cell if the query point is within `reach` standard
        # deviations (of the widest axis) of the cell's bounding box
        widest = np.sqrt(np.linalg.eigvalsh(np.linalg.inv(self.sigma_inv)).max())
        self.reach_radius = reach * widest
        self.cells = {}   # cell index tuple -> complex64 bundle
        self.splats = []  # ground truth only

    def _cell_of(self, mu):
        return tuple((np.asarray(mu) // self.cell_size).astype(int))

    def add_splat(self, mu, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        c = self._cell_of(mu)
        if c not in self.cells:
            self.cells[c] = np.zeros(self.dim, dtype=np.complex64)
        self.cells[c] += np.complex64(alpha) * \
            np.exp(1j * (self.W @ mu)).astype(np.complex64)
        self.splats.append((mu, float(alpha)))

    def eval(self, points):
        from . import accel
        points = np.asarray(points, dtype=np.float32)
        masked = []
        for c, S in self.cells.items():
            lo = np.array(c, dtype=np.float32) * self.cell_size
            nearest = np.clip(points, lo, lo + self.cell_size)
            mask = ((points - nearest) ** 2).sum(axis=1) \
                <= self.reach_radius ** 2
            if mask.any():
                masked.append((mask, S))
        self.last_consults_per_point = \
            sum(int(m.sum()) for m, _ in masked) / max(len(points), 1)
        if accel.active():
            # batched masked GEMMs on the GPU: cell_decode computes
            # Re(E @ b^T), so b = conj(S)/d turns it into the readout
            cells = [(m, np.conj(S)[None, :] / self.dim) for m, S in masked]
            return accel.cell_decode(self.W, points, cells)[:, 0]
        out = np.zeros(len(points), dtype=np.float32)
        for mask, S in masked:
            out[mask] += accel.readout(points[mask], self.W, S)
        return out

    def exact(self, points):
        points = np.asarray(points, dtype=np.float64)
        out = np.zeros(len(points))
        for mu, alpha in self.splats:
            delta = points - mu
            out += alpha * np.exp(-0.5 * np.einsum(
                "ij,jk,ik->i", delta, self.sigma_inv, delta))
        return out


def _rot2(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def demo(dim=4096, seed=0, save_png=True):
    rng = np.random.default_rng(seed + 8)

    # -- multi-band: broad isotropic blobs + thin rotated streaks --------
    print(f"== Multi-band field: per-splat covariance classes (d={dim}) ==")
    sigmas = [np.eye(2) * 0.05 ** 2,
              _rot2(35) @ np.diag([0.09 ** 2, 0.008 ** 2]) @ _rot2(35).T]
    mb = MultiBandSplatField(dim, sigmas, seed=seed)
    for _ in range(25):
        mb.add_splat(rng.uniform(0.15, 0.85, 2), rng.uniform(0.5, 1), band=0)
    for _ in range(25):
        mb.add_splat(rng.uniform(0.15, 0.85, 2), rng.uniform(0.5, 1), band=1)
    grid = 110
    xs = np.linspace(0, 1, grid)
    P2 = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    truth2 = mb.exact(P2)
    approx2 = mb.eval(P2)
    rmse = np.sqrt(np.mean((approx2 - truth2) ** 2))
    print(f"  2 bands x 25 splats, RMSE {rmse:.3f} "
          f"({rmse/truth2.max():.1%} of peak); "
          f"storage: {len(sigmas)} bundles = {len(sigmas)*8*dim} bytes")

    # -- chunked 3-D scene ----------------------------------------------
    print(f"== Chunked 3-D field: one bundle per octree cell (d={dim}) ==")
    n_splats = 1500
    mus = rng.uniform(0.05, 0.95, size=(n_splats, 3))
    alphas = rng.uniform(0.5, 1.0, size=n_splats)
    sigma3 = np.eye(3) * 0.03 ** 2

    chunked = ChunkedSplatField(dim, sigma3, cell_size=0.125, seed=seed)
    glob = GaussianSplatField(dim, sigma3, seed=seed)
    for mu, a in zip(mus, alphas):
        chunked.add_splat(mu, a)
        glob.add_splat(mu, a)

    grid3 = 100
    xs = np.linspace(0, 1, grid3)
    X, Y = np.meshgrid(xs, xs)
    slice_pts = np.stack([X, Y, np.full_like(X, 0.5)], axis=-1).reshape(-1, 3)
    truth = chunked.exact(slice_pts)
    import time
    t0 = time.time()
    est_g = glob.eval(slice_pts)
    tg = time.time() - t0
    t0 = time.time()
    est_c = chunked.eval(slice_pts)
    tc = time.time() - t0
    rg = np.sqrt(np.mean((est_g - truth) ** 2))
    rc = np.sqrt(np.mean((est_c - truth) ** 2))
    n_cells = len(chunked.cells)
    touched = chunked.last_consults_per_point * 8 * dim
    print(f"  {n_splats} splats, z=0.5 slice, {n_cells} occupied cells "
          f"(~{chunked.last_consults_per_point:.1f} consulted/query)")
    print(f"  {'':14} {'RMSE':>7} {'rel peak':>9} {'bytes/query':>12} "
          f"{'eval s':>7}")
    print(f"  {'global bundle':<14} {rg:>7.3f} {rg/truth.max():>9.1%} "
          f"{8*dim:>12,} {tg:>7.2f}")
    print(f"  {'chunked':<14} {rc:>7.3f} {rc/truth.max():>9.1%} "
          f"{int(touched):>12,} {tc:>7.2f}")
    print(f"  same d per vector: ~{chunked.last_consults_per_point:.0f} "
          f"inner products/query instead of 1 buys "
          f"{rg/rc:.1f}x lower error — distant-splat crosstalk is zero, "
          "not merely small")
    equal = n_cells * dim
    print(f"  (a SINGLE bundle with equal storage, d={equal:,}, would be "
          f"statistically stronger still, but every query and every "
          f"update would touch all {8*equal//2**20}MB; cells keep "
          "mutation and sync local — see crdt.py)")

    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not available; skipping image")
            print()
            return
        import os
        os.makedirs("out", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
        for ax, (title, img) in zip(axes, [
                ("ground truth (2 covariance classes)",
                 truth2.reshape(grid, grid)),
                (f"holographic, 2 bands, d={dim}",
                 approx2.reshape(grid, grid))]):
            ax.imshow(img, origin="lower", cmap="magma",
                      vmin=0, vmax=truth2.max())
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle("Per-splat covariance via frequency bands", fontsize=11)
        fig.tight_layout()
        fig.savefig("out/multiband.png", dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        for ax, (title, img) in zip(axes, [
                ("ground truth (z=0.5 slice)", truth.reshape(grid3, grid3)),
                (f"ONE global bundle, d={dim}\nRMSE {rg:.2f}",
                 est_g.reshape(grid3, grid3)),
                (f"chunked: {n_cells} cell bundles, d={dim}\nRMSE {rc:.2f}",
                 est_c.reshape(grid3, grid3))]):
            ax.imshow(img, origin="lower", cmap="magma",
                      vmin=0, vmax=truth.max())
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"{n_splats} splats in 3-D: distant splats add noise "
                     "to a global hologram; chunking zeroes it", fontsize=11)
        fig.tight_layout()
        fig.savefig("out/chunked3d.png", dpi=110)
        plt.close(fig)
        print("  saved out/multiband.png, out/chunked3d.png")
    print()
