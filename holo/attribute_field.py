"""Splats with attributes: role-filler binding meets the splat field.

`field.py` bundles bare positions; here every splat carries a payload:

    S = sum_k alpha_k * bind(pos(mu_k), A_k),    pos(mu) = e^{i W mu}

where A_k is an attribute codeword (or a whole role-filler *record*).
Because bind distributes over the bundle and unbinding is exact, one
vector answers three different queries with pure algebra:

    what_is_at(p):  unbind(S, pos(p)) ~ sum_k alpha_k K(p - mu_k) A_k + noise
                    -> cleanup against the attribute codebook. The kernel
                    does the addressing: only splats covering p vote.
    where_is(A):    unbind(S, A) ~ sum_{k: A_k = A} alpha_k pos(mu_k) + noise
                    -> a positional hologram of ONLY the matching splats,
                    renderable like any GaussianSplatField. Semantic
                    filtering of a scene without a list traversal.
    is_there(A, p): sim(S, bind(pos(p), A)) — a joint existence test.

With records as payloads the composition goes one level deeper:
unbind the position, then unbind a role — "what COLOR is the thing
here?" — two exact inverses applied to a single complex64 vector.

Crosstalk budget: every query pays ~sqrt(N R / (2d)) where R is the
payload's component power (1 for codewords, #fields for records).
"""

import hashlib

import numpy as np

from .fhrr import FHRR, ItemMemory


class AttributeSplatField:
    def __init__(self, space, sigma):
        """space: FHRR; sigma: shared splat covariance ((k,k) matrix, or
        a scalar STD DEV -> sigma^2 * I in 2-D), as in GaussianSplatField.

        W is derived from a hash of the space's (dim, seed) — the same
        recipe as FHRR.label_vector — so any replica of the space builds
        the identical frequency matrix with no coordination and no
        dependence on how much of space.rng anyone has consumed (the CRDT
        layer needs W to agree across peers; see crdt.py). Hashing rather
        than reusing the bare seed keeps W uncorrelated with every other
        stream derived from that seed: two generators built from one seed
        emit identical, hence correlated, streams.
        """
        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        if sigma.shape == (1, 1):
            sigma = np.eye(2) * float(sigma[0, 0]) ** 2   # std -> covariance
        self.space = space
        self.k = sigma.shape[0]
        self.sigma_inv = np.linalg.inv(sigma)
        B = np.linalg.cholesky(self.sigma_inv)
        digest = hashlib.blake2b(b"attribute-field-W",
                                 key=str((space.dim, space.seed)).encode(),
                                 digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        self.W = (rng.standard_normal((space.dim, self.k)) @ B.T) \
            .astype(np.float32)
        self.S = space.zeros()
        self.attrs = ItemMemory(space, "attributes")
        self.splats = []  # (mu, alpha, label_or_None), for ground truth only

    def pos(self, p):
        """Fractional-power position encoding e^{i W p}."""
        p = np.asarray(p, dtype=np.float32)
        return np.exp(1j * (self.W @ p)).astype(np.complex64)

    def add_splat(self, mu, attr, alpha=1.0):
        """attr: a label (drawn from the codebook) or a raw hypervector
        such as a RecordSpace record."""
        if isinstance(attr, str):
            payload, label = self.attrs.get(attr), attr
        else:
            payload, label = attr, None
        self.S += np.complex64(alpha) * FHRR.bind(self.pos(mu), payload)
        self.splats.append((np.asarray(mu, dtype=np.float32), float(alpha),
                            label))

    # -- queries ----------------------------------------------------------

    def at(self, p):
        """The (noisy) payload sitting at p: unbind the position."""
        return FHRR.unbind(self.S, self.pos(p))

    def what_is_at(self, p):
        """Best attribute label at p, with its kernel-weighted score."""
        return self.attrs.cleanup(self.at(p))

    def where_query(self, attr_hv):
        """Positional hologram of splats whose payload matches attr_hv."""
        return FHRR.unbind(self.S, attr_hv)

    def where_is(self, label):
        return self.where_query(self.attrs.get(label))

    def is_there(self, label, p):
        """Joint test: ~alpha if a `label` splat covers p, ~0 otherwise."""
        return self.space.sim(self.S, FHRR.bind(self.pos(p),
                                                self.attrs.get(label)))

    def eval_positions(self, hologram, points, chunk=8192):
        """Render a positional hologram (e.g. from where_is) as a field."""
        from .accel import readout
        return readout(points, self.W, hologram, chunk=chunk)

    def exact_class_field(self, label, points):
        """Ground truth for where_is: the mixture of that class only."""
        points = np.asarray(points, dtype=np.float64)
        out = np.zeros(len(points))
        for mu, alpha, lab in self.splats:
            if lab == label:
                delta = points - mu
                out += alpha * np.exp(-0.5 * np.einsum(
                    "ij,jk,ik->i", delta, self.sigma_inv, delta))
        return out


def _scene(space, seed, n_splats, labels, sigma=0.04):
    field = AttributeSplatField(space, sigma)
    rng = np.random.default_rng(seed + 11)
    for i in range(n_splats):
        field.add_splat(rng.uniform(0.08, 0.92, size=2),
                        labels[i % len(labels)],
                        alpha=float(rng.uniform(0.5, 1.0)))
    return field


def demo(dim=4096, seed=0, save_png=True):
    print(f"== Attribute field: role-filler payloads on splats (d={dim}) ==")
    labels = ["tree", "rock", "water", "house", "path"]

    # -- capacity: distant splats still cost full-power crosstalk ---------
    # probes: 100 well-separated splats (vote margin ~1 by construction);
    # background: N splats two boxes away, zero kernel overlap with any
    # probe. In a single global bundle their noise arrives at full power
    # anyway (the locality argument for chunking, see spatial.py), so
    # what_is_at accuracy tracks sqrt(N_total/2d) alone — pure SNR, no
    # scene-geometry ambiguity mixed in.
    print(f"  {'background N':>13} {'crosstalk √(N∕2d)':>18} {'accuracy':>9}")
    for n_bg in [0, 1000, 4000, 16000]:
        space = FHRR(dim, seed=seed)
        field = AttributeSplatField(space, 0.04)
        rng = np.random.default_rng(seed + 23)
        probes = []
        for i in range(10):
            for j in range(10):
                mu = (np.array([0.05 + 0.1 * i, 0.05 + 0.1 * j])
                      + rng.uniform(-0.01, 0.01, 2))
                lab = labels[(i * 10 + j) % len(labels)]
                field.add_splat(mu, lab, alpha=1.0)
                probes.append((mu, lab))
        for _ in range(n_bg):
            field.add_splat(rng.uniform([2.0, 0.0], [3.0, 1.0]),
                            labels[int(rng.integers(len(labels)))],
                            alpha=float(rng.uniform(0.5, 1.0)))
        ok = sum(field.what_is_at(mu)[0] == lab for mu, lab in probes)
        noise = np.sqrt((len(probes) + n_bg) / (2 * dim))
        print(f"  {n_bg:>13} {noise:>18.2f} {ok/len(probes):>9.0%}")

    # -- records as payloads: two exact unbinds deep ----------------------
    from .record import RecordSpace
    space = FHRR(dim, seed=seed)
    rs = RecordSpace(space)
    field = AttributeSplatField(space, 0.04)
    kinds, colors = ["tree", "rock", "house"], ["green", "gray", "red"]
    rng = np.random.default_rng(seed + 5)
    stored = []
    for i in range(60):
        mu = rng.uniform(0.08, 0.92, size=2)
        kind, color = kinds[i % 3], colors[(i // 3) % 3]
        field.add_splat(mu, rs.encode({"kind": kind, "color": color}))
        stored.append((mu, kind, color))
    ok_kind = ok_color = 0
    for mu, _, _ in stored:
        vk, vc = {}, {}
        for mu2, kind2, color2 in stored:
            delta = np.asarray(mu, dtype=np.float64) - mu2
            k = float(np.exp(-0.5 * delta @ field.sigma_inv @ delta))
            vk[kind2] = vk.get(kind2, 0.0) + k
            vc[color2] = vc.get(color2, 0.0) + k
        payload = field.at(mu)
        ok_kind += rs.get(payload, "kind")[0] == max(vk, key=vk.get)
        ok_color += rs.get(payload, "color")[0] == max(vc, key=vc.get)
    print(f"  record payloads (60 splats): position-unbind then role-unbind"
          f" agrees with the mixture's kernel vote -> kind {ok_kind}/60, "
          f"color {ok_color}/60")

    # -- "where is": semantic filtering by one unbind ---------------------
    space = FHRR(dim, seed=seed)
    field = _scene(space, seed, 60, labels)
    grid = 120
    xs = np.linspace(0, 1, grid)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    shown = labels[:3]
    pairs = []
    for lab in shown:
        truth = field.exact_class_field(lab, P)
        holo = field.eval_positions(field.where_is(lab), P)
        rmse = float(np.sqrt(np.mean((holo - truth) ** 2)))
        pairs.append((lab, truth, holo, rmse))
        print(f"  where_is({lab!r}): heatmap RMSE {rmse:.3f} "
              f"(peak {truth.max():.2f})")

    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping image")
            print()
            return
        fig, axes = plt.subplots(2, len(pairs), figsize=(4 * len(pairs), 7.6))
        vmax = max(t.max() for _, t, _, _ in pairs)
        for col, (lab, truth, holo, _) in enumerate(pairs):
            for row, img in enumerate((truth, holo)):
                ax = axes[row, col]
                ax.imshow(img.reshape(grid, grid), origin="lower",
                          extent=(0, 1, 0, 1), cmap="magma", vmin=0, vmax=vmax)
                ax.set_xticks([]), ax.set_yticks([])
            axes[0, col].set_title(f"{lab}: class mixture (truth)", fontsize=10)
            axes[1, col].set_title(f'unbind(S, "{lab}") rendered', fontsize=10)
        fig.suptitle("One vector holds 60 labeled splats — unbinding a label "
                     "leaves a hologram of just that class", fontsize=11)
        fig.tight_layout()
        import os
        os.makedirs("out", exist_ok=True)
        path = os.path.join("out", "attribute_field.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"saved {path}")
    print()
