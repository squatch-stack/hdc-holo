"""Core FHRR (Fourier Holographic Reduced Representation) operations.

A hypervector is a dense array of d unit-magnitude complex numbers
("phasors"), stored as complex64. The algebra:

    bind(a, b)     elementwise complex multiply  (phases add)
    unbind(a, b)   multiply by conjugate         (exact inverse of bind)
    bundle         complex addition              (superposition)
    permute        fixed random permutation      (protects order/position)
    sim(a, b)      real inner product / d        (~1 for a match, ~0 for random)

Two independent random phasor vectors have sim 0 with std 1/sqrt(2d).
That noise floor is the budget every structure in this package spends:
bundling N items on top of a target adds crosstalk with std ~sqrt(N/(2d)).
"""

import hashlib

import numpy as np


class FHRR:
    """A hypervector space: fixed dimensionality + RNG for drawing vectors."""

    def __init__(self, dim=4096, seed=0):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def random(self, n=None):
        """Draw n (or one) random unit-phasor hypervectors."""
        shape = (self.dim,) if n is None else (n, self.dim)
        phases = self.rng.uniform(-np.pi, np.pi, shape)
        return np.exp(1j * phases).astype(np.complex64)

    def label_vector(self, label):
        """Deterministic codeword for a symbolic label: the label itself is
        hashed into the RNG seed. Any replica of this space (same dim and
        seed) derives the identical vector for 'alice' with no coordination
        — conflict-free codebooks, which distributed use (see crdt.py)
        requires. A sequential-RNG codebook would assign vectors by
        *creation order* and two replicas would silently disagree."""
        digest = hashlib.blake2b(repr(label).encode(),
                                 key=str((self.dim, self.seed)).encode(),
                                 digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        phases = rng.uniform(-np.pi, np.pi, self.dim)
        return np.exp(1j * phases).astype(np.complex64)

    def zeros(self):
        return np.zeros(self.dim, dtype=np.complex64)

    @staticmethod
    def bind(*vs):
        out = vs[0]
        for v in vs[1:]:
            out = out * v
        return out.astype(np.complex64)

    @staticmethod
    def unbind(a, b):
        """Recover x from bind(x, b): exact inverse because |b_j| = 1."""
        return (a * np.conj(b)).astype(np.complex64)

    @staticmethod
    def bundle(*vs):
        out = vs[0].astype(np.complex64).copy()
        for v in vs[1:]:
            out += v
        return out

    @staticmethod
    def normalize(v):
        """Project each component back onto the unit circle (phase-only)."""
        mag = np.abs(v)
        mag[mag == 0] = 1.0
        return (v / mag).astype(np.complex64)

    def sim(self, a, b):
        """Re<a, b>/d. Unit-phasor match -> 1; for a bundle vs. a stored
        item, returns roughly that item's bundled weight."""
        return float(np.real(np.vdot(b, a))) / self.dim

    def cos(self, a, b):
        """Cosine similarity (magnitude-normalized), for comparing bundles."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.real(np.vdot(b, a))) / (na * nb)


class Permutation:
    """A fixed random permutation rho with signed powers: rho^k(v)."""

    def __init__(self, space, seed=1):
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(space.dim)
        self.inv = np.argsort(self.perm)

    def __call__(self, v, power=1):
        idx = self.perm if power >= 0 else self.inv
        for _ in range(abs(power)):
            v = v[idx]
        return v


class ItemMemory:
    """Codebook mapping symbolic labels to random hypervectors, with
    cleanup: nearest-codeword search that turns noisy vectors back into
    symbols. This is the 'decoder' half of every holographic structure."""

    def __init__(self, space, name="items"):
        self.space = space
        self.name = name
        self.labels = []
        self._index = {}
        self._vectors = []
        self._matrix = None  # cache, rebuilt when items are added

    def __len__(self):
        return len(self.labels)

    def __contains__(self, label):
        return label in self._index

    def get(self, label):
        """Return the codeword for label, deriving it deterministically
        from the label (see FHRR.label_vector) so codebooks built in any
        order — or on any replica — agree."""
        if label not in self._index:
            self._index[label] = len(self.labels)
            self.labels.append(label)
            self._vectors.append(self.space.label_vector(label))
            self._matrix = None
        return self._vectors[self._index[label]]

    def matrix(self):
        if self._matrix is None:
            self._matrix = np.stack(self._vectors)
        return self._matrix

    def scores(self, q):
        """sim(q, codeword) for every stored label."""
        return np.real(self.matrix().conj() @ q) / self.space.dim

    def cleanup(self, q):
        """Best-matching label and its score."""
        s = self.scores(q)
        k = int(np.argmax(s))
        return self.labels[k], float(s[k])

    def matches(self, q, threshold=0.5):
        """All labels scoring above threshold, best first."""
        s = self.scores(q)
        order = np.argsort(-s)
        return [(self.labels[k], float(s[k])) for k in order if s[k] >= threshold]
