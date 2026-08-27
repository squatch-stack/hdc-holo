"""The near-enough dispatcher: rule tables without Boolean gates.

A rule engine where conditions are hypervector patterns over messy
text, dispatch is similarity, and the acceptance threshold is POLICY,
not logic — below it the engine abstains ("route to a human"), an
outcome a Boolean if-table cannot express. Three engines, one algebra:

  matrix   cosine of the input's trigram profile against per-rule
           condition profiles; argmax over rules. The shape of an
           embedding "semantic router", but algebraic: deterministic,
           hash-derived, no learned model. Cost O(N) inner products.
  bundle   THE WHOLE RULE TABLE IS ONE VECTOR:
               R = sum_rules bind(cond_hat, action_codeword)
           dispatch = cleanup(unbind(R, input_hat)) over the action
           codebook — K inner products regardless of N. Pays the one
           law: crosstalk ~ sqrt(N/(2d)) under every readout, so a flat
           bundle dies as N grows past the budget.
  banded   the scene medicine (holo/spatial.py) applied to rules:
           split the table into B bundles. Random bands cut per-readout
           load to N/B at B*K readouts; CLUSTERED bands + top-r
           centroid routing add the reach analog — consult only bands
           the query is near. Measured on a topic-structured 4096-rule
           book: top-1 routing reaches 0.98 accuracy at ~43x less
           compute than matrix and 64x fewer stored vectors (reproduce:
           examples/near_enough_rules.py --scale). Clustered routing
           needs rulebooks with topic structure exactly the way
           spatial cells need scenes with spatial locality.

Because bundles add, banded rule tables MERGE: two peers' rulebooks
superpose per band with no coordination (holo/crdt.py's writer-sharded
recipe applies verbatim; action codewords are hash-derived).

Budget (capacity is API): matrix accuracy is limited only by condition
confusability under corruption — trigram profiles hold ~1.00 through
30% character typos where exact AND scores 0.00 (see demo). Flat
bundle accuracy tracks the sqrt(N/(2d)) crosstalk floor; banding
restores it at sqrt((N/B)/(2d)) per readout, minus a slowly-growing
max-over-readouts penalty (~sqrt(2 ln BK)).

Failure modes: a flat bundle past its budget dispatches near-randomly
(use bands or matrix); clustered routing degrades toward random-band
behavior on rulebooks with no topic structure; trigram profiles ignore
word ORDER beyond the trigram horizon ("transfer to savings" ~ "savings
to transfer") — order-sensitive conditions need permuted position tags
(holo/sequence.py's recipe) in the condition encoding.
"""

import numpy as np

from .fhrr import FHRR, ItemMemory
from .ngram import NGramEncoder


class FastNGramProfiler:
    """Vectorized twin of NGramEncoder.profile — identical math on
    ASCII text (same letter codewords, same permutation powers) via
    table lookups and one fused product, ~1000x faster on book-sized
    text. ASCII letters and space only; other characters are dropped
    (near enough for routing — NGramEncoder itself keeps any
    isalpha(), so the twins diverge on non-ASCII input)."""

    def __init__(self, space, n=3):
        self.space = space
        self.enc = NGramEncoder(space, n=n)
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        self.lut = {c: i for i, c in enumerate(alphabet)}
        self.tables = [
            np.stack([self.enc.rho(self.enc.letters.get(c), power=n - 1 - j)
                      for c in alphabet])
            for j in range(n)]
        self.n = n

    def profile(self, text):
        t = np.array([self.lut[c] for c in text.lower() if c in self.lut],
                     dtype=np.int32)
        n_wins = len(t) - self.n + 1
        if n_wins <= 0:
            return self.space.zeros()
        g = self.tables[0][t[:n_wins]]
        for j in range(1, self.n):
            g = g * self.tables[j][t[j:j + n_wins]]
        return g.sum(axis=0).astype(np.complex64)

    def unit_profile(self, text):
        p = self.profile(text)
        return (p / max(np.linalg.norm(p), 1e-9)).astype(np.complex64)


class NearEnoughDispatcher:
    """Rules as (condition_text, action_label) pairs; graded dispatch.

    dispatch(text, threshold=None) -> (action_label | None, score).
    A None action is an ABSTENTION: the best score fell below the
    threshold, and 'unsure' is the correct answer. matrix mode scores
    are cosines in [~0, 1]; bundle mode scores are cleanup similarities
    with crosstalk std ~ sqrt(N/(2d)) — threshold accordingly.
    """

    def __init__(self, rules, space=None, dim=2048, seed=0):
        self.space = space or FHRR(dim, seed=seed)
        self.prof = FastNGramProfiler(self.space)
        self.actions = ItemMemory(self.space, "actions")
        self.rule_action = [a for _, a in rules]
        self.cond = np.stack([self.prof.unit_profile(c) for c, _ in rules])
        acts = np.stack([self.actions.get(a) for a in self.rule_action])
        self.R = (self.cond * acts).sum(axis=0).astype(np.complex64)

    @property
    def n_rules(self):
        return len(self.rule_action)

    def dispatch_matrix(self, text, threshold=None):
        q = self.prof.unit_profile(text)
        s = np.real(self.cond.conj() @ q)
        i = int(np.argmax(s))
        if threshold is not None and s[i] < threshold:
            return None, float(s[i])
        return self.rule_action[i], float(s[i])

    def dispatch_bundle(self, text, threshold=None):
        q = self.prof.unit_profile(text)
        label, score = self.actions.cleanup(self.R * np.conj(q))
        if threshold is not None and score < threshold:
            return None, float(score)
        return label, float(score)


class BandedDispatcher:
    """Split a dispatcher's rule table into B band bundles; optionally
    k-means-cluster the conditions so top-r centroid routing consults
    only nearby bands (the spatial-cells analog for rule space)."""

    def __init__(self, dispatcher, n_bands, clustered=False,
                 kmeans_iters=10, seed=7):
        self.d = dispatcher
        n = dispatcher.n_rules
        if clustered:
            rng = np.random.default_rng(seed)
            X = np.concatenate([dispatcher.cond.real,
                                dispatcher.cond.imag], axis=1)
            C = X[rng.choice(n, n_bands, replace=False)].copy()
            for _ in range(kmeans_iters):
                assign = np.argmax(X @ C.T, axis=1)
                for b in range(n_bands):
                    m = assign == b
                    if m.any():
                        c = X[m].mean(axis=0)
                        C[b] = c / max(np.linalg.norm(c), 1e-9)
            self.assign = np.argmax(X @ C.T, axis=1)
            dim = dispatcher.cond.shape[1]
            self.centroids = (C[:, :dim] + 1j * C[:, dim:]) \
                .astype(np.complex64)
        else:
            self.assign = np.arange(n) % n_bands
            self.centroids = None
        acts = np.stack([dispatcher.actions.get(a)
                         for a in dispatcher.rule_action])
        bound = dispatcher.cond * acts
        dim = dispatcher.cond.shape[1]
        self.bundles = np.stack([
            bound[self.assign == b].sum(axis=0) if (self.assign == b).any()
            else np.zeros(dim, np.complex64)
            for b in range(n_bands)]).astype(np.complex64)
        self.n_bands = n_bands

    def dispatch(self, text, top_r=None, threshold=None):
        q = self.d.prof.unit_profile(text)
        if top_r is not None and self.centroids is not None:
            bands = np.argsort(-np.abs(self.centroids.conj() @ q))[:top_r]
        else:
            bands = range(self.n_bands)
        best, best_s = None, -1e9
        for b in bands:
            label, s = self.d.actions.cleanup(self.bundles[b] * np.conj(q))
            if s > best_s:
                best, best_s = label, float(s)
        if threshold is not None and best_s < threshold:
            return None, best_s
        return best, best_s


# ---------------------------------------------------------------------------
# Demo: the brittleness cliff, the capacity law, the rescue, the policy
# ---------------------------------------------------------------------------

def _mkword(rng):
    return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),
                              rng.integers(4, 10)))


def _corrupt(rng, kws, vocab, typo, drop, distract=2):
    kws = list(kws)
    for _ in range(drop):
        kws.pop(rng.integers(len(kws)))
    kws += list(rng.choice(vocab, distract, replace=False))
    rng.shuffle(kws)
    out = []
    for w in kws:
        out.append("".join(chr(ord("a") + rng.integers(26))
                           if rng.random() < typo else c for c in w))
    return " ".join(out)


def demo(dim=2048, seed=0, save_png=True):
    print(f"== near-enough dispatch: rules without Boolean gates "
          f"(d={dim}) ==")
    rng = np.random.default_rng(seed)
    n_actions, per_topic = 16, 40
    topics = [[_mkword(rng) for _ in range(per_topic)]
              for _ in range(n_actions)]
    vocab = [w for t in topics for w in t]

    def build(n_rules):
        rules = []
        for i in range(n_rules):
            a = i % n_actions
            kws = rng.choice(topics[a], 6, replace=False)
            rules.append((" ".join(kws), f"route-{a}"))
        return rules

    rules = build(256)
    d = NearEnoughDispatcher(rules, dim=dim, seed=seed)
    print(f"  {'typo':>6} {'exact-AND':>10} {'matrix':>7} {'bundle':>7}"
          "   (256 rules, 1 keyword dropped)")
    for typo in [0.0, 0.1, 0.3]:
        hits = dict(exact=0, matrix=0, bundle=0)
        for _ in range(150):
            ci = rng.integers(len(rules))
            text = _corrupt(rng, rules[ci][0].split(), vocab, typo, 1)
            toks = set(text.split())
            exact = next((a for c, a in rules
                          if all(k in toks for k in c.split())), None)
            hits["exact"] += exact == rules[ci][1]
            hits["matrix"] += d.dispatch_matrix(text)[0] == rules[ci][1]
            hits["bundle"] += d.dispatch_bundle(text)[0] == rules[ci][1]
        print(f"  {typo:>6.1f} {hits['exact']/150:>10.2f} "
              f"{hits['matrix']/150:>7.2f} {hits['bundle']/150:>7.2f}")

    print("  -- banding rescues the bundle (typo 0.1, 1 dropped) --")
    print(f"  {'N':>6} {'flat':>6} {'B=32':>6} {'clustered top-1':>16} "
          f"{'floor sqrt(N/2d)':>17}")
    for n in [256, 2048]:
        rules_n = build(n)
        dn = NearEnoughDispatcher(rules_n, dim=dim, seed=seed)
        b32 = BandedDispatcher(dn, 32)
        cl = BandedDispatcher(dn, 32, clustered=True)
        hits = dict(flat=0, b32=0, cl=0)
        for _ in range(150):
            ci = rng.integers(n)
            text = _corrupt(rng, rules_n[ci][0].split(), vocab, 0.1, 1)
            a = rules_n[ci][1]
            hits["flat"] += dn.dispatch_bundle(text)[0] == a
            hits["b32"] += b32.dispatch(text)[0] == a
            hits["cl"] += cl.dispatch(text, top_r=1)[0] == a
        print(f"  {n:>6} {hits['flat']/150:>6.2f} {hits['b32']/150:>6.2f} "
              f"{hits['cl']/150:>16.2f} {np.sqrt(n/(2*dim)):>17.2f}")

    print("  -- abstention: threshold is policy, not logic --")
    ok, bad = [], []
    for _ in range(400):
        ci = rng.integers(len(rules))
        text = _corrupt(rng, rules[ci][0].split(), vocab, 0.35, 3,
                        distract=4)
        a, s = d.dispatch_matrix(text)
        (ok if a == rules[ci][1] else bad).append(s)
    ok, bad = np.array(ok), np.array(bad)
    alls = np.concatenate([ok, bad])
    for th in [0.0, 0.2]:
        answered = alls >= th
        prec = (ok >= th).sum() / max(answered.sum(), 1)
        print(f"  θ={th:.1f}: answers {answered.mean():>4.0%} of inputs "
              f"at {prec:.0%} precision"
              + ("" if th == 0 else "  (the rest escalate — by design)"))
    print()


__all__ = ["FastNGramProfiler", "NearEnoughDispatcher", "BandedDispatcher"]
