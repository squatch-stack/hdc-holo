"""Observed-remove holographic containers: deletion as set membership.

crdt.py's retraction is ARITHMETIC — subtract the addend from your
shard. Correct alone, wrong together: two peers concurrently retracting
the same item both subtract, and the merged sum over-cancels into a
negative phantom (the PN-Counter anomaly). The OR-Set recipe fixes
deletion by changing its type: removals are not subtractions performed
by writers, they are TOMBSTONES — entries in a grow-only set — and the
subtraction is derived, once, by every reader.

    add:     tag each addend with a unique id "<name>/<peer>.<epoch>/<i>"
             and record its descriptor (the recipe for its vector) in a
             replicated index. Adds accumulate into per-peer EPOCH
             bundles ("<name>/<peer>.<epoch>" -> blob), sealed every
             epoch_size adds or explicitly (a brush stroke, a batch).
    remove:  insert the observed id into the tombstone map. Concurrent
             duplicate removes write the same key — set union — so the
             item is subtracted exactly once at read time, ever.
    merged:  sum the non-tombstoned epoch blobs, then subtract the
             re-encoded descriptors of item-tombstones. Removing a whole
             epoch is pure EXCLUSION from the sum — no arithmetic at all.

Two classic OR-Set properties fall out:
  * idempotent removal — over-cancellation is structurally impossible,
    because readers subtract per unique tombstone, not per remove op;
  * add-wins — a remove only covers ids it OBSERVED, so a concurrent
    re-add (fresh id) survives the merge.

Reconstruction (encode(descriptor)) works on any peer because codewords
and frequency matrices are hash/seed-derived (see crdt.py). Note that
re-encoding is deterministic only to ~1 ulp (numpy SIMD paths), so
merged() results agree across peers semantically, not byte-for-byte —
compare with allclose, digest only blob bytes (see SDK.md). Item
tombstones cost one re-encode per read; compact() lets each peer fold
tombstoned items out of its OWN epoch blobs (owner-only keys, so no
races) and records them as folded so readers stop subtracting them.

Epochs are the honest capacity knob: item-level removal needs the index
(per-item descriptors — metadata, not vectors), while removal by epoch
(undo a stroke) needs nothing but the id. Coarser epochs, more
holographic; finer epochs, more surgical deletion.
"""

import json

import numpy as np

from .crdt import pack_bundle, unpack_bundle
from .demokit import banner
from .fhrr import FHRR, ItemMemory
from .field import GaussianSplatField


class ORStore:
    """Observed-remove layer over a HoloReplica. `encode` turns a
    JSON-able descriptor into its (channels, d) complex64 vector."""

    def __init__(self, replica, name, encode, epoch_size=16, channels=1):
        assert "/" not in name, "container names must not contain '/'"
        self.replica = replica
        self.name = name
        self.encode = encode
        self.epoch_size = epoch_size
        self.channels = channels
        self.dim = replica.space.dim
        own = [k for k in self._index().keys()
               if k.startswith(f"{name}/{replica.peer}.")]
        self.epoch = 1 + max((int(k.rsplit(".", 1)[1]) for k in own),
                             default=-1)
        self._items = []
        self._bundle = self._zeros()
        self._pending = False
        replica.flush_hooks.append(self._publish)

    def _zeros(self):
        return np.zeros((self.channels, self.dim), dtype=np.complex64)

    def _blobs(self):
        return self.replica.doc.get_map("or-bundles")

    def _index(self):
        return self.replica.doc.get_map("or-index")

    def _tombs(self):
        return self.replica.doc.get_map("or-tombs")

    def _folded(self):
        return self.replica.doc.get_map("or-folded")

    def _epoch_key(self, epoch=None):
        which = self.epoch if epoch is None else epoch
        return f"{self.name}/{self.replica.peer}.{which}"

    # -- adding ----------------------------------------------------------

    def add(self, descriptor):
        """Returns this addend's unique id (keep it to remove later)."""
        self._bundle += np.atleast_2d(self.encode(descriptor))
        self._items.append(descriptor)
        self._pending = True
        add_id = f"{self._epoch_key()}/{len(self._items) - 1}"
        if len(self._items) >= self.epoch_size:
            self.seal()
        return add_id

    def _publish(self):
        if not self._pending:
            return
        self._blobs().insert(self._epoch_key(), pack_bundle(self._bundle))
        self._index().insert(self._epoch_key(), json.dumps(self._items))
        self._pending = False

    def seal(self):
        """Close the current epoch (e.g. end a brush stroke); returns its
        id, the unit that remove_epoch() deletes."""
        key = self._epoch_key()
        if not self._items:
            return None
        self._publish()
        self.epoch += 1
        self._items = []
        self._bundle = self._zeros()
        return key

    # -- observing and removing -----------------------------------------

    def observed(self):
        """Live (add_id, descriptor) pairs across all peers."""
        self.replica.flush()
        tombs, folded = set(self._tombs().keys()), self._folded_ids()
        out = []
        for key in sorted(self._index().keys()):
            if not key.startswith(f"{self.name}/") or key in tombs:
                continue
            for i, desc in enumerate(json.loads(self._index().get(key).value)):
                add_id = f"{key}/{i}"
                if add_id not in tombs and add_id not in folded:
                    out.append((add_id, desc))
        return out

    def remove(self, add_id):
        """Tombstone one observed addend. Idempotent under concurrency:
        N peers removing the same id still subtract it once."""
        self.replica.flush()          # our own pending adds become removable
        self._tombs().insert(add_id, True)
        self.replica.doc.commit()

    def remove_where(self, predicate):
        ids = [i for i, d in self.observed() if predicate(d)]
        for add_id in ids:
            self._tombs().insert(add_id, True)
        self.replica.doc.commit()
        return ids

    def remove_epoch(self, epoch_key):
        """Exclude a whole epoch (stroke/batch) from the merged sum."""
        self._tombs().insert(epoch_key, True)
        self.replica.doc.commit()

    # -- reading ---------------------------------------------------------

    def _folded_ids(self):
        folded = set()
        for key in self._folded().keys():
            if key.startswith(f"{self.name}/"):
                folded.update(json.loads(self._folded().get(key).value))
        return folded

    def merged(self):
        """(channels, d) merged bundle: sum of live epochs minus
        re-encoded item tombstones."""
        self.replica.flush()
        tombs = set(self._tombs().keys())
        folded = self._folded_ids()
        total = self._zeros()
        for key in sorted(self._blobs().keys()):
            if not key.startswith(f"{self.name}/") or key in tombs:
                continue
            total += np.atleast_2d(
                unpack_bundle(self._blobs().get(key).value))
        for tk in sorted(tombs):
            if not tk.startswith(f"{self.name}/") or tk.count("/") != 2:
                continue                       # not an item id of ours
            epoch_key, i = tk.rsplit("/", 1)
            if epoch_key in tombs or tk in folded or \
                    self._blobs().get(epoch_key) is None:
                continue
            desc = json.loads(self._index().get(epoch_key).value)[int(i)]
            total -= np.atleast_2d(self.encode(desc))
        return total

    # -- maintenance -----------------------------------------------------

    def compact(self):
        """Fold item tombstones out of OUR OWN epoch blobs (owner-only
        keys — no races) and drop fully-tombstoned own epochs. Readers
        skip folded ids, so merged() is unchanged. Returns #folded."""
        self.replica.flush()
        tombs = set(self._tombs().keys())
        folded = self._folded_ids()
        count = 0
        own = f"{self.name}/{self.replica.peer}."
        for key in [k for k in self._blobs().keys() if k.startswith(own)]:
            if key in tombs:
                self._blobs().delete(key)
                continue
            items = json.loads(self._index().get(key).value)
            doomed = [i for i in range(len(items))
                      if f"{key}/{i}" in tombs and f"{key}/{i}" not in folded]
            if not doomed:
                continue
            blob = np.atleast_2d(
                unpack_bundle(self._blobs().get(key).value)).copy()
            for i in doomed:
                blob -= np.atleast_2d(self.encode(items[i]))
            self._blobs().insert(key, pack_bundle(blob))
            already = json.loads(self._folded().get(key).value) \
                if self._folded().get(key) is not None else []
            self._folded().insert(key, json.dumps(
                sorted(set(already) | {f"{key}/{i}" for i in doomed})))
            count += len(doomed)
        self.replica.doc.commit()
        return count


class ORHoloMap:
    """The replicated key/value map with OR-Set deletion semantics."""

    def __init__(self, replica, name="orkv"):
        self.replica = replica
        self.name = name
        self.keys = ItemMemory(replica.space, "keys")
        self.values = ItemMemory(replica.space, "values")
        self.store = ORStore(replica, name, self._encode)

    def _encode(self, desc):
        return FHRR.bind(self.keys.get(desc[0]), self.values.get(desc[1]))

    def put(self, key, value):
        self.replica.register_label(self.name, value)
        return self.store.add([key, value])

    def remove(self, key):
        """Remove every OBSERVED pair under key. A concurrent re-put
        (unseen id) survives: add-wins."""
        return self.store.remove_where(lambda d: d[0] == key)

    def _codebook(self):
        for label in self.replica.known_labels(self.name):
            self.values.get(label)

    def get(self, key):
        self._codebook()
        v = FHRR.unbind(self.store.merged()[0], self.keys.get(key))
        return self.values.cleanup(v)

    def get_all(self, key, threshold=0.5):
        self._codebook()
        v = FHRR.unbind(self.store.merged()[0], self.keys.get(key))
        return self.values.matches(v, threshold)


class ORStrokeScene:
    """A 2-D color splat scene where each brush stroke is an epoch:
    undo_stroke() is observed-remove by exclusion, safe under concurrent
    duplicate undos from any number of peers."""

    def __init__(self, replica, sigma, name="orscene"):
        space = replica.space
        proto = GaussianSplatField(space.dim, sigma, seed=space.seed)
        self.W = proto.W
        self.sigma_inv = proto.sigma_inv
        self.store = ORStore(replica, name, self._encode,
                             epoch_size=10 ** 9, channels=3)
        self.splats = []   # local ground-truth mirror for demos/tests

    def _encode(self, desc):
        mu = np.asarray(desc[0], dtype=np.float32)
        rgb = np.asarray(desc[1], dtype=np.float32)
        pos = np.exp(1j * (self.W @ mu)).astype(np.complex64)
        return (float(desc[2]) * rgb)[:, None] * pos[None, :]

    def add_splat(self, mu, rgb, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        self.splats.append((mu, np.asarray(rgb, np.float32), float(alpha)))
        return self.store.add([[float(x) for x in mu],
                               [float(c) for c in rgb], float(alpha)])

    def end_stroke(self):
        return self.store.seal()

    def strokes(self):
        """Live (not undone) stroke ids across ALL peers, ordered by
        (peer, epoch-number) — numeric on the epoch so peers that sort
        independently agree on 'the earliest stroke' past epoch 9."""
        self.store.replica.flush()
        tombs = set(self.store._tombs().keys())

        def order(key):
            peer, epoch = key.rsplit("/", 1)[1].rsplit(".", 1)
            return (peer, int(epoch))

        return sorted(
            (k for k in self.store._blobs().keys()
             if k.startswith(f"{self.store.name}/") and k not in tombs),
            key=order)

    def undo_stroke(self, stroke_id):
        """Observed-remove by exclusion: any peer may undo any stroke it
        has seen; concurrent duplicate undos subtract it exactly once."""
        self.store.remove_epoch(stroke_id)

    def eval_rgb(self, points, chunk=8192):
        from .accel import readout
        return readout(points, self.W, self.store.merged(), chunk=chunk)


def _demo_plot(before, after_a, naive):
    """The point of the whole module in three panels: merged strokes,
    a concurrent double-undo handled correctly, and the negative hole
    arithmetic retraction would have left."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import os
    os.makedirs("out", exist_ok=True)
    vmax = float(before.max())
    panels = [("5 strokes, merged", before),
              ("concurrent double-undo of one stroke\n(observed-remove:"
               " subtracted once)", after_a),
              ("what arithmetic retraction would do\n(double-subtract:"
               " negative hole)", naive)]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.clip(img / vmax, 0, 1), origin="lower")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Deletion as set membership: tombstones, not "
                 "subtractions", fontsize=12)
    fig.tight_layout()
    fig.savefig("out/orset_undo.png", dpi=110)
    plt.close(fig)
    print("  saved out/orset_undo.png")


def demo(dim=4096, seed=0, save_png=True):
    try:
        from .crdt import HAVE_LORO, HoloReplica, ReplicatedHoloMap
    except ImportError:
        HAVE_LORO = False
    if not HAVE_LORO:
        print("== OR-Set demo skipped: pip install loro ==\n")
        return
    banner("Observed-remove holographic containers", dim)

    # -- the anomaly, then the fix --------------------------------------
    A, B = HoloReplica(FHRR(dim, seed=seed)), HoloReplica(FHRR(dim, seed=seed))
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    kv_a.put("doomed", "v")
    A.sync(B)
    kv_a.delete("doomed", "v")          # both peers retract, concurrently
    kv_b.delete("doomed", "v")
    A.sync(B)
    print(f"  arithmetic retraction (crdt.py), concurrent double-delete: "
          f"score {kv_a.get('doomed')[1]:+.2f}  <- negative phantom")

    A, B = HoloReplica(FHRR(dim, seed=seed)), HoloReplica(FHRR(dim, seed=seed))
    or_a, or_b = ORHoloMap(A), ORHoloMap(B)
    or_a.put("doomed", "v")
    A.sync(B)
    or_a.remove("doomed")               # same concurrent double-remove
    or_b.remove("doomed")
    A.sync(B)
    # `+ 0.0` normalizes a -0.00 that carries no information: this score
    # is crosstalk around zero, and its SIGN moves with allocation
    # layout (the ~1 ulp caveat in SDK.md), so printing it signed made
    # the demo's own output depend on unrelated imports
    print(f"  observed-remove (orset.py), same scenario:      "
          f"score {round(or_a.get('doomed')[1], 2) + 0.0:+.2f}"
          f"  <- clean zero")

    # -- add-wins --------------------------------------------------------
    or_a.put("config", "v1")
    A.sync(B)
    or_a.remove("config")               # A removes what it observed...
    id2 = or_b.put("config", "v2")      # ...while B concurrently re-adds
    A.sync(B)
    va, vb = or_a.get("config"), or_b.get("config")
    print(f"  add-wins: concurrent remove vs re-put -> A reads {va[0]!r} "
          f"({va[1]:.2f}), B reads {vb[0]!r} ({vb[1]:.2f})")
    assert va[0] == vb[0] == "v2" and id2

    # -- stroke undo in a painted scene ---------------------------------
    A, B = HoloReplica(FHRR(dim, seed=seed)), HoloReplica(FHRR(dim, seed=seed))
    sa, sb = (ORStrokeScene(A, np.eye(2) * 0.03 ** 2),
              ORStrokeScene(B, np.eye(2) * 0.03 ** 2))
    rng = np.random.default_rng(seed + 30)
    import colorsys
    strokes_a = []
    for _s in range(3):
        p = rng.uniform([0.1, 0.1], [0.5, 0.9])
        ang = rng.uniform(0, 2 * np.pi)
        rgb = colorsys.hsv_to_rgb(rng.uniform(0, 0.15), 0.9, 1.0)
        for _ in range(14):
            sa.add_splat(np.clip(p, 0.04, 0.96), rgb)
            p = p + 0.035 * np.array([np.cos(ang), np.sin(ang)])
            ang += 0.15
        strokes_a.append(sa.end_stroke())
    for _s in range(2):
        p = rng.uniform([0.5, 0.1], [0.9, 0.9])
        ang = rng.uniform(0, 2 * np.pi)
        rgb = colorsys.hsv_to_rgb(rng.uniform(0.5, 0.7), 0.9, 1.0)
        for _ in range(14):
            sb.add_splat(np.clip(p, 0.04, 0.96), rgb)
            p = p + 0.035 * np.array([np.cos(ang), np.sin(ang)])
            ang += 0.15
        sb.end_stroke()
    A.sync(B)
    res = 130
    xs = np.linspace(0, 1, res, dtype=np.float32)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    before = sa.eval_rgb(P).reshape(res, res, 3)
    naive = before - 2 * (before - _without(sa, strokes_a[1], P, res))
    sa.undo_stroke(strokes_a[1])        # BOTH peers undo the same stroke
    sb.undo_stroke(strokes_a[1])
    A.sync(B)
    after_a = sa.eval_rgb(P).reshape(res, res, 3)
    after_b = sb.eval_rgb(P).reshape(res, res, 3)
    print(f"  stroke undo: A and B both undo A's 2nd stroke; peers agree: "
          f"{bool(np.allclose(after_a, after_b))}, min value "
          f"{after_a.min():+.2f} (naive double-subtract would dip to "
          f"{naive.min():+.2f})")

    # -- compaction ------------------------------------------------------
    n = or_a.store.compact()
    va2 = or_a.get("config")
    print(f"  compact(): folded {n} tombstoned addends into owner blobs; "
          f"reads unchanged ({va2[0]!r}, {va2[1]:.2f})")

    if save_png:
        _demo_plot(before, after_a, naive)
    print()


def _without(scene, stroke_id, P, res):
    """Render the scene as if stroke_id were cleanly excluded (for the
    demo's 'what naive subtraction would look like' comparison)."""
    tomb = scene.store._tombs().get(stroke_id) is not None
    assert not tomb
    scene.store.remove_epoch(stroke_id)
    img = scene.eval_rgb(P).reshape(res, res, 3)
    # roll back the tombstone locally: this is demo-only surgery
    scene.store._tombs().delete(stroke_id)
    scene.store.replica.doc.commit()
    return img
