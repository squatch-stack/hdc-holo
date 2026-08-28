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
from collections import OrderedDict

import numpy as np

from .crdt import LOSSLESS_CODECS, pack_bundle, unpack_bundle
from .demokit import banner
from .fhrr import FHRR, ItemMemory
from .field import GaussianSplatField

#: Own epochs whose EXACT (never-quantised) blob is still in hand. Under a
#: lossy codec compact() can then subtract the doomed from the exact value
#: and quantise ONCE — cheap like the lossless path and flat like the
#: rebuild — instead of re-encoding every live item, which measured 158x
#: slower on a 5,000-item epoch. Bounded, because a long editing session
#: would otherwise hold one (channels, d) array per stroke forever; a miss
#: simply falls back to the rebuild, which is always correct.
EXACT_CACHE_EPOCHS = 64


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
        self._item_cells = []         # parallel to _items; None when unkeyed
        self._cells = {}              # cell -> this epoch's partial bundle
        self._exact = OrderedDict()   # own blob key -> exact blob
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

    @staticmethod
    def _blob_key(epoch_key, cell=None):
        """Where one epoch's bundle lives, optionally per cell.

        `@` and not `/`: an item id is `<name>/<peer>.<epoch>/<i>` and is
        told apart from an epoch key by its slash count, so a cell in the
        slash namespace would be read as an item index.

        The cell rides in the KEY of one flat map rather than in a child
        container per cell. Loro's own guidance is the reason: two peers
        lazily creating the same child container concurrently get
        conflicting container ids, which "prevents automatic merging and
        may result in data loss". A flat map creates no child containers,
        so that hazard cannot arise.
        """
        return epoch_key if cell is None else "%s@%s" % (epoch_key, cell)

    @staticmethod
    def _epoch_of(blob_key):
        """The epoch a blob belongs to — one tombstone on this excludes
        every cell the stroke touched, however many that was."""
        return blob_key.split("@", 1)[0]

    # -- adding ----------------------------------------------------------

    def add(self, descriptor, cell=None):
        """Returns this addend's unique id (keep it to remove later).

        `cell` partitions one epoch's bundle by space. The accumulator
        then holds only the cells THIS stroke touched, not the scene: at
        capture scale a store per cell would cost ~690 MB of (channels,
        d) accumulators on saguaro before any editing happened.
        """
        if cell is None:
            self._bundle += np.atleast_2d(self.encode(descriptor))
        else:
            key = str(cell)
            if key not in self._cells:
                self._cells[key] = self._zeros()
            self._cells[key] += np.atleast_2d(self.encode(descriptor))
        self._items.append(descriptor)
        self._item_cells.append(None if cell is None else str(cell))
        self._pending = True
        add_id = f"{self._epoch_key()}/{len(self._items) - 1}"
        if len(self._items) >= self.epoch_size:
            self.seal()
        return add_id

    def _remember_exact(self, key, blob):
        """Keep the unquantised blob for one of our own epochs, newest
        first, discarding the oldest past the bound."""
        self._exact[key] = blob
        self._exact.move_to_end(key)
        while len(self._exact) > EXACT_CACHE_EPOCHS:
            self._exact.popitem(last=False)

    def _publish(self):
        if not self._pending:
            return
        epoch_key = self._epoch_key()
        parts = ([(None, self._bundle)] if not self._cells
                 else list(self._cells.items()))
        for cell, bundle in parts:
            key = self._blob_key(epoch_key, cell)
            self._remember_exact(key, bundle.copy())
            self._blobs().insert(key, pack_bundle(bundle,
                                                  self.replica.codec))
        # An unkeyed epoch still writes a bare list, so docs written
        # before cells existed read back unchanged; only a cell-keyed
        # epoch pays for the richer form.
        payload = (self._items if not self._cells
                   else {"items": self._items, "cells": self._item_cells})
        self._index().insert(epoch_key, json.dumps(payload))
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
        self._item_cells = []
        self._cells = {}
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
            for i, desc in enumerate(self._read_index(key)[0]):
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
        """Exclude a whole epoch (stroke/batch) from the merged sum.

        ONE tombstone however many cells the stroke touched, because the
        tombstone names the epoch and the blob keys hang off it. Forty
        separate tombstones could also land in forty separate syncs, and
        a peer that saw half of them would render half an undo.
        """
        self._tombs().insert(epoch_key, True)
        self.replica.doc.commit()

    # -- reading ---------------------------------------------------------

    def _read_index(self, epoch_key):
        """(descriptors, cells) for one epoch; cells are None when the
        epoch was written without them."""
        entry = self._index().get(epoch_key)
        if entry is None:
            return [], []
        data = json.loads(entry.value)
        if isinstance(data, list):
            return data, [None] * len(data)
        return data["items"], data["cells"]

    def _folded_ids(self):
        folded = set()
        for key in self._folded().keys():
            if key.startswith(f"{self.name}/"):
                folded.update(json.loads(self._folded().get(key).value))
        return folded

    def merged(self, cell=None):
        """(channels, d) merged bundle: sum of live epochs minus
        re-encoded item tombstones.

        With `cell`, only that cell's blobs — the capture-scale read,
        where a scene has thousands of cells and a view wants a few. A
        tombstone is on the EPOCH, so undoing a stroke that touched forty
        cells is one tombstone and forty exclusions, not forty
        tombstones applied non-atomically.
        """
        self.replica.flush()
        tombs = set(self._tombs().keys())
        folded = self._folded_ids()
        total = self._zeros()
        want = None if cell is None else str(cell)
        for key in sorted(self._blobs().keys()):
            if not key.startswith(f"{self.name}/"):
                continue
            epoch_key, _, this_cell = key.partition("@")
            if epoch_key in tombs:
                continue
            if want is not None and this_cell != want:
                continue
            total += np.atleast_2d(
                unpack_bundle(self._blobs().get(key).value))
        for tk in sorted(tombs):
            desc = self._tombstoned_item(tk, tombs, folded, want)
            if desc is not None:
                total -= np.atleast_2d(self.encode(desc))
        return total

    def _tombstoned_item(self, tk, tombs, folded, want):
        """The descriptor a reader must subtract for tombstone `tk`, or
        None when it does not apply to the cell being summed.

        The cell check is not an optimisation: an item lives in exactly
        one cell's blob, so subtracting it while summing a DIFFERENT cell
        would remove something that was never added there.
        """
        if not tk.startswith(f"{self.name}/") or tk.count("/") != 2:
            return None                        # not an item id of ours
        epoch_key, i = tk.rsplit("/", 1)
        if epoch_key in tombs or tk in folded:
            return None
        items, cells = self._read_index(epoch_key)
        if int(i) >= len(items):
            return None
        item_cell = cells[int(i)]
        if want is not None and item_cell != want:
            return None
        if self._blobs().get(self._blob_key(epoch_key, item_cell)) is None:
            return None
        return items[int(i)]

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
            epoch_key, _, this_cell = key.partition("@")
            if epoch_key in tombs:
                self._blobs().delete(key)
                continue
            items, cells = self._read_index(epoch_key)
            mine = [i for i in range(len(items))
                    if cells[i] == (this_cell or None)]
            doomed = [i for i in mine
                      if f"{epoch_key}/{i}" in tombs
                      and f"{epoch_key}/{i}" not in folded]
            if not doomed:
                continue
            # How the doomed items leave the blob depends on whether the
            # wire codec is lossless.
            #
            # Lossless ("raw"): decode, subtract, re-pack is EXACT, and it
            # costs one encode per doomed item. That is the cheap path and
            # it was never wrong.
            #
            # Lossy ("hg8"): the subtract moves values off the
            # quantisation grid, so re-packing quantises an adjusted
            # approximation and each compaction adds a little more error —
            # 0.0119 rising to 0.0271 over six staged batches. Rebuilding
            # from the descriptor index instead quantises the true value
            # exactly once however often this runs, flat at 0.0079. It
            # costs an encode per LIVE item — 158x the cheap path on a
            # 5,000-item epoch, linear in epoch size against constant —
            # so it is the last resort, taken only when the exact blob
            # is not in hand.
            if self.replica.codec in LOSSLESS_CODECS:
                blob = np.atleast_2d(
                    unpack_bundle(self._blobs().get(key).value)).copy()
                for i in doomed:
                    blob -= np.atleast_2d(self.encode(items[i]))
            elif key in self._exact:
                # the exact blob is still in hand, so subtract from THAT
                # and quantise once: cheap and drift-free at the same time
                blob = np.atleast_2d(self._exact[key]).copy()
                for i in doomed:
                    blob -= np.atleast_2d(self.encode(items[i]))
                self._remember_exact(key, blob)
            else:
                blob = self._zeros()
                for i in mine:
                    if f"{epoch_key}/{i}" not in tombs:
                        blob += np.atleast_2d(self.encode(items[i]))
                self._remember_exact(key, blob)
            self._blobs().insert(key, pack_bundle(blob, self.replica.codec))
            already = json.loads(self._folded().get(key).value) \
                if self._folded().get(key) is not None else []
            self._folded().insert(key, json.dumps(
                sorted(set(already) | {f"{epoch_key}/{i}" for i in doomed})))
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

    def __init__(self, replica, sigma, name="orscene", cell_size=None):
        space = replica.space
        proto = GaussianSplatField(space.dim, sigma, seed=space.seed)
        self.W = proto.W
        self.sigma_inv = proto.sigma_inv
        self.cell_size = cell_size
        self.store = ORStore(replica, name, self._encode,
                             epoch_size=10 ** 9, channels=3)
        self.splats = []   # local ground-truth mirror for demos/tests

    def cell_of(self, mu):
        """Which cell a splat belongs to, or None when the scene is
        unpartitioned. Same floor-divide rule as capture.encode_bands, so
        a stroke lands in the cells a capture would have put it in."""
        if self.cell_size is None:
            return None
        return tuple(int(np.floor(x / self.cell_size)) for x in mu)

    def _encode(self, desc):
        mu = np.asarray(desc[0], dtype=np.float32)
        rgb = np.asarray(desc[1], dtype=np.float32)
        pos = np.exp(1j * (self.W @ mu)).astype(np.complex64)
        return (float(desc[2]) * rgb)[:, None] * pos[None, :]

    def add_splat(self, mu, rgb, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        self.splats.append((mu, np.asarray(rgb, np.float32), float(alpha)))
        return self.store.add([[float(x) for x in mu],
                               [float(c) for c in rgb], float(alpha)],
                              cell=self.cell_of(mu))

    def cells(self):
        """Cells this scene has written blobs for, across all peers."""
        self.store.replica.flush()
        out = set()
        for key in self.store._blobs().keys():
            if key.startswith(f"{self.store.name}/") and "@" in key:
                out.add(key.split("@", 1)[1])
        return sorted(out)

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

        # one stroke is one epoch however many cells it wrote blobs for,
        # so collapse the cell suffix before listing
        epochs = {ORStore._epoch_of(k) for k in self.store._blobs().keys()
                  if k.startswith(f"{self.store.name}/")}
        return sorted((e for e in epochs if e not in tombs), key=order)

    def undo_stroke(self, stroke_id):
        """Observed-remove by exclusion: any peer may undo any stroke it
        has seen; concurrent duplicate undos subtract it exactly once."""
        self.store.remove_epoch(stroke_id)

    def eval_rgb(self, points, chunk=8192, cell=None):
        """Read the field back. With `cell`, only that cell's strokes —
        the capture-scale read, where a view wants a handful of cells out
        of thousands rather than the whole scene summed."""
        from .accel import readout
        return readout(points, self.W, self.store.merged(cell=cell),
                       chunk=chunk)


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
