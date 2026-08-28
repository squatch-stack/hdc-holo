"""Replicated holographic state on Loro CRDTs (https://loro.dev).

Superposition is already *almost* a CRDT merge: bundles add, addition
commutes and associates, so replicas can accumulate independently and
any merge order converges. What addition is NOT is idempotent — a naive
'send me your bundle and I'll add it in' protocol double-counts on
redelivery. The G-Counter recipe closes the gap: shard each container's
accumulator BY WRITER.

    Loro map "bundles":  key "<container>::<peer>" -> complex64 blob

Each peer only ever writes its own keys (its running partial sum), so
Loro's per-key last-writer-wins is trivially safe, and Loro's version
vectors make update delivery idempotent and exactly-once. The merged
value of a container is the SUM over all peers' blobs, computed at read
time — order-free, so replicas that have seen the same updates read the
same hologram bit for bit. Loro contributes exactly the parts vector
algebra lacks: causal versioning, delta sync (send only what the peer
hasn't seen), snapshots/persistence, and key-set convergence.

Retraction = subtracting an addend and republishing your blob. The sum
cancels it no matter who stored the original, but this is counter
semantics, not observed-remove: two peers concurrently retracting the
same item over-cancel, leaving a negative phantom (the PN-Counter
anomaly). Retract from your own additions, or coordinate above.

Two things must be deterministic across peers with NO coordination:
codewords (FHRR.label_vector hashes the label into its vector) and the
field frequency matrix W (drawn from the shared space seed). Labels
themselves replicate in a grow-only Loro map so peers can build cleanup
codebooks for values they have never seen locally.

Practical note: every flush() rewrites this peer's blob as a fresh Loro
op, and doc history keeps old blobs — batch writes (flush per sync, not
per put) and compact long-lived docs with ExportMode.ShallowSnapshot.
"""

import json
import struct

import numpy as np

from .demokit import banner
from .fhrr import FHRR, ItemMemory
from .field import GaussianSplatField
from .phase import pack_polar, unpack
from .record import RecordSpace

try:
    from loro import ExportMode, LoroDoc
    HAVE_LORO = True
except ImportError:  # structures below then need an explicit doc-like stub
    HAVE_LORO = False


# -- wire format v1 ---------------------------------------------------------
#
# Every bundle blob (G-Counter shards AND observed-remove epochs) carries a
# 12-byte self-describing header, and every doc carries a format record
# ("format" map, key "holo") naming (wire version, dim, seed) — the
# universe. Replicas refuse to read or write a doc from a different
# universe instead of silently producing garbage. These layouts, plus the
# key schemes ("container::peer", "name/peer.epoch[/i]") and the label
# registry ("namespace::label" -> True), ARE the compatibility surface:
# changing any of them bumps WIRE_VERSION.

WIRE_VERSION = 1
_BLOB_MAGIC = b"HB"
_BLOB_HEADER = struct.Struct("<2sBBHIH")  # magic, ver, dtype, channels, dim, reserved
_DTYPE_COMPLEX64 = 0
_DTYPE_HG8 = 1

#: Wire codecs, by the name callers pass. "hg8" is gamma-companded polar
#: (storage.md's HG-8): a quarter of the bytes at 0.010 drift against the
#: uncompressed decode, which is the faithful codec that study settled on.
#: A NEW DTYPE CODE, NOT A NEW LAYOUT — the header always had a dtype
#: field, so WIRE_VERSION does not move and a build that predates this
#: refuses the blob loudly with "unknown dtype code 1" rather than
#: decoding it as complex64.
CODECS = ("raw", "hg8")
_CODEC_DTYPE = {"raw": _DTYPE_COMPLEX64, "hg8": _DTYPE_HG8}

#: Codecs that survive a decode/re-encode round unchanged. "raw" is the
#: complex64 bytes themselves, so adjusting a decoded blob and re-packing
#: it is exact; a lossy codec re-quantises an adjusted approximation and
#: drifts, which is why ORStore.compact() picks its strategy from this.
LOSSLESS_CODECS = frozenset({"raw"})


def pack_bundle(arr, codec="raw"):
    """complex64 bundle (d,) or (channels, d) -> tagged blob bytes.

    Peers may choose different codecs without coordinating: what
    replicates is the BLOB, so every replica decodes the same bytes to
    the same values and convergence is untouched. The choice is a local
    bytes/fidelity trade, not part of the compatibility surface.
    """
    if codec not in _CODEC_DTYPE:
        raise ValueError("unknown wire codec %r; expected one of %s"
                         % (codec, ", ".join(CODECS)))
    a = np.ascontiguousarray(arr, dtype=np.complex64)
    channels, dim = (1, a.shape[0]) if a.ndim == 1 else a.shape
    head = _BLOB_HEADER.pack(_BLOB_MAGIC, WIRE_VERSION, _CODEC_DTYPE[codec],
                             channels, dim, 0)
    if codec == "raw":
        return head + a.tobytes()
    rows = a.reshape(channels, dim)
    return head + b"".join(pack_polar(rows[c], bits=8)
                           for c in range(channels))


def unpack_bundle(buf):
    """Tagged blob bytes -> read-only complex64 array, (d,) when it holds
    one channel, else (channels, d). Refuses untagged or foreign bytes."""
    if len(buf) < _BLOB_HEADER.size or buf[:2] != _BLOB_MAGIC:
        raise ValueError(
            "not a holo bundle blob: missing the 'HB' format tag (wire "
            "format v1 prefixes every blob with a 12-byte header; legacy "
            "untagged blobs must be re-encoded)")
    _, ver, dtype, channels, dim, _ = _BLOB_HEADER.unpack_from(buf)
    if ver != WIRE_VERSION:
        raise ValueError(f"bundle blob is wire version {ver}; this build "
                         f"speaks {WIRE_VERSION}")
    if dtype == _DTYPE_COMPLEX64:
        data = np.frombuffer(buf, np.complex64, count=channels * dim,
                             offset=_BLOB_HEADER.size)
        return data if channels == 1 else data.reshape(channels, dim)
    if dtype == _DTYPE_HG8:
        # every channel is a self-describing HG blob of identical length
        # for a given (bits, dim), so the payload splits evenly
        payload = len(buf) - _BLOB_HEADER.size
        if channels <= 0 or payload % channels:
            raise ValueError(
                "hg8 bundle blob payload of %d bytes does not divide into "
                "%d channels" % (payload, channels))
        each = payload // channels
        rows = [unpack(buf[_BLOB_HEADER.size + c * each:
                           _BLOB_HEADER.size + (c + 1) * each])
                for c in range(channels)]
        out = np.asarray(rows, dtype=np.complex64)
        return out[0] if channels == 1 else out
    raise ValueError(f"bundle blob has unknown dtype code {dtype}")


class HoloReplica:
    """One peer's handle on a replicated set of holographic containers."""

    def __init__(self, space, doc=None, codec="raw"):
        if doc is None:
            if not HAVE_LORO:
                raise RuntimeError("pip install loro")
            doc = LoroDoc()
        if codec not in _CODEC_DTYPE:
            raise ValueError("unknown wire codec %r; expected one of %s"
                             % (codec, ", ".join(CODECS)))
        self.space = space
        self.codec = codec
        self.doc = doc
        self.peer = str(doc.peer_id)
        self.local = {}     # container -> this peer's own partial bundle
        self.dirty = set()
        self.flush_hooks = []   # layers (e.g. orset.ORStore) publish here

    # -- writing (always to our own shard) ------------------------------

    def add(self, container, vec):
        buf = self.local.get(container)
        if buf is None:
            stored = self._bundles().get(f"{container}::{self.peer}")
            buf = (unpack_bundle(stored.value).copy()
                   if stored is not None else self.space.zeros())
            self.local[container] = buf
        buf += vec
        self.dirty.add(container)

    def retract(self, container, vec):
        self.add(container, -vec)

    def register_label(self, namespace, label):
        self._labels().insert(f"{namespace}::{label}", True)

    def flush(self):
        """Publish dirty containers (one map op per container) and run
        any registered layer hooks, then commit — a no-op when clean.

        The format record is written only in the SAME commit as actual
        content ops: an empty flush must create no ops at all, or a
        version() snapshot taken before the first write would cover ops
        the peer never received and every later delta would arrive with
        missing causal deps (Loro queues them forever)."""
        self._check_format(writing=False)
        for hook in list(self.flush_hooks):
            hook()
        m = self._bundles()
        for c in sorted(self.dirty):
            m.insert(f"{c}::{self.peer}", pack_bundle(self.local[c],
                                                      self.codec))
        self.dirty.clear()
        if self.doc.get_pending_txn_len() > 0:
            self._check_format(writing=True)
        self.doc.commit()

    def _check_format(self, writing):
        """Assert the doc belongs to this replica's universe: same wire
        version, dim, and seed (seed must be JSON-serializable). Writes
        the format record on first flush; raises ValueError on mismatch."""
        m = self.doc.get_map("format")
        mine = {"wire": WIRE_VERSION, "dim": self.space.dim,
                "seed": self.space.seed}
        stored = m.get("holo")
        if stored is None:
            if writing:
                m.insert("holo", json.dumps(mine))
            return
        theirs = json.loads(stored.value)
        if theirs != mine:
            raise ValueError(
                f"holo wire-format mismatch: the doc carries {theirs}, this "
                f"replica speaks {mine}. Peers must share (wire, dim, seed) "
                "— refusing to read or write.")

    # -- reading (merge = sum of every peer's shard) --------------------

    def merged(self, container):
        self.flush()
        total = self.space.zeros()
        m = self._bundles()
        prefix = f"{container}::"
        for k in sorted(m.keys()):
            if k.startswith(prefix):
                total += unpack_bundle(m.get(k).value)
        return total

    def containers(self, prefix=""):
        self.flush()
        return sorted({k.rsplit("::", 1)[0] for k in self._bundles().keys()
                       if k.startswith(prefix)})

    def known_labels(self, namespace):
        pre = f"{namespace}::"
        return [k[len(pre):] for k in self._labels().keys()
                if k.startswith(pre)]

    # -- replication ----------------------------------------------------

    def updates_for(self, other):
        """Delta: every op this doc has that `other` has not seen."""
        self.flush()
        if not self._reachable_by_delta(other.doc.oplog_vv):
            self._refuse_stale(other.doc.oplog_vv)
        return self.doc.export(ExportMode.Updates(from_=other.doc.oplog_vv))

    def version(self):
        """Snapshot of this doc's version vector. Wire protocol: record
        version() after each exchange; the next updates_since(that) is
        exactly the ops the peer lacks — no access to the peer's doc."""
        self.flush()
        return self.doc.oplog_vv

    def updates_since(self, vv):
        """Delta of every op not covered by version vector vv — the
        socket-transportable form of updates_for (see examples/live_sync.py).

        Refuses when this doc's history has been trimmed past `vv`;
        without that check Loro hands back a plausible-looking frame the
        peer imports successfully and reads NOTHING from."""
        self.flush()
        if not self._reachable_by_delta(vv):
            self._refuse_stale(vv)
        return self.doc.export(ExportMode.Updates(from_=vv))

    def apply(self, update_bytes):
        """Import a peer's delta, then validate the doc's format record.
        A ValueError here means a foreign universe's ops are already in
        the local doc — treat this replica as poisoned and rebuild it
        from a trusted snapshot."""
        self.doc.import_(update_bytes)
        self._check_format(writing=False)

    def sync(self, other):
        """Bidirectional delta exchange."""
        other.apply(self.updates_for(other))
        self.apply(other.updates_for(self))

    def snapshot(self):
        self.flush()
        return self.doc.export(ExportMode.Snapshot())

    # -- history trimming -----------------------------------------------

    def _reachable_by_delta(self, vv):
        """Can a peer at `vv` be brought up to date by a DELTA?

        Only if it already holds everything before the trim point. On a
        doc that was never trimmed `shallow_since_vv` is empty and this
        is vacuously true, so ordinary replicas never touch this path.
        """
        return (not self.doc.is_shallow()
                or vv.includes_vv(self.doc.shallow_since_vv))

    def _refuse_stale(self, vv):
        raise ValueError(
            "this replica's history was trimmed at %s, past the "
            "requesting peer's version %s: a delta cannot carry it "
            "forward and Loro " % (self.doc.shallow_since_vv, vv) +
            "will import one WITHOUT error while leaving the peer with "
            "nothing. Re-onboard that peer from snapshot() instead — a "
            "snapshot of a trimmed doc is complete and the peer can "
            "write back normally. See docs/sync.md, 'Trimming history'.")

    def checkpoint(self, at=None):
        """A snapshot with history before `at` discarded — what you hand
        a newcomer or archive to cold storage. `at` defaults to this
        doc's current frontiers (trim everything).

        Trim only past a version EVERY peer already holds: see
        docs/sync.md for what happens to one that is behind."""
        self.flush()
        if at is None:
            at = self.doc.vv_to_frontiers(self.doc.oplog_vv)
        return self.doc.export(ExportMode.ShallowSnapshot(at))

    def trim_history(self, at=None):
        """Discard this replica's own history before `at`, in place.
        Returns (bytes_before, bytes_after) of a full snapshot.

        Rebuilding the doc gives it a FRESH peer id, which would strand
        every `<container>::<peer>` blob this peer has already written
        under a name it no longer writes to — so the id is restored.
        The shallow snapshot carries the op counter with it, so writing
        resumes at the next sequence number rather than colliding with
        the trimmed history.
        """
        before = len(self.snapshot())
        shallow = self.checkpoint(at)
        doc = LoroDoc()
        doc.import_(shallow)
        doc.peer_id = int(self.peer)
        self.doc = doc
        self.local, self.dirty = {}, set()
        self._check_format(writing=False)
        return before, len(shallow)

    def _bundles(self):
        return self.doc.get_map("bundles")

    def _labels(self):
        return self.doc.get_map("labels")


class ReplicatedHoloMap:
    """hashmap.HoloMap, multi-writer. Concurrent puts under one key
    superpose (both retrievable — multi-value-register semantics; the
    app picks a winner and retracts the loser if it wants LWW)."""

    def __init__(self, replica, name="kv"):
        self.replica = replica
        self.name = name
        self.keys = ItemMemory(replica.space, "keys")
        self.values = ItemMemory(replica.space, "values")

    def put(self, key, value):
        self.replica.add(self.name, FHRR.bind(self.keys.get(key),
                                              self.values.get(value)))
        self.replica.register_label(self.name, value)

    def delete(self, key, value):
        """Retract a known pair (see module docstring for semantics)."""
        self.replica.retract(self.name, FHRR.bind(self.keys.get(key),
                                                  self.values.get(value)))

    def _codebook(self):
        for label in self.replica.known_labels(self.name):
            self.values.get(label)   # deterministic; order irrelevant

    def get(self, key):
        self._codebook()
        v = FHRR.unbind(self.replica.merged(self.name), self.keys.get(key))
        return self.values.cleanup(v)

    def get_all(self, key, threshold=0.5):
        """All values bundled under key (concurrent writes show up here)."""
        self._codebook()
        v = FHRR.unbind(self.replica.merged(self.name), self.keys.get(key))
        return self.values.matches(v, threshold)


class ReplicatedSplatScene:
    """A chunked splat field whose cells are replicated containers: the
    chunk is the CRDT sync unit. Peers paint splats offline; syncing
    exchanges only dirty cells; every peer renders the same scene."""

    def __init__(self, replica, sigma, cell_size=0.25, reach=3.0,
                 name="scene"):
        space = replica.space
        proto = GaussianSplatField(space.dim, sigma, seed=space.seed)
        self.replica = replica
        self.name = name
        self.W = proto.W                    # deterministic from space seed
        self.sigma_inv = proto.sigma_inv
        self.cell_size = cell_size
        widest = np.sqrt(np.linalg.eigvalsh(
            np.linalg.inv(self.sigma_inv)).max())
        self.reach_radius = reach * widest

    def _container(self, cell):
        return f"{self.name}:" + ",".join(str(i) for i in cell)

    def add_splat(self, mu, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        cell = tuple((mu // self.cell_size).astype(int))
        vec = np.complex64(alpha) * np.exp(1j * (self.W @ mu)) \
                                      .astype(np.complex64)
        self.replica.add(self._container(cell), vec)

    def eval(self, points):
        points = np.asarray(points, dtype=np.float32)
        out = np.zeros(len(points), dtype=np.float32)
        for name in self.replica.containers(prefix=f"{self.name}:"):
            cell = tuple(int(i) for i in name.split(":", 1)[1].split(","))
            S = self.replica.merged(name)
            lo = np.array(cell, dtype=np.float32) * self.cell_size
            nearest = np.clip(points, lo, lo + self.cell_size)
            mask = ((points - nearest) ** 2).sum(axis=1) \
                <= self.reach_radius ** 2
            if not mask.any():
                continue
            from .accel import readout
            out[mask] += readout(points[mask], self.W, S)
        return out


class ReplicatedRecordSpace:
    """record.RecordSpace made coordination-free: every vector is already
    hash-derived from its label, so the only thing peers must share is
    which role/filler labels EXIST. encode() registers them in the
    grow-only registry; decode-side rebuilds both codebooks from it
    before cleanup. Any peer can then decode any record — including the
    schema itself (known_roles), discovered rather than agreed on."""

    def __init__(self, replica, name="records"):
        self.replica = replica
        self.name = name
        self.rs = RecordSpace(replica.space)

    def encode(self, fields):
        for role, filler in fields.items():
            self.replica.register_label(f"{self.name}::roles", role)
            self.replica.register_label(f"{self.name}::fillers", filler)
        return self.rs.encode(fields)

    def _codebooks(self):
        for role in self.replica.known_labels(f"{self.name}::roles"):
            self.rs.roles.get(role)      # hash-derived; order irrelevant
        for filler in self.replica.known_labels(f"{self.name}::fillers"):
            self.rs.fillers.get(filler)

    def known_roles(self):
        return self.replica.known_labels(f"{self.name}::roles")

    def get(self, record_hv, role):
        self._codebooks()
        return self.rs.get(record_hv, role)

    def fields(self, record_hv, threshold=0.5):
        """Decode every registered role scoring above threshold."""
        self._codebooks()
        out = {}
        for role in self.known_roles():
            label, score = self.rs.get(record_hv, role)
            if score >= threshold:
                out[role] = (label, score)
        return out


class ReplicatedAttributeScene:
    """ReplicatedSplatScene with role-filler payloads: every splat binds
    an attribute codeword to its position encoding (attribute_field.py),
    cells stay the CRDT sync unit, and labels travel in the grow-only
    label registry — so a peer can answer what_is_at(p) with codewords
    for labels it has never used locally. Everything that must agree
    across peers is hash-derived from the shared space (dim, seed):
    codewords via FHRR.label_vector, W via AttributeSplatField.

    Locality bonus: a query consults only cells within reach, so payload
    crosstalk follows the LOCAL cell load, not N_total — the chunking
    argument of spatial.py, extended to semantic queries."""

    def __init__(self, replica, sigma, cell_size=0.25, reach=3.0,
                 name="attrs"):
        from .attribute_field import AttributeSplatField
        self.replica = replica
        self.name = name
        self.proto = AttributeSplatField(replica.space, sigma)
        widest = np.sqrt(np.linalg.eigvalsh(
            np.linalg.inv(self.proto.sigma_inv)).max())
        self.cell_size = cell_size
        self.reach_radius = reach * widest
        self.records = ReplicatedRecordSpace(replica, name=f"{name}-records")

    def _container(self, cell):
        return f"{self.name}:" + ",".join(str(i) for i in cell)

    def add_splat(self, mu, label, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        cell = tuple((mu // self.cell_size).astype(int))
        vec = np.complex64(alpha) * FHRR.bind(self.proto.pos(mu),
                                              self.proto.attrs.get(label))
        self.replica.add(self._container(cell), vec)
        self.replica.register_label(self.name, label)

    def add_splat_record(self, mu, fields, alpha=1.0):
        """Bind a whole role-filler record to the position (see
        attribute_field.py). encode() registers every role and filler
        label, so the record decodes on any peer after sync."""
        mu = np.asarray(mu, dtype=np.float32)
        cell = tuple((mu // self.cell_size).astype(int))
        vec = np.complex64(alpha) * FHRR.bind(self.proto.pos(mu),
                                              self.records.encode(fields))
        self.replica.add(self._container(cell), vec)

    def record_at(self, p, role):
        """Field `role` of the record at p: position-unbind, role-unbind,
        cleanup — codebooks rebuilt from the replicated registry."""
        return self.records.get(self.at(p), role)

    def fields_at(self, p, threshold=0.5):
        """Every registered role decodable at p — the schema comes from
        the registry, not from anything this peer stored."""
        return self.records.fields(self.at(p), threshold)

    def _codebook(self):
        for label in self.replica.known_labels(self.name):
            self.proto.attrs.get(label)   # hash-derived; order irrelevant

    def _cell_masks(self, points):
        """Yield (container, mask of points within reach of that cell)."""
        for cname in self.replica.containers(prefix=f"{self.name}:"):
            cell = tuple(int(i) for i in cname.split(":", 1)[1].split(","))
            lo = np.array(cell, dtype=np.float32) * self.cell_size
            nearest = np.clip(points, lo, lo + self.cell_size)
            mask = ((points - nearest) ** 2).sum(axis=1) \
                <= self.reach_radius ** 2
            if mask.any():
                yield cname, mask

    def at(self, p):
        """Merged (noisy) payload at p, from in-reach cells only."""
        p = np.asarray(p, dtype=np.float32)
        total = self.replica.space.zeros()
        for cname, _ in self._cell_masks(p[None, :]):
            total += self.replica.merged(cname)
        return FHRR.unbind(total, self.proto.pos(p))

    def what_is_at(self, p):
        self._codebook()
        return self.proto.attrs.cleanup(self.at(p))

    def eval_where(self, label, points):
        """Render where_is(label) across every peer's cells."""
        self._codebook()
        A = self.proto.attrs.get(label)
        points = np.asarray(points, dtype=np.float32)
        out = np.zeros(len(points), dtype=np.float32)
        from .accel import readout
        for cname, mask in self._cell_masks(points):
            H = FHRR.unbind(self.replica.merged(cname), A)
            out[mask] += readout(points[mask], self.proto.W, H)
        return out


def _demo_plot(view_a, view_b, merged_a, attr_b, P, grid):
    """The demo's two figures: divergence-then-merge, and a label only
    one peer ever used being rendered by the other."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping image")
        return

    vmax = merged_a.max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (title, img) in zip(axes, [
            ("peer A before sync (left half)", view_a),
            ("peer B before sync (right half)", view_b),
            ("either peer after Loro sync", merged_a)]):
        ax.imshow(img.reshape(grid, grid), origin="lower", cmap="magma",
                  vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Holographic scene as a CRDT: bundles merge by "
                 "addition, Loro makes the exchange exactly-once",
                 fontsize=11)
    fig.tight_layout()
    import os
    os.makedirs("out", exist_ok=True)
    fig.savefig("out/crdt_scene.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, lab in zip(axes, ["tree", "rock"]):
        img = attr_b.eval_where(lab, P).reshape(grid, grid)
        ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=1.1)
        ax.set_title(f'peer B renders where_is("{lab}")', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Attributed splats over CRDT sync — B renders "
                 "'tree', a label only A ever used", fontsize=11)
    fig.tight_layout()
    fig.savefig("out/crdt_attributes.png", dpi=110)
    plt.close(fig)
    print("  saved out/crdt_scene.png, out/crdt_attributes.png")


def demo(dim=4096, seed=0, save_png=True):
    if not HAVE_LORO:
        print("== CRDT demo skipped: pip install loro ==\n")
        return
    banner("Loro-replicated holographic structures", dim)

    # -- two peers, offline divergence, delta sync ----------------------
    A = HoloReplica(FHRR(dim, seed=seed))
    B = HoloReplica(FHRR(dim, seed=seed))
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    rng = np.random.default_rng(seed + 9)
    pairs_a = {f"a-key{i}": f"val{rng.integers(64)}" for i in range(200)}
    pairs_b = {f"b-key{i}": f"val{rng.integers(64)}" for i in range(200)}
    for k, v in pairs_a.items():
        kv_a.put(k, v)
    for k, v in pairs_b.items():
        kv_b.put(k, v)

    upd = A.updates_for(B)
    A.sync(B)
    both = {**pairs_a, **pairs_b}
    ok_a = sum(kv_a.get(k)[0] == v for k, v in both.items())
    ok_b = sum(kv_b.get(k)[0] == v for k, v in both.items())
    identical = np.array_equal(A.merged("kv"), B.merged("kv"))
    print(f"  offline puts: A 200 + B 200; after delta sync "
          f"({len(upd):,} bytes) A reads {ok_a}/400, B reads {ok_b}/400, "
          f"merged bundles bit-identical: {identical}")

    # -- idempotent redelivery (the reason for writer-sharding) ---------
    before = B.merged("kv").copy()
    B.apply(upd)                      # replay the SAME update
    print(f"  redelivering the same update: state unchanged = "
          f"{np.array_equal(before, B.merged('kv'))} "
          "(Loro version vectors reject replayed ops; a naive "
          "add-your-bundle merge would have double-counted)")

    # -- concurrent writes to one key (clean container for clarity) -----
    cfg_a, cfg_b = (ReplicatedHoloMap(A, name="cfg"),
                    ReplicatedHoloMap(B, name="cfg"))
    cfg_a.put("shared-key", "from-A")
    cfg_b.put("shared-key", "from-B")
    A.sync(B)
    values = cfg_a.get_all("shared-key")
    print(f"  concurrent put('shared-key'): both survive superposed -> "
          f"{[(label, round(s, 2)) for label, s in values]} "
          "(multi-value register; app resolves, loser gets retracted)")

    # -- retraction ------------------------------------------------------
    victim = next(iter(pairs_b))
    kv_b.delete(victim, pairs_b[victim])
    A.sync(B)
    print(f"  B retracts {victim!r}; A now reads score "
          f"{kv_a.get(victim)[1]:.2f} (was ~1.0)")

    # -- replicated splat scene: two painters, one hologram -------------
    P_A = HoloReplica(FHRR(dim, seed=seed))
    P_B = HoloReplica(FHRR(dim, seed=seed))
    sigma = np.eye(2) * 0.035 ** 2
    scene_a = ReplicatedSplatScene(P_A, sigma)
    scene_b = ReplicatedSplatScene(P_B, sigma)
    rng = np.random.default_rng(seed + 10)
    for _ in range(150):   # A paints the left half, B the right
        scene_a.add_splat(rng.uniform([0.05, 0.05], [0.5, 0.95]),
                          rng.uniform(0.5, 1.0))
        scene_b.add_splat(rng.uniform([0.5, 0.05], [0.95, 0.95]),
                          rng.uniform(0.5, 1.0))
    grid = 110
    xs = np.linspace(0, 1, grid)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    view_a = scene_a.eval(P)
    view_b = scene_b.eval(P)
    upd_scene = P_A.updates_for(P_B)
    P_A.sync(P_B)
    merged_a, merged_b = scene_a.eval(P), scene_b.eval(P)
    cells = len(P_A.containers(prefix="scene:"))
    print(f"  scene: 150 + 150 splats across {cells} cell containers; "
          f"cell-level delta sync = {len(upd_scene):,} bytes; "
          f"peers render identically: "
          f"{bool(np.allclose(merged_a, merged_b))}")

    # -- attributed scene: labels travel with the splats ----------------
    Q_A = HoloReplica(FHRR(dim, seed=seed))
    Q_B = HoloReplica(FHRR(dim, seed=seed))
    attr_a = ReplicatedAttributeScene(Q_A, 0.025)
    attr_b = ReplicatedAttributeScene(Q_B, 0.025)
    rng = np.random.default_rng(seed + 12)
    placed = []
    for i in range(5):
        for j in range(10):
            jit = rng.uniform(-0.005, 0.005, 2)
            mu = np.array([0.05 + 0.09 * i, 0.05 + 0.09 * j]) + jit
            lab = ["tree", "water"][(i + j) % 2]
            attr_a.add_splat(mu, lab)          # A labels the left half
            placed.append((mu, lab))
            mu = np.array([0.55 + 0.09 * i, 0.05 + 0.09 * j]) + jit
            lab = ["rock", "path"][(i + j) % 2]
            attr_b.add_splat(mu, lab)          # B labels the right half
            placed.append((mu, lab))
    rec_mu = np.array([0.5, 0.5])
    attr_a.add_splat_record(rec_mu, {"kind": "oak", "height": "12m"})
    Q_A.sync(Q_B)
    agree = sum(attr_a.what_is_at(mu)[0] == lab
                and attr_b.what_is_at(mu)[0] == lab for mu, lab in placed)
    tree_map = attr_b.eval_where("tree", P)    # B never used 'tree'
    left_peak = tree_map[P[:, 0] < 0.5].max()
    right_peak = tree_map[P[:, 0] >= 0.5].max()
    print(f"  attributed scene: A paints 50 tree/water splats left, B 50 "
          f"rock/path right; after sync both peers answer what_is_at "
          f"{agree}/100; B renders where_is('tree') — a label it never "
          f"used — peak {left_peak:.2f} on A's half, {right_peak:.2f} on "
          f"its own")
    got = attr_b.fields_at(rec_mu)
    decoded = ", ".join(f"{r}={l} ({s:.2f})"
                        for r, (l, s) in sorted(got.items()))
    print(f"  B decodes A's record splat at (0.5, 0.5): {decoded} — "
          f"roles and fillers rebuilt from the registry, schema included")

    if save_png:
        _demo_plot(view_a, view_b, merged_a, attr_b, P, grid)
    print()
