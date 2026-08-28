# Observed-remove deletion and undo

*[← docs index](README.md) · collaboration & persistence*

**What.** Deletion with its type changed. Arithmetic retraction
([sync.md](sync.md)) subtracts at WRITE time — correct alone, wrong
together (two peers retracting the same item over-cancel into a
negative phantom, the PN-Counter anomaly). The OR-Set recipe ported to
superposition:

- **add**: tag each addend `<name>/<peer>.<epoch>/<i>`, record its
  descriptor (the recipe for its vector) in a replicated index; adds
  accumulate into per-peer EPOCH bundles, sealed per batch or per brush
  stroke.
- **remove**: insert the observed id into a grow-only tombstone map.
- **read**: sum non-tombstoned epochs, subtract re-encoded item
  tombstones — ONCE per unique tombstone, ever. Removing a whole epoch
  is pure EXCLUSION: no arithmetic at all.

Two classic properties fall out: *idempotent removal* (N peers removing
the same id still subtract it once — over-cancellation is structurally
impossible, because readers subtract per tombstone, not per remove op)
and *add-wins* (a remove covers only ids it OBSERVED; a concurrent
re-add has a fresh id and survives).

```mermaid
sequenceDiagram
    participant A as peer A
    participant B as peer B
    Note over A,B: item "k" exists as id k1 (observed by both)
    A->>A: remove(k) — tombstone {k1}
    B->>B: remove(k) — tombstone {k1}  (concurrent)
    B->>B: put(k) — fresh id k2 (concurrent re-add)
    A-->>B: sync
    Note over A,B: tombstones = {k1} (a SET — two removes, one entry)<br/>read: subtract k1 once; k2 was never observed → survives<br/>idempotent removal + add-wins, by construction
```

**Measured.** Concurrent double-delete: arithmetic scores -1.00
(phantom), observed-remove scores +0.00. Concurrent double-undo of a
brush stroke across two OS processes over TCP: both peers pick the same
stroke independently, tombstone it twice, it vanishes once — field
floor -0.30 (crosstalk noise) vs -2.21 for naive double-subtraction.

**Cells: one stroke, one tombstone.** `add(descriptor, cell=)` partitions
an epoch's bundle by space, and `merged(cell=)` reads one back. It exists
for capture scale, where a scene has thousands of cells and a brush
stroke crosses dozens, and it buys two things a store-per-cell does not.

The accumulator holds only the cells THIS stroke touched: 40 cells at
d=8192 and four channels is **10.0 MB against 655.2 MB**, 66x, for one
`(channels, d)` array per cell of the whole capture. And undoing a
stroke stays **one tombstone** however many cells it wrote, because the
tombstone names the epoch and the blob keys hang off it — N tombstones
could otherwise arrive across N syncs, leaving a peer rendering half an
undo.

The cell rides in the blob KEY of the one flat map (`<name>/<peer>.
<epoch>@<cell>`), not in a child container per cell. That is Loro's own
advice: two peers lazily creating the same child container concurrently
get conflicting container ids, which "prevents automatic merging and may
result in data loss". A flat map creates no child containers, so the
hazard cannot arise. `@` rather than `/` because an item id is
`<name>/<peer>.<epoch>/<i>` and is told from an epoch key by its slash
count.

An item lives in exactly one cell's blob, so a reader summing a cell
subtracts only the tombstones belonging to it — subtracting one from
another cell would remove something that was never added there. An epoch
written without cells still stores the old bare-list index, so documents
predating this read back unchanged.

**Costs and maintenance.** Item tombstones cost one re-encode per read
until the OWNER `compact()`s them into its own blobs (owner-only keys,
no races; readers skip folded ids — merged() unchanged, verified).
Epochs are the capacity knob: coarser = more holographic, finer = more
surgical deletion. Reconstruction is deterministic only to ~1 ulp —
compare merged() with `allclose`, digest blob bytes.

**Capture scale.** `ORStrokeScene(replica, sigma, cell_size=)` partitions
a scene the way `capture.encode_bands` does — the same floor-divide, so a
stroke lands in the cells a capture would have put it in — and
`eval_rgb(points, cell=)` reads one back. `cells()` lists what has been
written to. `cell_size=None` is the unpartitioned 2-D demo path and is
unchanged.

**API.**
```python
from holo import FHRR, HoloReplica, ORHoloMap, ORStrokeScene
kv = ORHoloMap(HoloReplica(FHRR(4096, seed=0)))
add_id = kv.put("k", "v")
kv.remove("k")                        # tombstones all OBSERVED ids
scene = ORStrokeScene(replica, sigma)
scene.add_splat(mu, rgb); stroke = scene.end_stroke()
scene.undo_stroke(stroke)             # any peer; duplicates are safe
```

**Evidence.** Before / concurrent double-undo / what arithmetic would
have done:

![observed-remove stroke undo vs the arithmetic phantom](../out/orset_undo.png)

`tests/test_orset.py` (phantom regression pin, idempotence, add-wins,
observed-coverage, epoch undo, compaction); `holo-demos orset`;
undo over the wire in `out/live_sync.png` ([sync.md](sync.md)).
