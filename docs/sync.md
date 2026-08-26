# CRDT replication on Loro

*[← docs index](README.md) · collaboration & persistence*

**What.** Superposition is *almost* a CRDT merge: bundles add, addition
commutes and associates, so replicas accumulate independently and any
merge order converges. What addition is NOT is idempotent — naive
"send me your bundle" double-counts on redelivery. The G-Counter recipe
closes the gap: shard each container's accumulator BY WRITER,

    Loro map "bundles":  "<container>::<peer>" -> complex64 blob

Each peer only writes its own keys (single-writer LWW is trivially
safe); Loro's version vectors make delivery exactly-once; the merged
hologram is the SUM over peers' blobs at read time — order-free, so
replicas that saw the same updates read the same bytes. Loro
contributes exactly what the algebra lacks: causal versioning, delta
sync, snapshots, key-set convergence.

**Coordination-free by construction.** Codewords are hash-derived from
labels and `W` from the space seed ([core.md](core.md)), so peers agree
on the entire algebra with NO negotiation; labels/roles replicate as
*names* in a grow-only registry. Consequences with tests behind them:
a peer answers `what_is_at` and renders `where_is` for labels it never
used; a peer decodes every field of records it never stored,
discovering the schema from the registry.

```mermaid
sequenceDiagram
    participant A as peer A
    participant DA as A's doc (Loro)
    participant DB as B's doc (Loro)
    participant B as peer B
    A->>DA: add(container, vec) — writes ONLY "container::A"
    B->>DB: add(container, vec) — writes ONLY "container::B"
    A->>DA: flush()  — one map op per dirty container
    DA->>DB: updates_since(B's vv) — delta frame
    Note over DB: version vectors make<br/>redelivery a no-op
    B->>DB: merged(container)
    Note over B: Σ over all peers' blobs<br/>= identical bytes on every replica
```

**Wire protocol.** `version()` snapshots the vv after each exchange;
`updates_since(vv)` exports exactly what the peer lacks — no access to
the peer's doc. `live_sync.py` runs it for real: two OS processes
co-paint one scene over TCP with length-prefixed delta frames
(~100KB/round, one stroke epoch), ending with identical state and
render digests.

**Format tags (wire v1).** Every blob carries a 12-byte `HB` header
(version, dtype, channels, dim) — `pack_bundle`/`unpack_bundle` are the
only codec, and untagged bytes are refused. Every doc carries a format
record (`format` map, key `holo`: JSON `{wire, dim, seed}` — the
universe), validated on every flush and after every `apply()`: peers
from a different universe are refused loudly instead of decoding
garbage. The record rides in the same commit as the doc's first content
ops so delta sync never strands it. Key schemes are part of v1;
changing any layout bumps `WIRE_VERSION`.

**Failure modes.** Retraction here is arithmetic (counter semantics):
concurrent duplicate retraction over-cancels into a negative phantom —
single-owner removal only; multi-writer deletion belongs to
[orset.md](orset.md). Concurrent same-key writes superpose (multi-value
register; the app picks and retracts). Digest bytes, not recomputed
sums. Batch writes (`flush()` per sync); compact long docs with
`ExportMode.ShallowSnapshot`.

**API.**
```python
from holo import FHRR, HoloReplica, ReplicatedHoloMap
A = HoloReplica(FHRR(4096, seed=0))   # same (dim, seed) on every peer
B = HoloReplica(FHRR(4096, seed=0))
kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
kv_a.put("k", "v"); A.sync(B)         # in-process
frame = A.updates_since(last_vv)      # over a socket
```

**Evidence.** Two peers paint halves of a scene offline and delta-sync
into one hologram:

![peer A before sync, peer B before sync, either peer after Loro sync](../out/crdt_scene.png)

![two OS processes co-painting over TCP, matching final digests](../out/live_sync.png)

`tests/test_crdt.py` (convergence, idempotent redelivery, remote record
decode), `tests/test_live_sync.py` (blob-only wire assembly on a third
replica; the two-process subprocess test); `holo-demos crdt`.
