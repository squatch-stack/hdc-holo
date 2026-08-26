"""Replication on Loro CRDTs (install extra: ``holo[crdt]``).

Two deletion models, chosen per container (see SDK.md):
  * G-Counter style (HoloReplica + Replicated*): writer-sharded blobs,
    arithmetic retraction — single-owner removal only.
  * Observed-remove (ORStore, ORHoloMap, ORStrokeScene): tombstone
    sets, idempotent concurrent removal, add-wins, epoch/stroke undo,
    owner compaction — multi-writer mutable state.

Wire protocol for sockets: HoloReplica.version() / updates_since().
See live_sync.py for two OS processes co-painting with undo over TCP.
"""

from .crdt import (HAVE_LORO, WIRE_VERSION, HoloReplica,  # noqa: F401
                   ReplicatedAttributeScene, ReplicatedHoloMap,
                   ReplicatedRecordSpace, ReplicatedSplatScene,
                   pack_bundle, unpack_bundle)
from .color import ReplicatedColorScene  # noqa: F401
from .orset import ORHoloMap, ORStore, ORStrokeScene  # noqa: F401

__all__ = ["HAVE_LORO", "HoloReplica", "ReplicatedHoloMap",
           "ReplicatedSplatScene", "ReplicatedAttributeScene",
           "ReplicatedRecordSpace", "ReplicatedColorScene",
           "ORStore", "ORHoloMap", "ORStrokeScene",
           "WIRE_VERSION", "pack_bundle", "unpack_bundle"]
