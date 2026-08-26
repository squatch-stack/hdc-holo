"""holo — the SDK for the holographic computing stack.

Implementation home AND public API, laid out per the charter in SDK.md.
The charter-named modules organize the surface:

    holo.core        FHRR algebra, spaces, codewords, cleanup memories
    holo.encode      FPE splat fields, covariance bands, spatial cells
    holo.structures  map / sketch / record / sequence / graph / FSM / SDM
    holo.scene       attribute- and color-carrying splat scenes
    holo.query       role-filler records and query patterns
    holo.render      projection-slice views straight from bundles
    holo.fit         ridge-fitting holograms from data, frequency bands
    holo.sync        Loro replication: G-Counter and observed-remove
    holo.storage     phase-only and quantized codecs
    holo.backend     numpy | MLX/Metal dispatch

with one implementation file per concept underneath (holo/fhrr.py,
holo/hashmap.py, ...). Every public name is also exported flat here:
``from holo import FHRR, HoloMap, ORStrokeScene, ...``. The ``hdc``
package remains as a compatibility shim over this one.
"""

__version__ = "0.2.1"

from .fhrr import FHRR, ItemMemory, Permutation
from .hashmap import HoloMap
from .sketch import MembershipFilter, FrequencySketch
from .record import RecordSpace
from .sequence import HoloStack, SequenceMemory
from .ngram import NGramEncoder
from .graph import HoloGraph
from .fsm import HoloFSM
from .sdm import SparseDistributedMemory
from .field import GaussianSplatField
from .attribute_field import AttributeSplatField
from .spatial import MultiBandSplatField, ChunkedSplatField
from .crdt import (HAVE_LORO, HoloReplica, ReplicatedHoloMap,
                   ReplicatedSplatScene, ReplicatedAttributeScene,
                   ReplicatedRecordSpace)
from .fit import FrequencyBands, HoloRegressor
from .render import render_orthographic, view_bundle, exact_projection
from .color import ColorSplatField, ReplicatedColorScene
from .orset import ORStore, ORHoloMap, ORStrokeScene
from .spectral import (SplatScene, sample_frequencies, decode_weights,
                       spectral_bundle, phasor_bundle, decode_field,
                       decode_field_phasor, translate_bundle)
from .capture import (load_splat, load_spz, load_scene_file, build_scene,
                      render_mip, band_codebooks, encode_bands,
                      decode_slice, exact_slice, render_xray, exact_xray)

from . import (backend, core, encode, fit, query, render, scene,  # noqa: E402
               storage, structures, sync)

__all__ = [
    "FHRR", "ItemMemory", "Permutation",
    "HoloMap", "MembershipFilter", "FrequencySketch", "RecordSpace",
    "HoloStack", "SequenceMemory", "NGramEncoder", "HoloGraph",
    "HoloFSM", "SparseDistributedMemory", "GaussianSplatField",
    "AttributeSplatField", "MultiBandSplatField", "ChunkedSplatField",
    "HAVE_LORO", "HoloReplica", "ReplicatedHoloMap", "ReplicatedSplatScene",
    "ReplicatedAttributeScene", "ReplicatedRecordSpace",
    "FrequencyBands", "HoloRegressor",
    "render_orthographic", "view_bundle", "exact_projection",
    "ColorSplatField", "ReplicatedColorScene",
    "ORStore", "ORHoloMap", "ORStrokeScene",
    "SplatScene", "sample_frequencies", "decode_weights", "spectral_bundle",
    "phasor_bundle", "decode_field", "decode_field_phasor",
    "translate_bundle",
    "load_splat", "load_spz", "load_scene_file", "build_scene", "render_mip",
    "band_codebooks", "encode_bands", "decode_slice", "exact_slice",
    "render_xray", "exact_xray",
    "core", "encode", "structures", "scene", "query", "render",
    "fit", "sync", "storage", "backend",
]
