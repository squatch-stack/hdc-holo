"""holo — the SDK for the holographic computing stack.

Implementation home AND public API, laid out per the charter in SDK.md.
The charter-named modules organize the surface:

    holo.core        FHRR algebra, spaces, codewords, cleanup memories
    holo.encode      FPE splat fields, covariance bands, spatial cells
    holo.structures  map / sketch / record / sequence / graph / FSM /
                     SDM / near-enough dispatch
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

from . import (
                   backend,
                   core,
                   encode,
                   fit,
                   query,
                   render,
                   scene,
                   storage,
                   structures,
                   sync,
)
from .attribute_field import AttributeSplatField
from .capture import (
                   band_codebooks,
                   build_scene,
                   decode_slice,
                   encode_bands,
                   exact_slice,
                   exact_xray,
                   load_ply_sh,
                   load_scene_file,
                   load_splat,
                   load_spz,
                   render_mip,
                   render_xray,
                   save_ply,
                   save_spz,
)
from .color import ColorSplatField, ReplicatedColorScene
from .crdt import (
                   HAVE_LORO,
                   HoloReplica,
                   ReplicatedAttributeScene,
                   ReplicatedHoloMap,
                   ReplicatedRecordSpace,
                   ReplicatedSplatScene,
)
from .dispatch import BandedDispatcher, FastNGramProfiler, NearEnoughDispatcher
from .fhrr import FHRR, ItemMemory, Permutation
from .field import GaussianSplatField
from .fit import FrequencyBands, HoloRegressor
from .fsm import HoloFSM
from .graph import HoloGraph
from .hashmap import HoloMap
from .ngram import NGramEncoder
from .orset import ORHoloMap, ORStore, ORStrokeScene
from .record import RecordSpace
from .render import exact_projection, render_orthographic, view_bundle
from .sdm import SparseDistributedMemory
from .sequence import HoloStack, SequenceMemory
from .sketch import FrequencySketch, MembershipFilter
from .sog import save_sog
from .spatial import ChunkedSplatField, MultiBandSplatField
from .spectral import (
                   SplatScene,
                   decode_field,
                   decode_field_phasor,
                   decode_weights,
                   phasor_bundle,
                   sample_frequencies,
                   spectral_bundle,
                   translate_bundle,
)

__all__ = [
                   "FHRR",
                   "HAVE_LORO",
                   "AttributeSplatField",
                   "BandedDispatcher",
                   "ChunkedSplatField",
                   "ColorSplatField",
                   "FastNGramProfiler",
                   "FrequencyBands",
                   "FrequencySketch",
                   "GaussianSplatField",
                   "HoloFSM",
                   "HoloGraph",
                   "HoloMap",
                   "HoloRegressor",
                   "HoloReplica",
                   "HoloStack",
                   "ItemMemory",
                   "MembershipFilter",
                   "MultiBandSplatField",
                   "NGramEncoder",
                   "NearEnoughDispatcher",
                   "ORHoloMap",
                   "ORStore",
                   "ORStrokeScene",
                   "Permutation",
                   "RecordSpace",
                   "ReplicatedAttributeScene",
                   "ReplicatedColorScene",
                   "ReplicatedHoloMap",
                   "ReplicatedRecordSpace",
                   "ReplicatedSplatScene",
                   "SequenceMemory",
                   "SparseDistributedMemory",
                   "SplatScene",
                   "backend",
                   "band_codebooks",
                   "build_scene",
                   "core",
                   "decode_field",
                   "decode_field_phasor",
                   "decode_slice",
                   "decode_weights",
                   "encode",
                   "encode_bands",
                   "exact_projection",
                   "exact_slice",
                   "exact_xray",
                   "fit",
                   "load_ply_sh",
                   "load_scene_file",
                   "load_splat",
                   "load_spz",
                   "phasor_bundle",
                   "query",
                   "render",
                   "render_mip",
                   "render_orthographic",
                   "render_xray",
                   "sample_frequencies",
                   "save_ply",
                   "save_sog",
                   "save_spz",
                   "scene",
                   "spectral_bundle",
                   "storage",
                   "structures",
                   "sync",
                   "translate_bundle",
                   "view_bundle",
]
