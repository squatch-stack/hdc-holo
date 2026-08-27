"""Classical data structures rebuilt as holograms, thin on holo.core.

Capacity is API: every structure's noise budget is ~sqrt(N/(2d)) —
see each class docstring and the demo capacity tables (`hdc-demos`).
"""

from .dispatch import (
                       BandedDispatcher,
                       FastNGramProfiler,
                       NearEnoughDispatcher,
)
from .fsm import HoloFSM
from .graph import HoloGraph
from .hashmap import HoloMap
from .ngram import NGramEncoder
from .record import RecordSpace
from .sdm import SparseDistributedMemory
from .sequence import HoloStack, SequenceMemory
from .sketch import FrequencySketch, MembershipFilter

__all__ = [
                       "BandedDispatcher",
                       "FastNGramProfiler",
                       "FrequencySketch",
                       "HoloFSM",
                       "HoloGraph",
                       "HoloMap",
                       "HoloStack",
                       "MembershipFilter",
                       "NGramEncoder",
                       "NearEnoughDispatcher",
                       "RecordSpace",
                       "SequenceMemory",
                       "SparseDistributedMemory",
]
