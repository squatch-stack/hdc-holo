"""Classical data structures rebuilt as holograms, thin on holo.core.

Capacity is API: every structure's noise budget is ~sqrt(N/(2d)) —
see each class docstring and the demo capacity tables (`hdc-demos`).
"""

from .fsm import HoloFSM  # noqa: F401
from .graph import HoloGraph  # noqa: F401
from .hashmap import HoloMap  # noqa: F401
from .ngram import NGramEncoder  # noqa: F401
from .record import RecordSpace  # noqa: F401
from .sdm import SparseDistributedMemory  # noqa: F401
from .sequence import HoloStack, SequenceMemory  # noqa: F401
from .sketch import FrequencySketch, MembershipFilter  # noqa: F401

__all__ = ["HoloMap", "MembershipFilter", "FrequencySketch", "RecordSpace",
           "HoloStack", "SequenceMemory", "NGramEncoder", "HoloGraph",
           "HoloFSM", "SparseDistributedMemory"]
