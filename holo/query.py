"""Query patterns over holographic state.

Structured queries are METHODS on the state they interrogate, because
the query IS algebra on the vector: RecordSpace.get / .analogy (role
unbinding, Kanerva analogies), AttributeSplatField.what_is_at /
.where_is / .is_there (kernel-addressed unbinding), HoloGraph.neighbors,
HoloFSM.step. This module re-exports the record machinery — the piece
that composes with everything else as a payload — and serves as the
index to the rest.
"""

from .record import RecordSpace

__all__ = ["RecordSpace"]
