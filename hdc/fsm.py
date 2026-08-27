"""Compatibility shim: the implementation lives in holo/fsm.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.fsm")
