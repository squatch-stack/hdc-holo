"""Compatibility shim: the implementation lives in holo/accel.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.accel")
