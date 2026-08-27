"""Compatibility shim: the spectral encoder now lives in holo.spectral.

Kept so the research scripts (examples/run_prototype.py, examples/run_mog.py,
examples/run_real_scene.py) and any notebooks keep working unchanged. New code
should import from `holo` / `holo.encode`.
"""

from holo.spectral import *                                   # noqa: F401,F403
from holo.spectral import _decode, _self_test                 # noqa: F401

if __name__ == "__main__":
    _self_test()
