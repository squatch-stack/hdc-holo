"""Compatibility shim: the spectral encoder now lives in holo.spectral.

Kept so the research scripts (examples/run_prototype.py, examples/run_mog.py,
examples/run_real_scene.py) and any notebooks keep working unchanged. New code
should import from `holo` / `holo.encode`.
"""

import importlib


def __getattr__(name):
    """Resolve against holo.spectral on every access, so this shim can
    never hand back an object that was replaced at runtime."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(importlib.import_module("holo.spectral"), name)


def __dir__():
    return sorted(dir(importlib.import_module("holo.spectral")))


if __name__ == "__main__":
    __getattr__("_self_test")()
