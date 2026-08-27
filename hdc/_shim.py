"""One delegation helper for every `hdc.*` compatibility shim.

The shims used to re-export with `from holo.x import *`, which binds
the objects that exist AT IMPORT TIME. That is invisible until someone
replaces one at runtime — an out-of-tree GPU backend patching
`holo.accel.readout` is the real case — and then the shim, and
anything that reached the kernel through it, keeps calling the
original while results stay plausibly correct. The GPU simply never
engages, silently.

`delegate` resolves on every attribute access instead (PEP 562), so a
shim can never hold a stale object. It also means one implementation
of the shim contract rather than twenty copies of a star import.
"""

import importlib

__all__ = ["delegate"]


def delegate(module_name):
    """(__getattr__, __dir__) that forward to `module_name` live.

    Bind them at a shim's module level:

        __getattr__, __dir__ = delegate("holo.fhrr")
    """

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            # dunders are the module's own business; forwarding them
            # confuses pickling and introspection
            raise AttributeError(name)
        return getattr(importlib.import_module(module_name), name)

    def __dir__():
        return sorted(dir(importlib.import_module(module_name)))

    return __getattr__, __dir__
