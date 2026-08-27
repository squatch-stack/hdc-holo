"""Project-structure rules, enforced instead of merely documented.

Every rule here was a written convention first and drifted anyway —
which is the argument for checking it. Rules are constants rather than
config: there is one repo, the rules are short, and a config file for
three rules is more moving parts than the rules themselves.

  root-clutter      the repo root is for metadata, config, and entry
                    documents. Exactly one .py may live there (the
                    hdc_splat compat shim); drivers belong in examples/.
  module-test-pair  tests/TESTING.md requires one test file per holo
                    module — `holo/foo.py` <-> `tests/test_foo.py`.
  driver-leaf       examples/ drivers wrap the public API; nothing in
                    holo/ may import one, or an example becomes a
                    dependency of the library it demonstrates.
  shim-unreachable  the root compat shims sit at the repo root, which
                    is NOT on sys.path when a driver runs as
                    `python examples/foo.py` — importing one there
                    raises ModuleNotFoundError. Import `holo.*`.
"""

import ast
import glob
import os

__all__ = ["NO_TEST_REQUIRED", "ROOT_PY_ALLOWLIST", "check_structure"]

ROOT_PY_ALLOWLIST = {"hdc_splat.py"}

# Facades re-export their implementation modules; the implementation's
# own test file is the coverage, so a facade needs no twin.
NO_TEST_REQUIRED = {
    "__init__.py", "core.py", "encode.py", "structures.py", "scene.py",
    "query.py", "render.py", "fit.py", "sync.py", "storage.py",
    "backend.py", "cli.py",
}


def _root_clutter(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*.py"))):
        name = os.path.basename(path)
        if name not in ROOT_PY_ALLOWLIST:
            out.append(("FAIL", "root-clutter", name,
                        "root .py files belong in examples/ (allowed: %s)"
                        % ", ".join(sorted(ROOT_PY_ALLOWLIST))))
    return out


def _module_test_pairs(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "holo", "*.py"))):
        name = os.path.basename(path)
        if name in NO_TEST_REQUIRED:
            continue
        if not os.path.exists(os.path.join(root, "tests", "test_%s" % name)):
            out.append(("WARN", "module-test-pair", "holo/%s" % name,
                        "no tests/test_%s — TESTING.md requires one test "
                        "file per module" % name))
    return out


def _imported_modules(path):
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _driver_leaf(root):
    out = []
    pattern = os.path.join(root, "holo", "**", "*.py")
    for path in sorted(glob.glob(pattern, recursive=True)):
        for mod in _imported_modules(path):
            if mod == "examples" or mod.startswith("examples."):
                out.append(("FAIL", "driver-leaf",
                            os.path.relpath(path, root),
                            "imports %s — examples are leaves, never "
                            "library dependencies" % mod))
    return out


def _shim_unreachable(root):
    """Every example that imports a ROOT-level module, which is exactly
    what `python examples/foo.py` cannot resolve.

    This rule exists because the root-layout move introduced it: two
    drivers moved into examples/ kept `from hdc_splat import ...` and
    crashed on every invocation the docs recommend, silently, because
    nobody ran them for weeks.
    """
    shims = {name[:-3] for name in os.listdir(root)
             if name.endswith(".py")}
    out = []
    for path in sorted(glob.glob(os.path.join(root, "examples", "*.py"))):
        rel = os.path.relpath(path, root)
        for mod in _imported_modules(path):
            if mod.split(".")[0] in shims:
                out.append(("FAIL", "shim-unreachable", rel,
                            "imports %r, a repo-root module that is not "
                            "importable from examples/ — import holo.* "
                            "instead" % mod))
    return out


def check_structure(root):
    """[(level, code, path, message)] — empty when the tree is clean."""
    return (_root_clutter(root) + _module_test_pairs(root)
            + _driver_leaf(root) + _shim_unreachable(root))
