"""The code graph: this repo's own structure, queryable in Cypher.

Structure questions are graph questions — "what would break if I
changed this", "what does nothing call", "which module is the hub" —
and grep answers none of them. `holo-quality graph build` walks the
tree with Python's own `ast` and writes an embedded Kùzu database
(`quality/codegraph/`, gitignored, rebuilt in seconds), which
`holo-quality graph query` then answers in Cypher.

Schema, deliberately small — a graph nobody can hold in their head
gets queried by nobody:

    (:Module   {path, package, lane, basename, needs_test,
                loc, functions})
    (:Function {qualname, module, name, lineno, loc, complexity, args})
    (:Class    {qualname, module, name, lineno, methods})

    (:Module)-[:IMPORTS {names}]->(:Module)      resolved, in-repo only
    (:Module)-[:DEFINES]->(:Function|:Class)
    (:Class)-[:HAS_METHOD]->(:Function)
    (:Function)-[:CALLS {count}]->(:Function)    name-resolved

HONEST LIMITS. `CALLS` is resolved by name against the functions this
repo defines: a call to a same-named method on two classes cannot be
told apart, calls through variables/getattr are invisible, and calls
into numpy or any dependency are simply absent (only in-repo edges are
stored). So CALLS is a strong hint and a weak proof — good for "what
is definitely reachable", never for "this is definitely dead". IMPORTS,
DEFINES, and the node properties are exact, being syntax rather than
inference. Complexity is McCabe counted the same way ruff's C901 does
(one per branch point), so graph numbers and lint findings agree.
"""

import ast
import os
import shutil

from .structure import NO_TEST_REQUIRED

__all__ = ["CHECKS", "DB_PATH", "SCHEMA", "build", "connect",
           "query"]

DB_PATH = os.path.join("quality", "codegraph")

SCHEMA = [
    """CREATE NODE TABLE Module(path STRING, package STRING, lane STRING,
       basename STRING, needs_test BOOLEAN, loc INT64, functions INT64,
       PRIMARY KEY(path))""",
    """CREATE NODE TABLE Function(qualname STRING, module STRING,
       name STRING, lineno INT64, loc INT64, complexity INT64,
       args INT64, PRIMARY KEY(qualname))""",
    """CREATE NODE TABLE Class(qualname STRING, module STRING,
       name STRING, lineno INT64, methods INT64, PRIMARY KEY(qualname))""",
    "CREATE REL TABLE IMPORTS(FROM Module TO Module, names STRING)",
    """CREATE REL TABLE DEFINES(FROM Module TO Function,
       FROM Module TO Class)""",
    "CREATE REL TABLE HAS_METHOD(FROM Class TO Function)",
    "CREATE REL TABLE CALLS(FROM Function TO Function, count INT64)",
]

# Canned questions. Each is (name, description, cypher) and runs under
# `holo-quality graph checks`; they are the reason the graph exists.
CHECKS = [
    ("complexity-hotspots",
     "functions above the C901 cap, worst first",
     """MATCH (f:Function) WHERE f.complexity > 10
        RETURN f.qualname AS fn, f.complexity AS cx, f.loc AS loc
        ORDER BY cx DESC LIMIT 15"""),
    ("uncalled-functions",
     "defined in holo/, called by nothing in-repo — a HINT, not proof: "
     "the public API is called by users, not by this repo",
     """MATCH (f:Function) WHERE f.module STARTS WITH 'holo/'
          AND NOT f.name STARTS WITH '_'
          AND NOT f.name STARTS WITH 'demo'
          AND f.name <> 'main'
          AND NOT EXISTS { MATCH (:Function)-[:CALLS]->(f) }
        RETURN f.qualname AS fn, f.module AS module, f.loc AS loc
        ORDER BY loc DESC LIMIT 20"""),
    ("import-hubs",
     "modules the most other modules depend on — the blast radius list",
     """MATCH (m:Module)<-[:IMPORTS]-(other:Module)
        RETURN m.path AS module, count(other) AS dependents
        ORDER BY dependents DESC LIMIT 10"""),
    ("lane-leakage",
     "coupling BETWEEN infrastructure lanes — examples importing the "
     "library is the point of examples, so those are excluded",
     """MATCH (a:Module)-[:IMPORTS]->(b:Module)
        WHERE a.lane <> b.lane
          AND a.lane IN ['facts', 'quality', 'capture']
          AND b.lane IN ['facts', 'quality', 'capture']
        RETURN a.path AS importer, a.lane AS from_lane,
               b.path AS imported, b.lane AS to_lane
        ORDER BY importer"""),
    ("god-modules",
     "modules carrying an outsized share of the code",
     """MATCH (m:Module) WHERE m.loc > 400
        RETURN m.path AS module, m.loc AS loc, m.functions AS functions
        ORDER BY loc DESC LIMIT 10"""),
    ("untested-modules",
     "holo modules with no test twin (the structure rule, as a query)",
     """MATCH (m:Module) WHERE m.needs_test
        AND NOT EXISTS {
            MATCH (t:Module) WHERE t.package = 'tests'
              AND t.basename = 'test_' + m.basename }
        RETURN m.path AS module, m.loc AS loc ORDER BY loc DESC LIMIT 15"""),
]


def _lane(path):
    """Which lane a module belongs to — the ownership axis this repo's
    sessions actually coordinate along."""
    if path.startswith("holo/facts/"):
        return "facts"
    if path.startswith("holo/quality/"):
        return "quality"
    if os.path.basename(path) in ("capture.py", "sog.py", "spectral.py",
                                  "spatial.py"):
        return "capture"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("examples/") or path.startswith("bench/"):
        return "examples"
    return "core"


def _complexity(node):
    """McCabe, counted as ruff's C901 counts it: one per branch point."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While,
                              ast.ExceptHandler, ast.With, ast.AsyncWith,
                              ast.Assert, ast.IfExp)):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
        elif isinstance(child, comprehension_types()):
            score += 1
    return score


def comprehension_types():
    return (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _module_name(rel):
    return rel[:-3].replace(os.sep, ".").replace(".__init__", "")


def _walk_python(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", ".venv", "__pycache__", "data",
                                    "out", "results", "gpubench", "node_modules"}
                       and not d.startswith(".")]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.relpath(os.path.join(dirpath, name), root)


def _record_function(child, rel, cls, functions, calls):
    qual = "%s::%s%s" % (rel, cls + "." if cls else "", child.name)
    end = getattr(child, "end_lineno", child.lineno)
    functions.append({
        "qualname": qual, "module": rel, "name": child.name,
        "lineno": child.lineno, "loc": end - child.lineno + 1,
        "complexity": _complexity(child),
        "args": len(child.args.args) + len(child.args.kwonlyargs),
        "class": cls})
    for sub in ast.walk(child):
        if isinstance(sub, ast.Call):
            target = getattr(sub.func, "id", None) or \
                getattr(sub.func, "attr", None)
            if target:
                calls.append((qual, target))


def _collect_defs(node, rel, functions, classes, calls, cls=None):
    """Walk one body level, recursing into classes for their methods."""
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _record_function(child, rel, cls, functions, calls)
        elif isinstance(child, ast.ClassDef):
            classes.append({
                "qualname": "%s::%s" % (rel, child.name), "module": rel,
                "name": child.name, "lineno": child.lineno,
                "methods": sum(1 for g in child.body
                               if isinstance(g, (ast.FunctionDef,
                                                 ast.AsyncFunctionDef)))})
            _collect_defs(child, rel, functions, classes, calls,
                          cls=child.name)


def _parse(root, rel):
    """(module_row, functions, classes, imports, calls) for one file."""
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    loc = sum(1 for line in src.split("\n")
              if line.strip() and not line.strip().startswith("#"))
    functions, classes, imports, calls = [], [], [], []

    _collect_defs(tree, rel, functions, classes, calls)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.append((node.module or "",
                            ",".join(a.name for a in node.names)))
        elif isinstance(node, ast.ImportFrom):        # relative import
            pkg = os.path.dirname(rel).replace(os.sep, ".")
            imports.append(((pkg + "." + (node.module or "")).rstrip("."),
                            ",".join(a.name for a in node.names)))
        elif isinstance(node, ast.Import):
            for a in node.names:
                imports.append((a.name, a.name))

    # the SAME predicate structure.py enforces, carried into the graph
    # so the rule and the query cannot drift apart
    base = os.path.basename(rel)
    needs_test = (rel == "holo/" + base and base not in NO_TEST_REQUIRED)
    module = {"path": rel, "package": rel.split(os.sep)[0],
              "lane": _lane(rel), "basename": base,
              "needs_test": needs_test,
              "loc": loc, "functions": len(functions)}
    return module, functions, classes, imports, calls


def _fresh_db(root, db_path):
    """Clear whatever shape the previous database left behind: Kuzu
    writes a single file plus sidecars now, a directory in older
    versions."""
    import kuzu
    path = os.path.join(root, db_path or DB_PATH)
    for candidate in (path, path + ".wal", path + ".tmp"):
        if os.path.isdir(candidate):
            shutil.rmtree(candidate)
        elif os.path.exists(candidate):
            os.remove(candidate)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = kuzu.Database(path)
    conn = kuzu.Connection(db)
    for stmt in SCHEMA:
        conn.execute(stmt)
    return db, conn


def _scan(root):
    """Every parseable module in the tree, as parallel row lists."""
    modules, functions, classes, imports, calls = [], [], [], [], []
    for rel in _walk_python(root):
        try:
            m, f, c, i, k = _parse(root, rel)
        except (SyntaxError, UnicodeDecodeError):
            continue
        modules.append(m)
        functions.extend(f)
        classes.extend(c)
        imports.extend((rel, mod, names) for mod, names in i)
        calls.extend(k)
    return modules, functions, classes, imports, calls


def _insert_nodes(conn, modules, functions, classes):
    for m in modules:
        conn.execute(
            "CREATE (:Module {path: $p, package: $k, lane: $l, "
            "basename: $b, needs_test: $t, loc: $n, functions: $f})",
            {"p": m["path"], "k": m["package"], "l": m["lane"],
             "b": m["basename"], "t": m["needs_test"], "n": m["loc"],
             "f": m["functions"]})
    for f in functions:
        conn.execute(
            "CREATE (:Function {qualname: $q, module: $m, name: $n, "
            "lineno: $l, loc: $o, complexity: $c, args: $a})",
            {"q": f["qualname"], "m": f["module"], "n": f["name"],
             "l": f["lineno"], "o": f["loc"], "c": f["complexity"],
             "a": f["args"]})
    for c in classes:
        conn.execute(
            "CREATE (:Class {qualname: $q, module: $m, name: $n, "
            "lineno: $l, methods: $t})",
            {"q": c["qualname"], "m": c["module"], "n": c["name"],
             "l": c["lineno"], "t": c["methods"]})


def _insert_defines(conn, functions, classes):
    for f in functions:
        conn.execute(
            "MATCH (m:Module {path: $m}), (f:Function {qualname: $q}) "
            "CREATE (m)-[:DEFINES]->(f)",
            {"m": f["module"], "q": f["qualname"]})
        if f["class"]:
            conn.execute(
                "MATCH (c:Class {qualname: $c}), (f:Function {qualname: $q}) "
                "CREATE (c)-[:HAS_METHOD]->(f)",
                {"c": "%s::%s" % (f["module"], f["class"]),
                 "q": f["qualname"]})
    for c in classes:
        conn.execute(
            "MATCH (m:Module {path: $m}), (c:Class {qualname: $q}) "
            "CREATE (m)-[:DEFINES]->(c)",
            {"m": c["module"], "q": c["qualname"]})


def _insert_imports(conn, modules, imports):
    """Only in-repo edges: an import of numpy is not a graph fact here."""
    name_to_path = {_module_name(m["path"]): m["path"] for m in modules}
    n = 0
    for src, mod, names in imports:
        target = name_to_path.get(mod)
        if target is None or target == src:
            continue
        conn.execute(
            "MATCH (a:Module {path: $a}), (b:Module {path: $b}) "
            "CREATE (a)-[:IMPORTS {names: $n}]->(b)",
            {"a": src, "b": target, "n": names[:200]})
        n += 1
    return n


def _insert_calls(conn, functions, calls):
    """Name resolution, with ambiguity DROPPED rather than guessed: a
    name defined twice yields no edge at all."""
    by_name = {}
    for f in functions:
        by_name.setdefault(f["name"], []).append(f["qualname"])
    pairs = {}
    for caller, target in calls:
        matches = by_name.get(target, [])
        if len(matches) != 1 or matches[0] == caller:
            continue
        key = (caller, matches[0])
        pairs[key] = pairs.get(key, 0) + 1
    for (caller, callee), count in pairs.items():
        conn.execute(
            "MATCH (a:Function {qualname: $a}), (b:Function {qualname: $b}) "
            "CREATE (a)-[:CALLS {count: $c}]->(b)",
            {"a": caller, "b": callee, "c": count})
    return len(pairs)


def build(root, db_path=None):
    """(re)build the graph; returns a counts dict."""
    try:
        import kuzu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "the code graph needs kuzu — pip install 'hdc-holo[quality]'"
        ) from e

    db, conn = _fresh_db(root, db_path)
    modules, functions, classes, imports, calls = _scan(root)
    _insert_nodes(conn, modules, functions, classes)
    _insert_defines(conn, functions, classes)
    n_imports = _insert_imports(conn, modules, imports)
    n_calls = _insert_calls(conn, functions, calls)

    # Kuzu checkpoints on close; without this the writes stay in the WAL
    # and a fresh process opens an empty database (measured — the first
    # build only "worked" because the interpreter exited).
    for handle in (conn, db):
        closer = getattr(handle, "close", None)
        if closer:
            closer()

    return {"modules": len(modules), "functions": len(functions),
            "classes": len(classes), "imports": n_imports,
            "calls": n_calls}


def connect(root, db_path=None):
    import kuzu
    path = os.path.join(root, db_path or DB_PATH)
    if not os.path.exists(path):
        raise RuntimeError("no code graph — run: holo-quality graph build")
    return kuzu.Connection(kuzu.Database(path))


def query(conn, cypher):
    """(columns, rows) — plain Python, ready to print or assert on."""
    result = conn.execute(cypher)
    columns = result.get_column_names()
    rows = []
    while result.has_next():
        rows.append(list(result.get_next()))
    return columns, rows
