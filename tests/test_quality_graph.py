"""The Cypher code graph (holo/quality/graph.py).

Dedicated file because kuzu is the optional `quality` extra
(importorskip at file scope per tests/TESTING.md). Tests build a real
graph over a small synthetic tree rather than over this repo: the
assertions then state exact numbers instead of drifting with every
commit.
"""

import os

import pytest

pytest.importorskip("kuzu")

from holo.quality import graph

TREE = {
    "holo/alpha.py": (
        "import os\n"
        "from holo.beta import helper\n"
        "\n"
        "class Engine:\n"
        "    def run(self, x):\n"
        "        if x > 0:\n"
        "            for i in range(x):\n"
        "                helper(i)\n"
        "        return x\n"
        "\n"
        "def spare():\n"
        "    return 1\n"
    ),
    "holo/beta.py": "def helper(i):\n    return i * 2\n",
    "tests/test_alpha.py": "def test_alpha():\n    assert True\n",
    "examples/driver.py": "from holo.alpha import Engine\n",
}


@pytest.fixture
def built(tmp_path):
    for rel, src in TREE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)
    counts = graph.build(str(tmp_path))
    return str(tmp_path), counts


def test_build_counts_every_node_kind(built):
    _, counts = built
    assert counts["modules"] == 4
    assert counts["classes"] == 1
    # helper, spare, Engine.run, test_alpha
    assert counts["functions"] == 4
    assert counts["imports"] == 2      # alpha->beta, driver->alpha (os is external)


def test_imports_edge_resolves_in_repo_only(built):
    root, _ = built
    conn = graph.connect(root)
    _, rows = graph.query(conn, """
        MATCH (a:Module)-[:IMPORTS]->(b:Module)
        RETURN a.path AS importer, b.path AS imported ORDER BY importer""")
    assert [tuple(r) for r in rows] == [
        ("examples/driver.py", "holo/alpha.py"),
        ("holo/alpha.py", "holo/beta.py"),
    ]


def test_calls_edge_links_caller_to_callee(built):
    root, _ = built
    conn = graph.connect(root)
    _, rows = graph.query(conn, """
        MATCH (a:Function)-[:CALLS]->(b:Function)
        RETURN a.name AS caller, b.name AS callee""")
    assert [tuple(r) for r in rows] == [("run", "helper")]


def test_complexity_matches_mccabe_branch_counting(built):
    root, _ = built
    conn = graph.connect(root)
    _, rows = graph.query(conn, """
        MATCH (f:Function) RETURN f.name AS name, f.complexity AS cx
        ORDER BY name""")
    scores = dict(rows)
    # base 1 + the `if` + the `for`
    assert scores["run"] == 3
    assert scores["helper"] == 1


def test_class_owns_its_methods(built):
    root, _ = built
    conn = graph.connect(root)
    _, rows = graph.query(conn, """
        MATCH (c:Class)-[:HAS_METHOD]->(f:Function)
        RETURN c.name AS cls, f.name AS method""")
    assert [tuple(r) for r in rows] == [("Engine", "run")]


def test_needs_test_carries_the_structure_rule(built):
    root, _ = built
    conn = graph.connect(root)
    _, rows = graph.query(conn, """
        MATCH (m:Module) WHERE m.needs_test
        RETURN m.path AS path ORDER BY path""")
    # holo modules only, facades exempt, subpackage files excluded
    assert [r[0] for r in rows] == ["holo/alpha.py", "holo/beta.py"]

    _, missing = graph.query(conn, graph.CHECKS[-1][2])
    assert [r[0] for r in missing] == ["holo/beta.py"]   # alpha has a twin


def test_build_is_durable_across_processes(built):
    # kuzu keeps writes in the WAL until close; a fresh connection
    # reading an empty database was a real bug here
    root, _ = built
    conn = graph.connect(root)
    _, rows = graph.query(conn, "MATCH (m:Module) RETURN count(m)")
    assert rows[0][0] == 4


def test_connect_without_a_build_says_what_to_run(tmp_path):
    with pytest.raises(RuntimeError, match="graph build"):
        graph.connect(str(tmp_path))


def test_every_canned_check_is_valid_cypher(built):
    root, _ = built
    conn = graph.connect(root)
    for _name, description, cypher in graph.CHECKS:
        graph.query(conn, cypher)          # raises on a bad query
        assert description and not description.endswith(".")


def test_the_database_is_not_committed():
    # a 26 MB rebuildable artifact has no business in git history
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".gitignore")) as f:
        assert "quality/codegraph" in f.read()
