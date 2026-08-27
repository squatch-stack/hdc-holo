"""Demo output helpers (holo/demokit.py).

The point of these tests is byte-equivalence with the hand-rolled
f-strings demokit replaced: eighteen demos print capacity tables, and
a formatting change here would rewrite all of their evidence at once.
"""

import holo.demokit as demokit
from holo.demokit import Table, banner


def test_banner_matches_the_hand_rolled_form(capsys):
    banner("HoloMap: hash map in superposition", 4096)
    out = capsys.readouterr().out
    assert out == "== HoloMap: hash map in superposition (d=4096) ==\n"


def test_banner_without_a_dimension_omits_it(capsys):
    banner("Codec rate-distortion: bytes vs task fidelity")
    assert capsys.readouterr().out == (
        "== Codec rate-distortion: bytes vs task fidelity ==\n")


def test_table_reproduces_the_original_fstring_exactly(capsys):
    # the literal hashmap demo header and row, pre-demokit
    table = Table(("pairs N", 8), ("load N/d", 9, ".2f"),
                  ("pred noise", 11, ".3f"), ("accuracy", 9, ".1%"))
    table.header()
    table.row(500, 500 / 4096, 0.247, 0.57)
    out = capsys.readouterr().out.split("\n")
    assert out[0] == (f"{'pairs N':>8} {'load N/d':>9} "
                      f"{'pred noise':>11} {'accuracy':>9}")
    assert out[1] == (f"{500:>8} {500 / 4096:>9.2f} "
                      f"{0.247:>11.3f} {0.57:>9.1%}")


def test_indent_prefixes_every_line(capsys):
    table = Table(("d", 6), ("codec", 6), indent="  ")
    table.header()
    table.row(8192, "HG")
    for line in capsys.readouterr().out.rstrip("\n").split("\n"):
        assert line.startswith("  ")


def test_row_rejects_the_wrong_column_count():
    table = Table(("a", 4), ("b", 4))
    try:
        table.row(1)
    except ValueError as e:
        assert "2 columns" in str(e)
    else:
        raise AssertionError("a short row must not print silently")


def test_unknown_keyword_is_refused():
    try:
        Table(("a", 4), indnet="  ")          # typo'd kwarg
    except TypeError as e:
        assert "indnet" in str(e)
    else:
        raise AssertionError("a typo'd kwarg must not be swallowed")


def test_module_documents_its_own_test_file():
    # this file is named in demokit's docstring; keep that honest
    assert "tests/test_demokit.py" in demokit.__doc__
