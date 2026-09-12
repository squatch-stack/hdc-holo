"""Deterministic checks of storage arithmetic and strict count provenance."""

import json

import pytest

from bench.storage_position import main, markdown, read_sweep, synthetic_sweep


def test_table_arithmetic_on_a_fixture(tmp_path, capsys):
    fixture = [{"scene": "fixture", "splats_encoded": 1000, "cells": 10,
                "cells_per_band": {"xfine": 4, "fine": 3, "mid": 2, "coarse": 1}}]
    source = tmp_path / "sweep.json"
    output = tmp_path / "rows.json"
    source.write_text(json.dumps(fixture))
    main(["--sweep", str(source), "--json", str(output)])
    row, = json.loads(output.read_text())
    assert row["splats"] == 1000
    assert row["cells"] == 10
    assert row["spz_bytes"] == 22000
    assert row["complex64_bytes"] == 655360
    assert row["hg8_bytes"] == row["knee_bytes"] == 163840
    assert row["knee_ratio"] == pytest.approx(7.4472727273, rel=1e-10)
    assert row["complex64_ratio"] == pytest.approx(29.7890909091, rel=1e-10)
    assert "Median ratio at the knee: 7.45x SPZ." in capsys.readouterr().out


@pytest.mark.parametrize("key", ["splats_encoded", "cells", "cells_per_band",
                                 "cells_per_band.mid"])
def test_refuses_when_counts_are_missing(tmp_path, key):
    fixture = synthetic_sweep()
    if "." in key:
        parent, child = key.split(".")
        del fixture[0][parent][child]
    else:
        del fixture[0][key]
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="missing key " + key.split(".")[-1]):
        read_sweep(path)


def test_markdown_has_one_row_per_capture(tmp_path, capsys):
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(synthetic_sweep()))
    rows = read_sweep(source)
    table = markdown(rows)
    assert len([line for line in table.splitlines() if line.startswith("|")]) == 5
    for row in rows:
        assert table.count("| " + row["capture"] + " |") == 1
    main(["--synthetic"])
    assert capsys.readouterr().out.strip() == table


@pytest.mark.parametrize("change", ["mismatch", "negative", "zero", "duplicate"])
def test_refuses_inconsistent_counts(tmp_path, change):
    fixture = synthetic_sweep()
    if change == "mismatch":
        fixture[0]["cells"] += 1
    elif change == "negative":
        fixture[0]["cells_per_band"]["mid"] = -1
    elif change == "zero":
        fixture[0]["splats_encoded"] = 0
    else:
        fixture.append(fixture[0])
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):
        read_sweep(source)
