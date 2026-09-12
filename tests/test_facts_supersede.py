"""Claim supersession and append-only merge integration on isolated fixtures."""

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from holo.facts.cli import main


def _claim(value=1, **updates):
    obj = {"id": "tests.count", "status": "current", "kind": "count",
           "value": value, "as_of": {"date": "2026-01-01", "version": "0.3.0"},
           "check": {"fn": "count_tests"}, "cites": [], "evidence": [],
           "source": {}, "units": "tests"}
    obj.update(updates)
    return obj


def _fixture(tmp_path, entries, count=3):
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims/config.json").write_text('{"surfaces": []}')
    registry = tmp_path / "claims/registry.jsonl"
    registry.write_text("# fixture\n" + "".join(json.dumps(c) + "\n" for c in entries))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_fixture.py").write_text("\n".join(
        "def test_%d():\n    pass" % i for i in range(count)))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "tests")
    return registry


def _git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Squatch Stack",
         "-c", "user.email=", "-c", "commit.gpgsign=false", *args],
        text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _run(root, *args):
    return main(["supersede", "tests.count", "--root", str(root), *args])


def _entries(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def _strict(root):
    assert main(["check", "--strict", "--root", str(root)]) == 0


def test_one_current_suffix_rollover_and_fields(tmp_path):
    old = _claim(0, id="tests.count@0.3.0-z", status="superseded",
                 superseded_by="tests.count", check={})
    registry = _fixture(tmp_path, [old, _claim(supersedes=old["id"])])
    assert _run(tmp_path, "--value", "3", "--note", "fixture update") == 0
    history, retired, current = _entries(registry)
    assert history == old
    assert retired["id"] == "tests.count@0.3.0-aa"
    assert retired["notes"] == "fixture update"
    assert retired["superseded_by"] == "tests.count"
    assert retired["supersedes"] == old["id"]
    assert not {"check", "cites", "evidence", "source", "units"} & retired.keys()
    assert current["supersedes"] == retired["id"]
    assert current["as_of"]["date"] == date.today().isoformat()
    assert current["value"] == 3
    _strict(tmp_path)


def test_union_currents_form_one_chain_using_newest_metadata(tmp_path):
    newer = _claim(3, as_of={"date": "2026-02-01", "version": "0.3.0"},
                   lane="newer")
    registry = _fixture(tmp_path, [newer, _claim(2, lane="older")])
    assert _run(tmp_path, "--auto") == 0
    newest, oldest, current = _entries(registry)
    assert oldest["id"] == "tests.count@0.3.0-a"
    assert newest["id"] == "tests.count@0.3.0-b"
    assert newest["supersedes"] == oldest["id"]
    assert current["supersedes"] == newest["id"]
    assert current["lane"] == "newer"
    assert sum(c["status"] == "current" for c in _entries(registry)) == 1
    _strict(tmp_path)


def test_auto_derives_tracked_tests(tmp_path):
    registry = _fixture(tmp_path, [_claim()], count=4)
    (tmp_path / "tests/test_untracked.py").write_text("def test_extra(): pass\n")
    assert _run(tmp_path, "--auto") == 0
    assert _entries(registry)[-1]["value"] == 4
    _strict(tmp_path)


def test_unchanged_is_byte_identical(tmp_path, capsys):
    registry = _fixture(tmp_path, [_claim(3)])
    before = registry.read_bytes()
    assert _run(tmp_path, "--auto") == 0
    assert "unchanged" in capsys.readouterr().out
    assert registry.read_bytes() == before
    _strict(tmp_path)


@pytest.mark.parametrize("unknown", [True, False])
def test_refuses_unknown_or_missing_derivation(tmp_path, capsys, unknown):
    registry = _fixture(tmp_path, [_claim(check={})])
    before = registry.read_bytes()
    cid = "unknown" if unknown else "tests.count"
    assert main(["supersede", cid, "--auto", "--root", str(tmp_path)]) == 2
    assert ("unknown id" if unknown else "no registered derivation") in (
        capsys.readouterr().err)
    assert registry.read_bytes() == before


def test_comments_and_unrelated_lines_preserved(tmp_path):
    registry = _fixture(tmp_path, [_claim()])
    prefix = b'# unusual spacing  \r\n\r\n{ "id":"other", "value": "x" }\r\n'
    registry.write_bytes(prefix + registry.read_bytes())
    assert _run(tmp_path, "--auto") == 0
    assert registry.read_bytes().startswith(prefix + b"# fixture\n")
    _strict(tmp_path)


def test_dry_run_does_not_write(tmp_path, capsys):
    registry = _fixture(tmp_path, [_claim()])
    before = registry.read_bytes()
    assert _run(tmp_path, "--auto", "--dry-run") == 0
    preview = capsys.readouterr().out
    assert "dry-run: tests.count@0.3.0-a -> tests.count; value = 3" in preview
    assert registry.read_bytes() == before
    assert _run(tmp_path, "--auto") == 0
    assert capsys.readouterr().out == preview.removeprefix("dry-run: ")
    _strict(tmp_path)


def test_no_current_revives_history_with_explicit_value(tmp_path):
    old = _claim(1, id="tests.count@0.3.0-z", status="superseded",
                 superseded_by="tests.count", check={})
    registry = _fixture(tmp_path, [old])
    assert _run(tmp_path, "--value", "3") == 0
    history, current = _entries(registry)
    assert history == old
    assert current["supersedes"] == old["id"]
    assert "superseded_by" not in current
    _strict(tmp_path)


def test_invalid_result_is_refused_without_write(tmp_path):
    registry = _fixture(tmp_path, [_claim(supersedes="missing")])
    before = registry.read_bytes()
    assert _run(tmp_path, "--auto") == 2
    assert registry.read_bytes() == before


def test_union_merge(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "base")
    attributes = Path(__file__).resolve().parents[1] / ".gitattributes"
    (tmp_path / ".gitattributes").write_bytes(attributes.read_bytes())
    (tmp_path / "docs").mkdir()
    (tmp_path / "SDK.md").write_text("# Findings\n")
    (tmp_path / "docs/figures.md").write_text("| Figure | Command |\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "fixture base")
    for branch in ("left", "right"):
        _git(tmp_path, "checkout", "-qb", branch, "base")
        with (tmp_path / "SDK.md").open("a") as stream:
            stream.write("- %s finding\n" % branch)
        with (tmp_path / "docs/figures.md").open("a") as stream:
            stream.write("| %s.png | fixture |\n" % branch)
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-qm", branch)
    transcript = _git(tmp_path, "merge", "--no-edit", "left")
    print(transcript)
    assert _git(tmp_path, "ls-files", "-u") == ""
    for branch in ("left", "right"):
        assert "- %s finding" % branch in (tmp_path / "SDK.md").read_text()
        assert "| %s.png" % branch in (tmp_path / "docs/figures.md").read_text()
