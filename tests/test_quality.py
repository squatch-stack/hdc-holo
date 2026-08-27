"""Structure rules and the lint ratchet (holo/quality/).

The ratchet's comparison logic is tested on synthetic counts rather
than by shelling out to ruff: the interesting behavior is "what counts
as a regression", and pinning that to the real tree would make the
test drift every time someone writes a line of code.
"""

import os

from holo.quality import check_structure, compare
from holo.quality.ratchet import load_baseline, save_baseline
from holo.quality.structure import NO_TEST_REQUIRED, ROOT_PY_ALLOWLIST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ratchet_flags_increases_not_existing_debt():
    baseline = {"a.py::E501": 3, "b.py::C901": 1}
    same = {"a.py::E501": 3, "b.py::C901": 1}
    regressions, improvements = compare(same, baseline)
    assert regressions == [] and improvements == []

    worse = {"a.py::E501": 4, "b.py::C901": 1}
    regressions, _ = compare(worse, baseline)
    assert regressions == [("a.py::E501", 3, 4)]


def test_ratchet_flags_a_brand_new_pair():
    regressions, _ = compare({"new.py::B023": 1}, {"a.py::E501": 3})
    assert regressions == [("new.py::B023", 0, 1)]


def test_ratchet_reports_fixes_as_improvements_not_failures():
    regressions, improvements = compare({"a.py::E501": 1},
                                        {"a.py::E501": 3})
    assert regressions == []
    assert improvements == [("a.py::E501", 3, 1)]


def test_baseline_round_trips(tmp_path):
    counts = {"z.py::F401": 2, "a.py::E501": 1}
    save_baseline(str(tmp_path), counts)
    assert load_baseline(str(tmp_path)) == counts
    # keys land sorted so the file diffs cleanly when debt changes
    text = (tmp_path / "quality" / "baseline.json").read_text()
    assert text.index("a.py::E501") < text.index("z.py::F401")


def test_missing_baseline_reads_as_none(tmp_path):
    assert load_baseline(str(tmp_path)) is None


def test_root_clutter_rule_catches_a_stray_driver(tmp_path):
    (tmp_path / "run_something.py").write_text("x = 1\n")
    (tmp_path / "hdc_splat.py").write_text("x = 1\n")
    codes = [(lvl, code, path) for lvl, code, path, _ in
             check_structure(str(tmp_path))]
    assert ("FAIL", "root-clutter", "run_something.py") in codes
    assert not any(p == "hdc_splat.py" for _, _, p in codes)


def test_driver_leaf_rule_catches_a_library_importing_an_example(tmp_path):
    holo_dir = tmp_path / "holo"
    holo_dir.mkdir()
    (holo_dir / "leaky.py").write_text("from examples.run_demos import x\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_leaky.py").write_text("")
    findings = check_structure(str(tmp_path))
    assert any(code == "driver-leaf" and lvl == "FAIL"
               for lvl, code, _, _ in findings)


def test_shim_unreachable_rule_catches_a_root_import(tmp_path):
    # the bug this rule exists for: a driver moved into examples/ kept
    # importing a repo-root shim and crashed on every documented
    # invocation, silently, for weeks
    (tmp_path / "hdc_splat.py").write_text("x = 1\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "run_thing.py").write_text(
        "from hdc_splat import x\n")
    (tmp_path / "examples" / "fine.py").write_text(
        "from holo.spectral import spectral_bundle\n")
    findings = check_structure(str(tmp_path))
    offenders = [path for lvl, code, path, _ in findings
                 if code == "shim-unreachable" and lvl == "FAIL"]
    assert offenders == ["examples/run_thing.py"]


def test_this_repo_satisfies_its_own_structure_rules():
    # the gate applied to the tree it ships in
    fails = [f for f in check_structure(ROOT) if f[0] == "FAIL"]
    assert fails == [], fails


def test_allowlists_stay_small_enough_to_read():
    # a rule nobody can hold in their head stops being enforced
    assert len(ROOT_PY_ALLOWLIST) <= 3
    assert "hdc_splat.py" in ROOT_PY_ALLOWLIST
    assert "backend.py" in NO_TEST_REQUIRED   # facades need no twin
