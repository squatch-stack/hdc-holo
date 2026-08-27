"""docs/figures.md — every figure records how to regenerate it.

A figure is a measured number that happens to be a picture, so it gets
the same treatment as one: the record must stay complete as figures are
added, and the commands in it must name things that exist. Without a
test this page is a snapshot that rots; with one it is a contract.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "docs", "figures.md")
IMAGE_DIRS = ("results", "out")
IMAGE_EXT = (".png", ".gif", ".jpg", ".svg")


def _record():
    with open(RECORD, encoding="utf-8") as f:
        return f.read()


def _figures_on_disk():
    found = set()
    for d in IMAGE_DIRS:
        path = os.path.join(ROOT, d)
        if not os.path.isdir(path):
            continue
        for name in os.listdir(path):
            if name.lower().endswith(IMAGE_EXT):
                found.add(name)
    return found


def test_every_figure_records_its_provenance():
    """A new figure with no entry fails here — which is the point: the
    moment to record where a picture came from is when it is made."""
    text = _record()
    missing = sorted(n for n in _figures_on_disk() if n not in text)
    assert not missing, (
        "figures with no entry in docs/figures.md: " + ", ".join(missing))


def test_recorded_drivers_exist():
    """Every `python examples/<driver>.py` named in the record must be
    a file that is actually there, so the commands stay runnable across
    renames and moves."""
    mods = set(re.findall(r"python -m (examples\.[\w-]+)", _record()))
    assert mods, "no drivers recorded — the table lost its commands"
    missing = sorted(m for m in mods if not os.path.isfile(
        os.path.join(ROOT, m.replace(".", os.sep) + ".py")))
    assert not missing, "recorded drivers that do not exist: " + str(missing)


def test_recorded_demo_targets_exist():
    """`hdc-demos <name>` entries must name registered demos."""
    from holo.cli import DEMOS
    names = set(re.findall(r"python -m holo\.cli ([\w-]+)", _record()))
    assert names, "no demo targets recorded"
    unknown = sorted(n for n in names if n not in DEMOS)
    assert not unknown, "recorded demos that are not registered: " + str(unknown)


def test_commands_are_module_form_not_paths():
    """From a worktree, `python examples/foo.py` and `hdc-demos foo`
    both import the SHARED checkout — the editable install's meta-path
    finder outranks sys.path, and a script puts its own directory on
    sys.path[0] rather than the repo root. Since all work happens in
    worktrees, a path-form command here would regenerate figures from
    main's code while looking correct. Measured, not assumed."""
    # only the table rows carry commands; the prose deliberately
    # quotes the wrong forms as counter-examples
    rows = [ln for ln in _record().splitlines() if ln.startswith("|")]
    bad = [ln.strip() for ln in rows
           if re.search(r"python examples/[\w-]+\.py", ln)
           or re.search(r"hdc-demos [\w-]+", ln)]
    assert not bad, ("figure commands must be module form "
                     "(python -m examples.x / python -m holo.cli x): "
                     + str(bad))


def test_the_one_unregenerable_figure_is_declared():
    """failure_herringbone.png cannot be regenerated without
    reintroducing the bug it documents. That has to be stated, or the
    next person will try to rebuild it and quietly fail."""
    text = _record()
    assert "failure_herringbone.png" in text
    assert "not regenerable" in text
