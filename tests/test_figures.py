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
    drivers = set(re.findall(r"python (examples/[\w./-]+\.py)", _record()))
    assert drivers, "no drivers recorded — the table lost its commands"
    missing = sorted(d for d in drivers
                     if not os.path.isfile(os.path.join(ROOT, d)))
    assert not missing, "recorded drivers that do not exist: " + str(missing)


def test_recorded_demo_targets_exist():
    """`hdc-demos <name>` entries must name registered demos."""
    from holo.cli import DEMOS
    names = set(re.findall(r"`hdc-demos ([\w-]+)`", _record()))
    assert names, "no hdc-demos targets recorded"
    unknown = sorted(n for n in names if n not in DEMOS)
    assert not unknown, "recorded demos that are not registered: " + str(unknown)


def test_the_one_unregenerable_figure_is_declared():
    """failure_herringbone.png cannot be regenerated without
    reintroducing the bug it documents. That has to be stated, or the
    next person will try to rebuild it and quietly fail."""
    text = _record()
    assert "failure_herringbone.png" in text
    assert "not regenerable" in text
