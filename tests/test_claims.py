"""Claims registry + stale-claim checker (holo/facts/).

The last test here is the suite-level stale-claim gate: pytest alone
fails when a registered claim drifts anywhere in the tree, before the
CI step even runs. Cross-surface consistency precedent:
tests/test_holo_facade.py.
"""

import os

from holo.facts import load_registry, run, validate
from holo.facts.check import DERIVATIONS, _is_historical, load_config
from holo.facts.normalize import Paragraph, canon, normalize_markdown
from holo.facts.registry import base_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "claims", "registry.jsonl")


def test_registry_parses_and_validates():
    claims = load_registry(REGISTRY)
    assert len(claims) >= 20
    assert validate(claims) == []
    ids = [c.id for c in claims]
    assert len(ids) == len(set(ids))


def test_supersession_links_resolve_to_current_generations():
    claims = load_registry(REGISTRY)
    by_id = {c.id: c for c in claims}
    for c in claims:
        if c.status == "superseded":
            cur = by_id[c.superseded_by or base_id(c.id)]
            assert cur.status == "current"


def test_every_derived_check_matches_the_tree():
    # the registry pinned to code/tree ground truth (the
    # test_holo_facade.py species of cross-surface consistency)
    claims = load_registry(REGISTRY)
    for c in claims:
        fn = DERIVATIONS.get(c.check.get("fn")) if c.check else None
        if fn is None:
            continue
        derived = fn(ROOT)
        accepted = {canon(v) for v in c.accepted_values()}
        assert canon(derived) in accepted, (c.id, derived, c.value)


def test_normalize_rejoins_hard_wrapped_numbers():
    paras = normalize_markdown("the pipeline ran 13\nmin on NumPy and 24\ns after.", "f.md")
    assert len(paras) == 1
    assert "13 min" in paras[0].text
    assert "24 s" in paras[0].text


def test_normalize_extracts_mermaid_labels():
    # the historical "3 scale bands" bug lived in a mermaid node label
    md = 'x\n\n```mermaid\nflowchart LR\n  A["4 scale bands<br/>by max axis"] --> B["cells"]\n```\n'
    paras = normalize_markdown(md, "f.md")
    mermaid = [p for p in paras if p.kind == "mermaid"]
    assert len(mermaid) == 1
    assert "4 scale bands" in mermaid[0].text
    assert "<br/>" not in mermaid[0].text


def test_normalize_splits_table_rows_and_strips_markup():
    md = "| `holo/fit.py` | **~70x** held-out |\n|---|---|\n"
    rows = [p for p in normalize_markdown(md, "f.md") if p.kind == "table"]
    assert "~70x held-out" in rows[0].text
    assert "`" not in rows[0].text and "*" not in rows[0].text


def test_historical_marker_and_pragma_allow_old_values():
    claims = load_registry(REGISTRY)
    old = next(c for c in claims if c.id == "capture.bands@0.2.0")
    config = load_config(ROOT)
    marked = Paragraph("docs/x.md", 1, 1, "prose",
                       "Historical note: three scale bands were correct "
                       "at their date.")
    plain = Paragraph("docs/x.md", 1, 1, "prose",
                      "the pipeline uses three scale bands.")
    pragma = Paragraph("docs/x.md", 1, 1, "prose",
                       "the pipeline uses three scale bands.",
                       {"allow capture.bands@0.2.0"})
    assert _is_historical(marked, old, config, [])
    assert not _is_historical(plain, old, config, [])
    assert _is_historical(pragma, old, config, [])


def test_changelog_version_scoping():
    claims = load_registry(REGISTRY)
    old = next(c for c in claims if c.id == "tests.count@0.1.0")
    config = load_config(ROOT)
    sections = [((0, 2, 0), 1, 9), ((0, 1, 0), 10, 20)]
    in_010 = Paragraph("CHANGELOG.md", 12, 12, "prose", "56 tests")
    in_020 = Paragraph("CHANGELOG.md", 5, 5, "prose", "56 tests")
    assert _is_historical(in_010, old, config, sections)
    assert not _is_historical(in_020, old, config, sections)


def test_head_has_no_stale_claims():
    # THE gate: a FAIL here means a registered claim drifted in the
    # working tree — run `holo-facts check` for the findings.
    result = run(ROOT)
    assert result.fails == [], "\n".join(f.render() for f in result.fails)
