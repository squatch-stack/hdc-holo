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


def test_chunker_splits_bullet_runs():
    # consecutive bullets have no blank lines between them; merging
    # them made one 216-line "paragraph" out of SDK.md's log before
    # this was a rule (masking the SDK dated-record zone besides)
    import os
    import tempfile
    from holo.facts.chunk import chunk_file
    md = ("# log\n\n" +
          "- first entry about the encode kernel " + "alpha " * 40 + "\n" +
          "- second entry about storage codecs " + "beta " * 40 + "\n" +
          "- third entry about the render path " + "gamma " * 40 + "\n")
    with tempfile.NamedTemporaryFile("w", suffix=".md",
                                     delete=False) as f:
        f.write(md)
        path = f.name
    try:
        chunks = chunk_file(path, "f.md")
    finally:
        os.unlink(path)
    assert len(chunks) == 3
    assert all(c.heading == "log" for c in chunks)
    assert "alpha" in chunks[0].text and "alpha" not in chunks[1].text


def test_hg8_roundtrip_preserves_profile_ranking():
    # HG-8 is the codec measured faithful on wide-dynamic-range
    # bundles; a profile row must come back ranking-equivalent
    import numpy as np
    from holo import FHRR
    from holo.dispatch import FastNGramProfiler
    from holo.storage import pack_polar, unpack
    space = FHRR(dim=2048, seed=0)
    prof = FastNGramProfiler(space, n=3)
    texts = ["the encode kernel runs far faster on the metal backend",
             "tombstone sets make deletion idempotent across peers",
             "ridge fitting treats the bundle as a weight vector"]
    rows = [prof.unit_profile(t) for t in texts]
    back = [unpack(pack_polar(v, bits=8)).astype(np.complex64)
            for v in rows]
    back = [b / np.linalg.norm(b) for b in back]
    for v, b in zip(rows, back):
        assert np.real(np.vdot(v, b)) > 0.99   # HG-8 drift ~0.01
    q = prof.unit_profile("how quick is encoding on the gpu backend")
    orig_top = int(np.argmax([np.real(np.vdot(v, q)) for v in rows]))
    back_top = int(np.argmax([np.real(np.vdot(b, q)) for b in back]))
    assert orig_top == back_top == 0


def test_search_claims_ranks_the_right_record():
    # MCP tool logic, stdlib-only (the server shim is a thin binding;
    # the wire is proven in test_facts_mcp.py under the facts extra)
    from holo.facts.query import search_claims
    out = search_claims(ROOT, "encode kernel speedup on the gpu")
    ids = [r["id"] for r in out["results"]]
    assert ids and ids[0] == "accel.encode_speedup"
    assert "37x" in str(out["results"][0]["statement"])


def test_get_claim_returns_chain_derivation_and_cite_sites():
    from holo.facts.query import get_claim
    rec = get_claim(ROOT, "capture.err_redrock")
    assert [c["id"] for c in rec["chain"]] == \
        ["capture.err_redrock", "capture.err_redrock@0.2.1"]
    counted = get_claim(ROOT, "tests.count")
    # convention: the registry value derives from the committed tree,
    # so the live derivation must agree whatever the number is today
    assert counted["derivation"]["matches"] is True
    assert any(s["file"] == "CONTRIBUTING.md"
               for s in counted["cite_sites"])
    missing = get_claim(ROOT, "no.such.claim")
    assert "error" in missing


def test_search_kb_reports_unconfigured_honestly():
    import os
    from holo.facts.query import search_kb
    old = os.environ.pop("HOLO_KB_PATH", None)
    try:
        out = search_kb(ROOT, "plunge region")
    finally:
        if old is not None:
            os.environ["HOLO_KB_PATH"] = old
    # config kb_path is null until math-kb exists (phase 4)
    assert out["configured"] is False
    assert "note" in out


def test_fuzzy_retrieves_paraphrase_and_scrambled_noise_abstains():
    # the fuzzy layer's whole contract: a digit-free paraphrase of a
    # claim scores above threshold on the right chunk, while true
    # noise (character-scrambled text — word order barely moves a
    # trigram profile, so word shuffles are NOT noise) stays below
    import numpy as np
    from holo import FHRR
    from holo.dispatch import FastNGramProfiler
    space = FHRR(dim=2048, seed=0)
    prof = FastNGramProfiler(space, n=3)
    corpus = [
        "the mlx encode kernel measures thirty seven times faster than "
        "numpy on an m one max at full dimension",
        "observed remove tombstones keep concurrent deletion idempotent "
        "and add wins under merge",
        "the saguaro turntable renders thirty six frames from the cell "
        "bundles at about five seconds per frame",
    ]
    mat = np.stack([prof.unit_profile(t) for t in corpus])
    q = prof.unit_profile(
        "the encode kernel is roughly thirtyseven fold quicker than "
        "numpy on the m one max")
    scores = np.real(mat.conj() @ q)
    assert int(np.argmax(scores)) == 0
    assert scores[0] > 0.30          # calibrated signal median ~0.39
    rng = np.random.default_rng(7)
    chars = np.array(list(corpus[0].replace(" ", "")))
    rng.shuffle(chars)
    noise = np.real(mat.conj() @ prof.unit_profile("".join(chars)))
    assert noise.max() < 0.18        # calibrated noise p95 ~0.10
