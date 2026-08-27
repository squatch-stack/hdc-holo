# The claims registry

Every measured number this repo states in prose is a **registered
claim**: a typed value with provenance, a supersession chain, and the
list of files that cite it. `holo-facts check` verifies the prose
against the registry — and the registry against code/tree ground truth
— warning at pre-commit and blocking in CI. The full design lives in
[docs/facts.md](../docs/facts.md).

## Files

- `registry.jsonl` — one JSON object per claim per line (`#` comments
  allowed). Superseded entries are **never deleted**: they are the
  historical allowlist that keeps "Historical note" prose legal.
- `config.json` — surface globs, historical markers, the dated-record
  zone (`historical_after_heading`: SDK.md's running log), gitignored
  evidence prefixes, fuzzy threshold, KB path.
- `index/` — built fuzzy-index artifacts (gitignored; rebuilt by
  `holo-facts index`).

## Authoring a claim

Start from `holo-facts new` (prints a template line). The bar mirrors
the SDK's "proven" definition: a claim should carry (a) where the
number came from (`source`: SDK.md log anchor and/or generator
command), (b) evidence (figure/test path), and (c) enough `patterns`
to find its restatements in normalized prose.

Key semantics:

- `kind`: `count` (integer, exact) · `floor` ("N+" citations pass when
  N ≤ current; lag >10% warns) · `measurement` (tolerance / accepted
  spellings) · `identifier` (exact string) · `text` (pattern presence).
- `status`: `current` | `superseded` | `retracted`. To update a value:
  re-id the old line `base.id@<old-version>` with `status:
  "superseded"`, add the new line under the base id, link them via
  `supersedes`/`superseded_by`.
- `cites`: files that MUST state the current value (drift or silent
  deletion there fails CI).
- `patterns`: regexes over *normalized* text (markdown/mermaid markup
  stripped, wrapped lines joined — see `holo/facts/normalize.py`);
  first capture group is the value.
- `check.fn`: optional derivation pinning the claim to ground truth
  (`count_tests`, `bands_len`, `license_id`, …) — a mismatch means the
  *registry* is stale.

## Historical contexts (where old numbers stay legal)

1. SDK.md's running log is a **dated-record zone** — entries were
   correct at their date and are never rewritten.
2. CHANGELOG sections are version-scoped: a superseded value under a
   heading ≤ its `as_of.version` passes.
3. Prose containing a historical marker ("Historical note", "were
   correct at", "retraction", …) passes.
4. Explicit pragma: `<!-- claims: allow tests.count@0.1.0 -->` (and
   `<!-- claims: ignore -->` to silence unregistered-number warns).
