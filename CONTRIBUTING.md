# Contributing

This repo is developed by several concurrent sessions — human and
agent — editing one working tree. The conventions below are what keep
that from being chaos; most of them exist because their absence
already bit us once.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'        # + '.[gpu]' on Apple silicon,
                                         # + '.[crdt]' for Loro sync
.venv/bin/python -m pytest tests/ -q     # 90+ tests, a few seconds
```

NumPy is pinned `<2.0`: the Accelerate-backed 2.0 wheels on macOS
corrupt float32 GEMV with heap-dependent NaNs. Don't lift the pin
without running the suite several times in a row.

## Where things go

- Implementation lives in `holo/` — one file per concept under the
  charter facades (`core/encode/structures/scene/query/render/fit/
  sync/storage/backend`). `hdc/` and the root `hdc_splat.py` are
  compatibility shims: never edit them.
- The SDK charter is [SDK.md](SDK.md) — the proven-technique inventory,
  the failure-mode record, and the 0.2 running log. It is amended in
  place, append-preferred. **A technique is "proven" when it has (a) a
  quantitative comparison against ground truth or theory, (b) a
  deterministic test, and (c) a documented failure mode.** Negative
  results go in the log with numbers; they are findings, not failures.
- Docs are one page per technique in [docs/](docs/README.md): math,
  API, measured budget, failure modes, evidence figures embedded
  inline. Mermaid diagrams use the GitHub-safe subset (flowchart,
  sequence, state, class, er, pie, xychart-beta, packet-beta) with no
  custom themes — GitHub handles dark mode.
- Demos register in `holo.cli.DEMOS`; example drivers live in
  `examples/` (never the repo root) and wrap the package's public API
  only. Root holds project metadata, config, and the `hdc_splat.py`
  shim — nothing else; `holo-quality structure` enforces it.

## Tests

Read [tests/TESTING.md](tests/TESTING.md) first. The short version:
one test file per `holo` module (never append to another module's
file); `importorskip` only at the top of a dedicated file; every test
seeds its own RNG; statistical assertions sit 3-4 sigma inside the
`sqrt(N R / 2d)` crosstalk budget with the margin derived in-line.
Run the suite on BOTH backends before committing:

```bash
.venv/bin/python -m pytest tests/ -q
HDC_BACKEND=numpy .venv/bin/python -m pytest tests/ -q
```

## CI economics (private repo)

macOS runners bill at 10x. The macOS CI job is therefore gated to
release tags and manual dispatch; Linux (NumPy-fallback proof) and
gitleaks run on every push. **Batch pushes** — accumulate local
commits and push once per work session, not per commit.

## Concurrency between sessions

- Announce file claims before starting multi-file work; hold clear of
  claimed paths until the owner's completion note lands in SDK.md.
- Check `ps` for running >4GB pipeline jobs before launching heavy
  encodes (two concurrent real-scene runs have OOM-killed each other).
- Replicated bundle blobs MUST go through `pack_bundle`/`unpack_bundle`
  (wire v1) — readers refuse raw complex64 bytes.

## Commits

Present-tense summary line stating the capability or finding, body
explaining the why and the measured numbers. Evidence figures are
committed under `results/` (real-scene strand) or `out/` (demos).

## Claims

Every measured number stated in prose (README, docs/, docstrings —
mermaid labels included) must be a registered claim in
[claims/registry.jsonl](claims/registry.jsonl) or carry a
`<!-- claims: ignore -->` pragma. `holo-facts check` verifies the
prose against the registry and blocks CI when a claim goes stale; opt
into the local warn hook once per clone with
`git config core.hooksPath .githooks`. To change a value, supersede it
(old line re-id'd `base.id@<version>`, `status: "superseded"`) — never
delete it. Authoring guide: [claims/README.md](claims/README.md);
design: [docs/facts.md](docs/facts.md). Commit checklist addition:
`holo-facts check` clean (or its warns understood).

## License

FSL-1.1-Apache-2.0 ([LICENSE.md](LICENSE.md)): free for everything
except competing use, converting to Apache-2.0 two years after each
release. Contributions are accepted under the same terms.
