# Test suite rules

These rules exist because this repo is edited by several concurrent
sessions (human and agent). The old single `test_structures.py` made
every session append to one file — merge clobbering, a mid-file
`importorskip` that silently skipped half the suite, and order-dependent
failures. Don't recreate it.

## Layout: one test file per holo module

- `holo/foo.py` is tested by `tests/test_foo.py` and nowhere else.
- Adding a module means adding a NEW test file, never appending to an
  existing one. This is the anti-clobber rule: concurrent sessions touch
  disjoint files.
- Cross-module integration tests live in the file of the *highest-level*
  module involved (e.g. replicated scenes -> `test_crdt.py`).
- Shared fixtures live in `conftest.py` only. Test files never import
  from each other.

## Optional dependencies

- Gate at the TOP of the dedicated file, first statement after imports:
  `pytest.importorskip("loro")`. Never mid-file — module-level
  importorskip skips every test after it, silently.

## Determinism and isolation

- Every test seeds its own RNG explicitly. No test reads global state,
  writes module-level state, or depends on run order. Each test builds
  its own space/structures (the `space` fixture is per-test).
- Suite must pass file-by-file (`pytest tests/test_foo.py`) and in any
  order. If a test fails only in the full run, treat it as a real bug
  (this suite caught a macOS Accelerate GEMV heap bug exactly that way —
  do not paper over order effects with reordering).

## Statistical assertions

- Pick parameters comfortably INSIDE capacity: assert margins should sit
  >= 3-4 sigma from the crosstalk noise floor sqrt(N/(2d)) so fixed
  seeds never flake. Where tolerance is meaningful, derive it from the
  theory in-line (see the frequency-sketch test) rather than guessing a
  constant.
- Demos, not tests, push structures past their capacity cliffs.

## Style

- One behavior per test; the name states it:
  `test_<subject>_<behavior>`. A reader should predict the asserts from
  the name.
- Keep tests linear — arrange, act, assert. No branching on outcomes,
  no try/except around asserts, no helper indirection that hides the
  behavior (target cyclomatic complexity ~1; simple loops over probe
  sets are fine).
- Short comments explain WHY a bound holds (the sigma math), not what
  the code does.

## The claims gate

`tests/test_claims.py` is the cross-surface consistency file for the
claims registry (`claims/registry.jsonl` ↔ prose ↔ code ground truth,
in the mold of `test_holo_facade.py`). Its last test fails whenever a
registered claim is stale anywhere in the tree — so adding tests, or
changing a measured number in docs, can legitimately fail it: update
the registry (supersede, never delete) per
[claims/README.md](../claims/README.md).
