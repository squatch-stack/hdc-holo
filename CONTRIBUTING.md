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
.venv/bin/python -m pytest tests/ -q     # 260+ tests, a few seconds
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

### Check what landed before you claim

The worktree discipline below prevents two sessions editing one file.
It does not prevent two sessions doing one job — a distinct failure
that has cost this repo real hours. Before opening a lane:
`git fetch github`, then `git log github/main -5 --oneline -- <area>`
and `gh pr list --state all --limit 10` for work in flight or recently
merged. Read what exists before writing a replacement, and announce a
lane when you open it rather than when you land it: a claim that
arrives with the PR is a notification, not coordination. Identify the
session you are addressing by its announcement, never by inferring
authorship from commit timing — several sessions share one committer
identity.

### Every lane gets a worktree

Several sessions share one checkout, so the checkout itself is common
ground: its files, index, and HEAD belong to no one session. **Do all
work in your own `git worktree`, and address it by path rather than
`cd`-ing into it** — the shell is shared too, and a session that
changes directory drags the others' relative paths with it.

```bash
git fetch github
git worktree add /tmp/wt-<lane> -b <lane>/<topic> github/main

git -C /tmp/wt-<lane> status                     # git takes -C
env -C /tmp/wt-<lane> .venv/bin/python -m pytest tests -q
env -C /tmp/wt-<lane> .venv/bin/holo-facts   check --strict
env -C /tmp/wt-<lane> .venv/bin/holo-quality check

git -C /tmp/wt-<lane> commit -m "..."            # commit freely, here
git -C /tmp/wt-<lane> push github HEAD:refs/heads/<lane>/<topic>
gh pr create -R squatch-stack/hdc-holo --base main --head <lane>/<topic>

# after the merge, close the lane
git worktree remove /tmp/wt-<lane> && git worktree prune
git push github --delete <lane>/<topic>
```

Rebase inside the worktree as well (`git -C <wt> rebase github/main`),
where a conflict cannot disturb anyone else's files.

Use `env -C` for anything that imports `holo`, not `PYTHONPATH`: the
shared `.venv` installs the package through an editable *meta-path
finder*, which Python consults before `sys.path`, so a run started
from the shared checkout imports the shared checkout's code however
`PYTHONPATH` is set — your worktree's changes silently absent while
the suite still passes. (`--root` flags aim the gates' analysis at a
tree; that is not the same as importing from it.) `env -C` sets the
working directory of one child process and never moves your shell.

In the shared checkout: **no destructive git commands** — no
`reset --hard`, no `checkout --` over edits you did not write, no
stashing someone else's work. Check `git status` before syncing and
notice *whose* changes are present; sync with
`git merge --ff-only github/main`, and when it refuses, fix the real
cause (uncommitted edits, or untracked files that main now tracks)
rather than forcing past it. A `reset --hard` here has already
destroyed another session's uncommitted work once.

- Announce file claims before starting multi-file work; hold clear of
  claimed paths until the owner's completion note lands in SDK.md.
- **Check headroom before a heavy encode, and sweep in ONE process.**
  `holo.budget.require_headroom(gb)` is the mechanical form of this rule
  — it reads actual free memory, lists every job over 4 GB by argv, and
  refuses with the pids named (`--force-memory` on the examples overrides
  it); the heavy examples call it on entry and report peak RSS on the way
  out. The rule used to be "check `ps` yourself", and twice it was not
  checked: two concurrent real-scene runs OOM-killed each other, then a
  lambda sweep launched as parallel processes did it again beside a 15 GB
  splat trainer. A sweep shares the band Gram AND its eigendecomposition,
  so one process per setting pays N times for both — it is slower as well
  as fatter.
- **Heavy runs leave a record.** `holo.runlog.record(label, need_gb=)`
  wraps the headroom check and writes two lines to a gitignored
  `out/runs/<date>.jsonl`: one when the run starts, one when it ends. A
  start with no end is a run that was KILLED — SIGKILL runs no `atexit`
  handler, so the evidence has to be on disk before the process dies.
  `python -m holo.runlog` lists them and flags the killed ones with
  whatever was holding the machine at the time; `--killed` shows only
  those. The start record carries the commit, whether the tree was
  dirty, the backend and the contending jobs, because a number whose
  code and machine are unknown is not reproducible and a wall-clock
  without them is not comparable. **The record goes to the SHARED
  checkout's `out/runs/`, not to your worktree** — it used to be
  package-relative, which meant `git worktree remove` deleted the
  telemetry along with the lane, and every record written before
  2026-08-28 was lost that way without anyone noticing, because a
  missing file looks exactly like a run nobody launched. `HDC_RUN_DIR`
  overrides the location.
- **Memory is not the only thing you share with a GPU trainer.**
  `holo/accel.py` picks the MLX/Metal backend whenever mlx imports, so a
  heavy encode runs on the same GPU as any splat training on the box.
  When another process faults the GPU, Metal's recovery discards *your*
  command buffer too and the run dies with
  `kIOGPUCommandBufferCallbackErrorInnocentVictim` — nothing to do with
  your code, and the headroom check cannot see it coming. If a trainer
  is running and you only need a number, force `HDC_BACKEND=numpy`;
  otherwise wait for it.
- Replicated bundle blobs MUST go through `pack_bundle`/`unpack_bundle`
  (wire v1) — readers refuse raw complex64 bytes.
- **Never `pip install -e .` from a clone or worktree.** The shared
  `.venv` has ONE editable mapping for `holo`; installing from a
  temporary tree silently repoints it, so every other session imports
  your in-flight code. Repair with `.venv/bin/pip install -e .
  --no-deps` from the real repo.
- **`PYTHONPATH` cannot aim an import at a worktree** — measured, and
  it is the reason the rule above is `env -C`. The editable install
  works through a meta-path *finder*, which Python consults BEFORE
  anything on `sys.path`, so neither `PYTHONPATH` nor pytest's rootdir
  insertion outranks it:

  ```
  PYTHONPATH=<wt> .venv/bin/python -m pytest <wt>/tests  -> shared checkout
  env -C <wt>     .venv/bin/python -m pytest tests       -> the worktree
  ```

  A whole session's local verification can pass while testing code the
  branch does not contain. Probe it when in doubt: a throwaway test
  that raises `holo.__file__` says which tree actually loaded.

## Commits

Present-tense summary line stating the capability or finding, body
explaining the why and the measured numbers. Evidence figures are
committed under `results/` (real-scene strand) or `out/` (demos).

## Quality gates

`holo-quality check` enforces the project-structure rules and a lint
ratchet: existing violations are frozen in `quality/baseline.json` and
CI fails only on NEW ones, so nothing has to be cleaned up before you
can commit. Complexity is capped at 10, drivers belong in `examples/`,
and nothing in `holo/` may import an example. Install with
`pip install -e '.[quality]'`; editor setup (ruff server +
basedpyright, both reading `pyproject.toml`) and the full rule
rationale are in [docs/quality.md](docs/quality.md). Paying debt down
(`ruff check --fix` then `holo-quality baseline`) is welcome and
separate from feature work.

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

Apache-2.0 ([LICENSE.md](LICENSE.md)). Contributions are accepted under
the same terms — opening a PR licenses your work that way, and there is
no CLA. Releases 0.2.0 and 0.2.1 shipped under FSL-1.1-Apache-2.0 and
stay that way; everything from 0.3 is Apache-2.0.
<!-- claims: allow project.license@0.2.1 -->
