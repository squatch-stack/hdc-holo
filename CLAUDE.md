# Working in this repo

Several sessions — human and agent — share this one checkout. The rules
below exist because every one of them was learned by breaking something.
Full contributor detail lives in [CONTRIBUTING.md](CONTRIBUTING.md).

## Before you open a lane: is this already done?

A worktree stops two sessions writing the same *file*. It says nothing
about two sessions doing the same *work*, and that is a different
failure with the same cost. Both happened here on the same day: one
session re-did a landed citations-and-figures pass without noticing it
was on main, and another opened a file claim addressed to the wrong
session because it inferred authorship from commit timing.

So before creating the branch, spend ten seconds:

```bash
git fetch github
git log github/main -5 --oneline -- <the area you are about to touch>
gh pr list -R <repo> --state all --limit 10   # open AND recently merged
```

If someone landed it, read theirs before writing yours; if it is in
flight, message that session first. Neither of those collisions cost
anything to prevent and both cost an hour to discover.

Two habits that make this work:

- **Address a session by evidence, not inference.** Commit timing does
  not identify an author when several sessions share one identity;
  ask, or check who announced the lane.
- **Announce a lane when you open it**, not when you land it. A claim
  that arrives with the PR is a notification, not coordination.

## Work in a worktree. Never in the shared checkout.

The shared checkout is common ground: its files, its index, and its
HEAD belong to no one session. Do all work in your own git worktree,
and **address it by path — never `cd` into it**, so the shell you and
every other session share keeps pointing at the same place.

```bash
# 1. claim a lane: branch + worktree off the current remote main
git fetch github
git worktree add /tmp/wt-<lane> -b <lane>/<topic> github/main

# 2. work there by PATH. git takes -C; anything that IMPORTS holo
#    needs `env -C`, which sets the working directory of that one
#    child process and never moves your shell.
git -C /tmp/wt-<lane> status
env -C /tmp/wt-<lane> .venv/bin/python -m pytest tests -q
env -C /tmp/wt-<lane> .venv/bin/holo-facts   check --strict
env -C /tmp/wt-<lane> .venv/bin/holo-quality check

# 3. commit in the worktree, as often as you like
git -C /tmp/wt-<lane> add <paths>
git -C /tmp/wt-<lane> commit -m "..."

# 4. push and open the PR when the gates are green
git -C /tmp/wt-<lane> push github HEAD:refs/heads/<lane>/<topic>
gh pr create -R squatch-stack/hdc-holo --base main --head <lane>/<topic> ...

# 5. after the merge, close the lane
git worktree remove /tmp/wt-<lane>   # --force only if you mean it
git worktree prune
git push github --delete <lane>/<topic>
```

Rebase inside the worktree too (`git -C <wt> rebase github/main`); a
conflict there cannot disturb anyone else's files.

**`PYTHONPATH=<wt>` is not enough, and fails quietly.** The shared
`.venv` installs `holo` as an editable package through a *meta-path
finder*, which Python consults before `sys.path` — so a run started
from the shared checkout imports the shared checkout's code no matter
what `PYTHONPATH` says, and your worktree's changes are simply absent
while the tests still pass. `env -C <wt>` is what actually works
(`--root` flags aim the gates' *analysis* at a tree, which is not the
same as importing from it). If a test of brand-new code passes
suspiciously, check `python -c "import holo.capture as c;
print(c.__file__)"` before believing it.

## Rules for the shared checkout

- **Never run a destructive git command in it.** No `reset --hard`, no
  `checkout --` over another session's edits, no `stash` of work you did
  not write. A `reset --hard` here has already destroyed another
  session's uncommitted work once.
- **Check `git status` before any sync** and look at *whose* changes are
  present. Modified tracked files you did not touch belong to someone
  else — ask, don't clear.
- **Sync with `git merge --ff-only github/main`**, never `reset`. If it
  refuses, something real is in the way (uncommitted edits, or untracked
  files that main now tracks); resolve that, don't force past it.
- Its HEAD may lag. Read code from your worktree, not from here, once
  your lane is open.

## Environment

- **Never `pip install -e .` from a worktree or clone.** The shared
  `.venv` holds ONE editable mapping for `holo`; installing from a
  temporary tree silently repoints it and every other session then
  imports your in-flight code. Use `PYTHONPATH=<worktree>` instead.
  Repair with `.venv/bin/pip install -e . --no-deps` from this repo.
- `data/` is gitignored and holds real captures; `results/` and `out/`
  hold committed evidence figures.

## Working agreements

- Announce a file claim before multi-file work, and say when it lands.
- Nothing unproven enters the SDK surface: a technique needs a
  quantitative comparison, a deterministic test, and a documented
  failure mode ([SDK.md](SDK.md)).
- Measured numbers stated in prose are registered claims — `holo-facts`
  gates them, so update the registry in the same commit that changes
  the number.
- The public identity is "Squatch Stack". No personal names, emails, or
  domains in tracked files, commit messages, or artifacts.
