# Working in this repo

Several sessions — human and agent — share this one checkout. The rules
below exist because every one of them was learned by breaking something.
Full contributor detail lives in [CONTRIBUTING.md](CONTRIBUTING.md).

## Work in a worktree. Never in the shared checkout.

The shared checkout is common ground: its files, its index, and its
HEAD belong to no one session. Do all work in your own git worktree,
and **address it by path — never `cd` into it**, so the shell you and
every other session share keeps pointing at the same place.

```bash
# 1. claim a lane: branch + worktree off the current remote main
git fetch github
git worktree add /tmp/wt-<lane> -b <lane>/<topic> github/main

# 2. work there by PATH — git takes -C, the gates take --root,
#    python takes PYTHONPATH. Nothing needs a cd.
git -C /tmp/wt-<lane> status
PYTHONPATH=/tmp/wt-<lane> .venv/bin/python -m pytest /tmp/wt-<lane>/tests -q
.venv/bin/holo-facts   check --root /tmp/wt-<lane> --strict
.venv/bin/holo-quality check --root /tmp/wt-<lane>

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
