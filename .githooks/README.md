# Repo git hooks

Opt in once per clone:

```
git config core.hooksPath .githooks
```

- `pre-commit` — runs `holo-facts check` in warn mode. It never blocks
  a commit; the CI step (`holo-facts check --strict`) is the gate that
  keeps stale claims off main. See [claims/README.md](../claims/README.md).
