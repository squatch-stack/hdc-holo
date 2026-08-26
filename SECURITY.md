# Security

- **Reporting**: this is currently a private research repo; report
  anything sensitive to the repository owner directly (GitHub
  `@squatch-stack`).
- **Secret scanning**: gitleaks runs over the full history on every
  push and pull request (`.github/workflows/ci.yml`, config in
  `.gitleaks.toml`). Test fixtures author synthetic binary formats and
  print hex digests; if a rule flags one of these high-entropy strings,
  extend the allowlist in `.gitleaks.toml` rather than rewriting the
  fixture.
- **Supply chain**: GitHub Actions are pinned to full commit SHAs, CI
  runs with `permissions: contents: read`, and runtime dependencies are
  minimal (`numpy`, optional `loro`/`mlx`/`matplotlib`).
- **Data**: `data/` (scene captures) is gitignored and never pushed;
  the wire format validates a per-doc universe record before decoding
  any peer's bytes, and untagged blobs are refused.
