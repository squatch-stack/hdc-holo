# Archived reports

Findings published to the gpugate reports channel by sessions running on
the GPU host, kept here because **that channel is not durable storage**.

`20260826-b978b2` was cited as the generator for `pipeline.rtx5090_e2e`
(the 6.92 s end-to-end figure). On 2026-08-29 it began returning HTTP
404 while its same-day sibling `20260826-ed92e7` still resolved — so
the loss was a deletion of one specific report, not eviction by age,
kind, or the automated-note flood that keeps the 100-entry listing
churning. No retention rule explains it.

The text below was recovered from a local session transcript and
verified byte-identical against a copy that happened to survive in a
scratch directory. Both are outside version control and one is in a
temp directory that gets purged, so neither was a durable copy — the
figure was one `rm -rf /tmp` away from being unsupportable.

**Cite these files, not report IDs.** An off-repo ID is a promise
another system has to keep; a committed file is evidence.

| file | claim it supports |
|---|---|
| `20260826-b978b2.md` | `pipeline.rtx5090_e2e` — 6.92 s end to end |
| `20260826-ed92e7.md` | the optimisation chain it completes (to 8.25 s) |

Both were checked for hostnames, IP addresses, usernames, filesystem
paths and domains before being committed.
