"""The stale-claim checker: exact matching over normalized surfaces.

Tiers (docs/facts.md carries the full contract):

FAIL (exit 1 under --strict) —
  superseded-value   a superseded/retracted value outside a historical
                     context (CHANGELOG version scoping, marker
                     proximity, or an explicit allow pragma)
  cite-drift         a cite file shows a different value than the claim
  cite-missing       a cite file no longer states the claim at all
  cite-overclaim     a floor-form citation exceeds the derived value
  figure-missing     a referenced evidence figure does not exist
  derived-mismatch   the REGISTRY is stale against code/tree ground
                     truth ("update the claim, then its cites")
  version-skew       holo.__version__ vs CHANGELOG / CITATION.cff

WARN (never fails) —
  floor-lag            floor citation lags current by >10%
  unregistered-number  high-signal number attributable to no claim
  orphan-figure        results/ or out/ file cited nowhere
  unverifiable-evidence  claim evidence is gitignored (absent in CI)

The exact tier owns the gate; fuzzy recall (phase 2) is WARN-only —
trigram cosine cannot distinguish a corrected restatement from a stale
one.
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field

from .normalize import normalize_file, figure_refs, canon
from .registry import load_registry, validate, base_id

__all__ = ["Finding", "CheckResult", "run", "load_config", "DERIVATIONS"]


@dataclass
class Finding:
    level: str      # FAIL | WARN
    code: str
    claim_id: str
    file: str
    line: int
    message: str

    def render(self):
        loc = "%s:%s" % (self.file, self.line) if self.file else "-"
        cid = self.claim_id or "-"
        return "%-4s %-22s %-28s %s" % (self.level, self.code, cid,
                                        "%s %s" % (loc, self.message))


@dataclass
class CheckResult:
    findings: list = field(default_factory=list)

    @property
    def fails(self):
        return [f for f in self.findings if f.level == "FAIL"]

    @property
    def warns(self):
        return [f for f in self.findings if f.level == "WARN"]

    def to_json(self):
        return json.dumps({
            "findings": [vars(f) for f in self.findings],
            "counts": {"fail": len(self.fails), "warn": len(self.warns)},
        }, indent=2, default=list)


# ---------------------------------------------------------------- derivations

def _count_tests(root):
    """Tracked test files only: the claim is about the COMMITTED suite,
    so an untracked WIP test file in a shared working tree does not move
    the number until its lane lands it (and bumps the claim)."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", root, "ls-files",
                              "tests/test_*.py"],
                             capture_output=True, text=True, check=True)
        files = [os.path.join(root, rel) for rel in out.stdout.split()]
    except Exception:
        files = glob.glob(os.path.join(root, "tests", "test_*.py"))
    n = 0
    for path in files:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                n += len(re.findall(r"^def test_", f.read(), re.M))
    return n


def _bands_len(root):
    from holo import capture
    return len(capture.BANDS)


def _capture_dim(root):
    from holo import capture
    return capture.DIM


def _render_dim(root):
    from holo import capture
    return capture.DIM_R


def _version(root):
    import holo
    return holo.__version__


def _license_id(root):
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
        m = re.search(r'license\s*=\s*\{\s*text\s*=\s*"([^"]+)"', f.read())
    return m.group(1) if m else None


def _storage_version(root):
    from holo import phase
    return phase.STORAGE_VERSION


def _wire_version(root):
    from holo import crdt
    return crdt.WIRE_VERSION


DERIVATIONS = {
    "count_tests": _count_tests,
    "bands_len": _bands_len,
    "capture_dim": _capture_dim,
    "render_dim": _render_dim,
    "version": _version,
    "license_id": _license_id,
    "storage_version": _storage_version,
    "wire_version": _wire_version,
}


# ------------------------------------------------------------------- helpers

def load_config(root):
    with open(os.path.join(root, "claims", "config.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _surface_files(root, config):
    seen, out = set(), []
    for pat in config["surfaces"]:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            rel = os.path.relpath(path, root)
            if rel not in seen and os.path.isfile(path):
                seen.add(rel)
                out.append(rel)
    return out


def _changelog_sections(root):
    """[(version tuple or None for Unreleased, first line, last line)]."""
    path = os.path.join(root, "CHANGELOG.md")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    heads = []
    for i, line in enumerate(lines):
        m = re.match(r"##\s+(?:\[)?(\d+\.\d+\.\d+|Unreleased)", line)
        if m:
            v = m.group(1)
            heads.append((None if v == "Unreleased" else _semver(v), i + 1))
    out = []
    for j, (v, start) in enumerate(heads):
        end = heads[j + 1][1] - 1 if j + 1 < len(heads) else len(lines)
        out.append((v, start, end))
    return out


def _semver(s):
    try:
        return tuple(int(p) for p in s.split("."))
    except ValueError:
        return None


def _is_historical(par, claim, config, changelog_sections, hist_after=None):
    """May a superseded/retracted value legally appear in this paragraph?

    Dated log records are first-class: config's historical_after_heading
    marks a file's tail (e.g. SDK.md's running log) as a dated record
    zone where superseded numbers were correct at their date and stay."""
    if hist_after:
        cut = hist_after.get(par.file)
        if cut is not None and par.line_start >= cut:
            return True
    for pragma in par.pragmas:
        if pragma.startswith("allow") and claim.id in pragma:
            return True
    if os.path.basename(par.file) in claim.allow_historical_in \
            or par.file in claim.allow_historical_in:
        # blanket per-file allowance still requires version scoping when
        # the file is the CHANGELOG (below) — for any other file it's a
        # direct allow.
        if os.path.basename(par.file) != "CHANGELOG.md":
            return True
    text = par.text.lower()
    for marker in config.get("historical_markers", []):
        if marker.lower() in text:
            return True
    if os.path.basename(par.file) == "CHANGELOG.md":
        as_of = _semver(claim.as_of.get("version", "")) if claim.as_of else None
        for version, start, end in changelog_sections:
            if start <= par.line_start <= end:
                if version is not None and as_of is not None \
                        and version <= as_of:
                    return True
    return False


def _match_values(claim, text):
    """[(canon value, floor?)] captured by the claim's patterns in text.
    A pattern with no group is a presence match (value = claim's own)."""
    if claim.context_any:
        low = text.lower()
        if not any(c.lower() in low for c in claim.context_any):
            return []
    out = []
    for pat in claim.patterns:
        for m in re.finditer(pat, text, re.I):
            if m.groups() and m.group(1) is not None:
                end = m.end(1)
                floor = end < len(text) and text[end] == "+"
                out.append((canon(m.group(1)), floor))
            else:
                out.append((canon(claim.value), False))
    return out


def _value_ok(claim, got):
    return got in {canon(v) for v in claim.accepted_values()}


# ---------------------------------------------------------------------- run

def run(root, strict=False, config=None, claims=None):
    config = config or load_config(root)
    claims = claims if claims is not None else \
        load_registry(os.path.join(root, "claims", "registry.jsonl"))
    result = CheckResult()

    for err in validate(claims):
        result.findings.append(Finding("FAIL", "registry-invalid", "",
                                       "claims/registry.jsonl", 0, err))
    if result.fails:
        return result

    current = [c for c in claims if c.status == "current"]
    stale = [c for c in claims if c.status in ("superseded", "retracted")]
    by_id = {c.id: c for c in claims}
    sections = _changelog_sections(root)

    hist_after = {}
    for rel, heading in config.get("historical_after_heading", {}).items():
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if line.strip().startswith(heading):
                        hist_after[rel] = i + 1
                        break

    surfaces = _surface_files(root, config)
    paras = {}   # rel path -> [Paragraph]
    raw = {}     # rel path -> raw text

    def get_paras(rel):
        if rel not in paras:
            paras[rel] = normalize_file(os.path.join(root, rel))
            with open(os.path.join(root, rel), encoding="utf-8",
                      errors="replace") as f:
                raw[rel] = f.read()
        return paras[rel]

    for rel in surfaces:
        get_paras(rel)

    # (b) citation checks for current claims
    for claim in current:
        if not claim.patterns:
            continue
        cur_canon = {canon(v) for v in claim.accepted_values()}
        stale_values = {canon(v): s for s in stale
                        if base_id(s.id) == claim.id
                        for v in s.accepted_values()}
        for cite in claim.cites:
            hits = []
            for par in get_paras(cite):
                for got, floor in _match_values(claim, par.text):
                    hits.append((par, got, floor))
            if not hits:
                result.findings.append(Finding(
                    "FAIL", "cite-missing", claim.id, cite, 0,
                    "no longer states this claim (current: %r)" % claim.value))
                continue
            ok = False
            for par, got, floor in hits:
                if got in cur_canon:
                    ok = True
                    continue
                if floor and claim.kind in ("count", "floor"):
                    try:
                        cited, curv = int(got), int(canon(claim.value))
                    except ValueError:
                        cited = curv = None
                    if cited is not None:
                        if cited > curv:
                            result.findings.append(Finding(
                                "FAIL", "cite-overclaim", claim.id, cite,
                                par.line_start,
                                "floor %s+ exceeds current %s" % (cited, curv)))
                        else:
                            ok = True
                            if curv and (curv - cited) / curv > 0.10:
                                result.findings.append(Finding(
                                    "WARN", "floor-lag", claim.id, cite,
                                    par.line_start,
                                    "floor %s+ lags current %s by >10%%"
                                    % (cited, curv)))
                    continue
                if got in stale_values:
                    s = stale_values[got]
                    if _is_historical(par, s, config, sections, hist_after):
                        continue
                    result.findings.append(Finding(
                        "FAIL", "cite-drift", claim.id, cite, par.line_start,
                        "cites superseded %r (current: %r)"
                        % (got, claim.value)))
                    continue
                # a value that belongs to ANOTHER claim stated in the same
                # paragraph is not drift of this one (e.g. "encode 37x,
                # decode 106x ... render scale 65x" in one paragraph)
                if any(c2.id != claim.id
                       and got in {canon(v) for v in c2.accepted_values()}
                       and any(g == got for g, _ in
                               _match_values(c2, par.text))
                       for c2 in claims if c2.patterns):
                    continue
                result.findings.append(Finding(
                    "FAIL", "cite-drift", claim.id, cite, par.line_start,
                    "cites %r but current is %r" % (got, claim.value)))
            if not ok and not any(f.claim_id == claim.id and f.file == cite
                                  for f in result.findings):
                result.findings.append(Finding(
                    "FAIL", "cite-missing", claim.id, cite, 0,
                    "states no current value (current: %r)" % claim.value))

    # (a) superseded/retracted values anywhere outside historical context
    for claim in stale:
        if not claim.patterns:
            continue
        stale_canon = {canon(v) for v in claim.accepted_values()}
        cur = by_id.get(claim.superseded_by or base_id(claim.id))
        cur_canon = {canon(v) for v in cur.accepted_values()} if cur else set()
        for rel in surfaces:
            for par in get_paras(rel):
                for got, _ in _match_values(claim, par.text):
                    if got in cur_canon and got not in stale_canon:
                        continue
                    if got not in stale_canon:
                        continue
                    if _is_historical(par, claim, config, sections, hist_after):
                        continue
                    result.findings.append(Finding(
                        "FAIL", "superseded-value", claim.id, rel,
                        par.line_start,
                        "%s value %r appears un-annotated%s"
                        % (claim.status, got,
                           " (current: %r)" % cur.value if cur else "")))

    # derived checks: the registry against code/tree ground truth
    for claim in current:
        fn = DERIVATIONS.get(claim.check.get("fn")) if claim.check else None
        if not fn:
            continue
        try:
            derived = fn(root)
        except Exception as e:  # import guards: derivation unavailable here
            result.findings.append(Finding(
                "WARN", "derivation-error", claim.id, "", 0,
                "%s: %s" % (claim.check.get("fn"), e)))
            continue
        if canon(derived) not in {canon(v) for v in claim.accepted_values()}:
            result.findings.append(Finding(
                "FAIL", "derived-mismatch", claim.id, "", 0,
                "registry says %r but tree derives %r — update the claim, "
                "then its cites" % (claim.value, derived)))

    # (d1) structural, hard
    referenced = set()
    for rel in surfaces:
        if not rel.endswith(".md"):
            continue
        for ref in figure_refs(raw[rel]):
            referenced.add(ref)
            if not os.path.exists(os.path.join(root, ref)):
                line = next((i + 1 for i, l in
                             enumerate(raw[rel].split("\n")) if ref in l), 0)
                result.findings.append(Finding(
                    "FAIL", "figure-missing", "", rel, line,
                    "references %s which does not exist" % ref))

    try:
        import holo
        pkg_version = holo.__version__
    except Exception:
        pkg_version = None
    if pkg_version:
        rel_heads = [v for v, _, _ in sections if v is not None]
        if rel_heads and rel_heads[0] != _semver(pkg_version):
            result.findings.append(Finding(
                "FAIL", "version-skew", "project.version", "CHANGELOG.md", 0,
                "newest release heading %s != holo.__version__ %s"
                % (".".join(map(str, rel_heads[0])), pkg_version)))
        cff = os.path.join(root, "CITATION.cff")
        if os.path.exists(cff):
            with open(cff, encoding="utf-8") as f:
                m = re.search(r"^version:\s*[\"']?(\d+\.\d+\.\d+)", f.read(),
                              re.M)
            if m and m.group(1) != pkg_version:
                result.findings.append(Finding(
                    "FAIL", "version-skew", "project.version", "CITATION.cff",
                    0, "cff version %s != holo.__version__ %s"
                    % (m.group(1), pkg_version)))

    # (d2) structural, soft
    for d in ("results", "out"):
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith((".png", ".gif")):
                continue
            rel = "%s/%s" % (d, name)
            if rel not in referenced:
                result.findings.append(Finding(
                    "WARN", "orphan-figure", "", rel, 0, "cited nowhere"))

    prefixes = tuple(config.get("evidence_unverifiable_prefixes", []))
    for claim in current:
        for ev in claim.evidence:
            if prefixes and ev.startswith(prefixes):
                result.findings.append(Finding(
                    "WARN", "unverifiable-evidence", claim.id, ev, 0,
                    "evidence is gitignored — unverifiable in CI"))

    # (c) unregistered high-signal numbers (WARN, capped)
    candidates = [
        (re.compile(r"\b\d+(?:\.\d+)?x\b"),
         ("faster", "speedup", "encode", "decode", "slower")),
        (re.compile(r"\b\d+\s+tests\b"),
         ()),
        (re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|ms|min)\b"),
         ("pipeline", "end to end", "end-to-end", "per frame", "s/frame",
          "wall")),
    ]
    all_patterned = [c for c in claims if c.patterns]
    unregistered = []
    skip_c = set(config.get("unregistered_skip", []))
    for rel in surfaces:
        if not rel.endswith(".md") or rel in skip_c:
            continue
        for par in get_paras(rel):
            if par.kind == "verbatim" or any(p.startswith("ignore")
                                             for p in par.pragmas):
                continue
            low = par.text.lower()
            hit = False
            for pat, ctx in candidates:
                if pat.search(par.text) and \
                        (not ctx or any(c in low for c in ctx)):
                    hit = True
                    break
            if not hit:
                continue
            if any(_match_values(c, par.text) for c in all_patterned):
                continue
            unregistered.append(Finding(
                "WARN", "unregistered-number", "", rel, par.line_start,
                "high-signal number with no registered claim: %r"
                % (par.text[:90] + ("…" if len(par.text) > 90 else ""))))
    result.findings.extend(unregistered[:20])
    if len(unregistered) > 20:
        result.findings.append(Finding(
            "WARN", "unregistered-number", "", "", 0,
            "…and %d more (run with --json for all)"
            % (len(unregistered) - 20)))

    return result
