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

from .normalize import canon, figure_refs, normalize_file
from .registry import base_id, load_registry, validate

__all__ = ["DERIVATIONS", "CheckResult", "Finding", "load_config", "run"]


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


_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def _front_matter_findings(root, config, claims):
    """Knowledge-base front-matter validation, config-gated: files
    matching `front_matter_surfaces` globs must carry parseable flat
    front-matter whose `claims:` ids exist in the registry, whose
    `arxiv:` ids are well-formed, and whose `swept:` date parses —
    the dated-arXiv-sweep convention made structural."""
    from .normalize import front_matter
    globs_ = config.get("front_matter_surfaces", [])
    if not globs_:
        return []
    ids = {c.id for c in claims}
    findings = []
    for pat in globs_:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            rel = os.path.relpath(path, root)
            fm = front_matter(path)
            if not fm:
                findings.append(Finding(
                    "WARN", "front-matter-missing", "", rel, 1,
                    "surface expects flat front-matter (--- block)"))
                continue
            for cid in fm.get("claims", []) or []:
                if cid not in ids:
                    findings.append(Finding(
                        "FAIL", "front-matter-claim", cid, rel, 1,
                        "front-matter names unregistered claim id"))
            for aid in fm.get("arxiv", []) or []:
                if not _ARXIV_ID.match(aid):
                    findings.append(Finding(
                        "FAIL", "front-matter-arxiv", "", rel, 1,
                        "malformed arXiv id %r" % aid))
            swept = fm.get("swept")
            if swept and not re.match(r"^\d{4}-\d{2}-\d{2}$",
                                      str(swept)):
                findings.append(Finding(
                    "FAIL", "front-matter-swept", "", rel, 1,
                    "swept: %r is not YYYY-MM-DD" % swept))
    return findings


# ---------------------------------------------------------------------- run

class _Scan:
    """Everything the individual checks share, resolved once.

    `run` used to hold all of this in one 259-line scope with a closure
    over a paragraph cache; the checks below take a _Scan instead, which
    is what let them become separate functions at all.
    """

    def __init__(self, root, config, claims):
        self.root = root
        self.config = config
        self.claims = claims
        self.current = [c for c in claims if c.status == "current"]
        self.stale = [c for c in claims
                      if c.status in ("superseded", "retracted")]
        self.by_id = {c.id: c for c in claims}
        self.sections = _changelog_sections(root)
        self.hist_after = _historical_zones(root, config)
        self.surfaces = _surface_files(root, config)
        self.patterned = [c for c in claims if c.patterns]
        self._paras = {}
        self._raw = {}

    def paras(self, rel):
        """Normalized paragraphs for a surface, parsed at most once."""
        if rel not in self._paras:
            ps = normalize_file(os.path.join(self.root, rel))
            for p in ps:
                p.file = rel      # zone/allowlist keys are repo-relative
            self._paras[rel] = ps
            with open(os.path.join(self.root, rel), encoding="utf-8",
                      errors="replace") as f:
                self._raw[rel] = f.read()
        return self._paras[rel]

    def raw(self, rel):
        self.paras(rel)
        return self._raw[rel]

    def is_historical(self, par, claim):
        return _is_historical(par, claim, self.config, self.sections,
                              self.hist_after)


def _historical_zones(root, config):
    """{file: first line of its dated-record zone} — below that line,
    superseded values are history rather than drift."""
    zones = {}
    for rel, heading in config.get("historical_after_heading", {}).items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if line.strip().startswith(heading):
                    zones[rel] = i + 1
                    break
    return zones


def _floor_finding(claim, cite, par, got):
    """(ok, finding) for an 'N+' style citation against a count claim."""
    try:
        cited, current = int(got), int(canon(claim.value))
    except ValueError:
        return False, None
    if cited > current:
        return False, Finding(
            "FAIL", "cite-overclaim", claim.id, cite, par.line_start,
            "floor %s+ exceeds current %s" % (cited, current))
    if current and (current - cited) / current > 0.10:
        return True, Finding(
            "WARN", "floor-lag", claim.id, cite, par.line_start,
            "floor %s+ lags current %s by >10%%" % (cited, current))
    return True, None


def _belongs_to_another_claim(scan, claim, par, got):
    """A value another claim states in the SAME paragraph is not drift
    of this one — "encode 37x, decode 106x" is one sentence, two claims."""
    for other in scan.patterned:
        if other.id == claim.id:
            continue
        if got not in {canon(v) for v in other.accepted_values()}:
            continue
        if any(g == got for g, _ in _match_values(other, par.text)):
            return True
    return False


def _classify_hit(scan, claim, cite, par, got, floor, cur_canon,
                  stale_values):
    """One matched value in one cite file: (states_current, findings)."""
    if got in cur_canon:
        return True, []
    if floor and claim.kind in ("count", "floor"):
        ok, finding = _floor_finding(claim, cite, par, got)
        return ok, [finding] if finding else []
    if got in stale_values:
        superseded = stale_values[got]
        if scan.is_historical(par, superseded):
            return False, []
        return False, [Finding(
            "FAIL", "cite-drift", claim.id, cite, par.line_start,
            "cites superseded %r (current: %r)" % (got, claim.value))]
    if _belongs_to_another_claim(scan, claim, par, got):
        return False, []
    return False, [Finding(
        "FAIL", "cite-drift", claim.id, cite, par.line_start,
        "cites %r but current is %r" % (got, claim.value))]


def _check_one_citation(scan, claim, cite, cur_canon, stale_values):
    hits = [(par, got, floor)
            for par in scan.paras(cite)
            for got, floor in _match_values(claim, par.text)]
    if not hits:
        return [Finding("FAIL", "cite-missing", claim.id, cite, 0,
                        "no longer states this claim (current: %r)"
                        % claim.value)]
    findings, states_current = [], False
    for par, got, floor in hits:
        ok, new = _classify_hit(scan, claim, cite, par, got, floor,
                                cur_canon, stale_values)
        states_current = states_current or ok
        findings.extend(new)
    if not states_current and not findings:
        findings.append(Finding(
            "FAIL", "cite-missing", claim.id, cite, 0,
            "states no current value (current: %r)" % claim.value))
    return findings


def _check_citations(scan):
    """(b) every cite file must still state its claim's current value."""
    findings = []
    for claim in scan.current:
        if not claim.patterns:
            continue
        cur_canon = {canon(v) for v in claim.accepted_values()}
        stale_values = {canon(v): s for s in scan.stale
                        if base_id(s.id) == claim.id
                        for v in s.accepted_values()}
        for cite in claim.cites:
            findings.extend(_check_one_citation(scan, claim, cite,
                                                cur_canon, stale_values))
    return findings


def _superseded_in_paragraph(scan, claim, cur, rel, par, stale_canon):
    findings = []
    for got, _ in _match_values(claim, par.text):
        if got not in stale_canon or scan.is_historical(par, claim):
            continue
        findings.append(Finding(
            "FAIL", "superseded-value", claim.id, rel, par.line_start,
            "%s value %r appears un-annotated%s"
            % (claim.status, got,
               " (current: %r)" % cur.value if cur else "")))
    return findings


def _check_superseded(scan):
    """(a) a retired value anywhere outside a historical context."""
    findings = []
    for claim in scan.stale:
        if not claim.patterns:
            continue
        stale_canon = {canon(v) for v in claim.accepted_values()}
        cur = scan.by_id.get(claim.superseded_by or base_id(claim.id))
        for rel in scan.surfaces:
            for par in scan.paras(rel):
                findings.extend(_superseded_in_paragraph(
                    scan, claim, cur, rel, par, stale_canon))
    return findings


def _check_derivations(scan):
    """The REGISTRY against code/tree ground truth."""
    findings = []
    for claim in scan.current:
        fn = DERIVATIONS.get(claim.check.get("fn")) if claim.check else None
        if not fn:
            continue
        try:
            derived = fn(scan.root)
        except Exception as e:   # a derivation may need an absent extra
            findings.append(Finding(
                "WARN", "derivation-error", claim.id, "", 0,
                "%s: %s" % (claim.check.get("fn"), e)))
            continue
        if canon(derived) not in {canon(v) for v in claim.accepted_values()}:
            findings.append(Finding(
                "FAIL", "derived-mismatch", claim.id, "", 0,
                "registry says %r but tree derives %r — update the claim, "
                "then its cites" % (claim.value, derived)))
    return findings


def _check_figures(scan):
    """(findings, referenced) — every cited figure must exist."""
    findings, referenced = [], set()
    for rel in scan.surfaces:
        if not rel.endswith(".md"):
            continue
        raw = scan.raw(rel)
        for ref in figure_refs(raw):
            referenced.add(ref)
            if os.path.exists(os.path.join(scan.root, ref)):
                continue
            line = next((i + 1 for i, text in enumerate(raw.split("\n"))
                         if ref in text), 0)
            findings.append(Finding(
                "FAIL", "figure-missing", "", rel, line,
                "references %s which does not exist" % ref))
    return findings, referenced


def _check_versions(scan):
    """holo.__version__ against the CHANGELOG heading and CITATION.cff."""
    try:
        import holo
        version = holo.__version__
    except Exception:
        return []
    findings = []
    heads = [v for v, _, _ in scan.sections if v is not None]
    if heads and heads[0] != _semver(version):
        findings.append(Finding(
            "FAIL", "version-skew", "project.version", "CHANGELOG.md", 0,
            "newest release heading %s != holo.__version__ %s"
            % (".".join(map(str, heads[0])), version)))
    cff = os.path.join(scan.root, "CITATION.cff")
    if os.path.exists(cff):
        with open(cff, encoding="utf-8") as f:
            m = re.search(r"^version:\s*[\"']?(\d+\.\d+\.\d+)",
                          f.read(), re.M)
        if m and m.group(1) != version:
            findings.append(Finding(
                "FAIL", "version-skew", "project.version", "CITATION.cff", 0,
                "cff version %s != holo.__version__ %s"
                % (m.group(1), version)))
    return findings


def _check_orphan_figures(scan, referenced):
    findings = []
    for directory in ("results", "out"):
        base = os.path.join(scan.root, directory)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            rel = "%s/%s" % (directory, name)
            if name.endswith((".png", ".gif")) and rel not in referenced:
                findings.append(Finding("WARN", "orphan-figure", "", rel, 0,
                                        "cited nowhere"))
    return findings


def _check_evidence(scan):
    prefixes = tuple(scan.config.get("evidence_unverifiable_prefixes", []))
    if not prefixes:
        return []
    return [Finding("WARN", "unverifiable-evidence", claim.id, ev, 0,
                    "evidence is gitignored — unverifiable in CI")
            for claim in scan.current for ev in claim.evidence
            if ev.startswith(prefixes)]


_CANDIDATE_NUMBERS = [
    (re.compile(r"\b\d+(?:\.\d+)?x\b"),
     ("faster", "speedup", "encode", "decode", "slower")),
    (re.compile(r"\b\d+\s+tests\b"), ()),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|ms|min)\b"),
     ("pipeline", "end to end", "end-to-end", "per frame", "s/frame",
      "wall")),
]


def _is_candidate_number(par):
    low = par.text.lower()
    for pattern, context in _CANDIDATE_NUMBERS:
        if pattern.search(par.text) and \
                (not context or any(c in low for c in context)):
            return True
    return False


def _unregistered_in_surface(scan, rel):
    findings = []
    for par in scan.paras(rel):
        if par.kind == "verbatim" or \
                any(p.startswith("ignore") for p in par.pragmas):
            continue
        if not _is_candidate_number(par):
            continue
        if any(_match_values(c, par.text) for c in scan.patterned):
            continue
        text = par.text[:90] + ("…" if len(par.text) > 90 else "")
        findings.append(Finding(
            "WARN", "unregistered-number", "", rel, par.line_start,
            "high-signal number with no registered claim: %r" % text))
    return findings


def _check_unregistered(scan, cap=20):
    """(c) high-signal numbers attributable to no claim — candidates for
    registration, capped so one messy file cannot bury the other tiers."""
    skip = set(scan.config.get("unregistered_skip", []))
    found = []
    for rel in scan.surfaces:
        if rel.endswith(".md") and rel not in skip:
            found.extend(_unregistered_in_surface(scan, rel))
    if len(found) <= cap:
        return found
    return [*found[:cap], Finding(
        "WARN", "unregistered-number", "", "", 0,
        "…and %d more (run with --json for all)" % (len(found) - cap))]


def run(root, config=None, claims=None):
    """Every tier, in reporting order. Each check is independent and
    takes the shared _Scan; this function only sequences them."""
    config = config or load_config(root)
    claims = claims if claims is not None else \
        load_registry(os.path.join(root, "claims", "registry.jsonl"))
    result = CheckResult()

    for err in validate(claims):
        result.findings.append(Finding("FAIL", "registry-invalid", "",
                                       "claims/registry.jsonl", 0, err))
    if result.fails:
        return result           # a broken registry makes every tier lie

    scan = _Scan(root, config, claims)
    figure_findings, referenced = _check_figures(scan)

    result.findings.extend(_check_citations(scan))
    result.findings.extend(_check_superseded(scan))
    result.findings.extend(_check_derivations(scan))
    result.findings.extend(figure_findings)
    result.findings.extend(_check_versions(scan))
    result.findings.extend(_check_orphan_figures(scan, referenced))
    result.findings.extend(_front_matter_findings(root, config, claims))
    result.findings.extend(_check_evidence(scan))
    result.findings.extend(_check_unregistered(scan))
    return result
