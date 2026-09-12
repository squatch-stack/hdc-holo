"""Append claim generations while repairing union-merged current entries.

Newest means latest as_of.date, with registry order breaking ties. If no
current entry survives, explicit values can revive the newest historical
entry; auto requires a surviving registered derivation. Existing historical
lines are never rewritten. Suffixes advance beyond existing alphabetic
suffixes for the same version (legacy ad-hoc suffixes remain untouched).
"""

import copy
import json
import re
from datetime import date
from pathlib import Path

from .check import DERIVATIONS
from .registry import Claim, validate

_DROP = ("check", "cites", "evidence", "source", "units")


def _read(path):
    lines = path.read_bytes().splitlines(keepends=True)
    entries = []
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith(b"#"):
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            raise ValueError("registry line %d needs an object with an id"
                             % (index + 1))
        entries.append((index, obj))
    return lines, entries


def _age(entry):
    index, obj = entry
    return obj.get("as_of", {}).get("date", ""), index


def _suffix_number(suffix):
    number = 0
    for char in suffix:
        number = number * 26 + ord(char) - ord("a") + 1
    return number


def _allocate(prefix, ids):
    numbers = [_suffix_number(cid[len(prefix):]) for cid in ids
               if cid.startswith(prefix) and re.fullmatch("[a-z]+", cid[len(prefix):])]
    number = max(numbers, default=0) + 1
    suffix = ""
    while number:
        number, digit = divmod(number - 1, 26)
        suffix = chr(ord("a") + digit) + suffix
    cid = prefix + suffix
    ids.add(cid)
    return cid


def _value(template, value, auto, root):
    if not auto:
        return value
    fn = DERIVATIONS.get((template.get("check") or {}).get("fn"))
    if fn is None:
        raise ValueError("%s has no registered derivation" % template["id"])
    return fn(str(root))


def _validate(entries):
    fields = Claim.__dataclass_fields__
    claims = [Claim(**{key: val for key, val in obj.items() if key in fields})
              for _, obj in entries]
    errors = validate(claims)
    if errors:
        raise ValueError("invalid resulting registry: " + "; ".join(errors))


def supersede(root, claim_id, value=None, auto=False, note=None, dry_run=False):
    """Return a chain summary; dry_run computes the same edit without writing."""
    path = Path(root) / "claims" / "registry.jsonl"
    lines, entries = _read(path)
    family = [(i, obj) for i, obj in entries
              if obj["id"] == claim_id or obj["id"].startswith(claim_id + "@")]
    if not family:
        raise ValueError("unknown id: %s" % claim_id)
    current = [(i, obj) for i, obj in family
               if obj["id"] == claim_id and obj.get("status", "current") == "current"]
    template = copy.deepcopy(max(current or family, key=_age)[1])
    value = _value(template, value, auto, root)
    if len(current) == 1 and template.get("value") == value:
        return "unchanged: %s = %s" % (claim_id, json.dumps(value))
    today = date.today().isoformat()
    ids = {obj["id"] for _, obj in entries}
    previous = template.get("supersedes") if current else template["id"]
    retired = []
    for index, obj in sorted(current, key=_age):
        version = obj.get("as_of", {}).get("version")
        if not version:
            raise ValueError("%s needs as_of.version" % claim_id)
        obj["id"] = _allocate("%s@%s-" % (claim_id, version), ids)
        obj.update(status="superseded", superseded_by=claim_id,
                   notes=note if note is not None else
                   "Superseded by holo-facts supersede on " + today)
        if previous:
            obj["supersedes"] = previous
        for key in _DROP:
            obj.pop(key, None)
        previous = obj["id"]
        retired.append(previous)
        ending = b"\r\n" if lines[index].endswith(b"\r\n") else b"\n"
        lines[index] = json.dumps(obj, ensure_ascii=False).encode("utf-8") + ending
    template.update(id=claim_id, status="current", value=value, supersedes=previous)
    template.pop("superseded_by", None)
    template["as_of"] = {**template.get("as_of", {}), "date": today}
    entries.append((len(lines), template))
    _validate(entries)
    payload = b"".join(lines)
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    payload += json.dumps(template, ensure_ascii=False).encode("utf-8") + b"\n"
    if not dry_run:
        path.write_bytes(payload)
    chain = " -> ".join([*retired, claim_id])
    prefix = "dry-run: " if dry_run else ""
    return "%s%s; value = %s" % (prefix, chain, json.dumps(value))
