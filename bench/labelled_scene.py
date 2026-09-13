"""Annotated-scene seam; no dataset downloads or fixture accuracy claims.

API: load_boxes(path) -> rows; load_instance_points(ply, segs, aggregation)
-> (points, ids, classes); objects_from_instances(...) -> rows;
to_objects(rows, label_map=None) -> (semantic_memory.OBJECT array, report).
load_scene(manifest, label_map=None) joins these paths. All positions keep
source world units; no normalization or pooling. Extents are AABB half-widths.
Instance positions are centroids; their extents enclose points about that
centroid (not about the bounding-box midpoint). Both stores receive one array.

Verified formats: ARKitScenes data[].label / segments.obbAligned with centroid,
axesLengths (full sizes), normalizedAxes (row axes); ScanNet segIndices and
segGroups with objectId, segments, label. Sources read before implementation:
https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/box_utils.py
https://github.com/ScanNet/ScanNet#data-formats

Our JSON manifest uses format=boxes|instances|files. Boxes require annotations
and geometry paths; instances require geometry, segmentation, annotations;
files require objects=[{label, path}]. Paths are relative to the manifest.
File manifests are our interchange shape, not a verified dataset schema;
exact dataset field names still need confirming. Existing PLY/SPZ/splat loaders
are used; mesh faces are irrelevant to this centre-query experiment. Geometry
and annotations must already share the loader's coordinate frame and units.
An optional crop={lo:[x,y,z], extent:[x,y,z]} uses crop_scene_file, retains
objects whose centres lie in the crop, and never recomputes partial objects.

Label maps are JSON objects: source label -> replacement string or null to
exclude; unmapped labels survive unchanged. Vocabulary IDs are sorted and
shared query metadata, excluded equally from both numeric-ID stores. Their
UTF-8 JSON byte size is reported separately; add it to both for named queries.
One store per scene, no pooling. Below 128 surviving objects the run measures
the easy regime only at d=512; reaching 128 does not establish a knee. The
baseline's formal 50% knee differs from the brief's 90% operating region.
No real annotations have been measured here; all three gates remain untested
on real labels. Fixture tests establish the seam, not semantic accuracy.

Baseline module was absent in this worktree and recovered from local commit
ab59d7e. No results, figures, baseline tests or other-lane files were restored.
Frozen proof uses golden output from that untouched source: CLI stdout with a
zero clock and a sentinel for the unchanged continuous fork, exact table bytes
and hologram header, then float payloads and the actual fork at rtol=atol=2e-6.
Raw float byte hashes varied across allocations in this NumPy build; these
explicit tolerances avoid claiming cross-platform bitwise float reproducibility.
The synthetic algorithms, defaults, query contract and serialization stay intact.
Reproduce fixtures: HDC_BACKEND=numpy .venv/bin/python -m pytest
 tests/test_labelled_scene.py -q
Once data lands: HDC_BACKEND=numpy .venv/bin/python -m bench.semantic_memory
 --scene "$SCENES/scene.json" --label-map "$SCENES/labels.json" --dims 512
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bench import semantic_memory
from holo.capture import bbox_of, crop_scene_file, load_scene_file


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read annotation {path}: {exc}") from exc


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError(f"{name} exceeds float32 range")
    return result


def _row(label, position, extent):
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a nonempty string")
    position = _array(position, (3,), "position")
    extent = _array(extent, (3,), "extent")
    if np.any(extent < 0):
        raise ValueError("extent must be nonnegative")
    return {"label": label, "position": position, "extent": extent}


def load_boxes(path):
    """Read ARKitScenes OBBs, enclosing rotations in an AABB."""
    try:
        data = _json(path)
        if data.get("skipped", False):
            raise ValueError("annotation is marked skipped")
        rows = []
        for box in data["data"]:
            obb = box["segments"]["obbAligned"]
            size = _array(obb["axesLengths"], (3,), "axesLengths")
            if np.any(size <= 0):
                raise ValueError("axesLengths must be positive")
            axes = _array(obb.get("normalizedAxes", np.eye(3).ravel()),
                          (9,), "normalizedAxes").reshape(3, 3)
            if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6, rtol=0):
                raise ValueError("normalizedAxes must be orthonormal")
            rows.append(_row(box["label"], obb["centroid"],
                             np.abs(axes.T) @ (size / 2)))
        return rows
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed box annotation: {exc}") from exc


def load_instance_points(geometry, segmentation, annotations):
    """ScanNet segments -> per-point instance IDs; -1 is unannotated."""
    try:
        points = load_scene_file(geometry)[0]
        segments = np.asarray(_json(segmentation)["segIndices"])
        if segments.shape != (len(points),) or segments.dtype.kind not in "iu":
            raise ValueError("segIndices must be integer IDs aligned to vertices")
        ids = np.full(len(points), -1, dtype=np.int64)
        classes = {}
        for group in _json(annotations)["segGroups"]:
            instance = group["objectId"]
            if type(instance) is not int or instance < 0 or instance in classes:
                raise ValueError("objectId must be unique nonnegative integer")
            members = group["segments"]
            if not members or any(type(v) is not int for v in members):
                raise ValueError("segments must be nonempty integer IDs")
            if not set(members) <= set(segments.tolist()):
                raise ValueError("annotation references missing segment")
            mask = np.isin(segments, members)
            if np.any(ids[mask] != -1):
                raise ValueError("segment belongs to multiple instances")
            ids[mask] = instance
            classes[instance] = group["label"]
        return points, ids, classes
    except (KeyError, TypeError, OSError, AssertionError) as exc:
        raise ValueError(f"malformed instance annotation or geometry: {exc}") from exc


def objects_from_instances(points, instance_ids, classes):
    """Derive once, before either store is constructed."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    ids = np.asarray(instance_ids)
    if ids.shape != (len(points),) or ids.dtype.kind not in "iu":
        raise ValueError("instance IDs must be integers aligned to points")
    if np.any(ids < -1) or set(ids[ids >= 0].tolist()) != set(classes):
        raise ValueError("instance classes must match all non-background IDs")
    rows = []
    for instance in sorted(classes):
        selected = points[ids == instance]
        center = selected.mean(axis=0)
        extent = np.maximum(center - selected.min(0), selected.max(0) - center)
        rows.append(_row(classes[instance], center, extent))
    return rows


def to_objects(rows, label_map=None):
    """Return the existing structured array and counts; no store duplication."""
    rows = [_row(r["label"], r["position"], r["extent"]) for r in rows]
    mapping = {} if label_map is None else label_map
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or (v is not None and
        (not isinstance(v, str) or not v.strip())) for k, v in mapping.items()
    ):
        raise ValueError("label map must map strings to nonempty strings or null")
    kept = [(r, mapping.get(r["label"], r["label"])) for r in rows]
    kept = [(r, label) for r, label in kept if label is not None]
    vocabulary = sorted({label for _, label in kept})
    if not kept or len(vocabulary) > 65535:
        raise ValueError("scene must retain objects and 1..65535 classes")
    indices = {label: i for i, label in enumerate(vocabulary)}
    objects = np.empty(len(kept), dtype=semantic_memory.OBJECT)
    for i, (row, label) in enumerate(kept):
        objects[i] = (row["position"], row["extent"], indices[label])
    report = {"objects_before": len(rows), "objects_survived": len(kept),
              "classes_before": len({r["label"] for r in rows}),
              "classes_survived": len(vocabulary), "vocabulary": vocabulary,
              "vocabulary_bytes": len(json.dumps(vocabulary).encode("utf-8")),
              "class_counts": {label: sum(v == label for _, v in kept)
                               for label in vocabulary},
              "pooling": False, "easy_regime_only": len(kept) < 128}
    return objects, report


def _file_rows(entries, root):
    rows = []
    for entry in entries:
        lo, extent = bbox_of(root / entry["path"])
        rows.append(_row(entry["label"], lo + extent / 2, extent / 2))
    return rows


def _crop(rows, geometry, crop):
    lo = _array(crop["lo"], (3,), "crop lo")
    extent = _array(crop["extent"], (3,), "crop extent")
    with TemporaryDirectory() as directory:
        crop_scene_file(geometry, lo, extent, Path(directory) / "crop.ply")
    return [r for r in rows if np.all(r["position"] >= lo)
            and np.all(r["position"] <= lo + extent)]


def load_scene(manifest, label_map=None):
    """Load one explicit manifest; refuse missing/malformed inputs with reasons."""
    root = Path(manifest).parent
    try:
        spec = _json(manifest)
        kind = spec["format"]
        if kind == "boxes":
            load_scene_file(root / spec["geometry"])
            rows = load_boxes(root / spec["annotations"])
        elif kind == "instances":
            rows = objects_from_instances(*load_instance_points(
                root / spec["geometry"], root / spec["segmentation"],
                root / spec["annotations"]))
        elif kind == "files":
            rows = _file_rows(spec["objects"], root)
        else:
            raise ValueError(f"unknown annotation format: {kind}")
        before_crop = len(rows)
        if "crop" in spec:
            rows = _crop(rows, root / spec["geometry"], spec["crop"])
        objects, report = to_objects(rows, _json(label_map) if label_map else None)
        report["objects_before_crop"] = before_crop
        return objects, report
    except (KeyError, TypeError, OSError, AssertionError, AttributeError) as exc:
        raise ValueError(f"malformed scene manifest or geometry: {exc}") from exc


def run_scene(path, label_map, dimensions):
    objects, report = load_scene(path, label_map)
    print(json.dumps({"scene": report}), flush=True)
    rows = []
    for d in dimensions:
        if d <= 0:
            raise ValueError("dimensions must be positive")
        row = semantic_memory.measure(objects, d, classes=report["classes_survived"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"crossovers": semantic_memory.summarize(rows)["crossovers"],
                      "gates": "real-label gates not established by one scene; "
                      "four-balanced-class knee and continuous-proxy fork "
                      "require their preregistered design"}), flush=True)
