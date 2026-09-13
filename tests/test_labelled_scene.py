"""Authored annotation fixtures only; no downloaded data or accuracy claims."""

import hashlib
import json

import numpy as np
import pytest

from bench import labelled_scene as seam
from bench import semantic_memory as experiment
from holo.capture import load_scene_file, save_ply


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def geometry(tmp_path):
    rng = np.random.default_rng(17)
    points = rng.uniform(0.1, 0.9, (12, 3))
    path = tmp_path / "points.ply"
    save_ply(path, points, np.full((12, 3), 0.01),
             np.ones((12, 4)), np.tile([1., 0., 0., 0.], (12, 1)))
    return path


def boxes(path):
    angle = np.pi / 6
    axes = [[np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    obb = {"centroid": [0.5, 0.5, 0.5], "axesLengths": [0.2, 0.4, 0.6],
           "normalizedAxes": np.ravel(axes).tolist()}
    return write_json(path, {"data": [{"label": "chair", "segments": {
        "obbAligned": obb}}]})


def test_boxes_round_trip_to_objects(tmp_path, geometry):
    annotation = boxes(tmp_path / "boxes.json")
    rows = seam.load_boxes(annotation)
    objects, report = seam.to_objects(rows)
    np.testing.assert_allclose(objects["position"], [[0.5] * 3], atol=1e-7)
    np.testing.assert_allclose(objects["extent"],
                               [[0.1 * np.sqrt(3) / 2 + 0.1,
                                 0.05 + 0.2 * np.sqrt(3) / 2, 0.3]],
                               atol=1e-7)
    assert report["objects_survived"] == 1
    manifest = write_json(tmp_path / "scene.json", {
        "format": "boxes", "geometry": geometry.name,
        "annotations": annotation.name})
    np.testing.assert_array_equal(seam.load_scene(manifest)[0], objects)


def test_instances_derive_the_same_position_and_extent_for_both_stores(
    tmp_path, geometry, monkeypatch,
):
    segs = write_json(tmp_path / "segments.json", {"segIndices": [2] * 8 + [5] * 4})
    annotation = write_json(tmp_path / "aggregation.json", {"segGroups": [
        {"objectId": 0, "segments": [2], "label": "chair"},
        {"objectId": 9, "segments": [5], "label": "table"}]})
    loaded = seam.load_instance_points(geometry, segs, annotation)
    rows = seam.objects_from_instances(*loaded)
    selected = loaded[0][:8]
    np.testing.assert_allclose(rows[0]["position"], selected.mean(0), atol=1e-7)
    np.testing.assert_allclose(rows[0]["extent"],
                               np.abs(selected - selected.mean(0)).max(0),
                               atol=1e-7)
    objects, _ = seam.to_objects(rows)
    seen = []
    for name in ("encode_table", "encode_hologram"):
        original = getattr(experiment, name)

        def capture(value, *args, original=original, **kwargs):
            seen.append(value)
            return original(value, *args, **kwargs)

        monkeypatch.setattr(experiment, name, capture)
    experiment.measure(objects, 32, classes=2)
    assert len(seen) == 2
    assert all(value is objects for value in seen)
    manifest = write_json(tmp_path / "scene.json", {
        "format": "instances", "geometry": geometry.name,
        "segmentation": segs.name, "annotations": annotation.name})
    np.testing.assert_array_equal(seam.load_scene(manifest)[0], objects)


def test_label_map_collapses_rare_classes_and_reports_the_count():
    rows = [{"label": label, "position": [i, 0, 0], "extent": [1, 1, 1]}
            for i, label in enumerate(["chair", "chair", "vase", "lamp", "void"])]
    objects, report = seam.to_objects(rows, {
        "vase": "other", "lamp": "other", "void": None})
    assert len(objects) == report["objects_survived"] == 4
    assert report["objects_before"] == 5
    assert report["classes_before"] == 4
    assert report["classes_survived"] == 2
    assert report["class_counts"] == {"chair": 2, "other": 2}
    assert report["easy_regime_only"] and not report["pooling"]
    with pytest.raises(ValueError, match="retain objects"):
        seam.to_objects(rows, dict.fromkeys(r["label"] for r in rows))


def test_synthetic_path_is_unchanged(monkeypatch, capsys):
    # Golden hashes from untouched ab59d7e; clock=0 and fork sentinel.
    # The unchanged fork has platform-sensitive float text; test it below.
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(experiment, "encoding_fork",
                        lambda capture: {"frozen_fork": capture})
    experiment.main(["--synthetic", "--objects", "8", "--dims", "32"])
    output = capsys.readouterr().out
    assert hashlib.sha256(output.encode()).hexdigest() == (
        "60ba9eebee08e0f902ffcc0dce999534746463e68c0e0d65f356c22b6442a60a"
    )
    objects = experiment.object_scene(8, np.random.default_rng(experiment.SEED))
    table = experiment.encode_table(objects)[0]
    assert hashlib.sha256(table.serialize()).hexdigest() == (
        "30161aebb8f60b54f8637770b46890b85c9d162f15dd3f58a18be7139c384c05"
    )
    store = experiment.encode_hologram(objects, 32, experiment.SIGMA,
                                       experiment.FHRR(32, seed=experiment.SEED))[0]
    assert store.blob[:experiment.HEADER.size].hex() == (
        "534d012000000011000000040001000ad7233c0000"
    )
    # Float kernels can differ in their last bit with SIMD alignment / NumPy 2.
    frozen = [
        2.68301463, -2.53001833, 3.01202679, 1.43103814, -2.93560934,
        0.22990161, 0.90595621, 4.09688568, -1.55284083, 1.64093065,
        1.81256783, -2.36698151, 1.96816874, 3.08273435, -0.13832170,
        3.55292034, 2.69925666, -1.42549992, 0.59043658, -0.45170131,
        0.15364164, -0.37584877, -1.42482007, 1.76853037, -1.54118192,
        0.73187387, -1.55453098, 1.31590796, 1.49546885, 1.33122742,
        1.25740969, 1.04517651, 1.25624537, 0.63768804, 4.38910580,
        -1.15120769, 3.55727625, 1.09206390, 2.48136806, 2.11788130,
        1.98314166, 1.15235555, -0.21450120, -0.81996453, -0.70529020,
        -0.28427160, 1.08169782, 1.24945021, 2.68235922, 5.15245342,
        1.14736438, 3.18434238, 0.64159036, 0.22741568, -1.41489458,
        -3.22390556, -1.78709888, 2.25947523, 2.70979881, -2.06415844,
        0.62914199, -0.89629215, -0.32187140, -2.49561739,
    ]
    np.testing.assert_allclose(
        np.frombuffer(store.blob[experiment.HEADER.size:], dtype="<f4"),
        frozen, rtol=2e-6, atol=2e-6,
    )


@pytest.mark.parametrize("value, reason", [
    ({}, "malformed"), ({"format": "unknown"}, "unknown"),
    ({"format": "boxes", "geometry": "absent.ply"}, "geometry"),
])
def test_missing_or_malformed_annotation_is_refused_with_a_reason(
    tmp_path, value, reason,
):
    with pytest.raises(ValueError, match=reason):
        seam.load_scene(write_json(tmp_path / "bad.json", value))
    with pytest.raises(ValueError, match="cannot read annotation"):
        seam.load_boxes(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="positive"):
        seam.load_boxes(write_json(tmp_path / "negative.json", {"data": [{
            "label": "chair", "segments": {"obbAligned": {
                "axesLengths": [-1, 2, 3], "centroid": [0, 0, 0]}}}]}))
    with pytest.raises(ValueError, match="aligned"):
        seam.objects_from_instances([[0, 0, 0]], [], {})


def test_objects_feed_the_existing_experiment_unmodified(tmp_path, geometry):
    manifest = write_json(tmp_path / "scene.json", {
        "format": "files", "objects": [{"path": geometry.name, "label": "chair"}],
        "geometry": geometry.name, "crop": {"lo": [0, 0, 0], "extent": [1, 1, 1]}})
    objects, report = seam.load_scene(manifest)
    points = load_scene_file(geometry)[0]
    np.testing.assert_allclose(objects["position"][0],
                               (points.min(0) + points.max(0)) / 2, atol=1e-7)
    np.testing.assert_allclose(objects["extent"][0], np.ptp(points, axis=0) / 2,
                               atol=1e-7)
    assert objects.dtype == experiment.OBJECT
    result = experiment.measure(objects, 32, classes=report["classes_survived"])
    assert result["table"]["bytes"] == experiment.TABLE_HEADER.size + objects.nbytes


def test_scene_cli_reports_counts_first_and_supports_many_classes(
    tmp_path, geometry, capsys,
):
    manifest = write_json(tmp_path / "scene.json", {"format": "files", "objects": [
        {"path": geometry.name, "label": f"class{i}"} for i in range(6)]})
    mapping = write_json(tmp_path / "map.json", {"class5": "class4"})
    experiment.main(["--scene", str(manifest), "--label-map", str(mapping),
                     "--dims", "32"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["scene"]["classes_survived"] == 5
    assert lines[0]["scene"]["objects_survived"] == 6
    assert lines[1]["n"] == 6
    assert "gates" in lines[2]


def test_frozen_encoding_fork_numeric_output():
    actual = experiment.encoding_fork()
    assert actual['n'] == 512
    assert actual['d'] == 256
    np.testing.assert_allclose(actual['matching_mean'], 0.8375406265258789,
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual['nonmatching_mean'], 0.20615975558757782,
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual['floor'], 0.9302719831466675,
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual['separation'], 0.6313808560371399,
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual['separation_sigma'], 0.6787056554164718,
                               rtol=2e-6, atol=2e-6)
    assert actual['selected'] == 'codewords'


@pytest.mark.parametrize("groups, reason", [
    ([{"objectId": 1, "segments": [99], "label": "chair"}], "missing segment"),
    ([{"objectId": 1, "segments": [2], "label": "chair"},
      {"objectId": 2, "segments": [2], "label": "table"}], "multiple instances"),
    ([{"objectId": 1, "segments": [2], "label": "chair"},
      {"objectId": 1, "segments": [2], "label": "table"}], "unique"),
])
def test_inconsistent_instances_are_refused(tmp_path, geometry, groups, reason):
    segs = write_json(tmp_path / "segs.json", {"segIndices": [2] * 12})
    annotation = write_json(tmp_path / "agg.json", {"segGroups": groups})
    with pytest.raises(ValueError, match=reason):
        seam.load_instance_points(geometry, segs, annotation)


def test_bad_rotation_and_nonfinite_boxes_are_refused(tmp_path):
    annotation = boxes(tmp_path / "box.json")
    data = json.loads(annotation.read_text())
    obb = data["data"][0]["segments"]["obbAligned"]
    obb["normalizedAxes"] = [1] * 9
    with pytest.raises(ValueError, match="orthonormal"):
        seam.load_boxes(write_json(annotation, data))
    obb["normalizedAxes"] = np.eye(3).ravel().tolist()
    obb["centroid"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError, match="finite"):
        seam.load_boxes(write_json(annotation, data))


def test_crop_retains_whole_objects_by_center(tmp_path, geometry):
    annotation = boxes(tmp_path / "boxes.json")
    data = json.loads(annotation.read_text())
    outside = json.loads(json.dumps(data["data"][0]))
    outside["segments"]["obbAligned"]["centroid"] = [2, 2, 2]
    data["data"].append(outside)
    write_json(annotation, data)
    manifest = write_json(tmp_path / "scene.json", {
        "format": "boxes", "geometry": geometry.name,
        "annotations": annotation.name,
        "crop": {"lo": [0, 0, 0], "extent": [1, 1, 1]}})
    objects, report = seam.load_scene(manifest)
    assert len(objects) == 1
    assert report["objects_before_crop"] == 2
    expected, _ = seam.to_objects(seam.load_boxes(annotation)[:1])
    np.testing.assert_array_equal(objects, expected)
