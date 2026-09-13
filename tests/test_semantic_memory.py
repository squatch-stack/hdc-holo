"""Seeded object-store checks; no capture assets required."""

import numpy as np
import pytest

from bench.semantic_memory import (
    HEADER,
    OBJECT,
    SIGMA,
    TABLE_HEADER,
    decode_hologram,
    encode_hologram,
    encode_table,
    four_bit,
    object_scene,
    payloads,
    predicted_knee,
    query_what_is_at,
    query_where_is,
    table_objects,
)
from holo.fhrr import FHRR


def fixture(n=8):
    return object_scene(n, np.random.default_rng(17))


def test_what_is_at_is_exact_at_low_load():
    objects = fixture()
    holo, _ = encode_hologram(objects, 512, SIGMA, FHRR(512, seed=17))
    for store in (holo, four_bit(holo), encode_table(objects)[0]):
        np.testing.assert_array_equal(query_what_is_at(store, objects["position"]),
                                      objects["label"])


def test_accuracy_degrades_where_the_capacity_law_says_it_will():
    d = 128
    accuracies = []
    for n in (8, 128, 256, 512):
        objects = fixture(n)
        holo, _ = encode_hologram(objects, d, SIGMA, FHRR(d, seed=17))
        accuracy = np.mean(query_what_is_at(holo, objects["position"])
                           == objects["label"])
        accuracies.append((n, accuracy))
    assert accuracies[0][1] == 1
    knee = next(n for n, accuracy in accuracies if accuracy <= 0.5)
    assert 0.5 <= knee / predicted_knee(d) <= 2
    assert accuracies[-1][1] < accuracies[0][1]


def test_table_and_hologram_receive_identical_inputs():
    objects = fixture(16)
    before = objects.tobytes()
    table, _ = encode_table(objects)
    holo, _ = encode_hologram(objects, 256, SIGMA, FHRR(256, seed=17))
    assert objects.tobytes() == before
    assert table_objects(table).tobytes() == before
    # Reconstruct the expected sum independently from the serialized table.
    field, codes, _ = decode_hologram(holo)
    expected = np.zeros(256, dtype=np.complex64)
    for obj in table_objects(table):
        expected += field.pos(obj["position"]) * codes[obj["label"]]
    np.testing.assert_allclose(field.S, expected, rtol=2e-5, atol=2e-5)


def test_byte_accounting_is_complete():
    objects = fixture(9)
    table, table_bytes = encode_table(objects)
    holo, holo_bytes = encode_hologram(objects, 129, SIGMA, FHRR(129, seed=17))
    quant = four_bit(holo)
    assert OBJECT.itemsize == 26
    assert table_bytes == len(table.serialize()) == TABLE_HEADER.size + 9 * 26
    assert holo_bytes == len(holo.serialize()) == HEADER.size + 129 * 8
    # HG header is 16 B; each magnitude/phase stream rounds up separately.
    assert len(quant.serialize()) == HEADER.size + 16 + 2 * ((129 + 1) // 2)
    field, _, _ = decode_hologram(quant)
    assert field.S.shape == (129,)
    assert np.all(np.isfinite(field.S))


def test_where_is_returns_only_that_class():
    objects = fixture(16)
    table, _ = encode_table(objects)
    holo, _ = encode_hologram(objects, 512, SIGMA, FHRR(512, seed=17))
    for label in range(4):
        expected = objects["position"][objects["label"] == label]
        np.testing.assert_array_equal(query_where_is(table, label), expected)
        np.testing.assert_array_equal(
            query_where_is(holo, label, objects["position"]), expected)
    with pytest.raises(ValueError, match="external candidate"):
        query_where_is(holo, 0)


def test_record_payloads_multiply_the_budget_by_field_count():
    objects = fixture()
    base_d = 256
    for fields in (1, 3):
        d = base_d * fields
        holo, size = encode_hologram(objects, d, SIGMA, FHRR(d, seed=17),
                                    fields=fields)
        assert size - HEADER.size == fields * base_d * 8
        assert predicted_knee(d, fields) == predicted_knee(base_d)
        codes = payloads(FHRR(d, seed=17), 4, fields)
        # Independent role/filler sums have R-fold power, not unit power.
        np.testing.assert_allclose(np.mean(np.abs(codes) ** 2), fields,
                                   rtol=0.08, atol=0.01)
        np.testing.assert_array_equal(query_what_is_at(holo, objects["position"]),
                                      objects["label"])
