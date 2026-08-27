"""Trigram profiles: language identification with one vector per language."""

from holo import NGramEncoder
from holo.ngram import TEST, TRAIN


def test_ngram_language_id(space):
    enc = NGramEncoder(space, n=3)
    profiles = {lang: enc.profile(text) for lang, text in TRAIN.items()}
    correct = sum(
        max(profiles, key=lambda l: space.cos(enc.profile(s), profiles[l]))
        == lang
        for lang, s in TEST)
    assert correct >= len(TEST) - 1  # allow one miss on short sentences
