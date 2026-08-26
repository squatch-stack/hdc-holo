"""Shared fixtures for the holographic structure test suite.

This is the ONLY place shared test plumbing lives — see TESTING.md.
"""

import pytest

from holo import FHRR


@pytest.fixture
def space():
    """A fresh default hypervector space per test: d=4096, seed 0."""
    return FHRR(dim=4096, seed=0)
