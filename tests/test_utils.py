from __future__ import annotations

import pytest

from arguenaut.utils import hypothesis_hash, normalise_hypothesis


def test_normalise_collapses_whitespace():
    assert normalise_hypothesis("  Scaling  laws\nplateau before AGI.  ") == "Scaling laws plateau before AGI."


def test_normalise_rejects_short():
    with pytest.raises(ValueError):
        normalise_hypothesis("too short")


def test_normalise_rejects_long():
    with pytest.raises(ValueError):
        normalise_hypothesis("x" * 3000)


def test_hash_is_stable():
    a = hypothesis_hash("Scaling laws plateau before AGI.")
    b = hypothesis_hash("  Scaling   laws  plateau before AGI.  ")
    assert a == b
