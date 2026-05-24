"""Parser-only tests for PerspectiveGenerator — no live Groq call."""

from __future__ import annotations

import json

import pytest

from arguenaut.generation.perspectives import (
    DEFAULT_STANCES,
    PerspectiveGenerationError,
    _parse_perspective_json,
)


def _payload(items):
    return json.dumps({"perspectives": items})


def test_parse_happy_path():
    items = [{"stance": s, "text": f"text for {s}"} for s in DEFAULT_STANCES]
    parsed = _parse_perspective_json(_payload(items), DEFAULT_STANCES)
    assert [p.stance for p in parsed] == list(DEFAULT_STANCES)
    assert [p.position for p in parsed] == list(range(len(DEFAULT_STANCES)))


def test_parse_strips_json_fence():
    items = [{"stance": s, "text": "x"} for s in DEFAULT_STANCES]
    fenced = "```json\n" + _payload(items) + "\n```"
    parsed = _parse_perspective_json(fenced, DEFAULT_STANCES)
    assert len(parsed) == len(DEFAULT_STANCES)


def test_missing_stance_raises():
    items = [{"stance": s, "text": "x"} for s in DEFAULT_STANCES[:-1]]
    with pytest.raises(PerspectiveGenerationError):
        _parse_perspective_json(_payload(items), DEFAULT_STANCES)


def test_garbage_raises():
    with pytest.raises(PerspectiveGenerationError):
        _parse_perspective_json("not json at all", DEFAULT_STANCES)


def test_empty_perspectives_raises():
    with pytest.raises(PerspectiveGenerationError):
        _parse_perspective_json(json.dumps({"perspectives": []}), DEFAULT_STANCES)
