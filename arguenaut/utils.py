"""Small utilities shared across modules."""

from __future__ import annotations

import hashlib
import re

_WS_RE = re.compile(r"\s+")


def normalise_hypothesis(text: str) -> str:
    """Collapse whitespace, strip, and enforce a sensible length."""
    if text is None:
        raise ValueError("hypothesis is None")
    cleaned = _WS_RE.sub(" ", text).strip()
    if len(cleaned) < 12:
        raise ValueError(
            "Hypothesis is too short. Please write a complete claim "
            "(at least a dozen characters, ideally a full sentence)."
        )
    if len(cleaned) > 2000:
        raise ValueError("Hypothesis is too long (over 2000 characters). Trim it down to a single claim.")
    return cleaned


def hypothesis_hash(text: str) -> str:
    """Stable cache key for a normalised hypothesis."""
    return hashlib.sha256(normalise_hypothesis(text).encode("utf-8")).hexdigest()[:16]
