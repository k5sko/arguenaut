"""Persistent local state about the currently-provisioned Lambda instance."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from arguenaut.config import settings


@dataclass
class LambdaState:
    instance_id: str
    name: str
    ip: str
    ssh_user: str
    api_url: str
    api_port: int
    region: str
    instance_type: str
    launched_at: float
    bootstrap_complete: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _state_path() -> Path:
    override = os.environ.get("ARGUENAUT_LAMBDA_STATE")
    if override:
        return Path(override)
    return settings.data_dir / ".lambda-state.json"


def load_state() -> LambdaState | None:
    p = _state_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        return LambdaState(**raw)
    except Exception:
        # Corrupt state file — treat as no state.
        return None


def save_state(state: LambdaState) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2))


def clear_state() -> None:
    p = _state_path()
    if p.exists():
        p.unlink()


def touch_state(**overrides) -> LambdaState | None:
    s = load_state()
    if s is None:
        return None
    data = s.to_dict()
    data.update(overrides)
    new = LambdaState(**data)
    save_state(new)
    return new
