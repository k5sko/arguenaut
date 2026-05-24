"""Centralised configuration. Reads from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key, default)
    if val is None or val == "":
        return None
    return val


def _path(key: str, default: str) -> Path:
    raw = _env(key, default)
    assert raw is not None
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Settings:
    # ── Groq ────────────────────────────────────────────────────────────────
    groq_api_key: str | None = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"
    )

    # ── Model under study ───────────────────────────────────────────────────
    hf_model_id: str = field(
        default_factory=lambda: _env("HF_MODEL_ID", "Qwen/Qwen2.5-3B") or "Qwen/Qwen2.5-3B"
    )

    # ── Lambda backend ──────────────────────────────────────────────────────
    lambda_api_url: str = field(
        default_factory=lambda: _env("LAMBDA_API_URL", "http://localhost:8000") or "http://localhost:8000"
    )
    lambda_api_token: str | None = field(default_factory=lambda: _env("LAMBDA_API_TOKEN"))

    # ── Lambda Cloud provisioning ───────────────────────────────────────────
    lambda_cloud_api_key: str | None = field(default_factory=lambda: _env("LAMBDA_CLOUD_API_KEY"))
    lambda_ssh_key_name: str | None = field(default_factory=lambda: _env("LAMBDA_SSH_KEY_NAME"))
    lambda_ssh_key_path: str | None = field(default_factory=lambda: _env("LAMBDA_SSH_KEY_PATH"))
    lambda_instance_type: str = field(
        default_factory=lambda: _env("LAMBDA_INSTANCE_TYPE", "gpu_1x_a10") or "gpu_1x_a10"
    )
    lambda_region: str | None = field(default_factory=lambda: _env("LAMBDA_REGION"))
    lambda_file_system_name: str | None = field(default_factory=lambda: _env("LAMBDA_FILE_SYSTEM_NAME"))
    lambda_git_url: str | None = field(default_factory=lambda: _env("LAMBDA_GIT_URL"))
    lambda_git_ref: str = field(default_factory=lambda: _env("LAMBDA_GIT_REF", "main") or "main")
    lambda_auto_shutdown_minutes: int = field(
        default_factory=lambda: int(_env("LAMBDA_AUTO_SHUTDOWN_MINUTES", "15") or "15")
    )

    # ── Storage paths ───────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: _path("ARGUENAUT_DATA_DIR", "data"))
    db_path: Path = field(default_factory=lambda: _path("ARGUENAUT_DB_PATH", "data/arguenaut.sqlite"))
    hdf5_path: Path = field(default_factory=lambda: _path("ARGUENAUT_HDF5_PATH", "data/activations.h5"))

    # ── Pipeline knobs ──────────────────────────────────────────────────────
    n_perspectives_per_hypothesis: int = 6
    # Default candidate layers to inspect (mid-to-late, per PLAN.md §2.4).
    default_layers: tuple[int, ...] = (16, 20, 24, 28)
    default_n_components: int = 8
    axis_verification_threshold: float = 0.6
    axis_max_refinements: int = 3

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
