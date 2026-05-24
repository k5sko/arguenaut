"""HDF5 activation store.

Layout:
    /{hypothesis_id}/{perspective_id}/last_token   shape [n_layers, d_model], fp32
    /{hypothesis_id}/{perspective_id}/mean_pooled  shape [n_layers, d_model], fp32

Attrs on the root group:
    model_id, n_layers, d_model, dtype
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


class ActivationStore:
    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file: h5py.File | None = h5py.File(self.path, mode)

    @classmethod
    @contextmanager
    def open(cls, path: str | Path, mode: str = "a") -> Iterator["ActivationStore"]:
        store = cls(path, mode)
        try:
            yield store
        finally:
            store.close()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self) -> "ActivationStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── metadata ───────────────────────────────────────────────────────────
    def set_model_meta(self, model_id: str, n_layers: int, d_model: int) -> None:
        assert self.file is not None
        self.file.attrs["model_id"] = model_id
        self.file.attrs["n_layers"] = n_layers
        self.file.attrs["d_model"] = d_model

    @property
    def model_id(self) -> str | None:
        assert self.file is not None
        v = self.file.attrs.get("model_id")
        return v.decode() if isinstance(v, bytes) else v

    @property
    def n_layers(self) -> int | None:
        assert self.file is not None
        return int(self.file.attrs["n_layers"]) if "n_layers" in self.file.attrs else None

    @property
    def d_model(self) -> int | None:
        assert self.file is not None
        return int(self.file.attrs["d_model"]) if "d_model" in self.file.attrs else None

    # ── writes ─────────────────────────────────────────────────────────────
    def put(
        self,
        hypothesis_id: int,
        perspective_id: int,
        last_token: np.ndarray,
        mean_pooled: np.ndarray | None = None,
    ) -> None:
        assert self.file is not None
        grp = self.file.require_group(f"{hypothesis_id}/{perspective_id}")
        for name in ("last_token", "mean_pooled"):
            if name in grp:
                del grp[name]
        grp.create_dataset(
            "last_token", data=np.ascontiguousarray(last_token, dtype=np.float32), compression="gzip"
        )
        if mean_pooled is not None:
            grp.create_dataset(
                "mean_pooled",
                data=np.ascontiguousarray(mean_pooled, dtype=np.float32),
                compression="gzip",
            )

    # ── reads ──────────────────────────────────────────────────────────────
    def get(self, hypothesis_id: int, perspective_id: int, kind: str = "last_token") -> np.ndarray:
        assert self.file is not None
        return np.asarray(self.file[f"{hypothesis_id}/{perspective_id}/{kind}"], dtype=np.float32)

    def has(self, hypothesis_id: int, perspective_id: int, kind: str = "last_token") -> bool:
        assert self.file is not None
        return f"{hypothesis_id}/{perspective_id}/{kind}" in self.file

    def iter_keys(self) -> Iterator[tuple[int, int]]:
        """Yield (hypothesis_id, perspective_id) for every stored activation."""
        assert self.file is not None
        for hid_str, hgrp in self.file.items():
            if not isinstance(hgrp, h5py.Group):
                continue
            for pid_str in hgrp.keys():
                yield int(hid_str), int(pid_str)

    def stack_layer(
        self, layer: int, kind: str = "last_token"
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Return (matrix [N, d_model], list of (hypothesis_id, perspective_id))."""
        keys = list(self.iter_keys())
        if not keys:
            return np.zeros((0, 0), dtype=np.float32), []
        rows = []
        for hid, pid in keys:
            act = self.get(hid, pid, kind=kind)
            rows.append(act[layer])
        return np.stack(rows, axis=0), keys

    def stack_all_layers(
        self, kind: str = "last_token"
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Return (tensor [N, n_layers, d_model], keys)."""
        keys = list(self.iter_keys())
        if not keys:
            return np.zeros((0, 0, 0), dtype=np.float32), []
        rows = [self.get(hid, pid, kind=kind) for hid, pid in keys]
        return np.stack(rows, axis=0), keys
