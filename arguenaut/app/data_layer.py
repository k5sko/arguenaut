"""Shared data-loading helpers for the Streamlit pages.

Wraps SQLite + HDF5 + PCA cache behind a single facade so the pages don't
each re-implement the same loading code. Cached via @st.cache_resource so
opening the HDF5 file doesn't happen on every interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from arguenaut.analysis.cache import load_pca_cache
from arguenaut.analysis.pca import PCAResult
from arguenaut.config import settings
from arguenaut.storage import ActivationStore, Database


@dataclass
class HypothesisRow:
    id: int
    text: str
    topic: str | None


@dataclass
class PerspectiveRow:
    id: int
    hypothesis_id: int
    stance: str
    text: str
    position: int


@dataclass
class AxisRow:
    id: int
    layer: int
    component_idx: int
    label: str | None
    high_pole: str | None
    low_pole: str | None
    confidence: float | None
    explained_var: float | None


class AppData:
    def __init__(self):
        settings.ensure_dirs()
        self.db = Database(settings.db_path)
        self.store = ActivationStore(settings.hdf5_path, mode="r") if settings.hdf5_path.exists() else None
        pca_path = settings.data_dir / "pca_cache.npz"
        self.pca: dict[int, PCAResult] = load_pca_cache(pca_path) if pca_path.exists() else {}

    # ── lookups ────────────────────────────────────────────────────────────
    def hypotheses(self) -> list[HypothesisRow]:
        return [HypothesisRow(h.id, h.text, h.topic) for h in self.db.list_hypotheses()]

    def perspectives(self) -> list[PerspectiveRow]:
        return [
            PerspectiveRow(p.id, p.hypothesis_id, p.stance, p.text, p.position)
            for p in self.db.list_perspectives()
        ]

    def axes(self, layer: int | None = None) -> list[AxisRow]:
        return [
            AxisRow(
                id=a.id,
                layer=a.layer,
                component_idx=a.component_idx,
                label=a.label,
                high_pole=a.high_pole,
                low_pole=a.low_pole,
                confidence=a.confidence,
                explained_var=a.explained_var,
            )
            for a in self.db.list_axes(layer=layer)
        ]

    def axis_verifications(self, axis_id: int) -> list[dict]:
        return self.db.list_axis_verifications(axis_id)

    # ── PCA helpers ────────────────────────────────────────────────────────
    def available_layers(self) -> list[int]:
        return sorted(self.pca.keys())

    def pca_for_layer(self, layer: int) -> PCAResult | None:
        return self.pca.get(layer)

    def scores_dataframe(self, layer: int):
        import pandas as pd

        pca = self.pca.get(layer)
        if pca is None:
            return pd.DataFrame()
        hyp_by_id = {h.id: h for h in self.db.list_hypotheses()}
        persp_by_id = {p.id: p for p in self.db.list_perspectives()}
        rows = []
        for (hid, pid), scores in zip(pca.keys, pca.scores):
            h = hyp_by_id.get(hid)
            p = persp_by_id.get(pid)
            if h is None or p is None:
                continue
            row = {
                "hypothesis_id": hid,
                "perspective_id": pid,
                "hypothesis": h.text,
                "topic": h.topic or "—",
                "stance": p.stance,
                "text": p.text,
            }
            for i, s in enumerate(scores):
                row[f"PC{i + 1}"] = float(s)
            rows.append(row)
        return pd.DataFrame(rows)
