"""Smoke tests for SQLite + HDF5 wrappers — no GPU / no Groq required."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arguenaut.storage.db import Database
from arguenaut.storage.h5 import ActivationStore


def test_db_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")

    hyp_id = db.add_hypothesis("Scaling laws will plateau before AGI.", topic="ml_theory")
    same_id = db.add_hypothesis("Scaling laws will plateau before AGI.")
    assert hyp_id == same_id

    p1 = db.add_perspective(hyp_id, "strong_agree", "Compute is bounded.", 0)
    p2 = db.add_perspective(hyp_id, "strong_disagree", "We have not seen the wall.", 1)
    assert p1 != p2

    perspectives = db.list_perspectives(hyp_id)
    assert [p.position for p in perspectives] == [0, 1]
    assert perspectives[0].stance == "strong_agree"

    # Re-inserting at the same position updates rather than duplicating.
    p1_again = db.add_perspective(hyp_id, "qualified_agree", "Mostly true.", 0)
    assert p1_again == p1
    assert db.get_perspective(p1).stance == "qualified_agree"

    db.close()


def test_axes_upsert(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")
    axis_id = db.upsert_axis(layer=16, component_idx=0, explained_var=0.12)
    # second upsert with a label should merge, not duplicate
    axis_id2 = db.upsert_axis(
        layer=16, component_idx=0, explained_var=0.13,
        label="Empiricism vs Rationalism", high_pole="empirical", low_pole="theoretical",
        confidence=0.78, refinement_rounds=1,
    )
    assert axis_id == axis_id2
    axes = db.list_axes(layer=16)
    assert len(axes) == 1
    assert axes[0].label == "Empiricism vs Rationalism"
    assert axes[0].refinement_rounds == 1
    db.close()


def test_axis_verifications(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")
    aid = db.upsert_axis(layer=20, component_idx=2, explained_var=0.08)
    db.add_axis_verification(aid, round_=0, candidate="A vs B", score=0.4, detail={"n": 8})
    db.add_axis_verification(aid, round_=1, candidate="A vs C", score=0.7, detail={"n": 8})
    rows = db.list_axis_verifications(aid)
    assert [r["round"] for r in rows] == [0, 1]
    assert rows[1]["detail"] == {"n": 8}
    db.close()


def test_h5_round_trip(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path / "acts.h5", mode="w")
    store.set_model_meta("Qwen/Qwen2.5-3B", n_layers=4, d_model=8)

    a = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    b = np.random.RandomState(1).randn(4, 8).astype(np.float32)
    store.put(1, 10, a, b)
    store.put(1, 11, b, a)
    store.put(2, 20, a, b)

    assert store.has(1, 10)
    assert store.model_id == "Qwen/Qwen2.5-3B"
    assert store.n_layers == 4
    assert store.d_model == 8

    np.testing.assert_allclose(store.get(1, 10), a)
    np.testing.assert_allclose(store.get(1, 10, kind="mean_pooled"), b)

    layer1_mat, keys = store.stack_layer(1)
    assert layer1_mat.shape == (3, 8)
    assert (1, 10) in keys

    full, _ = store.stack_all_layers()
    assert full.shape == (3, 4, 8)
    store.close()
