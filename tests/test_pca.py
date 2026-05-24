"""PCA + scoring tests — uses synthetic data so no GPU/Groq needed."""

from __future__ import annotations

import numpy as np
import pytest

from arguenaut.analysis.pca import run_per_layer_pca, run_cross_layer_pca, rank_layers
from arguenaut.analysis.scoring import correlation_score, stance_to_pro_con


@pytest.fixture
def synthetic_activations():
    """Two clusters per layer: +axis vs -axis along a known direction.

    Layer 0 is noise; layer 1 has clean linear separation; layer 2 is moderate.
    """
    rng = np.random.RandomState(42)
    n = 24
    d = 16
    n_layers = 3

    acts = rng.randn(n, n_layers, d).astype(np.float32) * 0.1
    direction = np.zeros(d, dtype=np.float32)
    direction[0] = 1.0
    labels = np.array([+1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int32)
    # Layer 1: strong signal
    acts[:, 1, :] += labels[:, None] * direction * 3.0
    # Layer 2: weaker signal
    acts[:, 2, :] += labels[:, None] * direction * 1.0

    keys = [(0, i) for i in range(n)]
    pro_con = {(0, i): int(labels[i]) for i in range(n)}
    return acts, keys, pro_con


def test_per_layer_pca_shapes(synthetic_activations):
    acts, keys, _ = synthetic_activations
    results = run_per_layer_pca(acts, keys, n_components=4)
    assert set(results.keys()) == {0, 1, 2}
    for r in results.values():
        assert r.components.shape == (4, acts.shape[2])
        assert r.scores.shape == (acts.shape[0], 4)


def test_cross_layer_pca_concatenates(synthetic_activations):
    acts, keys, _ = synthetic_activations
    r = run_cross_layer_pca(acts, keys, layers=[1, 2], n_components=3)
    assert r.layer is None
    assert r.layers_used == (1, 2)
    assert r.components.shape == (3, 2 * acts.shape[2])


def test_rank_layers_finds_signal(synthetic_activations):
    acts, keys, pro_con = synthetic_activations
    results = run_per_layer_pca(acts, keys, n_components=2)
    ranking = rank_layers(results, pro_con)
    # Layer 1 was injected with the strongest signal — it should rank #1.
    assert ranking[0].layer == 1
    assert ranking[0].silhouette > ranking[-1].silhouette


def test_project_new_point(synthetic_activations):
    acts, keys, _ = synthetic_activations
    r = run_per_layer_pca(acts, keys, n_components=3)[1]
    proj = r.project(acts[0, 1, :])
    assert proj.shape == (1, 3)


def test_correlation_score():
    pred = np.array([+1, +1, +1, -1, -1, -1])
    actual = np.array([2.0, 1.5, 0.5, -0.5, -1.0, -2.0])
    assert correlation_score(pred, actual) > 0.9
    flipped = -actual
    assert correlation_score(pred, flipped) < -0.9


def test_stance_to_pro_con():
    assert stance_to_pro_con("strong_agree") == +1
    assert stance_to_pro_con("strong_disagree") == -1
    assert stance_to_pro_con("neutral_uncertain") == 0
    assert stance_to_pro_con("orthogonal_reframe") == 0
    assert stance_to_pro_con("nonsense") == 0
