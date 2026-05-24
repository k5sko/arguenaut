"""PCA over residual-stream activations.

Two modes:
  • per-layer  — fit a separate PCA on each layer's [N, d_model] matrix.
                 Useful for picking the layer(s) where disagreement structure
                 is cleanest (Phase 2.2 / 2.4).
  • cross-layer — concatenate selected layers into [N, k * d_model] and fit
                  one richer PCA over the joint space.

`rank_layers` ranks layers by how well PC1/PC2 separate opposing perspectives
within each hypothesis (silhouette-style score against pro/con labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    layer: int | None                 # None for cross-layer
    layers_used: tuple[int, ...]      # the layer(s) that fed this PCA
    n_components: int
    components: np.ndarray            # [n_components, d_feature]
    explained_variance_ratio: np.ndarray  # [n_components]
    mean: np.ndarray                  # [d_feature] for centering
    scores: np.ndarray                # [N, n_components] — projections of training points
    keys: list[tuple[int, int]]       # (hypothesis_id, perspective_id) per row

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project new vectors of shape [d_feature] or [B, d_feature] onto components."""
        x = np.atleast_2d(x).astype(np.float32)
        centered = x - self.mean
        return centered @ self.components.T


def run_per_layer_pca(
    activations: np.ndarray,          # [N, n_layers, d_model]
    keys: list[tuple[int, int]],
    layers: Sequence[int] | None = None,
    n_components: int = 8,
) -> dict[int, PCAResult]:
    """Fit one PCA per requested layer. Returns {layer_index: PCAResult}."""
    if activations.ndim != 3:
        raise ValueError(f"activations must be [N, n_layers, d_model], got {activations.shape}")
    n, n_layers, d = activations.shape
    if layers is None:
        layers = list(range(n_layers))

    results: dict[int, PCAResult] = {}
    for layer in layers:
        if not 0 <= layer < n_layers:
            raise IndexError(f"layer {layer} out of range [0,{n_layers})")
        X = activations[:, layer, :].astype(np.float32)
        n_comp = min(n_components, n, d)
        pca = PCA(n_components=n_comp, svd_solver="auto")
        scores = pca.fit_transform(X)
        results[layer] = PCAResult(
            layer=layer,
            layers_used=(layer,),
            n_components=n_comp,
            components=pca.components_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            scores=scores.astype(np.float32),
            keys=list(keys),
        )
    return results


def run_cross_layer_pca(
    activations: np.ndarray,
    keys: list[tuple[int, int]],
    layers: Sequence[int],
    n_components: int = 8,
) -> PCAResult:
    """Concatenate `layers` along the feature dim then fit one PCA."""
    if not layers:
        raise ValueError("layers must be non-empty")
    n, n_layers, d = activations.shape
    flat = np.concatenate([activations[:, l, :] for l in layers], axis=1).astype(np.float32)
    n_comp = min(n_components, n, flat.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="auto")
    scores = pca.fit_transform(flat)
    return PCAResult(
        layer=None,
        layers_used=tuple(layers),
        n_components=n_comp,
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        scores=scores.astype(np.float32),
        keys=list(keys),
    )


@dataclass
class LayerScore:
    layer: int
    silhouette: float
    explained_var_pc1: float
    explained_var_top2: float
    notes: str = ""
    kind: str = "unknown"  # "attention" / "deltanet" / "unknown"


def rank_layers(
    per_layer_results: dict[int, PCAResult],
    pro_con_labels: dict[tuple[int, int], int],
    layer_kinds: dict[int, str] | None = None,
) -> list[LayerScore]:
    """Rank layers by how well PC1/PC2 separate pro vs con (label in {-1, +1}).

    pro_con_labels maps (hypothesis_id, perspective_id) → -1 (con) / +1 (pro) / 0 (skip).
    Layers with too few labels collapse to silhouette = NaN (skipped).
    """
    from arguenaut.analysis.scoring import silhouette_pro_con

    layer_kinds = layer_kinds or {}
    out: list[LayerScore] = []
    for layer, res in per_layer_results.items():
        labels = np.array([pro_con_labels.get(k, 0) for k in res.keys], dtype=int)
        mask = labels != 0
        sil = float("nan")
        if mask.sum() >= 4 and len(set(labels[mask].tolist())) == 2:
            sil = silhouette_pro_con(res.scores[mask, :2], labels[mask])
        out.append(
            LayerScore(
                layer=layer,
                silhouette=sil,
                explained_var_pc1=float(res.explained_variance_ratio[0]),
                explained_var_top2=float(res.explained_variance_ratio[:2].sum()),
                kind=layer_kinds.get(layer, "unknown"),
            )
        )
    # Sort highest silhouette first, NaN last
    out.sort(key=lambda ls: (np.isnan(ls.silhouette), -ls.silhouette))
    return out


def project_new_point(result: PCAResult, vector: np.ndarray) -> np.ndarray:
    """Project a single feature vector onto a PCAResult, returning [n_components]."""
    return result.project(vector)[0]
