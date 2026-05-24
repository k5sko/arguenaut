"""Load PCA cache written by `arguenaut.scripts.run_pca`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arguenaut.analysis.pca import PCAResult


def load_pca_cache(path: str | Path) -> dict[int, PCAResult]:
    """Return {layer_idx: PCAResult} from the .npz produced by run_pca."""
    data = np.load(path, allow_pickle=False)
    layer_indices = sorted({int(k[1:].split("_", 1)[0]) for k in data.files if k.startswith("L")})
    out: dict[int, PCAResult] = {}
    for layer in layer_indices:
        keys_arr = data[f"L{layer}_keys"]
        keys = [(int(a), int(b)) for a, b in keys_arr]
        components = data[f"L{layer}_components"]
        out[layer] = PCAResult(
            layer=layer,
            layers_used=(layer,),
            n_components=components.shape[0],
            components=components,
            explained_variance_ratio=data[f"L{layer}_explvar"],
            mean=data[f"L{layer}_mean"],
            scores=data[f"L{layer}_scores"],
            keys=keys,
        )
    return out
