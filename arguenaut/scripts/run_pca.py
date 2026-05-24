"""CLI: run per-layer PCA over the stored activations and write axes to SQLite.

Usage:
    python -m arguenaut.scripts.run_pca --layers 16,20,24,28 --n-components 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from arguenaut.analysis.pca import run_per_layer_pca, rank_layers
from arguenaut.analysis.scoring import stance_to_pro_con
from arguenaut.config import settings
from arguenaut.extraction.layer_meta import schedule_3to1
from arguenaut.storage import ActivationStore, Database


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="all", help="Comma-separated layer indices, or 'all'")
    p.add_argument("--n-components", type=int, default=settings.default_n_components)
    p.add_argument("--kind", default="last_token", choices=["last_token", "mean_pooled"])
    p.add_argument("--report", type=Path, default=None, help="Write JSON ranking report to this path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    settings.ensure_dirs()
    db = Database(settings.db_path)
    store = ActivationStore(settings.hdf5_path, mode="r")

    acts, keys = store.stack_all_layers(kind=args.kind)
    if acts.size == 0:
        print("[run_pca] no activations found — run `arguenaut-extract` first", file=sys.stderr)
        return 1

    n, n_layers, d = acts.shape
    print(f"[run_pca] {n} perspectives × {n_layers} layers × {d}-dim ({args.kind})", file=sys.stderr)

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    results = run_per_layer_pca(acts, keys, layers=layers, n_components=args.n_components)

    # Build pro/con labels from stance strings.
    stance_by_key = {}
    for hyp in db.list_hypotheses():
        for p in db.list_perspectives(hyp.id):
            stance_by_key[(hyp.id, p.id)] = stance_to_pro_con(p.stance)

    kinds = schedule_3to1(n_layers)
    kind_by_layer = {k.layer: k.kind for k in kinds}
    ranking = rank_layers(results, stance_by_key, layer_kinds=kind_by_layer)

    print("[run_pca] layer ranking by silhouette (best first):", file=sys.stderr)
    for ls in ranking:
        print(
            f"           layer={ls.layer:2d} kind={ls.kind:<9} "
            f"silhouette={ls.silhouette:+.3f}  expl-var(PC1)={ls.explained_var_pc1:.3f}  "
            f"expl-var(top2)={ls.explained_var_top2:.3f}",
            file=sys.stderr,
        )

    # Persist axis metadata for every (layer, component) we evaluated.
    for layer, res in results.items():
        for c_idx in range(res.n_components):
            db.upsert_axis(
                layer=layer,
                component_idx=c_idx,
                explained_var=float(res.explained_variance_ratio[c_idx]),
            )

    # Cache the PCA models to disk so the Streamlit app and labeler can reuse them.
    cache_path = settings.data_dir / "pca_cache.npz"
    _save_pca_cache(cache_path, results)
    print(f"[run_pca] wrote PCA cache → {cache_path}", file=sys.stderr)

    if args.report:
        args.report.write_text(
            json.dumps(
                [
                    {
                        "layer": ls.layer,
                        "kind": ls.kind,
                        "silhouette": None if np.isnan(ls.silhouette) else ls.silhouette,
                        "explained_var_pc1": ls.explained_var_pc1,
                        "explained_var_top2": ls.explained_var_top2,
                    }
                    for ls in ranking
                ],
                indent=2,
            )
        )

    store.close()
    db.close()
    return 0


def _save_pca_cache(path: Path, results) -> None:
    """Persist {layer: PCAResult} as a single .npz so the app can reload it."""
    out: dict[str, np.ndarray] = {}
    for layer, res in results.items():
        out[f"L{layer}_components"] = res.components
        out[f"L{layer}_mean"] = res.mean
        out[f"L{layer}_scores"] = res.scores
        out[f"L{layer}_explvar"] = res.explained_variance_ratio
        out[f"L{layer}_keys"] = np.array(res.keys, dtype=np.int64)
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    raise SystemExit(main())
