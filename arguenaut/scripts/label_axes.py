"""CLI: label the top-k principal components per layer via the Groq labeling loop.

Usage:
    python -m arguenaut.scripts.label_axes --top-k 5 --layers 16,20,24,28
"""

from __future__ import annotations

import argparse
import logging
import sys

from arguenaut.analysis.cache import load_pca_cache
from arguenaut.config import settings
from arguenaut.extraction import make_extractor
from arguenaut.labeling import AxisLabeler
from arguenaut.storage import Database


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=5, help="Components per layer to label")
    p.add_argument("--layers", default=None, help="Comma-separated layers (defaults to cached set)")
    p.add_argument("--max-refinements", type=int, default=settings.axis_max_refinements)
    p.add_argument("--threshold", type=float, default=settings.axis_verification_threshold)
    p.add_argument("--remote", action="store_true",
                   help="Use the Lambda FastAPI server for verification extractions")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    settings.ensure_dirs()
    cache_path = settings.data_dir / "pca_cache.npz"
    if not cache_path.exists():
        print(f"[label_axes] PCA cache not found at {cache_path}. Run `arguenaut-pca` first.", file=sys.stderr)
        return 1
    cache = load_pca_cache(cache_path)
    if args.layers:
        wanted = {int(x) for x in args.layers.split(",")}
        cache = {l: r for l, r in cache.items() if l in wanted}

    db = Database(settings.db_path)
    extractor = make_extractor(remote=args.remote)
    extractor.load()

    labeler = AxisLabeler(
        db=db,
        extractor=extractor,
        max_refinements=args.max_refinements,
        threshold=args.threshold,
    )

    for layer, pca in sorted(cache.items()):
        print(f"\n[label_axes] === layer {layer} ===", file=sys.stderr)
        for c in range(min(args.top_k, pca.n_components)):
            print(f"[label_axes]  component {c} (expl-var={pca.explained_variance_ratio[c]:.3f})", file=sys.stderr)
            try:
                res = labeler.label_component(pca, c)
            except Exception as e:
                logging.exception("Labeling failed for layer %d component %d", layer, c)
                continue
            tag = "ACCEPTED" if res.accepted else "unlabeled"
            print(
                f"           → [{tag}] {res.final_label or '(none)'} "
                f"score={res.final_score:+.2f}  rounds={len(res.rounds)}",
                file=sys.stderr,
            )

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
