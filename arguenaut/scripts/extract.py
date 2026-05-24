"""CLI: ingest hypotheses end-to-end (perspectives + activations)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from arguenaut.config import settings
from arguenaut.extraction import make_extractor
from arguenaut.generation import PerspectiveGenerator
from arguenaut.pipeline import ingest_hypotheses
from arguenaut.storage import ActivationStore, Database


def _load_hypotheses(path: Path) -> list[tuple[str, str | None]]:
    """JSON file: either a list of strings, or a list of {text, topic} objects."""
    raw = json.loads(path.read_text())
    out: list[tuple[str, str | None]] = []
    for item in raw:
        if isinstance(item, str):
            out.append((item, None))
        elif isinstance(item, dict) and "text" in item:
            out.append((item["text"], item.get("topic")))
        else:
            raise ValueError(f"Bad hypothesis entry: {item!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest hypotheses: generate perspectives, extract activations")
    p.add_argument("--hypotheses", type=Path, required=True, help="JSON file with hypotheses")
    p.add_argument("--batch-size", type=int, default=8, help="Activation extraction batch size")
    p.add_argument("--limit", type=int, default=None, help="Only ingest first N hypotheses (dev)")
    p.add_argument("--force", action="store_true", help="Re-extract even if cached in HDF5")
    p.add_argument(
        "--remote", action="store_true",
        help="Use the Lambda FastAPI server for activation extraction "
             "(reads URL from data/.lambda-state.json or LAMBDA_API_URL)",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    hyps = _load_hypotheses(args.hypotheses)
    if args.limit:
        hyps = hyps[: args.limit]
    print(f"[extract] {len(hyps)} hypotheses queued from {args.hypotheses}", file=sys.stderr)

    settings.ensure_dirs()
    db = Database(settings.db_path)
    store = ActivationStore(settings.hdf5_path, mode="a")
    generator = PerspectiveGenerator()
    extractor = make_extractor(remote=args.remote)
    if args.remote:
        print(f"[extract] using REMOTE extractor at {extractor.api_url}", file=sys.stderr)

    try:
        stats = ingest_hypotheses(
            hyps,
            db=db,
            store=store,
            generator=generator,
            extractor=extractor,
            batch_size=args.batch_size,
            skip_if_cached=not args.force,
        )
    finally:
        store.close()
        db.close()

    print(f"[extract] done: {stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
