"""End-to-end pipeline validator (Phase 1.4).

Runs 3 test hypotheses through perspectives → activations and sanity-checks:
  • returned tensor shapes match (n_layers, d_model)
  • activations are NOT identical across perspectives (would imply broken hooks)
  • activations have reasonable scale (non-NaN, non-zero, not exploding)

Designed to be runnable as `python -m arguenaut.scripts.validate_pipeline`.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from arguenaut.config import settings
from arguenaut.extraction import make_extractor
from arguenaut.generation import PerspectiveGenerator
from arguenaut.storage import ActivationStore, Database

TEST_HYPOTHESES = [
    ("Neural scaling laws will hit diminishing returns before AGI.", "ml_theory"),
    ("Mathematical objects exist independently of human minds.", "math_foundations"),
    ("Alignment research will be solved by interpretability, not RLHF.", "ai_safety"),
]


def _check_activations(
    label: str, last_tokens: list[np.ndarray], n_layers_expected: int, d_model_expected: int
) -> list[str]:
    """Return a list of warning strings; empty list = healthy."""
    warns: list[str] = []
    for i, act in enumerate(last_tokens):
        if act.shape != (n_layers_expected, d_model_expected):
            warns.append(f"{label}#{i}: wrong shape {act.shape}, want ({n_layers_expected},{d_model_expected})")
        if not np.isfinite(act).all():
            warns.append(f"{label}#{i}: NaN/inf in activations")
        if (act == 0).all():
            warns.append(f"{label}#{i}: activations are all zero")
        if np.abs(act).max() > 1e4:
            warns.append(f"{label}#{i}: activations exploding (max={np.abs(act).max():.1f})")
    if len(last_tokens) >= 2:
        # All-pairs identical check: any two perspectives within one hypothesis
        # should NOT have identical activations.
        for i in range(len(last_tokens)):
            for j in range(i + 1, len(last_tokens)):
                if np.allclose(last_tokens[i], last_tokens[j]):
                    warns.append(f"{label}: perspectives {i} and {j} produced IDENTICAL activations")
    return warns


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persist", action="store_true", help="Also write into the real SQLite/HDF5 stores")
    p.add_argument("--remote", action="store_true",
                   help="Use Lambda FastAPI server instead of loading the model locally")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    print("[validate] loading model …", file=sys.stderr)
    extractor = make_extractor(remote=args.remote)
    extractor.load()
    print(f"[validate] model={extractor.model_id} n_layers={extractor.n_layers} d_model={extractor.d_model}", file=sys.stderr)

    generator = PerspectiveGenerator()

    db = None
    store = None
    if args.persist:
        settings.ensure_dirs()
        db = Database(settings.db_path)
        store = ActivationStore(settings.hdf5_path, mode="a")
        store.set_model_meta(extractor.model_id, extractor.n_layers, extractor.d_model)

    total_warns: list[str] = []
    for text, topic in TEST_HYPOTHESES:
        print(f"\n[validate] hypothesis: {text!r}", file=sys.stderr)
        perspectives = generator.generate(text)
        print(f"[validate]   got {len(perspectives)} perspectives", file=sys.stderr)
        for p in perspectives:
            print(f"           [{p.stance}] {p.text[:120]}…", file=sys.stderr)

        results = extractor.extract_batch([p.text for p in perspectives])
        last_tokens = [r.last_token for r in results]
        warns = _check_activations(
            label=text[:40], last_tokens=last_tokens,
            n_layers_expected=extractor.n_layers, d_model_expected=extractor.d_model,
        )
        if warns:
            for w in warns:
                print(f"[validate]   WARN: {w}", file=sys.stderr)
            total_warns.extend(warns)
        else:
            print("[validate]   OK — shapes, finiteness, and distinctness all pass", file=sys.stderr)

        if args.persist and db is not None and store is not None:
            hid = db.add_hypothesis(text, topic)
            for p, r in zip(perspectives, results):
                pid = db.add_perspective(hid, p.stance, p.text, p.position)
                store.put(hid, pid, r.last_token, r.mean_pooled)

    if store is not None:
        store.close()
    if db is not None:
        db.close()

    if total_warns:
        print(f"\n[validate] FAILED with {len(total_warns)} warnings", file=sys.stderr)
        return 1
    print("\n[validate] ALL OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
