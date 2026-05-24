"""High-level orchestration: hypothesis → perspectives → activations → storage.

Used by the extract CLI, the Lambda FastAPI server, and the live-query
path in the Streamlit app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from tqdm import tqdm

from arguenaut.config import settings
from arguenaut.extraction import ActivationExtractor
from arguenaut.generation import PerspectiveGenerator
from arguenaut.storage import ActivationStore, Database

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    hypotheses_seen: int = 0
    hypotheses_extracted: int = 0
    perspectives_generated: int = 0
    perspectives_extracted: int = 0
    skipped_cached: int = 0


def ingest_hypotheses(
    hypotheses: Iterable[tuple[str, str | None]],
    *,
    db: Database,
    store: ActivationStore,
    generator: PerspectiveGenerator,
    extractor: ActivationExtractor,
    batch_size: int = 8,
    skip_if_cached: bool = True,
    progress: bool = True,
) -> IngestStats:
    """Process (hypothesis_text, topic) pairs end-to-end.

    For each hypothesis:
      1. upsert into the DB
      2. ensure all stance perspectives exist (generate via Groq if missing)
      3. extract activations for any perspective that doesn't yet have them
         (or all, if skip_if_cached=False)
    """
    stats = IngestStats()
    hyp_list = list(hypotheses)
    if store.model_id is None:
        # We can only initialise model-meta once the extractor is loaded.
        extractor.load()
        store.set_model_meta(extractor.model_id, extractor.n_layers, extractor.d_model)

    bar = tqdm(hyp_list, desc="hypotheses", disable=not progress)
    for text, topic in bar:
        stats.hypotheses_seen += 1
        hyp_id = db.add_hypothesis(text, topic)

        existing = {p.position: p for p in db.list_perspectives(hyp_id)}
        if len(existing) < len(generator.stances):
            logger.info("Generating perspectives for hypothesis %d (have %d)", hyp_id, len(existing))
            perspectives = generator.generate(text)
            stats.perspectives_generated += len(perspectives)
            for p in perspectives:
                db.add_perspective(hyp_id, p.stance, p.text, p.position)
            existing = {p.position: p for p in db.list_perspectives(hyp_id)}

        # Build the list of (perspective_id, text) we still need activations for.
        to_extract = []
        for p in existing.values():
            if skip_if_cached and store.has(hyp_id, p.id, kind="last_token"):
                stats.skipped_cached += 1
                continue
            to_extract.append(p)

        if not to_extract:
            continue

        # Batch through the extractor.
        for i in range(0, len(to_extract), batch_size):
            chunk = to_extract[i : i + batch_size]
            results = extractor.extract_batch([p.text for p in chunk])
            for p, res in zip(chunk, results):
                store.put(hyp_id, p.id, res.last_token, res.mean_pooled)
                stats.perspectives_extracted += 1

        stats.hypotheses_extracted += 1
        bar.set_postfix(extracted=stats.perspectives_extracted, skipped=stats.skipped_cached)

    return stats


# ── live-query path used by the Streamlit app via Lambda ─────────────────────

def process_single_hypothesis(
    text: str,
    *,
    generator: PerspectiveGenerator,
    extractor: ActivationExtractor,
) -> dict:
    """Return perspectives + activations for a fresh user-entered hypothesis,
    without writing to disk. Used by the FastAPI /analyze endpoint."""
    perspectives = generator.generate(text)
    results = extractor.extract_batch([p.text for p in perspectives])
    return {
        "hypothesis": text,
        "perspectives": [
            {
                "stance": p.stance,
                "text": p.text,
                "position": p.position,
                "last_token": r.last_token.tolist(),
                "mean_pooled": r.mean_pooled.tolist(),
                "n_tokens": r.n_tokens,
            }
            for p, r in zip(perspectives, results)
        ],
        "model_id": extractor.model_id,
        "n_layers": extractor.n_layers,
        "d_model": extractor.d_model,
    }


def _default_components(*, with_extractor: bool = True):
    settings.ensure_dirs()
    db = Database(settings.db_path)
    store = ActivationStore(settings.hdf5_path, mode="a")
    generator = PerspectiveGenerator()
    extractor = ActivationExtractor() if with_extractor else None
    return db, store, generator, extractor
