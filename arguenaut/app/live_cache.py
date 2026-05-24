"""On-disk cache for live-query analyses so repeated probes don't re-call Lambda."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from arguenaut.app.lambda_client import LambdaAnalysis, LambdaPerspective
from arguenaut.config import settings
from arguenaut.utils import hypothesis_hash, normalise_hypothesis


_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_cache (
    hash       TEXT PRIMARY KEY,
    hypothesis TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    n_layers   INTEGER NOT NULL,
    d_model    INTEGER NOT NULL,
    payload    BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""


class LiveCache:
    """Trivial SQLite-backed cache. Payload is a JSON blob with the analysis."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or (settings.data_dir / "live_cache.sqlite"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.execute(_SCHEMA)

    def get(self, hypothesis: str) -> LambdaAnalysis | None:
        key = hypothesis_hash(hypothesis)
        row = self.conn.execute(
            "SELECT hypothesis, model_id, n_layers, d_model, payload FROM live_cache WHERE hash = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        hyp, model_id, n_layers, d_model, payload = row
        data = json.loads(payload)
        return LambdaAnalysis(
            hypothesis=hyp,
            model_id=model_id,
            n_layers=n_layers,
            d_model=d_model,
            perspectives=[
                LambdaPerspective(
                    stance=p["stance"],
                    text=p["text"],
                    position=p["position"],
                    last_token=np.asarray(p["last_token"], dtype=np.float32),
                    mean_pooled=np.asarray(p["mean_pooled"], dtype=np.float32),
                )
                for p in data["perspectives"]
            ],
        )

    def put(self, analysis: LambdaAnalysis) -> None:
        key = hypothesis_hash(analysis.hypothesis)
        payload = json.dumps(
            {
                "perspectives": [
                    {
                        "stance": p.stance,
                        "text": p.text,
                        "position": p.position,
                        "last_token": p.last_token.tolist(),
                        "mean_pooled": p.mean_pooled.tolist(),
                    }
                    for p in analysis.perspectives
                ]
            }
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO live_cache "
            "(hash, hypothesis, model_id, n_layers, d_model, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                normalise_hypothesis(analysis.hypothesis),
                analysis.model_id,
                analysis.n_layers,
                analysis.d_model,
                payload,
                time.time(),
            ),
        )

    def close(self) -> None:
        self.conn.close()
