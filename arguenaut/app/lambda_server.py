"""FastAPI server that runs on the Lambda A10. Exposes endpoints for live queries
and bulk activation extraction.

Endpoints:
  GET  /health    — liveness check (does not count against idle timer)
  GET  /meta      — model id, n_layers, d_model, idle stats
  POST /analyze   — hypothesis → perspectives → activations (full pipeline)
  POST /extract   — texts → activations (skip generation; used by remote extractor)

Start with:
    python -m arguenaut.app.lambda_server          # uvicorn on :8000
or:
    uvicorn arguenaut.app.lambda_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from arguenaut.cloud.auto_shutdown import maybe_install
from arguenaut.config import settings
from arguenaut.extraction import ActivationExtractor
from arguenaut.generation import PerspectiveGenerator
from arguenaut.pipeline import process_single_hypothesis

logger = logging.getLogger(__name__)

app = FastAPI(title="Arguenaut Lambda backend", version="0.1.0")

# Module-level singletons so the (heavy) model loads once per process.
_extractor: ActivationExtractor | None = None
_generator: PerspectiveGenerator | None = None
_watcher = None


class AnalyzeRequest(BaseModel):
    hypothesis: str = Field(..., min_length=4, max_length=2000)


class ExtractRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    max_length: int = Field(default=512, ge=8, le=2048)


def _check_auth(auth_header: str | None) -> None:
    expected = settings.lambda_api_token
    if not expected:
        return  # auth disabled
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "missing Bearer token")
    if auth_header.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(401, "bad Bearer token")


@app.on_event("startup")
def _warm() -> None:
    global _extractor, _generator, _watcher
    logger.info("Warming up: loading model %s", settings.hf_model_id)
    _extractor = ActivationExtractor()
    _extractor.load()
    _generator = PerspectiveGenerator()
    _watcher = maybe_install(app, settings)
    logger.info("Ready.")


@app.on_event("shutdown")
def _shutdown() -> None:
    if _watcher is not None:
        _watcher.stop()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_id": settings.hf_model_id,
        "loaded": _extractor is not None and _extractor._lm is not None,
    }


@app.get("/meta")
def meta() -> dict:
    if _extractor is None or _extractor._lm is None:
        raise HTTPException(503, "model not loaded yet")
    return {
        "model_id": _extractor.model_id,
        "n_layers": _extractor.n_layers,
        "d_model": _extractor.d_model,
        "idle_seconds": _watcher._seconds_idle() if _watcher else None,
        "auto_shutdown_minutes": settings.lambda_auto_shutdown_minutes,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest, authorization: str | None = Header(default=None)) -> dict:
    _check_auth(authorization)
    if _extractor is None or _generator is None:
        raise HTTPException(503, "server still warming up")
    return process_single_hypothesis(
        req.hypothesis, generator=_generator, extractor=_extractor
    )


@app.post("/extract")
def extract(req: ExtractRequest, authorization: str | None = Header(default=None)) -> dict:
    """Batch-extract activations for arbitrary texts. Used by RemoteActivationExtractor."""
    _check_auth(authorization)
    if _extractor is None:
        raise HTTPException(503, "server still warming up")
    results = _extractor.extract_batch(req.texts, max_length=req.max_length)
    return {
        "model_id": _extractor.model_id,
        "n_layers": _extractor.n_layers,
        "d_model": _extractor.d_model,
        "results": [
            {
                "last_token": r.last_token.tolist(),
                "mean_pooled": r.mean_pooled.tolist(),
                "n_tokens": r.n_tokens,
            }
            for r in results
        ],
    }


def main() -> None:
    import uvicorn

    host = os.environ.get("ARGUENAUT_HOST", "0.0.0.0")
    port = int(os.environ.get("ARGUENAUT_PORT", "8000"))
    uvicorn.run("arguenaut.app.lambda_server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
