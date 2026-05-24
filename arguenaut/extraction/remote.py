"""HTTP-backed extractor that satisfies the same interface as ActivationExtractor.

Scripts can swap this in transparently: anything that calls
`.load()`, `.extract_batch(texts)`, `.n_layers`, `.d_model`, `.model_id`
keeps working — except it never touches torch.
"""

from __future__ import annotations

import logging
from typing import Sequence

import httpx
import numpy as np

from arguenaut.config import settings
from arguenaut.extraction.activations import ExtractionResult

logger = logging.getLogger(__name__)


class RemoteActivationExtractor:
    def __init__(
        self,
        api_url: str | None = None,
        api_token: str | None = None,
        timeout: float = 300.0,
    ):
        self.api_url = (api_url or settings.lambda_api_url).rstrip("/")
        self.api_token = api_token or settings.lambda_api_token
        self.timeout = timeout
        self._model_id: str | None = None
        self._n_layers: int | None = None
        self._d_model: int | None = None

    # ── lifecycle (mirrors ActivationExtractor) ────────────────────────────
    def load(self) -> None:
        if self._model_id is not None:
            return
        try:
            r = httpx.get(f"{self.api_url}/meta", headers=self._headers(), timeout=30.0)
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Could not reach Lambda backend at {self.api_url}: {e}. "
                "Run `arguenaut-lambda up` first or check LAMBDA_API_URL."
            ) from e
        if r.status_code != 200:
            raise RuntimeError(f"GET /meta → {r.status_code}: {r.text[:200]}")
        meta = r.json()
        self._model_id = meta["model_id"]
        self._n_layers = int(meta["n_layers"])
        self._d_model = int(meta["d_model"])
        logger.info(
            "Remote extractor ready: model=%s n_layers=%d d_model=%d",
            self._model_id, self._n_layers, self._d_model,
        )

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self.load()
        assert self._model_id is not None
        return self._model_id

    @property
    def n_layers(self) -> int:
        if self._n_layers is None:
            self.load()
        assert self._n_layers is not None
        return self._n_layers

    @property
    def d_model(self) -> int:
        if self._d_model is None:
            self.load()
        assert self._d_model is not None
        return self._d_model

    # ── extraction ─────────────────────────────────────────────────────────
    def extract(self, text: str, max_length: int = 512) -> ExtractionResult:
        return self.extract_batch([text], max_length=max_length)[0]

    def extract_batch(
        self, texts: Sequence[str], max_length: int = 512
    ) -> list[ExtractionResult]:
        if not texts:
            return []
        if self._model_id is None:
            self.load()
        try:
            r = httpx.post(
                f"{self.api_url}/extract",
                json={"texts": list(texts), "max_length": max_length},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"POST /extract failed: {e}") from e
        if r.status_code != 200:
            raise RuntimeError(f"POST /extract → {r.status_code}: {r.text[:200]}")
        body = r.json()
        results: list[ExtractionResult] = []
        for entry in body["results"]:
            results.append(
                ExtractionResult(
                    last_token=np.asarray(entry["last_token"], dtype=np.float32),
                    mean_pooled=np.asarray(entry["mean_pooled"], dtype=np.float32),
                    n_tokens=int(entry["n_tokens"]),
                )
            )
        return results

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_token:
            h["authorization"] = f"Bearer {self.api_token}"
        return h


def make_extractor(remote: bool):
    """Factory used by scripts: returns either a local or remote extractor.

    If `remote` is True we also try to read the URL/token from the live Lambda
    state file written by `arguenaut-lambda up`, falling back to LAMBDA_API_URL.
    """
    if not remote:
        from arguenaut.extraction.activations import ActivationExtractor
        return ActivationExtractor()

    from arguenaut.cloud.state import load_state

    api_url = settings.lambda_api_url
    state = load_state()
    if state is not None:
        api_url = state.api_url
    return RemoteActivationExtractor(api_url=api_url)
