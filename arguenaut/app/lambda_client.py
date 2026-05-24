"""HTTP client for the FastAPI activation server running on Lambda.

Used by the Streamlit app's live-query path. If the Lambda server is
unreachable, callers should catch LambdaUnreachable and fall back to a
read-only "explore existing dataset" UX.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np

from arguenaut.config import settings

logger = logging.getLogger(__name__)


class LambdaUnreachable(RuntimeError):
    pass


@dataclass
class LambdaPerspective:
    stance: str
    text: str
    position: int
    last_token: np.ndarray   # [n_layers, d_model]
    mean_pooled: np.ndarray  # [n_layers, d_model]


@dataclass
class LambdaAnalysis:
    hypothesis: str
    perspectives: list[LambdaPerspective]
    model_id: str
    n_layers: int
    d_model: int


class LambdaClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or settings.lambda_api_url).rstrip("/")
        self.token = token or settings.lambda_api_token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.token:
            h["authorization"] = f"Bearer {self.token}"
        return h

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", headers=self._headers(), timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def analyze(self, hypothesis: str) -> LambdaAnalysis:
        try:
            r = httpx.post(
                f"{self.base_url}/analyze",
                json={"hypothesis": hypothesis},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise LambdaUnreachable(str(e)) from e
        if r.status_code != 200:
            raise LambdaUnreachable(f"{r.status_code}: {r.text[:200]}")
        data = r.json()
        return LambdaAnalysis(
            hypothesis=data["hypothesis"],
            model_id=data["model_id"],
            n_layers=data["n_layers"],
            d_model=data["d_model"],
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
