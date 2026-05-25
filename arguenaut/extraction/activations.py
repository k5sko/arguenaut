"""nnsight-based residual-stream activation extraction.

Designed to run on the Lambda A10 (24GB) box. Loads a base LM once and
streams text through it under `model.trace`, capturing the residual stream
output of every transformer block. Returns last-token and mean-pooled
summaries per layer.

Qwen2.5/3.5 base architecture exposes blocks at `model.model.layers[i]`
and each layer returns a tuple where index 0 is the residual-stream
hidden state of shape [batch, seq, d_model]. We hook the *output* of
each block (= residual stream after the block has written to it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from arguenaut.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Per-text result.

    last_token  : [n_layers, d_model]   activation at the final non-pad token
    mean_pooled : [n_layers, d_model]   mean over non-pad tokens
    n_tokens    : int                   tokens actually attended to
    """

    last_token: np.ndarray
    mean_pooled: np.ndarray
    n_tokens: int


class ActivationExtractor:
    """Wraps a HuggingFace base LM via nnsight.

    Typical usage on Lambda:
        ex = ActivationExtractor()
        ex.load()
        for text in texts:
            res = ex.extract(text)
    """

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str = "auto",
        dtype: str = "float16",
    ):
        self.model_id = model_id or settings.hf_model_id
        self.device_map = device_map
        self.dtype = dtype
        self._lm = None
        self._tokenizer = None
        self._n_layers: int | None = None
        self._d_model: int | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._lm is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ActivationExtractor needs `torch` and `transformers`. "
                "Install with `pip install -e '.[gpu]'`."
            ) from e

        torch_dtype = getattr(torch, self.dtype)
        logger.info("Loading %s (dtype=%s, device_map=%s)", self.model_id, self.dtype, self.device_map)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # transformers >=5 renamed `torch_dtype` to `dtype`; support both.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device_map, dtype=torch_dtype
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device_map, torch_dtype=torch_dtype
            )
        model.eval()
        self._lm = model

        # Inspect architecture for n_layers / d_model.
        cfg = model.config
        self._n_layers = int(getattr(cfg, "num_hidden_layers"))
        self._d_model = int(getattr(cfg, "hidden_size"))
        logger.info("Model loaded: %d layers, d_model=%d", self._n_layers, self._d_model)

    @property
    def n_layers(self) -> int:
        if self._n_layers is None:
            raise RuntimeError("Call .load() before reading n_layers")
        return self._n_layers

    @property
    def d_model(self) -> int:
        if self._d_model is None:
            raise RuntimeError("Call .load() before reading d_model")
        return self._d_model

    # ── extraction ─────────────────────────────────────────────────────────
    def extract(self, text: str, max_length: int = 512) -> ExtractionResult:
        """Single-text convenience wrapper around extract_batch."""
        results = self.extract_batch([text], max_length=max_length)
        return results[0]

    def extract_batch(
        self, texts: Sequence[str], max_length: int = 512
    ) -> list[ExtractionResult]:
        """Run a batch of texts through the model with output_hidden_states,
        returning per-text last-token + mean-pooled activations across every layer.

        Uses plain HuggingFace forward passes (not nnsight) under
        torch.inference_mode — `output_hidden_states` exposes the residual stream
        after each block, which is exactly what we want, and is stable across
        transformers/torch versions."""
        if self._lm is None:
            self.load()
        assert self._lm is not None and self._tokenizer is not None

        import torch

        if not texts:
            return []

        enc = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attn_mask = enc["attention_mask"]  # keep CPU copy for pooling
        device = next(self._lm.parameters()).device
        input_ids = enc["input_ids"].to(device)

        with torch.inference_mode():
            out = self._lm(
                input_ids=input_ids,
                attention_mask=attn_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )

        # hidden_states is a tuple of length n_layers+1: [0] is the embedding
        # output, [1:] are the residual stream after each transformer block.
        # Move each layer to CPU individually to keep peak GPU memory low.
        layer_acts = [
            h.detach().to(torch.float32).cpu().numpy() for h in out.hidden_states[1:]
        ]
        del out
        # Stack to [n_layers, batch, seq, d_model]
        stacked = np.stack(layer_acts, axis=0)
        del layer_acts

        results: list[ExtractionResult] = []
        attn_np = attn_mask.numpy().astype(bool)  # [batch, seq]
        for i in range(stacked.shape[1]):
            mask_i = attn_np[i]                              # [seq]
            n_tok = int(mask_i.sum())
            if n_tok == 0:
                raise ValueError(f"Text {i} produced zero tokens")
            # Robust to either padding side: the last attended position is the
            # rightmost index where the attention mask is True.
            last_idx = int(np.flatnonzero(mask_i)[-1])
            last = stacked[:, i, last_idx, :]                # [n_layers, d_model]
            seq = stacked[:, i, mask_i, :]                   # [n_layers, n_tok, d_model]
            mean = seq.mean(axis=1)                          # [n_layers, d_model]
            results.append(ExtractionResult(last_token=last, mean_pooled=mean, n_tokens=n_tok))

        return results
