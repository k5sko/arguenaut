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
            from nnsight import LanguageModel
        except ImportError as e:
            raise ImportError(
                "ActivationExtractor needs `nnsight` and `torch`. "
                "Install with `pip install -e '.[gpu]'`."
            ) from e

        torch_dtype = getattr(torch, self.dtype)
        logger.info("Loading %s (dtype=%s, device_map=%s) via nnsight", self.model_id, self.dtype, self.device_map)
        self._lm = LanguageModel(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            dispatch=True,
        )
        self._tokenizer = self._lm.tokenizer
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Inspect architecture for n_layers / d_model.
        cfg = self._lm.config
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
        """Run a batch of texts through the model under nnsight.trace, returning
        per-text last-token + mean-pooled activations across every layer."""
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
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # Resolve the layer modules. Qwen / Llama / Mistral all expose
        # `model.model.layers[i]`; fall back to common alternatives if needed.
        layers = _resolve_layers(self._lm)
        if len(layers) != self._n_layers:
            logger.warning(
                "Resolved %d layer modules but config says %d", len(layers), self._n_layers
            )

        with self._lm.trace(input_ids, attention_mask=attn_mask) as tracer:  # noqa: F841
            saved = []
            for layer in layers:
                # Block output is a tuple — first element is the hidden state.
                out = layer.output
                # `out` may be a tuple proxy; index [0] reliably.
                hidden = out[0]
                saved.append(hidden.save())

        # After the trace exits, .value holds the resolved tensor on CPU.
        # Shape per layer: [batch, seq, d_model]
        layer_acts = [s.value.detach().to(torch.float32).cpu().numpy() for s in saved]
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


def _resolve_layers(lm) -> list:
    """Find the list of transformer blocks across HF families.

    Tries the common attribute paths used by Qwen, Llama, Mistral, Falcon, GPT-NeoX, GPT-2.
    """
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("transformer", "blocks"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        node = lm
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            try:
                # nnsight modules behave like sequences
                return [node[i] for i in range(len(node))]
            except TypeError:
                continue
    raise RuntimeError("Could not locate transformer block list on this model")
