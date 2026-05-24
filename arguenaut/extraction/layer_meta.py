"""Layer metadata helpers — DeltaNet vs Attention separation per PLAN §2.3.

Qwen3.5 uses a 3:1 DeltaNet:Attention hybrid: every 4th layer is Attention,
the other three are DeltaNet. We don't always have introspection of which is
which, so we either:
  (a) infer from the loaded model's config / module class names, or
  (b) fall back to the 3:1 schedule (DeltaNet on 0,1,2,4,5,6,...; Attention on 3,7,11,...).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerKind:
    layer: int
    kind: str  # "attention" | "deltanet" | "unknown"


def infer_layer_kinds(model) -> list[LayerKind]:
    """Inspect block module classnames and tag each layer."""
    from arguenaut.extraction.activations import _resolve_layers

    try:
        blocks = _resolve_layers(model)
    except RuntimeError:
        return []
    kinds: list[LayerKind] = []
    for i, block in enumerate(blocks):
        cls = type(block).__name__.lower()
        # nnsight wraps modules; peek through if needed
        if hasattr(block, "_module"):
            cls = type(block._module).__name__.lower() or cls
        if "delta" in cls:
            kinds.append(LayerKind(i, "deltanet"))
        elif "attn" in cls or "attention" in cls:
            kinds.append(LayerKind(i, "attention"))
        else:
            # fall back to 3:1 schedule
            kinds.append(LayerKind(i, "attention" if (i + 1) % 4 == 0 else "deltanet"))
    return kinds


def schedule_3to1(n_layers: int) -> list[LayerKind]:
    """Pure 3:1 schedule fallback when no model is loaded."""
    return [LayerKind(i, "attention" if (i + 1) % 4 == 0 else "deltanet") for i in range(n_layers)]


def attention_layers(kinds: list[LayerKind]) -> list[int]:
    return [k.layer for k in kinds if k.kind == "attention"]


def deltanet_layers(kinds: list[LayerKind]) -> list[int]:
    return [k.layer for k in kinds if k.kind == "deltanet"]
