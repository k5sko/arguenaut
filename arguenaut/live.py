"""Per-prompt live axis discovery.

This is the heart of the per-prompt design: given a single hypothesis, discover
the axes of disagreement *for that specific debate*, on the fly — no precomputed
corpus, no frozen global map.

Flow (runs from the laptop; only activation extraction touches the GPU):

    prompt
      │  PerspectiveGenerator.generate_many       (Groq, local)
      ▼  ~32 diverse perspectives
      │  extractor.extract_batch                  (remote /extract on Lambda GPU)
      ▼  [N, n_layers, d_model] activations
      │  run_per_layer_pca on one chosen layer    (local, scikit-learn)
      ▼  per-perspective PC scores + components
      │  label_axis from the high/low perspectives (Groq, local)
      ▼
    Discovery
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from arguenaut.analysis.pca import run_per_layer_pca
from arguenaut.generation.perspectives import PerspectiveGenerator

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredAxis:
    component_idx: int          # 0-based PC index
    label: str                  # "<HIGH POLE> vs <LOW POLE>"
    high_pole: str
    low_pole: str
    rationale: str
    explained_variance: float   # fraction of variance this PC captures
    high_examples: list[str]    # perspectives scoring highest on this axis
    low_examples: list[str]     # perspectives scoring lowest
    outlier_driven: bool = False  # variance dominated by 1-2 points (suspect axis)


@dataclass
class PerspectivePoint:
    stance: str                 # "lens · stance" tag
    text: str
    scores: list[float]         # projection onto each retained PC


@dataclass
class Discovery:
    prompt: str
    model_id: str
    layer: int
    n_layers: int
    perspectives: list[PerspectivePoint]
    axes: list[DiscoveredAxis]
    explained_variance_ratio: list[float]
    # quality-filtering provenance (how the perspective set was whittled down)
    n_generated: int = 0          # raw perspectives Groq produced
    n_dropped_judge: int = 0      # cut by the relevance/substance judge
    n_dropped_duplicate: int = 0  # cut as near-duplicates
    n_dropped_outlier: int = 0    # cut as activation-space outliers


def _pick_layer(n_layers: int, layer: int | None, layer_frac: float) -> int:
    if layer is not None:
        if not 0 <= layer < n_layers:
            raise IndexError(f"layer {layer} out of range [0,{n_layers})")
        return layer
    # Default to a mid-late layer (PLAN.md: layers ~16-28 of 32 separate best).
    return max(0, min(n_layers - 1, round(layer_frac * (n_layers - 1))))


def discover_axes_for_prompt(
    prompt: str,
    *,
    extractor,
    generator: PerspectiveGenerator | None = None,
    n_perspectives: int = 32,
    n_axes: int = 4,
    layer: int | None = None,
    layer_frac: float = 0.7,
    do_label: bool = True,
    do_judge: bool = True,
    do_filter: bool = True,
) -> Discovery:
    """Discover and label the axes of disagreement for a single prompt.

    `extractor` is any object satisfying the ActivationExtractor interface
    (`load()`, `extract_batch(texts)`, `.model_id`, `.n_layers`) — typically a
    RemoteActivationExtractor pointed at the live Lambda box.

    Quality controls (all on by default):
      do_judge  — oversample perspectives, then keep the best `n_perspectives`
                  by an LLM relevance/substance score.
      do_filter — drop near-duplicate and activation-outlier perspectives before
                  PCA, and flag components whose variance is outlier-driven.
    """
    import math

    from arguenaut.analysis.filtering import dedup_and_trim, flag_outlier_components

    generator = generator or PerspectiveGenerator()

    # 1. Generate — oversample (up to the lens×stance grid cap) when judging so
    #    we have headroom to discard the weakest.
    gen_n = min(40, math.ceil(1.5 * n_perspectives)) if do_judge else n_perspectives
    logger.info("Generating %d candidate perspectives for %r", gen_n, prompt)
    perspectives = generator.generate_many(prompt, n=gen_n)
    n_generated = len(perspectives)

    # 2. Judge → keep the best n_perspectives.
    n_dropped_judge = 0
    if do_judge and len(perspectives) > n_perspectives:
        kept = generator.judge_and_select(prompt, perspectives, keep_n=n_perspectives)
        n_dropped_judge = len(perspectives) - len(kept)
        perspectives = kept
        logger.info("Judge kept %d / %d perspectives", len(perspectives), n_generated)

    # 3. Extract activations.
    logger.info("Extracting activations for %d perspectives", len(perspectives))
    results = extractor.extract_batch([p.text for p in perspectives])
    acts = np.stack([r.last_token for r in results], axis=0)  # [N, n_layers, d_model]
    n, n_layers, _ = acts.shape
    layer = _pick_layer(n_layers, layer, layer_frac)

    # 4. Dedup + outlier trim on the analysis layer, before PCA.
    n_dropped_duplicate = n_dropped_outlier = 0
    if do_filter:
        report = dedup_and_trim(acts[:, layer, :])
        if report.n_kept < n:
            mask = report.kept
            acts = acts[mask]
            perspectives = [p for p, k in zip(perspectives, mask) if k]
            n = acts.shape[0]
            n_dropped_duplicate = report.n_duplicates
            n_dropped_outlier = report.n_outliers
            logger.info(
                "Filter dropped %d outliers + %d duplicates → %d perspectives",
                report.n_outliers, report.n_duplicates, n,
            )

    # 5. PCA on the cleaned set.
    n_comp = min(max(n_axes, 2), n - 1)
    keys = [(0, i) for i in range(n)]
    pca = run_per_layer_pca(acts, keys, layers=[layer], n_components=n_comp)[layer]

    points = [
        PerspectivePoint(stance=p.stance, text=p.text, scores=pca.scores[i].tolist())
        for i, p in enumerate(perspectives)
    ]

    # 6. Flag outlier-dominated components so a single point can't pose as an axis.
    outlier_pc = flag_outlier_components(pca.scores) if do_filter else [False] * pca.n_components

    axes: list[DiscoveredAxis] = []
    for c in range(min(n_axes, pca.n_components)):
        comp_scores = pca.scores[:, c]
        order = np.argsort(comp_scores)
        k = min(5, n // 2)
        low_idx = order[:k].tolist()
        high_idx = order[-k:][::-1].tolist()
        high_texts = [perspectives[i].text for i in high_idx]
        low_texts = [perspectives[i].text for i in low_idx]

        if do_label:
            try:
                proposal = label_axis(generator.client, generator.model, high_texts, low_texts)
            except Exception as e:  # labeling is best-effort; never sink the whole run
                logger.warning("Axis %d labeling failed: %s", c, e)
                proposal = _fallback_label(c)
        else:
            proposal = _fallback_label(c)

        axes.append(
            DiscoveredAxis(
                component_idx=c,
                label=proposal["label"],
                high_pole=proposal["high_pole"],
                low_pole=proposal["low_pole"],
                rationale=proposal.get("rationale", ""),
                explained_variance=float(pca.explained_variance_ratio[c]),
                high_examples=high_texts,
                low_examples=low_texts,
                outlier_driven=bool(outlier_pc[c]) if c < len(outlier_pc) else False,
            )
        )

    return Discovery(
        prompt=prompt,
        model_id=extractor.model_id,
        layer=layer,
        n_layers=n_layers,
        perspectives=points,
        axes=axes,
        explained_variance_ratio=pca.explained_variance_ratio.tolist(),
        n_generated=n_generated,
        n_dropped_judge=n_dropped_judge,
        n_dropped_duplicate=n_dropped_duplicate,
        n_dropped_outlier=n_dropped_outlier,
    )


def _fallback_label(component_idx: int) -> dict:
    return {
        "label": f"PC{component_idx + 1} (unlabelled)",
        "high_pole": "high",
        "low_pole": "low",
        "rationale": "",
    }


# ── lightweight axis labeling (no DB, no verification loop) ──────────────────
# Reuses the prompt design from the corpus labeler but stays stateless so it can
# run inside the live request path.

from arguenaut.labeling.axis_labeler import (  # noqa: E402
    HYPOTHESIZE_SYSTEM,
    HYPOTHESIZE_USER,
    _format_block,
    _parse_label_json,
)


def label_axis(client, model: str, high_texts: list[str], low_texts: list[str]) -> dict:
    """Ask Groq to name the axis separating HIGH from LOW perspectives.

    Returns {label, high_pole, low_pole, rationale}. Stateless — unlike the
    corpus AxisLabeler, this does no verification round-trips (which would need
    extra GPU extractions and make the live request slow)."""
    messages = [
        {"role": "system", "content": HYPOTHESIZE_SYSTEM},
        {
            "role": "user",
            "content": HYPOTHESIZE_USER.format(
                k=len(high_texts),
                high_block=_format_block(high_texts),
                low_block=_format_block(low_texts),
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return _parse_label_json(resp.choices[0].message.content or "")
