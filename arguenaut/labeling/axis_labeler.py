"""Automated axis-labeling loop (PLAN.md Phase 3).

For a given (layer, PC) pair:
  1. Pull the 5 perspectives that score highest along the component, and the 5 lowest.
  2. Send both groups to Groq, ask for "X vs Y" plus per-pole descriptions.
  3. Verification: ask Groq to write 4 new texts for each pole. Extract their
     activations, project onto the PCA component, compute Spearman correlation
     between predicted polarity and actual projection.
  4. If r < threshold, show counterexamples back to Groq for a refined hypothesis.
     Up to `max_refinements` rounds, then mark "unlabeled" if still failing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from arguenaut.analysis.pca import PCAResult
from arguenaut.analysis.scoring import correlation_score
from arguenaut.config import settings
from arguenaut.extraction import ActivationExtractor
from arguenaut.generation.perspectives import TargetedPerspectiveGenerator
from arguenaut.storage import Database

logger = logging.getLogger(__name__)


HYPOTHESIZE_SYSTEM = """You are an interpretability researcher. You will be shown two groups of
scientific arguments. Group HIGH all score high on a hidden axis in a language model's residual
stream. Group LOW all score low on the same axis. Your job is to NAME this axis as a
trade-off between two intellectual positions: "<HIGH POLE> vs <LOW POLE>". Be specific —
"specific vs general" is too weak; "constructive proof vs non-constructive existence" is good.

Return STRICT JSON:
{
  "label": "<HIGH POLE> vs <LOW POLE>",
  "high_pole": "<short noun phrase capturing what HIGH arguments share>",
  "low_pole":  "<short noun phrase capturing what LOW  arguments share>",
  "rationale": "<one sentence explaining the contrast>"
}
"""


HYPOTHESIZE_USER = """HIGH group (top {k} along this axis):
{high_block}

LOW group (bottom {k} along this axis):
{low_block}

What axis does this represent? Return JSON."""


REFINE_SYSTEM = """You previously proposed an axis label that did not survive verification —
new texts you generated for one pole landed on the wrong side of the axis. Use these
counterexamples to refine your hypothesis. The axis IS real; you just named it wrong.

Return STRICT JSON with the same shape as before: label, high_pole, low_pole, rationale."""


REFINE_USER = """Previous label: {prev_label}
Previous high_pole: {prev_high}
Previous low_pole:  {prev_low}
Verification correlation: {score:+.2f} (need {threshold:+.2f})

These texts were generated FOR THE HIGH POLE but actually scored LOW:
{wrong_high}

These texts were generated FOR THE LOW POLE but actually scored HIGH:
{wrong_low}

Original training-set examples for context:
HIGH group:
{high_block}

LOW group:
{low_block}

Propose a refined label. Return JSON."""


@dataclass
class AxisLabelingRound:
    round_idx: int
    label: str
    high_pole: str
    low_pole: str
    rationale: str
    verification_score: float
    counterexamples: dict = field(default_factory=dict)


@dataclass
class AxisLabelingResult:
    layer: int
    component_idx: int
    final_label: str | None
    final_high_pole: str | None
    final_low_pole: str | None
    final_score: float
    rounds: list[AxisLabelingRound]
    accepted: bool


class AxisLabeler:
    def __init__(
        self,
        db: Database,
        extractor: ActivationExtractor,
        groq_client=None,
        model: str | None = None,
        threshold: float = settings.axis_verification_threshold,
        max_refinements: int = settings.axis_max_refinements,
        n_verify_per_pole: int = 4,
        top_k_examples: int = 5,
    ):
        self.db = db
        self.extractor = extractor
        self.threshold = threshold
        self.max_refinements = max_refinements
        self.n_verify_per_pole = n_verify_per_pole
        self.top_k_examples = top_k_examples
        self.targeted_gen = TargetedPerspectiveGenerator(client=groq_client, model=model)
        self.client = self.targeted_gen.client
        self.model = self.targeted_gen.model

    # ── label one component end-to-end ─────────────────────────────────────
    def label_component(self, pca: PCAResult, component_idx: int) -> AxisLabelingResult:
        layer = pca.layer if pca.layer is not None else -1
        # Top-k and bottom-k perspective texts along the component.
        scores = pca.scores[:, component_idx]
        order = np.argsort(scores)
        low_idx = order[: self.top_k_examples].tolist()
        high_idx = order[-self.top_k_examples:][::-1].tolist()
        high_texts = [self._lookup_perspective_text(pca.keys[i]) for i in high_idx]
        low_texts = [self._lookup_perspective_text(pca.keys[i]) for i in low_idx]

        rounds: list[AxisLabelingRound] = []
        prev = None
        wrong_high: list[str] = []
        wrong_low: list[str] = []
        for round_idx in range(self.max_refinements + 1):
            if round_idx == 0:
                proposal = self._hypothesize(high_texts, low_texts)
            else:
                proposal = self._refine(
                    prev_label=prev["label"],
                    prev_high=prev["high_pole"],
                    prev_low=prev["low_pole"],
                    score=prev["score"],
                    high_block=_format_block(high_texts),
                    low_block=_format_block(low_texts),
                    wrong_high=wrong_high,
                    wrong_low=wrong_low,
                )

            score, wrong_high, wrong_low = self._verify(
                pca=pca,
                component_idx=component_idx,
                label=proposal["label"],
                high_pole=proposal["high_pole"],
                low_pole=proposal["low_pole"],
            )
            rounds.append(
                AxisLabelingRound(
                    round_idx=round_idx,
                    label=proposal["label"],
                    high_pole=proposal["high_pole"],
                    low_pole=proposal["low_pole"],
                    rationale=proposal.get("rationale", ""),
                    verification_score=score,
                    counterexamples={"wrong_high": wrong_high, "wrong_low": wrong_low},
                )
            )
            prev = {**proposal, "score": score}
            if score >= self.threshold:
                break

        best = max(rounds, key=lambda r: r.verification_score)
        accepted = best.verification_score >= self.threshold

        # Persist into DB.
        axis_id = self.db.upsert_axis(
            layer=layer,
            component_idx=component_idx,
            explained_var=float(pca.explained_variance_ratio[component_idx]),
            label=best.label if accepted else None,
            high_pole=best.high_pole if accepted else None,
            low_pole=best.low_pole if accepted else None,
            confidence=best.verification_score,
            refinement_rounds=len(rounds) - 1,
        )
        for r in rounds:
            self.db.add_axis_verification(
                axis_id=axis_id,
                round_=r.round_idx,
                candidate=r.label,
                score=r.verification_score,
                detail={
                    "high_pole": r.high_pole,
                    "low_pole": r.low_pole,
                    "rationale": r.rationale,
                    "counterexamples": r.counterexamples,
                },
            )

        return AxisLabelingResult(
            layer=layer,
            component_idx=component_idx,
            final_label=best.label if accepted else None,
            final_high_pole=best.high_pole if accepted else None,
            final_low_pole=best.low_pole if accepted else None,
            final_score=best.verification_score,
            rounds=rounds,
            accepted=accepted,
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _lookup_perspective_text(self, key: tuple[int, int]) -> str:
        _, pid = key
        p = self.db.get_perspective(pid)
        return p.text if p is not None else "<missing>"

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
    def _chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _hypothesize(self, high_texts: list[str], low_texts: list[str]) -> dict:
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
        return _parse_label_json(self._chat(messages))

    def _refine(
        self,
        *,
        prev_label: str,
        prev_high: str,
        prev_low: str,
        score: float,
        high_block: str,
        low_block: str,
        wrong_high: list[str],
        wrong_low: list[str],
    ) -> dict:
        messages = [
            {"role": "system", "content": REFINE_SYSTEM},
            {
                "role": "user",
                "content": REFINE_USER.format(
                    prev_label=prev_label,
                    prev_high=prev_high,
                    prev_low=prev_low,
                    score=score,
                    threshold=self.threshold,
                    wrong_high=_format_block(wrong_high) or "(none)",
                    wrong_low=_format_block(wrong_low) or "(none)",
                    high_block=high_block,
                    low_block=low_block,
                ),
            },
        ]
        return _parse_label_json(self._chat(messages))

    def _verify(
        self,
        *,
        pca: PCAResult,
        component_idx: int,
        label: str,
        high_pole: str,
        low_pole: str,
    ) -> tuple[float, list[str], list[str]]:
        """Generate fresh texts, run them through the model, project, score."""
        high_new = self.targeted_gen.generate_for_pole(
            label=label, high_pole=high_pole, low_pole=low_pole,
            target_pole="HIGH", n=self.n_verify_per_pole,
        )
        low_new = self.targeted_gen.generate_for_pole(
            label=label, high_pole=high_pole, low_pole=low_pole,
            target_pole="LOW", n=self.n_verify_per_pole,
        )
        all_texts = high_new + low_new
        predicted = np.array([+1] * len(high_new) + [-1] * len(low_new), dtype=np.int32)

        results = self.extractor.extract_batch(all_texts)
        if pca.layer is None:
            raise ValueError("verification expects per-layer PCA, not cross-layer")
        feats = np.stack([r.last_token[pca.layer] for r in results], axis=0)
        projections = pca.project(feats)[:, component_idx]

        score = correlation_score(predicted, projections)

        # Identify counterexamples for the next refinement round.
        wrong_high: list[str] = []
        wrong_low: list[str] = []
        # We expected high_new to project positively. If they came out negative, they're wrong.
        for txt, proj in zip(high_new, projections[: len(high_new)]):
            if proj < 0:
                wrong_high.append(f"(actual={proj:+.2f}) {txt}")
        for txt, proj in zip(low_new, projections[len(high_new):]):
            if proj > 0:
                wrong_low.append(f"(actual={proj:+.2f}) {txt}")

        return score, wrong_high, wrong_low


# ── parsing helpers ──────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_label_json(raw: str) -> dict:
    text = _JSON_FENCE_RE.sub("", raw).strip()
    obj = json.loads(text)
    for key in ("label", "high_pole", "low_pole"):
        if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"Bad axis-label response, missing/empty {key!r}: {obj!r}")
    obj.setdefault("rationale", "")
    return obj


def _format_block(texts: list[str]) -> str:
    return "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(texts))
