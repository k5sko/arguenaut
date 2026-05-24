"""Groq-backed perspective generation.

Given a scientific hypothesis, returns N stance texts spanning the
agree/disagree spectrum. Each perspective is 2–4 sentences and argues
substantively rather than just declaring a position.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from arguenaut.config import settings

logger = logging.getLogger(__name__)


# Six stances along the agree/disagree spectrum (default N=6 in PLAN §1.2).
DEFAULT_STANCES: tuple[str, ...] = (
    "strong_agree",
    "qualified_agree",
    "neutral_uncertain",
    "qualified_disagree",
    "strong_disagree",
    "orthogonal_reframe",  # rejects the framing rather than picking a side
)


SYSTEM_PROMPT = """You are a panel of scientists and philosophers responding to a hypothesis.
For each requested stance you produce ONE 2–4 sentence argument that genuinely embodies that stance —
not a meta-comment, not a hedge, not just "I agree". Cite mechanisms, evidence, or theoretical
commitments. Different stances should disagree about substantive things, not just adjust adjectives.

Return STRICT JSON with this shape — no prose, no markdown fences:
{"perspectives": [{"stance": "<stance>", "text": "<argument>"}, ...]}
"""


USER_PROMPT_TEMPLATE = """Hypothesis: {hypothesis}

Generate one perspective for EACH of these stances, in order:
{stance_list}

Stance meanings:
- strong_agree: fully endorse the claim and explain the strongest reason it must be true.
- qualified_agree: agree but only under specific conditions; name them.
- neutral_uncertain: argue we genuinely cannot tell yet, and what evidence would resolve it.
- qualified_disagree: think it is likely wrong but acknowledge a real partial case for it.
- strong_disagree: argue the claim is clearly false; explain the central reason.
- orthogonal_reframe: reject the framing of the question itself; propose a better question.

Each text should be a self-contained 2–4 sentence argument. No headers, no labels inside the text.
"""


@dataclass
class Perspective:
    stance: str
    text: str
    position: int  # 0..N-1, matches order in DEFAULT_STANCES


class PerspectiveGenerationError(RuntimeError):
    pass


class PerspectiveGenerator:
    def __init__(
        self,
        client=None,
        model: str | None = None,
        stances: tuple[str, ...] = DEFAULT_STANCES,
    ):
        self.stances = stances
        self.model = model or settings.groq_model
        if client is None:
            try:
                from groq import Groq
            except ImportError as e:
                raise ImportError("`pip install groq` is required for PerspectiveGenerator") from e
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set; cannot construct PerspectiveGenerator")
            client = Groq(api_key=settings.groq_api_key)
        self.client = client

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
    def _call(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def generate(self, hypothesis: str) -> list[Perspective]:
        """Return one Perspective per stance, in stance order.

        Raises PerspectiveGenerationError if Groq returns malformed JSON or
        omits stances after retries.
        """
        if not hypothesis or not hypothesis.strip():
            raise ValueError("hypothesis must be non-empty")

        stance_list = "\n".join(f"  - {s}" for s in self.stances)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    hypothesis=hypothesis.strip(),
                    stance_list=stance_list,
                ),
            },
        ]
        raw = self._call(messages)
        parsed = _parse_perspective_json(raw, expected_stances=self.stances)
        return parsed


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_perspective_json(raw: str, expected_stances: tuple[str, ...]) -> list[Perspective]:
    text = _JSON_FENCE_RE.sub("", raw).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise PerspectiveGenerationError(f"Groq returned non-JSON: {raw[:300]!r}") from e

    items = obj.get("perspectives") if isinstance(obj, dict) else None
    if not isinstance(items, list) or not items:
        raise PerspectiveGenerationError(f"Expected JSON with 'perspectives' list, got: {obj!r}")

    by_stance = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        stance = str(item.get("stance", "")).strip()
        body = str(item.get("text", "")).strip()
        if stance and body:
            by_stance[stance] = body

    out: list[Perspective] = []
    missing: list[str] = []
    for pos, stance in enumerate(expected_stances):
        if stance not in by_stance:
            missing.append(stance)
            continue
        out.append(Perspective(stance=stance, text=by_stance[stance], position=pos))

    if missing:
        raise PerspectiveGenerationError(
            f"Missing stances from Groq response: {missing}. Got: {list(by_stance.keys())}"
        )
    return out


# ── ad-hoc generation used by the labeling loop ──────────────────────────────

LABEL_VERIFY_SYSTEM = """You generate scientific arguments designed to test an axis of disagreement.
You will receive a description of a hypothesized axis and must produce arguments that should land
at one specific pole of that axis. Each argument is a 2–4 sentence self-contained claim.

Return STRICT JSON: {"texts": ["<argument>", "<argument>", ...]}  — no prose, no markdown."""


LABEL_VERIFY_USER = """Hypothesized axis: {label}
The HIGH pole is: {high_pole}
The LOW  pole is: {low_pole}

Generate {n} arguments that should land at the {target_pole} pole of this axis.
They should be diverse — not paraphrases — and address different scientific domains where possible."""


class TargetedPerspectiveGenerator(PerspectiveGenerator):
    """Generates fresh perspectives intended to land at a specified pole of an axis.

    Used by the labeling verification loop (Phase 3.2).
    """

    def generate_for_pole(
        self,
        label: str,
        high_pole: str,
        low_pole: str,
        target_pole: str,  # "HIGH" or "LOW"
        n: int = 4,
    ) -> list[str]:
        if target_pole not in {"HIGH", "LOW"}:
            raise ValueError("target_pole must be 'HIGH' or 'LOW'")
        messages = [
            {"role": "system", "content": LABEL_VERIFY_SYSTEM},
            {
                "role": "user",
                "content": LABEL_VERIFY_USER.format(
                    label=label,
                    high_pole=high_pole,
                    low_pole=low_pole,
                    target_pole=target_pole,
                    n=n,
                ),
            },
        ]
        raw = self._call(messages)
        obj = json.loads(_JSON_FENCE_RE.sub("", raw).strip())
        texts = obj.get("texts") if isinstance(obj, dict) else None
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise PerspectiveGenerationError(f"Bad targeted-generation response: {obj!r}")
        return [t.strip() for t in texts if t.strip()]
