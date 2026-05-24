"""About the methodology — written for the Streamlit "About" page (PLAN.md §5)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Arguenaut — about", layout="centered")

st.title("About Arguenaut")

st.markdown(
    """
**Goal.** Given a scientific hypothesis, locate it in a space whose axes are the
fundamental intellectual trade-offs that real disagreements travel along — not
"agree vs disagree" sliders, but things like *constructive proof vs non-constructive
existence*, *gene-level selection vs group-level selection*, *capabilities-driven
safety vs interpretability-driven safety*.

**Method.**

1. **Perspectives.** A fast LLM (Groq) takes the hypothesis and produces 4–8
   short stance texts spanning the agree/disagree spectrum, including an
   "orthogonal reframe" that rejects the question's framing entirely.

2. **Activations.** Each perspective is fed through a small base LM
   (Qwen2.5-3B / Qwen3.5-4B-Base) on a Lambda GPU. We capture the residual-stream
   hidden state at every transformer block using `nnsight`. The last-token
   activation at each layer becomes a length-2560 vector summarising how the model
   represents that argument.

3. **Structure discovery.** We stack all those vectors into a matrix and run PCA
   per layer. Mid-to-late layers usually show the cleanest separation; we pick
   one or a few of them. Each principal component is then a candidate "axis of
   disagreement".

4. **Axis labelling.** For each top component we send the highest-scoring and
   lowest-scoring perspectives to Groq with the prompt: *"What hidden axis are
   these contrasting on?"*. Groq proposes a label of the form "HIGH pole vs LOW pole".

5. **Verification.** Groq then writes 4 NEW arguments for each pole. We extract
   their activations and project them onto the same component. If the predicted-high
   texts genuinely score high and the predicted-low texts score low (Spearman r ≥ 0.6),
   the label is accepted. Otherwise the counterexamples are fed back to Groq for
   refinement (max 3 rounds, then "unlabelled").

**Why a base model.** Instruction-tuned models inject their own stance into the
representation. A base model just *represents* the argument, which is what we
want when probing for structure in how disagreement is encoded.

**What this is not.** This is not an opinion-mining system or a fact-checker.
It's a microscope on how a base LM internally organises scientific contention.

**Risks.** PCA may capture surface-level variation (text length, vocabulary) before
semantic structure. The verification loop catches some plausible-but-wrong labels,
but a high verification score is necessary, not sufficient, for the label to be
*the* right way to describe the axis.

— *Source / replication: see `PLAN.md` and `README.md` in the repo.*
    """
)
