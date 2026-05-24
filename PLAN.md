# Arguenaut - Project Plan

## Overview

An interpretability system that discovers the fundamental axes of disagreement in scientific hypotheses by analyzing the internal activations of Qwen3.5-4B-Base.

**Core flow:**

1. User enters a scientific hypothesis/question
2. Groq generates opposing perspectives
3. Qwen3.5-4B-Base (on Lambda) processes those texts, activations are extracted via nnsight
4. PCA projects activations onto discovered axes
5. Groq labels those axes in human-readable terms via a hypothesis-verification loop
6. Streamlit displays where the hypothesis falls in the space and which axes it activates

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Streamlit   │────▶│  Groq API        │     │  Lambda Labs (GPU)    │
│  Frontend    │     │  - Perspective   │     │  - Qwen3.5-4B-Base   │
│              │◀────│    generation     │     │  - nnsight hooks      │
│  - Plotly    │     │  - Axis labeling │     │  - Activation cache   │
│  - Controls  │     │  - Verification  │     │                       │
└─────────────┘     └──────────────────┘     └───────────────────────┘
       │                                              │
       └──────────────── shared storage ──────────────┘
                    (activations + metadata)
```

---

## Tech Stack

| Component              | Tool                              |
|------------------------|-----------------------------------|
| Model                  | Qwen/Qwen3.5-4B-Base (fp16)      |
| Activation extraction  | nnsight                           |
| Compute                | Lambda Labs, A10 24GB             |
| Perspective generation | Groq API (fast LLM inference)     |
| Axis labeling          | Groq API                          |
| Dimensionality reduction | scikit-learn (PCA)              |
| Visualization          | Plotly                            |
| Frontend               | Streamlit                         |
| Storage                | HDF5 (activations), SQLite (metadata) |

---

## Phase 1: Infrastructure & Activation Extraction

### 1.1 Project scaffolding

- Set up Python project with pyproject.toml
- Directory structure:

  ```
  arguenaut/
  ├── extraction/       # nnsight activation extraction
  ├── generation/       # Groq perspective generation
  ├── analysis/         # PCA, axis discovery
  ├── labeling/         # Hypothesis-verification loop
  ├── storage/          # HDF5 + SQLite helpers
  ├── app/              # Streamlit frontend
  └── config.py         # API keys, model paths, constants
  ```

- Environment: Python 3.11+, CUDA 12.x on Lambda

### 1.2 Groq perspective generation

- Input: a scientific hypothesis (e.g., "Neural scaling laws will hit diminishing returns before AGI")
- Output: 4-8 perspective texts representing different positions:
  - Strong agree, qualified agree, neutral/uncertain, qualified disagree, strong disagree
  - Each perspective is a 2-4 sentence argument
- Prompt engineering to get diverse, substantive positions (not just "I agree/disagree")
- Store perspectives in SQLite keyed by (topic_id, perspective_id)

### 1.3 Activation extraction with nnsight

- Load Qwen3.5-4B-Base on Lambda GPU via HuggingFace
- Wrap with nnsight's LanguageModel
- For each perspective text:
  - Run through the model under a `model.trace()` context
  - Extract residual stream output at every layer (32 layers)
  - Save the **last token** activation (shape: [32, 2560]) — this is the most informationally rich position
  - Also experiment with **mean-pooled** activations across the sequence
- Store activation tensors in HDF5: key = `{topic_id}/{perspective_id}`, value = [32, 2560] tensor
- Batch processing: process multiple perspectives per forward pass for efficiency

### 1.4 Validate the pipeline end-to-end

- Pick 3 test hypotheses manually
- Generate perspectives, extract activations, verify tensor shapes and values look reasonable
- Sanity check: activations should NOT be identical across perspectives (if they are, something is wrong)
- Check memory usage — 32 layers × 2560 dims × fp32 = ~320KB per perspective, very manageable

---

## Phase 2: PCA & Structure Discovery

### 2.1 Build the initial dataset

- Curate 30-50 scientific hypotheses spanning:
  - ML theory (scaling laws, generalization, inductive biases)
  - Math foundations (constructivism vs formalism, axiom of choice)
  - AI safety (alignment approaches, capability vs safety tradeoffs)
  - Scientific methodology (frequentist vs Bayesian, replication crisis)
- Generate 6 perspectives per hypothesis → ~200-300 activation vectors
- Use Groq to help generate the hypothesis list itself

### 2.2 PCA analysis

- Flatten activations: either analyze per-layer or concatenate across layers
- **Per-layer PCA:** Run PCA on the [N, 2560] matrix for each layer separately
  - This tells you which layers encode disagreement structure most cleanly
  - Pick the layer(s) with the best separation for downstream analysis
- **Cross-layer PCA:** Concatenate selected layers → [N, 2560 * k] matrix
  - Richer representation but higher dimensional
- Compute explained variance ratio — how many components capture meaningful structure?

### 2.3 Validate structure

- Plot perspectives in PC1-PC2 space, colored by topic
- Check: do opposing perspectives on the same topic separate?
- Check: do perspectives across different topics align along consistent axes?
- Quantitative: silhouette score or similar clustering metric on pro/con labels
- **DeltaNet vs Attention layers:** Compare PCA results from DeltaNet layers (layers 0-2, 4-6, 8-10, ...) vs Attention layers (3, 7, 11, ...) — the 3:1 hybrid architecture may show different structure in each type

### 2.4 Layer selection

- Rank layers by how well their PCA separates perspectives
- Likely candidates: mid-to-late layers (layers 16-28 in a 32-layer model)
- Select 1-3 "best" layers for the interpretability pipeline

---

## Phase 3: Automated Interpretability Pipeline

### 3.1 Axis hypothesis generation

- For each top principal component:
  1. Identify the 5 perspectives scoring **highest** along that direction
  2. Identify the 5 perspectives scoring **lowest**
  3. Send both sets to Groq with a prompt like:

     ```
     Here are scientific arguments that score high on a hidden axis:
     [high-scoring perspectives]

     Here are arguments that score low on the same axis:
     [low-scoring perspectives]

     What fundamental intellectual trade-off or dimension does this
     axis seem to represent? Name it as "X vs Y" and explain why.
     ```

  4. Groq returns a candidate label (e.g., "Empiricism vs Rationalism", "Optimism about scaling vs Skepticism about scaling")

### 3.2 Hypothesis verification

- For each candidate label:
  1. Ask Groq to generate 4 NEW perspective texts that should score high on the axis, and 4 that should score low
  2. Run these through Qwen3.5-4B-Base, extract activations
  3. Project onto the principal component
  4. Check: do the predicted-high texts actually score high? Do predicted-low texts score low?
  5. Compute a verification score: correlation between predicted and actual ranking

### 3.3 Iterative refinement

- If verification score is below threshold (e.g., r < 0.6):
  1. Show Groq the counterexamples (texts that violated the prediction)
  2. Ask for a refined hypothesis
  3. Re-verify
  4. Max 3 refinement rounds, then mark the axis as "unlabeled" if still failing
- Store final labels with confidence scores in SQLite

### 3.4 Full axis catalog

- Run the above for top-k principal components (start with k=5, expand if results are good)
- Store: axis index, label, confidence score, high/low example perspectives
- This becomes the "map" of the disagreement space

---

## Phase 4: Streamlit Frontend

### 4.1 Core views

**Hypothesis Explorer (main page):**

- Text input: user enters a scientific hypothesis
- System generates perspectives via Groq
- Extracts activations via API call to Lambda instance
- Projects onto discovered axes
- Shows:
  - 2D scatter plot (Plotly) with the new perspectives plotted among existing ones
  - Labeled axes showing which fundamental trade-offs this hypothesis activates
  - Bar chart of axis scores (how much each axis is activated)

**Axis Browser (sidebar/second page):**

- List of all discovered axes with labels and confidence scores
- Click an axis to see:
  - High-scoring and low-scoring example perspectives
  - Distribution of all perspectives along this axis
  - The verification score and refinement history

**Space Overview (third page):**

- Full 2D/3D scatter of all perspectives, colored by topic
- Interactive: hover to see perspective text, click to see details
- Toggle between different axis pairs (PC1-PC2, PC1-PC3, etc.)

### 4.2 Deployment architecture

**Option A — All local (simpler):**

- Pre-compute activations for the curated dataset on Lambda
- Download HDF5 file + SQLite database
- Streamlit app runs locally or on Streamlit Cloud
- For new hypotheses: make an API call to a simple FastAPI server running on Lambda

**Option B — Lambda as backend:**

- FastAPI server on Lambda handles activation extraction
- Streamlit app calls this API
- More seamless but burns Lambda credits while running

Recommendation: **Option A** for the demo. Pre-compute the base dataset, download it, run Streamlit locally. Only spin up Lambda when you need activations for new user-entered hypotheses.

---

## Phase 5: Polish & Edge Cases

- Handle short/ambiguous hypothesis inputs gracefully
- Cache activations so repeated queries don't re-extract
- Add loading states and progress indicators in Streamlit
- Error handling for Groq API rate limits
- Write a few-paragraph explanation of the methodology for the app's "About" section

---

## Build Order (What to do first)

1. **Set up Lambda instance**, install dependencies, verify Qwen3.5-4B-Base loads and nnsight can extract activations from it
2. **Build the Groq perspective generator**, test on 3 hypotheses
3. **Extract activations** for those 3 hypotheses, verify shapes
4. **Run PCA** on the small dataset, see if anything separates at all
5. **Scale to 30-50 hypotheses**, re-run PCA, validate structure
6. **Build the labeling loop** (hypothesis generation + verification via Groq)
7. **Build Streamlit app** with pre-computed data
8. **Add live query support** (API call to Lambda for new hypotheses)

---

## Risk Factors

- **Qwen3.5's hybrid architecture may complicate things.** The DeltaNet layers work differently from attention layers. If PCA on residual stream doesn't show clean separation, try: (a) only using attention layer outputs, (b) using the output of the full DeltaNet+Attention block rather than individual layers, (c) using a different model.
- **Perspectives may not separate in activation space.** The base model processes these texts without any stance-taking, so the activations reflect how the model *represents* the text, not how it *argues*. This is actually what we want (representation, not generation), but it's possible the representation is too shallow at 4B parameters. Mitigation: try the 9B model if 4B doesn't work.
- **PCA may not find interpretable axes.** The top principal components might capture surface-level variation (text length, vocabulary) rather than semantic disagreement structure. Mitigation: normalize activations, try CCA or sparse dictionary learning as alternatives to PCA.
- **Groq labeling may hallucinate plausible-sounding but wrong axis labels.** The verification loop is specifically designed to catch this, but it depends on the verification prompts being good tests.
