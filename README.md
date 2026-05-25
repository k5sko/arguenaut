# Arguenaut

An interpretability system that discovers the fundamental axes of disagreement in scientific hypotheses by analyzing the internal activations of a small base LM (Qwen2.5-3B / Qwen3.5-4B-Base).

## Pipeline (per-prompt, on the fly)

Each prompt discovers its **own** axes of disagreement — there is no precomputed
corpus or frozen global map.

```
one hypothesis
   │
   ▼
Groq generates ~32 diverse perspectives (lens × stance grid)
   │
   ▼
Qwen base model (on Lambda) processes each under nnsight.trace
   │
   ▼
Last-token residual-stream activations at every layer
   │
   ▼
PCA over the [~32, d_model] matrix at one mid-late layer → top components
   │
   ▼
Groq labels each component "X vs Y" from its high/low perspectives
   │
   ▼
Streamlit plots the perspectives along the discovered axes
```

The orchestration lives in `arguenaut/live.py` (`discover_axes_for_prompt`); the
Streamlit main page is the whole UX. The batch corpus scripts (`arguenaut-extract`,
`-pca`, `-label-axes`) still exist for optional global-map analysis but are not
needed for the per-prompt flow.

## Layout

```
arguenaut/
├── arguenaut/
│   ├── config.py             # env-driven settings
│   ├── generation/           # Groq perspective generation
│   ├── extraction/           # nnsight activation extraction
│   ├── analysis/             # PCA, axis discovery, scoring
│   ├── labeling/             # axis-label hypothesize/verify loop
│   ├── storage/              # HDF5 + SQLite helpers
│   ├── app/                  # Streamlit + FastAPI Lambda server
│   └── scripts/              # CLI entry points
├── data/
│   ├── hypotheses.json       # seed dataset (Phase 2.1)
│   ├── arguenaut.sqlite      # perspectives, axes, metadata
│   └── activations.h5        # [topic_id/perspective_id] → [n_layers, d_model]
├── tests/
├── pyproject.toml
└── PLAN.md
```

## Install

Two install profiles — GPU box (Lambda) installs the heavy model deps;
laptop/Streamlit-Cloud installs only the analysis + frontend slice.

```bash
# On the Lambda A10 instance
pip install -e ".[gpu]"

# On your laptop / wherever Streamlit runs
pip install -e ".[client]"
```

## Quickstart (on-demand Lambda — recommended)

You run everything **from your laptop**. `arguenaut-lambda` spins a Lambda A10 up
on demand, bootstraps it from a git remote, and the GPU box self-terminates after
`LAMBDA_AUTO_SHUTDOWN_MINUTES` of idleness — so you never accidentally pay for
hours of nothing.

**One-time setup**

```bash
pip install -e ".[client]"
cp .env.example .env
# Fill in (at minimum):
#   GROQ_API_KEY              — from console.groq.com
#   LAMBDA_CLOUD_API_KEY      — from cloud.lambda.ai → Account → API keys
#   LAMBDA_SSH_KEY_NAME       — name of an SSH key you've added in the Lambda console
#   LAMBDA_SSH_KEY_PATH       — local path to that key, e.g. ~/.ssh/id_rsa
#   LAMBDA_GIT_URL            — https://github.com/<you>/arguenaut.git (must be pushed)
# Recommended:
#   LAMBDA_FILE_SYSTEM_NAME   — persistent FS for the HF cache (create once in the console)
```

**Each working session (per-prompt flow)**

```bash
arguenaut-lambda up --wait-healthy             # ~3-8 min first time, ~1 min on subsequent boots with PFS
streamlit run arguenaut/app/main.py            # type a hypothesis → discover its axes live
arguenaut-lambda down                          # (optional) stop billing now instead of waiting for idle timeout
```

That's the whole loop: bring the GPU up, open the app, type a claim, press
**Discover**. Each prompt spins up ~32 perspectives, extracts activations on the
GPU, runs PCA, and labels the axes — all live.

**Optional — build a global corpus map** (the original Phase 2–3 batch pipeline):

```bash
arguenaut-validate --remote                    # sanity-check 3 hypotheses through the GPU
arguenaut-extract --remote --hypotheses data/hypotheses.json
arguenaut-pca --layers 16,20,24,28             # runs locally, reads HDF5
arguenaut-label-axes --remote --top-k 5        # uses GPU for verification extractions
```

You can leave the instance running while you iterate — it'll self-terminate
after 15 idle minutes (override via `LAMBDA_AUTO_SHUTDOWN_MINUTES`). To kill
it immediately: `arguenaut-lambda down`.

**Other handy commands**

```bash
arguenaut-lambda status       # tracked + live state, /health, /meta, idle seconds
arguenaut-lambda logs -f      # tail the FastAPI server log over SSH
arguenaut-lambda ssh          # open an interactive SSH session
arguenaut-lambda types        # list instance types with current capacity
```

## Manual workflow (no Lambda automation)

If you'd rather drive Lambda yourself:

```bash
# On the Lambda A10 instance
pip install -e ".[gpu]"
python -m arguenaut.app.lambda_server         # listens on :8000

# On your laptop
pip install -e ".[client]"
export LAMBDA_API_URL=http://<lambda-ip>:8000
arguenaut-extract --remote --hypotheses data/hypotheses.json
arguenaut-pca --layers 16,20,24,28
arguenaut-label-axes --remote --top-k 5
streamlit run arguenaut/app/main.py
```

## Status

Phase 1–5 scaffolded. See `PLAN.md` for design intent and risk factors.
