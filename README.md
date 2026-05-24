# Arguenaut

An interpretability system that discovers the fundamental axes of disagreement in scientific hypotheses by analyzing the internal activations of a small base LM (Qwen2.5-3B / Qwen3.5-4B-Base).

## Pipeline

```
hypothesis
   │
   ▼
Groq generates 4–8 opposing perspective texts
   │
   ▼
Qwen base model (on Lambda) processes each text under nnsight.trace
   │
   ▼
Last-token residual-stream activations extracted at every layer → HDF5
   │
   ▼
PCA over the [N, 2560] matrix per layer → top principal components
   │
   ▼
Groq labels each component "X vs Y" then verifies on fresh perspectives
   │
   ▼
Streamlit plots where a new hypothesis falls in the discovered axis space
```

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

**Each working session**

```bash
arguenaut-lambda up --wait-healthy             # ~3-8 min first time, ~1 min on subsequent boots with PFS
arguenaut-validate --remote                    # sanity-check 3 hypotheses through the GPU
arguenaut-extract --remote --hypotheses data/hypotheses.json
arguenaut-pca --layers 16,20,24,28             # runs locally, reads HDF5
arguenaut-label-axes --remote --top-k 5        # uses GPU for verification extractions
streamlit run arguenaut/app/main.py            # uses GPU for live "probe a new hypothesis"
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
