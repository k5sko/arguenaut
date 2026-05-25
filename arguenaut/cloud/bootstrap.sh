#!/usr/bin/env bash
# Bootstrap an arguenaut Lambda instance from scratch (or update an existing one).
#
# Env vars (passed by the provisioner before exec):
#   ARGUENAUT_GIT_URL        — required, e.g. https://github.com/me/arguenaut.git
#   ARGUENAUT_GIT_REF        — branch/tag/commit. default: main
#   ARGUENAUT_ENV_B64        — base64 of the .env file to drop into the repo
#   ARGUENAUT_PFS            — path to persistent FS mount, e.g. /home/ubuntu/arguenaut-fs; "none" to disable
#   ARGUENAUT_PORT           — port for the FastAPI server. default: 8000
#   ARGUENAUT_LAMBDA_INSTANCE_ID — Lambda Cloud instance id (for auto-shutdown)
#
# Idempotent: re-running pulls latest code, reinstalls if needed, restarts the server.

set -euo pipefail

: "${ARGUENAUT_GIT_URL:?ARGUENAUT_GIT_URL must be set}"
GIT_REF="${ARGUENAUT_GIT_REF:-main}"
PFS="${ARGUENAUT_PFS:-none}"
API_PORT="${ARGUENAUT_PORT:-8000}"

WORKDIR="$HOME/arguenaut"
LOG_DIR="$WORKDIR/logs"
# NB: do NOT create LOG_DIR yet — that would make $WORKDIR non-empty and break a
# fresh `git clone` into it. It's created after the clone/update step below.

echo "[bootstrap] === arguenaut bootstrap starting ==="
echo "[bootstrap] git: $ARGUENAUT_GIT_URL @ $GIT_REF"
echo "[bootstrap] workdir: $WORKDIR"
echo "[bootstrap] persistent fs: $PFS"

# ── Clone or update ──────────────────────────────────────────────────────────
if [ -d "$WORKDIR/.git" ]; then
    cd "$WORKDIR"
    echo "[bootstrap] repo present, fetching $GIT_REF"
    git fetch --all --tags --quiet
    git checkout "$GIT_REF"
    git pull --ff-only --quiet || echo "[bootstrap] (non-ff or detached HEAD; leaving as-is)"
else
    echo "[bootstrap] cloning fresh"
    # $WORKDIR may exist but lack a .git (e.g. a leftover dir from a failed run or
    # a stock image). git clone refuses a non-empty target, so clear it first.
    # Safe: the persistent FS, if any, mounts elsewhere (see ARGUENAUT_PFS).
    rm -rf "$WORKDIR"
    git clone --quiet "$ARGUENAUT_GIT_URL" "$WORKDIR"
    cd "$WORKDIR"
    git checkout "$GIT_REF"
fi

# Repo now exists — safe to create the log dir inside it.
mkdir -p "$LOG_DIR"

# ── .env from base64 ────────────────────────────────────────────────────────
if [ -n "${ARGUENAUT_ENV_B64:-}" ]; then
    echo "[bootstrap] writing .env from ARGUENAUT_ENV_B64"
    echo "$ARGUENAUT_ENV_B64" | base64 -d > .env
else
    [ -f .env ] || touch .env
fi

# Append instance id so the server can self-terminate
if [ -n "${ARGUENAUT_LAMBDA_INSTANCE_ID:-}" ]; then
    grep -v '^ARGUENAUT_LAMBDA_INSTANCE_ID=' .env > .env.tmp || true
    echo "ARGUENAUT_LAMBDA_INSTANCE_ID=$ARGUENAUT_LAMBDA_INSTANCE_ID" >> .env.tmp
    mv .env.tmp .env
fi

# ── Persistent filesystem → HF cache ────────────────────────────────────────
if [ "$PFS" != "none" ] && [ -d "$PFS" ]; then
    echo "[bootstrap] using persistent FS at $PFS for model cache"
    mkdir -p "$PFS/hf-cache"
    grep -v -E '^(HF_HOME|TRANSFORMERS_CACHE|HF_HUB_CACHE)=' .env > .env.tmp || true
    {
        echo "HF_HOME=$PFS/hf-cache"
        echo "TRANSFORMERS_CACHE=$PFS/hf-cache"
        echo "HF_HUB_CACHE=$PFS/hf-cache"
    } >> .env.tmp
    mv .env.tmp .env
fi

# ── venv + install ──────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
    echo "[bootstrap] creating venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
echo "[bootstrap] pip install -e .[gpu]  (this may take a while on first run)"
# --ignore-requires-python: some Lambda images ship Python 3.10, and the code is
# 3.10-compatible (all modules use `from __future__ import annotations`), so the
# >=3.11 metadata floor is overly strict. Bypassing the check is safe here.
pip install --quiet --ignore-requires-python -e ".[gpu]"

# ── Restart server ──────────────────────────────────────────────────────────
echo "[bootstrap] stopping any existing arguenaut server"
pkill -f "arguenaut.app.lambda_server" || true
sleep 1

echo "[bootstrap] starting server on :$API_PORT"
nohup .venv/bin/python -m arguenaut.app.lambda_server \
    > "$LOG_DIR/server.log" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$WORKDIR/server.pid"

# ── Wait for /health ────────────────────────────────────────────────────────
echo "[bootstrap] waiting for /health …"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${API_PORT}/health" > /dev/null; then
        echo "[bootstrap] READY  (pid $SRV_PID, port $API_PORT)"
        exit 0
    fi
    sleep 5
done
echo "[bootstrap] FAILED — server did not become healthy in 5 minutes" >&2
tail -200 "$LOG_DIR/server.log" >&2 || true
exit 1
