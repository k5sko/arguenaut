"""Local SSH tunnel to the Lambda box's FastAPI server.

Lambda's firewall blocks inbound 8000, and the server is intentionally
unauthenticated, so we never expose it publicly. Instead `arguenaut-lambda up`
forwards a local port to the box over SSH, and points the app/scripts at
``http://localhost:<port>``.

The tunnel runs as a detached background process (survives the CLI exiting) and
auto-reconnects if SSH drops on a transient network blip. `arguenaut-lambda down`
tears it down.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path

from arguenaut.cloud.ssh import _COMMON_SSH_ARGS, _key_args
from arguenaut.config import settings

logger = logging.getLogger(__name__)


def _pid_path() -> Path:
    return settings.data_dir / ".lambda-tunnel.pid"


def _log_path() -> Path:
    return settings.data_dir / "tunnel.log"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def open_tunnel(
    *,
    ip: str,
    user: str,
    key_path: str | None,
    local_port: int,
    remote_port: int | None = None,
    wait_secs: float = 20.0,
) -> int | None:
    """Open localhost:local_port -> ip:remote_port over SSH, detached.

    Returns the supervising process PID, or None if the port never opened.
    Any existing tunnel is closed first.
    """
    remote_port = remote_port or local_port
    close_tunnel()

    ssh_cmd = [
        "ssh", "-N",
        "-L", f"{local_port}:localhost:{remote_port}",
        *_COMMON_SSH_ARGS,
        "-o", "ServerAliveCountMax=6",
        "-o", "ExitOnForwardFailure=yes",
        *_key_args(key_path),
        f"{user}@{ip}",
    ]
    # Wrap in a reconnect loop so a transient SSH drop doesn't kill the tunnel.
    inner = " ".join(shlex.quote(a) for a in ssh_cmd)
    supervisor = ["bash", "-c", f"while true; do {inner}; sleep 2; done"]

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    log = open(_log_path(), "ab")
    proc = subprocess.Popen(
        supervisor,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,  # detach into its own process group
    )
    _pid_path().write_text(str(proc.pid))
    logger.info("Tunnel starting: localhost:%d -> %s:%d (pid %d)", local_port, ip, remote_port, proc.pid)

    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if _port_open(local_port):
            logger.info("Tunnel up on localhost:%d", local_port)
            return proc.pid
        if proc.poll() is not None:
            logger.error("Tunnel supervisor exited early; see %s", _log_path())
            return None
        time.sleep(0.5)
    logger.warning("Tunnel port %d did not open within %.0fs", local_port, wait_secs)
    return proc.pid


def close_tunnel() -> bool:
    """Kill the tunnel process group, if any. Returns True if one was running."""
    p = _pid_path()
    if not p.exists():
        return False
    killed = False
    try:
        pid = int(p.read_text().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        killed = True
    except (ProcessLookupError, ValueError, PermissionError):
        pass
    p.unlink(missing_ok=True)
    return killed


def tunnel_running() -> bool:
    p = _pid_path()
    if not p.exists():
        return False
    try:
        pid = int(p.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, PermissionError):
        return False
