"""Thin subprocess wrappers around ssh / scp.

Uses the system `ssh` and `scp` so the user's existing config / keys / known_hosts
all "just work". We never modify ~/.ssh.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


_COMMON_SSH_ARGS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=~/.ssh/known_hosts",
    "-o", "ServerAliveInterval=30",
    "-o", "ConnectTimeout=10",
]


def _key_args(key_path: str | None) -> list[str]:
    return ["-i", key_path, "-o", "IdentitiesOnly=yes"] if key_path else []


def wait_for_ssh(
    ip: str, user: str = "ubuntu", key_path: str | None = None,
    timeout_secs: int = 300, poll_secs: int = 10,
) -> bool:
    """Block until `ssh user@ip echo ok` succeeds, or return False on timeout."""
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["ssh", *_COMMON_SSH_ARGS, *_key_args(key_path), f"{user}@{ip}", "echo ok"],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0 and "ok" in r.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(poll_secs)
    return False


def run_remote(
    ip: str, command: str, *, user: str = "ubuntu", key_path: str | None = None,
    check: bool = True, capture: bool = False, timeout: int | None = None,
) -> subprocess.CompletedProcess:
    args = ["ssh", *_COMMON_SSH_ARGS, *_key_args(key_path), f"{user}@{ip}", command]
    logger.debug("ssh: %s", " ".join(args))
    if capture:
        return subprocess.run(args, capture_output=True, text=True, check=check, timeout=timeout)
    return subprocess.run(args, check=check, timeout=timeout)


def scp_to_remote(
    local: str | Path, remote: str, *, ip: str, user: str = "ubuntu",
    key_path: str | None = None,
) -> subprocess.CompletedProcess:
    args = ["scp", *_COMMON_SSH_ARGS, *_key_args(key_path), str(local), f"{user}@{ip}:{remote}"]
    logger.debug("scp: %s", " ".join(args))
    return subprocess.run(args, check=True)


def scp_from_remote(
    remote: str, local: str | Path, *, ip: str, user: str = "ubuntu",
    key_path: str | None = None,
) -> subprocess.CompletedProcess:
    args = ["scp", *_COMMON_SSH_ARGS, *_key_args(key_path), f"{user}@{ip}:{remote}", str(local)]
    logger.debug("scp: %s", " ".join(args))
    return subprocess.run(args, check=True)
