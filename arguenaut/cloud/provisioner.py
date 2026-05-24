"""High-level Lambda provisioning: ensure_up → bootstrap → reachable URL."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from arguenaut.cloud.lambda_api import LambdaCloudClient, LambdaCloudError
from arguenaut.cloud.ssh import run_remote, scp_to_remote, wait_for_ssh
from arguenaut.cloud.state import LambdaState, clear_state, load_state, save_state
from arguenaut.config import settings

logger = logging.getLogger(__name__)


@dataclass
class InstanceInfo:
    instance_id: str
    name: str
    ip: str
    api_url: str
    is_new: bool


_BOOTSTRAP_PATH = Path(__file__).with_name("bootstrap.sh")


class LambdaProvisioner:
    """One source of truth for spinning the GPU instance up and down."""

    def __init__(
        self,
        api_key: str | None = None,
        ssh_key_name: str | None = None,
        ssh_key_path: str | None = None,
        instance_type: str | None = None,
        region: str | None = None,
        file_system: str | None = None,
        git_url: str | None = None,
        git_ref: str | None = None,
        instance_name: str = "arguenaut",
        api_port: int = 8000,
        ssh_user: str = "ubuntu",
    ):
        self.api = LambdaCloudClient(api_key or settings.lambda_cloud_api_key or "")
        self.ssh_key_name = ssh_key_name or settings.lambda_ssh_key_name
        self.ssh_key_path = ssh_key_path or settings.lambda_ssh_key_path
        self.instance_type = instance_type or settings.lambda_instance_type
        self.region = region or settings.lambda_region   # may be None → auto-pick
        self.file_system = file_system or settings.lambda_file_system_name
        self.git_url = git_url or settings.lambda_git_url
        self.git_ref = git_ref or settings.lambda_git_ref
        self.instance_name = instance_name
        self.api_port = api_port
        self.ssh_user = ssh_user

        if not self.ssh_key_name:
            raise LambdaCloudError("LAMBDA_SSH_KEY_NAME is not set")
        if not self.git_url:
            raise LambdaCloudError("LAMBDA_GIT_URL is not set (cannot bootstrap)")

    # ── public API ─────────────────────────────────────────────────────────
    def ensure_up(self, bootstrap: bool = True) -> InstanceInfo:
        """If an instance is already tracked + healthy, return it; else launch + bootstrap."""
        state = load_state()
        if state is not None:
            live = self._refresh(state.instance_id)
            if live is not None and live.get("status") in {"active", "booting"}:
                ip = live.get("ip") or state.ip
                api_url = f"http://{ip}:{state.api_port}"
                logger.info("Reusing existing instance %s @ %s", state.instance_id, ip)
                state = LambdaState(**{**state.to_dict(), "ip": ip, "api_url": api_url})
                save_state(state)
                if bootstrap and not state.bootstrap_complete:
                    self._bootstrap(state)
                return InstanceInfo(state.instance_id, state.name, ip, api_url, is_new=False)
            else:
                logger.info("Stale state for instance %s, clearing", state.instance_id)
                clear_state()

        # Need to launch.
        return self._launch_new(bootstrap=bootstrap)

    def down(self) -> bool:
        state = load_state()
        if state is None:
            logger.info("No instance recorded; nothing to terminate")
            return False
        logger.info("Terminating instance %s", state.instance_id)
        try:
            self.api.terminate_instances([state.instance_id])
        except LambdaCloudError as e:
            logger.warning("Termination call failed (%s); clearing state anyway", e)
        clear_state()
        return True

    def status(self) -> dict | None:
        state = load_state()
        if state is None:
            return None
        live = self._refresh(state.instance_id)
        if live is None:
            return {"state_file": state.to_dict(), "live": None}
        return {"state_file": state.to_dict(), "live": live}

    # ── internals ──────────────────────────────────────────────────────────
    def _refresh(self, instance_id: str) -> dict | None:
        try:
            return self.api.get_instance(instance_id)
        except LambdaCloudError:
            return None

    def _launch_new(self, bootstrap: bool) -> InstanceInfo:
        region = self.region or self.api.find_available_region(self.instance_type)
        if not region:
            raise LambdaCloudError(
                f"No region currently has capacity for {self.instance_type}. "
                "Try a different LAMBDA_INSTANCE_TYPE or wait."
            )
        logger.info("Launching %s in %s …", self.instance_type, region)
        ids = self.api.launch_instance(
            instance_type_name=self.instance_type,
            region_name=region,
            ssh_key_names=[self.ssh_key_name],
            name=self.instance_name,
            file_system_names=[self.file_system] if self.file_system else None,
        )
        if not ids:
            raise LambdaCloudError("Launch returned no instance ids")
        instance_id = ids[0]

        # Poll until the instance has an IP.
        ip = None
        deadline = time.time() + 600
        while time.time() < deadline:
            info = self._refresh(instance_id)
            if info and info.get("status") == "active" and info.get("ip"):
                ip = info["ip"]
                break
            time.sleep(10)
        if not ip:
            raise LambdaCloudError(f"Instance {instance_id} did not become active within 10 minutes")

        api_url = f"http://{ip}:{self.api_port}"
        state = LambdaState(
            instance_id=instance_id,
            name=self.instance_name,
            ip=ip,
            ssh_user=self.ssh_user,
            api_url=api_url,
            api_port=self.api_port,
            region=region,
            instance_type=self.instance_type,
            launched_at=time.time(),
            bootstrap_complete=False,
        )
        save_state(state)

        if bootstrap:
            self._bootstrap(state)

        return InstanceInfo(instance_id, self.instance_name, ip, api_url, is_new=True)

    def _bootstrap(self, state: LambdaState) -> None:
        logger.info("Waiting for SSH on %s …", state.ip)
        if not wait_for_ssh(state.ip, user=self.ssh_user, key_path=self.ssh_key_path):
            raise LambdaCloudError(f"SSH never came up on {state.ip}")
        # Upload bootstrap script
        logger.info("Uploading bootstrap.sh")
        scp_to_remote(_BOOTSTRAP_PATH, "/home/ubuntu/arguenaut-bootstrap.sh",
                      ip=state.ip, user=self.ssh_user, key_path=self.ssh_key_path)
        run_remote(state.ip, "chmod +x /home/ubuntu/arguenaut-bootstrap.sh",
                   user=self.ssh_user, key_path=self.ssh_key_path)

        # Build the env-var prefix for the remote command.
        local_env_b64 = self._build_remote_env_b64(state)
        pfs = self._pfs_mount_path() or "none"
        env_assigns = " ".join([
            f"ARGUENAUT_GIT_URL={_sh_quote(self.git_url)}",
            f"ARGUENAUT_GIT_REF={_sh_quote(self.git_ref)}",
            f"ARGUENAUT_ENV_B64={_sh_quote(local_env_b64)}",
            f"ARGUENAUT_PFS={_sh_quote(pfs)}",
            f"ARGUENAUT_PORT={state.api_port}",
            f"ARGUENAUT_LAMBDA_INSTANCE_ID={_sh_quote(state.instance_id)}",
        ])
        cmd = f"{env_assigns} bash /home/ubuntu/arguenaut-bootstrap.sh"
        logger.info("Running bootstrap on remote (may take several minutes on cold start) …")
        run_remote(state.ip, cmd, user=self.ssh_user, key_path=self.ssh_key_path, timeout=1800)

        # Mark bootstrap complete
        new_state = LambdaState(**{**state.to_dict(), "bootstrap_complete": True})
        save_state(new_state)
        logger.info("Bootstrap complete; API at %s", state.api_url)

    def _build_remote_env_b64(self, state: LambdaState) -> str:
        """Build the .env we want on the remote: forward our Groq key + add HF model id + Lambda creds for auto-shutdown."""
        lines = [
            f"GROQ_API_KEY={settings.groq_api_key or ''}",
            f"GROQ_MODEL={settings.groq_model}",
            f"HF_MODEL_ID={settings.hf_model_id}",
            f"LAMBDA_API_TOKEN={settings.lambda_api_token or ''}",
            # Server-side auto-shutdown
            f"LAMBDA_CLOUD_API_KEY={settings.lambda_cloud_api_key or ''}",
            f"ARGUENAUT_AUTO_SHUTDOWN_MINUTES={settings.lambda_auto_shutdown_minutes}",
            f"ARGUENAUT_LAMBDA_INSTANCE_ID={state.instance_id}",
        ]
        raw = "\n".join(lines).encode()
        return base64.b64encode(raw).decode()

    def _pfs_mount_path(self) -> str | None:
        if not self.file_system:
            return None
        # Lambda mounts persistent file systems at /home/ubuntu/<name>
        return f"/home/ubuntu/{self.file_system}"


def _sh_quote(s: str) -> str:
    """Single-quote a string for bash. Empty string allowed."""
    if s is None:
        s = ""
    return "'" + s.replace("'", "'\\''") + "'"
