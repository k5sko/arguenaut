"""Lambda Labs Cloud API client.

Docs: https://cloud.lambdalabs.com/api/v1/docs

Auth: HTTP Basic with the API key as the username and an empty password.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaCloudError(RuntimeError):
    pass


class LambdaCloudClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE, timeout: float = 30.0):
        if not api_key:
            raise LambdaCloudError("LAMBDA_CLOUD_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self._auth = (api_key, "")
        self._timeout = timeout

    # ── HTTP plumbing ──────────────────────────────────────────────────────
    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            r = httpx.request(method, url, auth=self._auth, timeout=self._timeout, **kwargs)
        except httpx.HTTPError as e:
            raise LambdaCloudError(f"{method} {path} failed: {e}") from e
        if r.status_code >= 400:
            raise LambdaCloudError(f"{method} {path} → {r.status_code}: {r.text[:300]}")
        if not r.content:
            return None
        body = r.json()
        # Lambda Cloud API wraps everything in {"data": ...} or {"error": ...}
        if isinstance(body, dict) and "error" in body and "data" not in body:
            raise LambdaCloudError(f"{method} {path} → {body['error']}")
        return body.get("data", body) if isinstance(body, dict) else body

    # ── reads ──────────────────────────────────────────────────────────────
    def list_instance_types(self) -> dict:
        """Returns {instance_type_name: {regions_with_capacity: [...], ...}}."""
        return self._request("GET", "/instance-types")

    def list_instances(self) -> list[dict]:
        return self._request("GET", "/instances")

    def get_instance(self, instance_id: str) -> dict:
        return self._request("GET", f"/instances/{instance_id}")

    def list_ssh_keys(self) -> list[dict]:
        return self._request("GET", "/ssh-keys")

    def list_file_systems(self) -> list[dict]:
        return self._request("GET", "/file-systems")

    # ── writes ─────────────────────────────────────────────────────────────
    def launch_instance(
        self,
        instance_type_name: str,
        region_name: str,
        ssh_key_names: list[str],
        name: str | None = None,
        file_system_names: list[str] | None = None,
        quantity: int = 1,
    ) -> list[str]:
        """Returns a list of instance ids that were launched."""
        body = {
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": ssh_key_names,
            "quantity": quantity,
        }
        if name:
            body["name"] = name
        if file_system_names:
            body["file_system_names"] = file_system_names
        data = self._request("POST", "/instance-operations/launch", json=body)
        return list(data.get("instance_ids", []))

    def terminate_instances(self, instance_ids: list[str]) -> list[dict]:
        data = self._request(
            "POST", "/instance-operations/terminate", json={"instance_ids": instance_ids}
        )
        return list(data.get("terminated_instances", []))

    # ── high-level helpers ─────────────────────────────────────────────────
    def find_available_region(self, instance_type: str) -> str | None:
        """Pick a region that currently has capacity for the given instance type."""
        types = self.list_instance_types()
        entry = types.get(instance_type)
        if not entry:
            return None
        regions = entry.get("regions_with_capacity_available", []) or []
        if not regions:
            return None
        # API returns a list of region dicts {name, description}
        first = regions[0]
        return first["name"] if isinstance(first, dict) else first

    def find_instance_by_name(self, name: str) -> dict | None:
        for ins in self.list_instances():
            if ins.get("name") == name:
                return ins
        return None
