"""arguenaut-lambda — provision / inspect / terminate the GPU instance.

Subcommands:
    up        Launch (or reuse) an instance, bootstrap arguenaut, wait until /health is green.
    down      Terminate the tracked instance and clear local state.
    status    Show what's tracked locally and what the Lambda API says about it.
    logs      Tail the server log over SSH.
    ssh       Open an interactive SSH session.
    types     List instance types currently with capacity.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time

import httpx

from arguenaut.cloud.lambda_api import LambdaCloudClient, LambdaCloudError
from arguenaut.cloud.provisioner import LambdaProvisioner
from arguenaut.cloud.ssh import _COMMON_SSH_ARGS, _key_args
from arguenaut.cloud.state import load_state
from arguenaut.config import settings


def _cmd_up(args: argparse.Namespace) -> int:
    prov = LambdaProvisioner(
        instance_type=args.instance_type,
        region=args.region,
        file_system=args.file_system,
        git_ref=args.git_ref,
    )
    info = prov.ensure_up(bootstrap=not args.no_bootstrap)
    print(f"instance:  {info.instance_id}")
    print(f"ip:        {info.ip}")
    print(f"new:       {info.is_new}")

    # Lambda's firewall blocks inbound 8000 and the server is unauthenticated, so
    # reach it through an SSH tunnel and point the app/scripts at localhost.
    api_url = info.api_url
    if not args.no_tunnel:
        from arguenaut.cloud.state import LambdaState, load_state, save_state
        from arguenaut.cloud.tunnel import open_tunnel

        st = load_state()
        port = st.api_port if st else 8000
        ip = st.ip if st else info.ip
        user = st.ssh_user if st else "ubuntu"
        pid = open_tunnel(
            ip=ip, user=user, key_path=settings.lambda_ssh_key_path,
            local_port=port, remote_port=port,
        )
        if pid is None:
            print("WARNING: tunnel failed to open; the app won't reach the box", file=sys.stderr)
        else:
            api_url = f"http://localhost:{port}"
            if st is not None:
                save_state(LambdaState(**{**st.to_dict(), "api_url": api_url}))
            print(f"tunnel:    localhost:{port} -> {ip}:{port}  (pid {pid})")
    print(f"api_url:   {api_url}")

    if args.wait_healthy:
        deadline = time.time() + 600
        print("waiting for /health …", file=sys.stderr)
        while time.time() < deadline:
            try:
                r = httpx.get(f"{api_url}/health", timeout=10)
                if r.status_code == 200 and r.json().get("loaded"):
                    print("HEALTHY", file=sys.stderr)
                    return 0
            except httpx.HTTPError:
                pass
            time.sleep(10)
        print("TIMEOUT waiting for /health", file=sys.stderr)
        return 1
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    from arguenaut.cloud.tunnel import close_tunnel

    if close_tunnel():
        print("tunnel closed")
    prov = LambdaProvisioner()
    ok = prov.down()
    print("terminated" if ok else "no tracked instance to terminate")
    return 0 if ok else 1


def _cmd_status(args: argparse.Namespace) -> int:
    state = load_state()
    if state is None:
        print("no tracked instance (state file empty)")
        return 1
    print("== local state ==")
    print(json.dumps(state.to_dict(), indent=2, default=str))
    if not settings.lambda_cloud_api_key:
        print("\n(set LAMBDA_CLOUD_API_KEY for live status)")
        return 0
    client = LambdaCloudClient(settings.lambda_cloud_api_key)
    try:
        live = client.get_instance(state.instance_id)
    except LambdaCloudError as e:
        print(f"\n== live status ==\nerror: {e}")
        return 1
    print("\n== live (Lambda API) ==")
    print(json.dumps(live, indent=2, default=str))

    # /health and /meta if reachable
    try:
        health = httpx.get(f"{state.api_url}/health", timeout=5).json()
        meta = httpx.get(f"{state.api_url}/meta", timeout=5).json()
        print("\n== server ==")
        print(json.dumps({"health": health, "meta": meta}, indent=2, default=str))
    except httpx.HTTPError as e:
        print(f"\n== server ==\nunreachable: {e}")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    state = load_state()
    if state is None:
        print("no tracked instance", file=sys.stderr)
        return 1
    follow = "tail -F" if args.follow else "tail -n 200"
    ssh_args = [
        "ssh", *_COMMON_SSH_ARGS, *_key_args(settings.lambda_ssh_key_path),
        f"{state.ssh_user}@{state.ip}",
        f"{follow} ~/arguenaut/logs/server.log",
    ]
    return subprocess.call(ssh_args)


def _cmd_ssh(args: argparse.Namespace) -> int:
    state = load_state()
    if state is None:
        print("no tracked instance", file=sys.stderr)
        return 1
    ssh_args = [
        "ssh", *_COMMON_SSH_ARGS, *_key_args(settings.lambda_ssh_key_path),
        f"{state.ssh_user}@{state.ip}",
    ]
    if args.command:
        ssh_args.append(args.command)
    return subprocess.call(ssh_args)


def _cmd_types(args: argparse.Namespace) -> int:
    if not settings.lambda_cloud_api_key:
        print("LAMBDA_CLOUD_API_KEY is not set", file=sys.stderr)
        return 1
    client = LambdaCloudClient(settings.lambda_cloud_api_key)
    types = client.list_instance_types()
    for name, entry in sorted(types.items()):
        regions = entry.get("regions_with_capacity_available", []) or []
        if not regions and args.only_available:
            continue
        region_names = ", ".join(r["name"] if isinstance(r, dict) else r for r in regions) or "(none)"
        descr = entry.get("instance_type", {}).get("description", "")
        print(f"{name:<24} avail: {region_names}   {descr}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    p = argparse.ArgumentParser(prog="arguenaut-lambda", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="launch + bootstrap (or reuse) an instance")
    up.add_argument("--instance-type", default=None, help="override LAMBDA_INSTANCE_TYPE")
    up.add_argument("--region", default=None, help="override LAMBDA_REGION (else auto-pick)")
    up.add_argument("--file-system", default=None, help="override LAMBDA_FILE_SYSTEM_NAME")
    up.add_argument("--git-ref", default=None, help="override LAMBDA_GIT_REF (branch/tag/sha)")
    up.add_argument("--no-bootstrap", action="store_true", help="just launch, skip the install step")
    up.add_argument("--no-tunnel", action="store_true", help="don't open the SSH tunnel to the server")
    up.add_argument("--wait-healthy", action="store_true", help="block until /health returns 200 + model loaded")
    up.set_defaults(func=_cmd_up)

    dn = sub.add_parser("down", help="terminate the tracked instance")
    dn.set_defaults(func=_cmd_down)

    st = sub.add_parser("status", help="show tracked + live status")
    st.set_defaults(func=_cmd_status)

    lg = sub.add_parser("logs", help="tail the server log over SSH")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.set_defaults(func=_cmd_logs)

    sh = sub.add_parser("ssh", help="open SSH (optionally run a command)")
    sh.add_argument("command", nargs="?")
    sh.set_defaults(func=_cmd_ssh)

    tp = sub.add_parser("types", help="list available instance types")
    tp.add_argument("--only-available", action="store_true")
    tp.set_defaults(func=_cmd_types)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except LambdaCloudError as e:
        print(f"lambda error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
