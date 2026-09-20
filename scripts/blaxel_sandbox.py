"""Drive a Blaxel sandbox for CPU jobs (dataset builds, API runs).

    python scripts/blaxel_sandbox.py up   [--name jevify-data] [--memory 4096]
    python scripts/blaxel_sandbox.py run  "bash -lc 'cd /work/Jevify && jevify-bench list'" [--env HF_TOKEN]
    python scripts/blaxel_sandbox.py down

Auth comes from BL_API_KEY / BL_WORKSPACE / BL_REGION in the environment. ``--env NAME``
forwards named variables from the local environment into the sandbox process, so secrets
never land on disk in the sandbox or in this repo.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

IMAGE = "blaxel/py-app:latest"


async def up(name: str, memory: int, ttl: str) -> None:
    from blaxel.core import SandboxInstance

    sb = await SandboxInstance.create_if_not_exists({
        "name": name, "image": IMAGE, "memory": memory, "ttl": ttl,
        "region": os.environ.get("BL_REGION", "us-was-1"), "labels": {"project": "jevify"},
    })
    print(f"{sb.metadata.name}: {sb.status}")


async def run(name: str, command: str, env_names: list[str], timeout: int, workdir: str | None) -> int:
    from blaxel.core import SandboxInstance

    sb = await SandboxInstance.get(name)
    env = {k: os.environ[k] for k in env_names if k in os.environ}
    missing = [k for k in env_names if k not in os.environ]
    if missing:
        print(f"warning: not set locally, not forwarded: {missing}", file=sys.stderr)
    req = {"command": command, "env": env, "wait_for_completion": False, "timeout": timeout}
    if workdir:
        req["working_dir"] = workdir
    proc = await sb.process.exec(req)
    pname = proc.name
    seen = 0
    status = getattr(proc, "status", None)
    while True:
        logs = await sb.process.logs(pname)
        if len(logs) > seen:
            sys.stdout.write(logs[seen:])
            sys.stdout.flush()
            seen = len(logs)
        info = await sb.process.get(pname)
        status = getattr(info, "status", None)
        if status in ("completed", "failed", "killed", "stopped"):
            logs = await sb.process.logs(pname)
            if len(logs) > seen:
                sys.stdout.write(logs[seen:])
            code = getattr(info, "exit_code", None)
            print(f"\n[{pname}] {status} exit={code}", file=sys.stderr)
            return int(code or (0 if status == "completed" else 1))
        time.sleep(3)


async def down(name: str) -> None:
    from blaxel.core import SandboxInstance

    await SandboxInstance.delete(name)
    print(f"{name}: deleted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "run", "down"])
    ap.add_argument("command", nargs="?", default="")
    ap.add_argument("--name", default="jevify-data")
    ap.add_argument("--memory", type=int, default=4096)
    ap.add_argument("--ttl", default="24h")
    ap.add_argument("--env", action="append", default=[], help="local env var to forward (repeatable)")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="seconds")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_intermixed_args()
    if a.cmd == "up":
        asyncio.run(up(a.name, a.memory, a.ttl))
        return 0
    if a.cmd == "down":
        asyncio.run(down(a.name))
        return 0
    return asyncio.run(run(a.name, a.command, a.env, a.timeout, a.workdir))


if __name__ == "__main__":
    raise SystemExit(main())
