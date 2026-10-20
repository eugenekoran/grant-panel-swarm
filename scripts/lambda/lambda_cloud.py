"""Minimal Lambda Cloud client: check capacity, poll-and-launch, list, terminate.

Reads LAMBDA_API_KEY from the environment or .env. Stdlib only.

    python scripts/lambda/lambda_cloud.py avail
    python scripts/lambda/lambda_cloud.py launch gpu_1x_h100_sxm5 gpu_1x_h100_pcie --name smoke --interval 7
    python scripts/lambda/lambda_cloud.py list
    python scripts/lambda/lambda_cloud.py terminate <instance-id>

`launch` takes instance types in order of preference, polls until one has capacity in any region,
launches it, waits for it to become active, and prints `ID=... IP=... TYPE=...` on the last line.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://cloud.lambdalabs.com/api/v1"
SSH_KEY_NAME = "lambda_key"


def _key() -> str:
    if k := os.environ.get("LAMBDA_API_KEY"):
        return k
    env = Path(__file__).resolve().parents[2] / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("LAMBDA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("LAMBDA_API_KEY not set")


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{_key()}:".encode()).decode())
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "aw-swarm/0.1")  # Cloudflare rejects the default urllib agent
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return {"error": json.loads(raw).get("error", {"code": e.code})}
        except ValueError:
            return {"error": {"code": e.code, "body": raw[:200].decode(errors="replace")}}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def avail() -> dict[str, list[str]]:
    data = call("GET", "/instance-types").get("data", {})
    return {k: [r["name"] for r in v["regions_with_capacity_available"]] for k, v in data.items()}


def launch(types: list[str], name: str, interval: float, timeout: float) -> None:
    deadline = time.time() + timeout
    n = 0
    while True:
        try:
            cap = avail()
        except Exception as e:  # network blips: keep polling
            cap = {}
            log(f"poll error: {e}")
        pick = next(((t, cap[t][0]) for t in types if cap.get(t)), None)
        if pick:
            t, region = pick
            log(f"capacity: {t} in {region}; launching")
            r = call("POST", "/instance-operations/launch", {
                "region_name": region, "instance_type_name": t, "ssh_key_names": [SSH_KEY_NAME],
                "quantity": 1, "name": name})
            ids = r.get("data", {}).get("instance_ids")
            if ids:
                break
            log(f"launch failed (capacity likely gone): {r.get('error')}")
        n += 1
        if n % 20 == 0:
            log(f"still polling ({n} polls); nothing for {types}")
        if time.time() > deadline:
            sys.exit("timed out waiting for capacity")
        time.sleep(interval)

    iid = ids[0]
    log(f"launched {iid}; waiting for active")
    while True:
        inst = call("GET", f"/instances/{iid}").get("data", {})
        if inst.get("status") == "active" and inst.get("ip"):
            print(f"ID={iid} IP={inst['ip']} TYPE={t} REGION={region}", flush=True)
            return
        if inst.get("status") in ("terminated", "unhealthy"):
            sys.exit(f"instance {iid} is {inst.get('status')}")
        time.sleep(10)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("avail")
    sub.add_parser("list")
    lp = sub.add_parser("launch")
    lp.add_argument("types", nargs="+")
    lp.add_argument("--name", default="aw-swarm")
    lp.add_argument("--interval", type=float, default=7.0)
    lp.add_argument("--timeout", type=float, default=6 * 3600)
    tp = sub.add_parser("terminate")
    tp.add_argument("ids", nargs="+")
    a = p.parse_args()

    if a.cmd == "avail":
        for k, regions in sorted(avail().items()):
            if regions:
                print(k, regions)
    elif a.cmd == "list":
        for i in call("GET", "/instances").get("data", []):
            print(i["id"], i.get("name"), i["instance_type"]["name"], i["status"], i.get("ip"), i["region"]["name"])
    elif a.cmd == "launch":
        launch(a.types, a.name, a.interval, a.timeout)
    elif a.cmd == "terminate":
        print(json.dumps(call("POST", "/instance-operations/terminate", {"instance_ids": a.ids})))


if __name__ == "__main__":
    main()
