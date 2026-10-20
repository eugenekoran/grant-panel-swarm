"""Append-only JSONL event log. See docs/HLD.md and swarm/README.md."""

from __future__ import annotations

import json
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")
        self.t0 = time.time()

    def write(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "t": round(time.time() - self.t0, 3), "event": event, **fields}
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
