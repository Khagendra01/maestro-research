"""Timestamped episode events and timing.

Uses monotonic clocks for timings and wall-clock timestamps for correlation.
Overlapping asynchronous work is not summed as if it were sequential.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


class EpisodeLogger:
    def __init__(self, path: str | Path, episode_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self._fh = self.path.open("a", encoding="utf-8")
        self.started = time.monotonic()

    def event(self, event_type: str, **fields: Any) -> None:
        record = {
            "episode_id": self.episode_id,
            "event": event_type,
            "wall_time": time.time(),
            "mono": time.monotonic() - self.started,
        }
        record.update(fields)
        self._fh.write(json.dumps(record, default=_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except Exception:
        pass
    return str(value)


@contextmanager
def timed(logger: Optional[EpisodeLogger], name: str, **fields: Any):
    start = time.perf_counter()
    try:
        yield
    finally:
        if logger is not None:
            logger.event("TIMING", name=name, seconds=time.perf_counter() - start, **fields)


def summarize_records(records: list, group_key: str = "suite") -> Dict[str, Any]:
    groups: Dict[str, Dict[str, int]] = {}
    for record in records:
        key = str(record.get(group_key, "unknown"))
        bucket = groups.setdefault(key, {"episodes": 0, "successes": 0})
        bucket["episodes"] += 1
        bucket["successes"] += int(bool(record.get("official_success")))
    total = sum(g["episodes"] for g in groups.values())
    successes = sum(g["successes"] for g in groups.values())
    for bucket in groups.values():
        bucket["success_rate"] = bucket["successes"] / bucket["episodes"] if bucket["episodes"] else None
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else None,
        "by_group": groups,
    }
