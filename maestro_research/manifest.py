"""Pinned evaluation manifests.

A manifest declares every episode before any rollout. Task names, initial-state
indices and horizons are frozen; results are reported for the declared set.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    suite: str
    task: str
    language: str
    initial_state_indices: tuple
    split: str = "development"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    task: TaskSpec
    initial_state_index: int


@dataclass
class Manifest:
    manifest_id: str
    path: Path
    raw: Dict
    tasks: List[TaskSpec]
    sha256: str

    @property
    def protocol(self) -> Dict:
        return self.raw["benchmark"]

    def episodes(self, limit_per_task: int | None = None) -> List[EpisodeSpec]:
        out: List[EpisodeSpec] = []
        for task in self.tasks:
            indices = task.initial_state_indices
            if limit_per_task is not None:
                indices = indices[:limit_per_task]
            for index in indices:
                out.append(
                    EpisodeSpec(
                        episode_id=f"{task.task_id}__state{index:03d}",
                        task=task,
                        initial_state_index=index,
                    )
                )
        return out

    def max_steps(self, suite: str) -> int:
        try:
            return int(self.protocol["task_max_steps"][suite])
        except KeyError as exc:
            raise KeyError(f"manifest does not pin max steps for suite {suite!r}") from exc

    def digest(self) -> str:
        return self.sha256


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    for key in ("manifest_id", "benchmark", "tasks"):
        if key not in raw:
            raise ValueError(f"manifest missing {key!r}")
    tasks = []
    seen = set()
    for entry in raw["tasks"]:
        for key in ("task_id", "suite", "task", "language", "initial_state_indices"):
            if key not in entry:
                raise ValueError(f"task entry missing {key!r}: {entry}")
        if entry["suite"] not in SUPPORTED_SUITES:
            raise ValueError(f"unsupported suite {entry['suite']!r}")
        if entry["task_id"] in seen:
            raise ValueError(f"duplicate task_id {entry['task_id']!r}")
        seen.add(entry["task_id"])
        indices = tuple(int(i) for i in entry["initial_state_indices"])
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate initial-state index in {entry['task_id']!r}")
        if not indices:
            raise ValueError(f"no initial states declared for {entry['task_id']!r}")
        tasks.append(
            TaskSpec(
                task_id=entry["task_id"],
                suite=entry["suite"],
                task=entry["task"],
                language=entry["language"],
                initial_state_indices=indices,
                split=entry.get("split", "development"),
            )
        )
    if not tasks:
        raise ValueError("manifest declares no tasks")
    return Manifest(
        manifest_id=raw["manifest_id"],
        path=path,
        raw=raw,
        tasks=tasks,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
