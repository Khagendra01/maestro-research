"""Episode runner: reset, select, execute, score, log.

Scoring reads the official success oracle after the executor has finished.
The executor never receives it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .contracts import TaskDecision
from .manifest import EpisodeSpec, Manifest
from .metrics import EpisodeLogger
from .selector import ScriptedSelector
from .skills import ExecutorConfig, StagedExecutor


def _save_frame(out_dir: Path, episode_id: str, name: str, rgb: np.ndarray) -> Optional[str]:
    try:
        from PIL import Image

        directory = out_dir / "frames" / episode_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
        return str(path.relative_to(out_dir))
    except Exception:
        return None


def run_episode(
    backend,
    episode: EpisodeSpec,
    manifest: Manifest,
    executor: StagedExecutor,
    out_dir: str | Path,
    settings: str = "oracle",
    save_frames: bool = True,
    execute_config: Optional[ExecutorConfig] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(out_dir / "episodes.jsonl", episode.episode_id)
    suite = episode.task.suite
    max_steps = manifest.max_steps(suite)
    started_wall = time.time()
    started = time.monotonic()
    logger.event(
        "EPISODE_START",
        suite=suite,
        task=episode.task.task,
        task_id=episode.task.task_id,
        initial_state_index=episode.initial_state_index,
        language=episode.task.language,
        settings=settings,
        max_steps=max_steps,
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest.digest(),
        backend=backend.info.name,
        executor_config=vars(executor.config) if execute_config is None else vars(execute_config),
    )
    record: Dict[str, Any] = {
        "episode_id": episode.episode_id,
        "task_id": episode.task.task_id,
        "suite": suite,
        "task": episode.task.task,
        "language": episode.task.language,
        "initial_state_index": episode.initial_state_index,
        "split": episode.task.split,
        "settings": settings,
        "manifest_id": manifest.manifest_id,
        "max_steps": max_steps,
        "backend": backend.info.name,
    }
    try:
        observation = backend.reset(episode.task, episode.initial_state_index)
        if save_frames:
            record["frame_start"] = _save_frame(out_dir, episode.episode_id, "start", observation.rgb)

        select_start = time.perf_counter()
        scene_positions = None
        scene_extents = None
        if settings == "oracle":
            positions = backend.oracle_positions()
            table_center = backend.table_center()
            decision = ScriptedSelector(positions, table_center).decide(episode.task.language)
            selector_kind = "scripted_oracle_control"
        elif settings == "vlm":
            from .perception.vlm import build_scene_from_vlm

            scene = build_scene_from_vlm(observation, backend, episode.task.language)
            scene_positions = scene["positions"]
            scene_extents = scene["extents"]
            decision = scene["decision"]
            selector_kind = "meta_muse_spark_contributor"
            record["vlm"] = scene["metadata"]
        elif settings == "rgbd":
            from .perception.localize import build_scene_from_rgbd

            scene = build_scene_from_rgbd(observation, backend=backend)
            scene_positions = scene["positions"]
            scene_extents = scene["extents"]
            decision = scene["selector"].decide(episode.task.language)
            selector_kind = "scripted_rgbd_control"
            record["perceived_objects"] = [
                {
                    "type": obj["type"],
                    "position": np.asarray(obj["position"]).tolist(),
                    "radius": float(obj["radius"]),
                    "n_points": int(obj["n_points"]),
                }
                for obj in scene["objects"]
            ]
        else:
            raise ValueError(f"unknown settings {settings!r}")
        select_seconds = time.perf_counter() - select_start
        record["selector_kind"] = selector_kind
        record["target_id"] = decision.target_id
        record["receptacle_id"] = decision.constraints.get("receptacle_id")
        record["select_seconds"] = select_seconds
        logger.event("SELECTED", stage="SELECT", decision=decision.__dict__, selector_kind=selector_kind)

        outcome = executor.run(
            backend,
            decision,
            max_steps=max_steps,
            logger=logger,
            oracle_geometry=(settings == "oracle"),
            scene_positions=scene_positions,
            scene_extents=scene_extents,
        )
        official = bool(backend.official_success())
        if save_frames:
            try:
                record["frame_end"] = _save_frame(out_dir, episode.episode_id, "end", backend.observe().rgb)
            except Exception:
                record["frame_end"] = None
        record.update(
            {
                "status": outcome.status,
                "reason_code": outcome.reason_code,
                "internal_success": outcome.internal_success,
                "official_success": official,
                "steps": outcome.steps,
                "grasp_attempts": outcome.grasp_attempts,
                "stage_errors": outcome.stage_errors,
                "detail": outcome.detail,
            }
        )
        logger.event("EPISODE_END", stage="SCORE", official_success=official, **{
            "status": outcome.status,
            "reason_code": outcome.reason_code,
            "internal_success": outcome.internal_success,
            "steps": outcome.steps,
        })
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "reason_code": f"exception:{type(exc).__name__}",
                "internal_success": False,
                "official_success": False,
                "steps": int(getattr(backend, "step_count", 0)),
                "error": repr(exc),
            }
        )
        logger.event("EPISODE_ERROR", stage="ERROR", error=repr(exc))
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["timestamp"] = started_wall
        logger.event("EPISODE_SUMMARY", **record)
        logger.close()
        if record.get("status") != "ok" or record.get("steps", 0) >= max_steps:
            pass
    return record


def write_summary(out_dir: str | Path, records: list, extra: Optional[Dict[str, Any]] = None) -> Path:
    from .metrics import summarize_records

    out_dir = Path(out_dir)
    summary = summarize_records(records)
    summary["records"] = records
    if extra:
        summary.update(extra)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return path
