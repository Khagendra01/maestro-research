"""OpenVLA-OFT learned-policy reference on the declared LIBERO dev smoke.

This module is imported only inside the Modal image that has the OpenVLA-OFT
repository, its custom transformers fork, and LIBERO installed. It calls the
reference evaluation primitives directly so the reference uses its own policy,
preprocessing, action chunks and official success check. It does not import
anything from the modular MAESTRO executor.

Task selection is by exact LIBERO task name from the frozen dev manifest, with
initial-state indices 0..4. The reference checkpoint is selected per suite.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CHECKPOINTS = {
    "libero_spatial": "/data/checkpoints/openvla-oft-libero-spatial",
    "libero_object": "/data/checkpoints/openvla-oft-libero-object",
    "libero_goal": "/data/checkpoints/openvla-oft-libero-goal",
}


class CountingEnv:
    """Count calls to step; everything else delegates to the wrapped env."""

    def __init__(self, env):
        self._env = env
        self.steps = 0

    def reset(self, *args, **kwargs):
        self.steps = 0
        return self._env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.steps += 1
        return self._env.step(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


def configure_libero_paths(root: str = "/opt/LIBERO") -> None:
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"LIBERO checkout not found at {root}")
    config = Path.home() / ".libero" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"benchmark_root: {root_path}/libero/libero\n"
        f"bddl_files: {root_path}/libero/libero/bddl_files\n"
        f"init_states: {root_path}/libero/libero/init_files\n"
        f"datasets: {root_path}/libero/datasets\n"
        f"assets: {root_path}/libero/libero/assets\n"
    )


def _task_id(bench, task_name: str) -> int:
    for index, task in enumerate(bench.tasks):
        if getattr(task, "name", None) == task_name:
            return index
    names = [getattr(task, "name", None) for task in bench.tasks]
    raise KeyError(f"task {task_name!r} not found; available names: {names}")


def _save_frame(out_dir: Path, episode_id: str, name: str, image: np.ndarray) -> None:
    try:
        from PIL import Image

        directory = out_dir / "frames" / episode_id
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(directory / f"{name}.png")
    except Exception:
        pass


def run_reference(
    manifest_path: str,
    out_dir: str,
    limit_per_task: int = 5,
    seed: int = 7,
    full_suite: bool = False,
    num_states: int = 50,
    commit_fn=None,
) -> Dict[str, Any]:
    import torch

    configure_libero_paths()
    from libero.libero import benchmark

    import experiments.robot.libero.run_libero_eval as eval_mod
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    manifest = json.loads(Path(manifest_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    original_get_action = eval_mod.get_action
    timer = {"seconds": 0.0, "calls": 0}

    def timed_get_action(*args, **kwargs):
        started = time.perf_counter()
        result = original_get_action(*args, **kwargs)
        timer["seconds"] += time.perf_counter() - started
        timer["calls"] += 1
        return result

    eval_mod.get_action = timed_get_action

    benchmark_dict = benchmark.get_benchmark_dict()
    if full_suite:
        suites = ["libero_spatial", "libero_object", "libero_goal"]
    else:
        suites = []
        for task in manifest["tasks"]:
            if task["suite"] not in suites:
                suites.append(task["suite"])

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "reference": "OpenVLA-OFT",
        "manifest_id": "libero_full_suite" if full_suite else manifest["manifest_id"],
        "manifest_sha256": None,
        "checkpoints": CHECKPOINTS,
        "num_actions_chunk": int(NUM_ACTIONS_CHUNK),
        "seed": seed,
        "records": [],
    }

    for suite in suites:
        checkpoint = CHECKPOINTS[suite]
        if not Path(checkpoint).is_dir():
            raise FileNotFoundError(f"missing checkpoint directory {checkpoint}")

        cfg = eval_mod.GenerateConfig(
            pretrained_checkpoint=checkpoint,
            task_suite_name=suite,
            num_steps_wait=int(manifest["benchmark"]["num_steps_wait"]),
            num_trials_per_task=limit_per_task,
            env_img_res=256,
            num_open_loop_steps=int(NUM_ACTIONS_CHUNK),
            seed=seed,
            local_log_dir=str(out / "logs"),
        )
        eval_mod.validate_config(cfg)
        set_seed_everywhere(cfg.seed)
        load_started = time.perf_counter()
        model, action_head, proprio_projector, _, processor = eval_mod.initialize_model(cfg)
        load_seconds = time.perf_counter() - load_started
        resize_size = get_image_resize_size(cfg)
        summary.setdefault("model_load_seconds", {})[suite] = load_seconds

        bench = benchmark_dict[suite]()
        if full_suite:
            tasks = [
                {
                    "task_id": f"{suite}_task{index:02d}",
                    "task": task.name,
                    "language": task.language,
                    "initial_state_indices": list(range(num_states)),
                }
                for index, task in enumerate(bench.tasks)
            ]
        else:
            tasks = [t for t in manifest["tasks"] if t["suite"] == suite]
        for task_spec in tasks:
            task_id = _task_id(bench, task_spec["task"])
            task = bench.get_task(task_id)
            initial_states = bench.get_task_init_states(task_id)
            env, description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            wrapped = CountingEnv(env)
            indices = task_spec["initial_state_indices"][:limit_per_task]
            for state_index in indices:
                episode_id = f"{task_spec['task_id']}__state{state_index:03d}"
                timer["seconds"] = 0.0
                timer["calls"] = 0
                started = time.perf_counter()
                success, replay_images = eval_mod.run_episode(
                    cfg,
                    wrapped,
                    description,
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    None,
                    initial_states[state_index],
                    None,
                )
                wall_seconds = time.perf_counter() - started
                policy_seconds = timer["seconds"]
                steps = max(0, int(wrapped.steps) - int(cfg.num_steps_wait))
                if replay_images:
                    _save_frame(out, episode_id, "start", replay_images[0])
                    _save_frame(out, episode_id, "end", replay_images[-1])
                record = {
                    "episode_id": episode_id,
                    "suite": suite,
                    "task_id": task_spec["task_id"],
                    "task": task_spec["task"],
                    "language": task_spec["language"],
                    "initial_state_index": int(state_index),
                    "checkpoint": checkpoint,
                    "official_success": bool(success),
                    "steps": steps,
                    "wall_seconds": wall_seconds,
                    "policy_seconds": policy_seconds,
                    "policy_calls": timer["calls"],
                    "sim_other_seconds": max(0.0, wall_seconds - policy_seconds),
                    "max_steps": int(eval_mod.TASK_MAX_STEPS[eval_mod.TaskSuite(suite)]),
                }
                records.append(record)
                print(
                    f"[oft] {episode_id} success={record['official_success']} "
                    f"steps={steps} wall={wall_seconds:.1f}s policy={policy_seconds:.1f}s",
                    flush=True,
                )
                with (out / "episodes.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if commit_fn is not None and len(records) % 100 == 0:
                    commit_fn()
            try:
                env.close()
            except Exception:
                pass

        if commit_fn is not None:
            commit_fn()
        del model, action_head, proprio_projector, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = len(records)
    successes = sum(1 for record in records if record["official_success"])
    by_suite: Dict[str, Dict[str, Any]] = {}
    for record in records:
        bucket = by_suite.setdefault(record["suite"], {"episodes": 0, "successes": 0})
        bucket["episodes"] += 1
        bucket["successes"] += int(record["official_success"])
    for bucket in by_suite.values():
        bucket["success_rate"] = bucket["successes"] / bucket["episodes"] if bucket["episodes"] else 0.0
    summary.update(
        {
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "by_suite": by_suite,
            "full_suite": full_suite,
            "num_states": num_states,
            "records": records,
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if commit_fn is not None:
        commit_fn()
    return summary
