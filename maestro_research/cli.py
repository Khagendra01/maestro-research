"""Single entry point: doctor / smoke / benchmark."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .hardware_gate import PHYSICAL_ENV_VAR, physical_authorized
from .manifest import load_manifest
from .skills import ExecutorConfig, StagedExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def doctor(as_json: bool = True) -> Dict[str, Any]:
    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "repo_root": str(REPO_ROOT),
        "git_sha": _git_sha(),
        "physical_execution_authorized": physical_authorized(),
        "physical_env_var": PHYSICAL_ENV_VAR,
        "modules": {
            name: _module_available(name)
            for name in ("numpy", "PIL", "scipy", "libero", "robosuite", "mujoco", "torch", "modal")
        },
        "credentials_present": {
            name: bool(os.environ.get(name)) for name in ("GEMINI_KEY", "OPENAI_API_KEY", "HF_TOKEN")
        },
        "maestro_config_present": (REPO_ROOT / "config" / "maestro.yaml").exists(),
        "notes": [
            "Physical execution is refused unless allow_physical is passed and "
            f"{PHYSICAL_ENV_VAR}=1.",
            "LIBERO must run inside the pinned Modal image on this machine.",
        ],
    }
    if as_json:
        print(json.dumps(report, indent=2))
    return report


def _backend(name: str, resolution: int, settings: str, convention: str = "bottom_left"):
    if name == "dryrun":
        from .backends.dryrun import DryRunBackend

        return DryRunBackend()
    if name == "libero":
        from .backends.libero import LiberoBackend

        return LiberoBackend(resolution=resolution, convention=convention, enable_depth=(settings in ("rgbd", "oracle", "vlm")))
    raise ValueError(f"unknown backend {name!r}")


def run(
    config: str | Path,
    backend_name: str = "dryrun",
    settings: str = "oracle",
    out_dir: str | Path | None = None,
    limit_per_task: int | None = 5,
    resolution: int = 256,
    executor_config: str | Path | None = None,
    save_frames: bool = True,
    max_episodes: int | None = None,
    convention: str = "bottom_left",
) -> Dict[str, Any]:
    from .episode_runner import run_episode, write_summary

    manifest = load_manifest(config)
    if out_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = REPO_ROOT / "runs" / "research" / f"{manifest.manifest_id}-{backend_name}-{settings}-{stamp}"
    out_dir = Path(out_dir)
    if backend_name == "dryrun" and settings != "oracle":
        raise ValueError("dryrun backend only supports oracle settings (it has no camera)")
    exec_cfg = ExecutorConfig()
    if executor_config is not None:
        exec_cfg = ExecutorConfig.from_dict(json.loads(Path(executor_config).read_text()))
    executor = StagedExecutor(exec_cfg)
    backend = _backend(backend_name, resolution, settings, convention)
    episodes = manifest.episodes(limit_per_task=limit_per_task)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    records = []
    print(f"manifest={manifest.manifest_id} sha256={manifest.digest()[:16]} episodes={len(episodes)}")
    print(f"backend={backend_name} settings={settings} out={out_dir}")
    try:
        for episode in episodes:
            record = run_episode(
                backend, episode, manifest, executor, out_dir,
                settings=settings, save_frames=save_frames, execute_config=exec_cfg,
            )
            records.append(record)
            print(
                f"{record['episode_id']:<48} official={record.get('official_success')} "
                f"status={record.get('status')} reason={record.get('reason_code')} "
                f"steps={record.get('steps')} wall={record.get('wall_seconds', 0):.1f}s"
            )
    finally:
        backend.close()
    summary_path = write_summary(
        out_dir,
        records,
        extra={
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.digest(),
            "settings": settings,
            "backend": backend_name,
            "limit_per_task": limit_per_task,
            "executor_config": vars(exec_cfg),
        },
    )
    summary = json.loads(summary_path.read_text())
    print(f"summary: {summary_path}")
    print(
        f"official successes: {summary['successes']}/{summary['episodes']} "
        f"({(summary['success_rate'] or 0.0) * 100:.1f}% on this declared smoke set)"
    )
    print("This smoke is an integration check, not a generalization result.")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="maestro_research")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="print an environment report")
    doctor_parser.add_argument("--text", action="store_true", help="human-readable instead of JSON")

    for name in ("smoke", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="pinned manifest JSON")
        p.add_argument("--backend", default="dryrun", choices=("dryrun", "libero"))
        p.add_argument("--settings", default="oracle", choices=("oracle", "rgbd", "vlm"))
        p.add_argument("--out", default=None)
        p.add_argument("--resolution", type=int, default=256)
        p.add_argument("--convention", default="bottom_left", choices=("bottom_left", "top_left"))
        p.add_argument("--executor-config", default=None)
        p.add_argument("--no-frames", action="store_true")
        if name == "smoke":
            p.add_argument("--limit-per-task", type=int, default=5)
            p.add_argument("--max-episodes", type=int, default=None)
        else:
            p.add_argument("--limit-per-task", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = doctor(as_json=True)
        if args.text:
            for key, value in report.items():
                print(f"{key}: {value}")
        return 0
    limit = args.limit_per_task if args.limit_per_task is not None else None
    run(
        config=args.config,
        backend_name=args.backend,
        settings=args.settings,
        out_dir=args.out,
        limit_per_task=limit,
        resolution=args.resolution,
        executor_config=args.executor_config,
        save_frames=not args.no_frames,
        max_episodes=getattr(args, "max_episodes", None),
        convention=args.convention,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
