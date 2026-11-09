"""Modal runner for the OpenVLA-OFT learned-policy reference.

The reference needs a CUDA GPU, the OpenVLA-OFT repository plus its custom
transformers fork, LIBERO, and three 7B checkpoints. This machine has no CUDA;
all of that runs in this pinned Modal image. The reference is evaluated only on
the frozen 15-episode dev smoke, with the same task names and initial-state
indices 0..4 as the modular runs.

Usage:
  modal run maestro_research/modal_oft.py::download_checkpoints
  modal run maestro_research/modal_oft.py::probe_oft --gl egl
  modal run maestro_research/modal_oft.py::run_oft_smoke --limit 5
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

app = modal.App("maestro-oft-libero")
volume = modal.Volume.from_name("maestro-research-data", create_if_missing=True)

REPOS = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
}
LOCAL_CHECKPOINTS = {
    "libero_spatial": "/data/checkpoints/openvla-oft-libero-spatial",
    "libero_object": "/data/checkpoints/openvla-oft-libero-object",
    "libero_goal": "/data/checkpoints/openvla-oft-libero-goal",
}

oft_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "clang",
        "libosmesa6-dev",
        "libgl1",
        "libglvnd0",
        "libegl1",
        "libgles2",
        "libglew-dev",
        "patchelf",
        "ffmpeg",
    )
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft",
        f"cd /opt/openvla-oft && git checkout {OFT_COMMIT}",
    )
    .pip_install(
        "numpy==1.23.5",
        "mujoco==2.3.7",
        "robosuite==1.4.1",
        "bddl==1.0.1",
        "gym==0.25.2",
        "cloudpickle==2.1.0",
        "future",
        "easydict",
        "hydra-core==1.2.0",
        "matplotlib",
        "termcolor",
        "numba",
        "scipy",
        "opencv-python-headless",
        "h5py",
        "Pillow",
        "pyyaml",
        "imageio[ffmpeg]",
    )
    .run_commands("cd /opt/openvla-oft && pip install -e .")
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        f"cd /opt/LIBERO && git checkout {LIBERO_COMMIT}",
        "pip install -e /opt/LIBERO",
    )
    .pip_install("numpy==1.23.5", "mujoco==2.3.7", "protobuf==4.21.6", "tensorflow-metadata==1.17.0")
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": "/opt/openvla-oft:/root",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "WANDB_MODE": "disabled",
        }
    )
    .add_local_dir("maestro_research", "/root/maestro_research")
)


@app.function(image=oft_image, timeout=600)
def versions() -> dict:
    import importlib.metadata as md

    out = {}
    for name in ("protobuf", "tensorflow", "tensorflow-datasets", "tensorflow-metadata", "transformers", "torch", "robosuite", "libero"):
        try:
            out[name] = md.version(name)
        except Exception as exc:
            out[name] = f"ERROR: {exc}"
    print(out, flush=True)
    return out


@app.function(image=oft_image, volumes={"/data": volume}, cpu=4, timeout=7200)
def download_checkpoints() -> dict:
    from huggingface_hub import snapshot_download

    out = {}
    for suite, repo in REPOS.items():
        local_dir = LOCAL_CHECKPOINTS[suite]
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=local_dir, max_workers=8)
        files = sorted(p.name for p in Path(local_dir).iterdir())
        out[suite] = {"repo": repo, "local_dir": local_dir, "files": files}
        print(f"[download] {suite}: {repo} -> {local_dir}", flush=True)
    volume.commit()
    return out


@app.function(image=oft_image, volumes={"/data": volume}, gpu="A100", timeout=1800)
def probe_oft(gl: str = "egl") -> dict:
    import os

    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    import numpy as np
    import torch
    from maestro_research.oft_reference import configure_libero_paths

    configure_libero_paths()
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env

    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    names = [task.name for task in bench.tasks]
    task = bench.get_task(0)
    env, description = get_libero_env(task, "openvla", resolution=256)
    obs = env.reset()
    image = np.asarray(obs["agentview_image"])
    env.close()
    result = {
        "cuda": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gl": gl,
        "task_names": names,
        "first_task": task.name,
        "description": description,
        "image_shape": list(image.shape),
        "image_min_max": [int(image.min()), int(image.max())],
        "checkpoints": {suite: Path(path).is_dir() for suite, path in LOCAL_CHECKPOINTS.items()},
    }
    print(json.dumps(result, default=str), flush=True)
    return result


@app.function(image=oft_image, volumes={"/data": volume}, gpu="A100", timeout=14400)
def run_oft_smoke(limit: int = 5, run_name: str = "", gl: str = "egl") -> dict:
    import os

    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    from maestro_research.oft_reference import run_reference

    stamp = run_name or time.strftime("%Y%m%d-%H%M%S")
    out = f"/data/runs/{stamp}-oft"
    summary = run_reference(
        manifest_path="/root/maestro_research/configs/dev_smoke_r2.json",
        out_dir=out,
        limit_per_task=limit,
        seed=7,
    )
    summary["out"] = out
    summary["gl"] = gl
    volume.commit()
    result = {
        "out": out,
        "episodes": summary["episodes"],
        "successes": summary["successes"],
        "success_rate": summary["success_rate"],
        "by_suite": summary["by_suite"],
        "model_load_seconds": summary.get("model_load_seconds", {}),
        "gl": gl,
        "records": summary["records"],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, default=str), flush=True)
    return result
