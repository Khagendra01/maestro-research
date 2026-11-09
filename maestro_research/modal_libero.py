"""Modal runner for the LIBERO adapter.

This machine has no CUDA GPU and 3.1 GiB free disk, so LIBERO physics/render and
any model inference run in a pinned Modal image. The local repository holds the
adapter, manifests and results. Nothing here touches physical hardware.

Usage:
  modal run maestro_research/modal_libero.py::probe
  modal run maestro_research/modal_libero.py::smoke --settings oracle --limit 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

app = modal.App("maestro-research-libero")


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
volume = modal.Volume.from_name("maestro-research-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libosmesa6-dev", "libgl1", "libglew-dev", "patchelf", "ffmpeg")
    .pip_install("torch==2.2.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        f"cd /opt/LIBERO && git checkout {LIBERO_COMMIT}",
    )
    .pip_install(
        "numpy==1.23.5",
        "mujoco==2.3.7",
        "robosuite==1.4.0",
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
    )
    .run_commands("pip install -e /opt/LIBERO")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": "/root"})
    .add_local_dir("maestro_research", "/root/maestro_research")
)


def _manifest_text() -> str:
    path = Path("/root/maestro_research/configs/dev_smoke_r2.json")
    if not path.exists():
        raise FileNotFoundError(f"manifest not found from cwd {Path.cwd()}: {path}")
    return path.read_text()


@app.function(image=image, volumes={"/data": volume}, secrets=[modal.Secret.from_name("meta-model-api")], cpu=8, memory=16384, timeout=7200)
def smoke(
    settings: str = "oracle",
    limit: int = 5,
    resolution: int = 256,
    convention: str = "bottom_left",
    executor_config: str = "",
    run_name: str = "",
) -> dict:
    import time

    from maestro_research.cli import run

    stamp = run_name or time.strftime("%Y%m%d-%H%M%S")
    out = f"/data/runs/{stamp}-{settings}"
    manifest_path = Path("/tmp/dev_smoke_r2.json")
    manifest_path.write_text(_manifest_text())
    exec_path = None
    if executor_config:
        exec_path = "/tmp/executor_config.json"
        Path(exec_path).write_text(executor_config)
    summary = run(
        config=manifest_path,
        backend_name="libero",
        settings=settings,
        out_dir=out,
        limit_per_task=limit,
        resolution=resolution,
        executor_config=exec_path,
        save_frames=True,
        convention=convention,
    )
    volume.commit()
    return {
        "out": out,
        "episodes": summary["episodes"],
        "successes": summary["successes"],
        "success_rate": summary["success_rate"],
        "by_group": summary["by_group"],
        "records": [
            {
                key: record.get(key)
                for key in (
                    "episode_id",
                    "suite",
                    "official_success",
                    "status",
                    "reason_code",
                    "steps",
                    "wall_seconds",
                    "grasp_attempts",
                    "stage_errors",
                    "target_id",
                    "receptacle_id",
                    "vlm",
                )
            }
            for record in summary["records"]
        ],
    }


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384, timeout=3600)
def rgbd_probe(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
    convention: str = "bottom_left",
) -> dict:
    from types import SimpleNamespace

    import numpy as np

    from maestro_research.backends.libero import LiberoBackend
    from maestro_research.perception.localize import _cluster_xy, _table_height, estimate_objects

    spec = SimpleNamespace(suite=suite, task=task, task_id="probe", language="")
    backend = LiberoBackend(resolution=256, enable_depth=True, convention=convention)
    obs = backend.reset(spec, state)
    points = backend.pointcloud(obs)
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    flat = flat[valid]
    rgb = np.asarray(obs.rgb).reshape(-1, 3)[valid]
    table_z = _table_height(flat)
    above = flat[:, 2] > table_z + 0.012
    objects = estimate_objects(obs, backend)
    raw_clusters = _cluster_xy(flat[above], rgb[above])
    return {
        "clusters": [
            {
                "n_points": int(pts.shape[0]),
                "mean_rgb": np.asarray(cols).mean(axis=0).tolist(),
                "center": np.median(pts, axis=0).tolist(),
                "span_z": float(np.ptp(pts[:, 2])),
                "radius": float(np.max(np.linalg.norm(pts[:, :2] - np.median(pts, axis=0)[:2], axis=1))),
            }
            for pts, cols in raw_clusters
        ],
        "depth_stats": [float(np.nanmin(obs.depth)), float(np.nanmedian(obs.depth)), float(np.nanmax(obs.depth))],
        "points_valid": int(flat.shape[0]),
        "z_min_median_max": [float(flat[:, 2].min()), float(np.median(flat[:, 2])), float(flat[:, 2].max())],
        "table_z": table_z,
        "above_count": int(above.sum()),
        "objects": [
            {
                "type": o["type"],
                "position": np.asarray(o["position"]).tolist(),
                "top_z": float(o["top_z"]),
                "bottom_z": float(o["bottom_z"]),
                "radius": float(o["radius"]),
                "n_points": int(o["n_points"]),
                "mean_rgb": np.asarray(o["mean_rgb"]).tolist(),
            }
            for o in objects
        ],
        "oracle": {k: v.tolist() for k, v in backend.oracle_positions().items()},
    }


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384, timeout=3600)
def pixel_probe(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
) -> dict:
    from types import SimpleNamespace

    import numpy as np

    from maestro_research.backends.libero import LiberoBackend

    spec = SimpleNamespace(suite=suite, task=task, task_id="probe", language="")
    backend = LiberoBackend(resolution=256, enable_depth=True)
    obs = backend.reset(spec, state)
    points = backend.pointcloud(obs)
    h, w = points.shape[:2]
    oracle = backend.oracle_positions()
    samples = {"target": (0.52, 0.42), "receptacle": (0.75, 0.27)}
    out = {}
    for name, (xn, yn) in samples.items():
        col = int(round(xn * (w - 1)))
        row_top = int(round(yn * (h - 1)))
        entry = {"col": col, "row_top": row_top}
        for label, row in (("top_left", row_top), ("bottom_left", h - 1 - row_top)):
            pt = points[row, col]
            nearest = min(
                (float(np.linalg.norm(pt - p)), k) for k, p in oracle.items()
            )
            entry[label] = {"xyz": pt.tolist(), "nearest": nearest}
        out[name] = entry
    return out


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384, timeout=3600)
def probe(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
    grasp_probe: bool = True,
    action_probe: bool = True,
) -> dict:
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from maestro_research.backends.libero import LiberoBackend
    from maestro_research.perception.cameras import unproject_depth
    from maestro_research.skills import ExecutorConfig, StagedExecutor

    spec = SimpleNamespace(suite=suite, task=task, task_id="probe", language="")
    backend = LiberoBackend(resolution=256, enable_depth=True)
    obs = backend.reset(spec, state)
    positions = backend.oracle_positions()
    report = {
        "suite": suite,
        "task": task,
        "state": state,
        "camera_pose": obs.camera_pose.tolist(),
        "intrinsics": obs.intrinsics.tolist(),
        "eef": obs.robot_state.eef_position.tolist(),
        "gripper_qpos": obs.robot_state.gripper_qpos.tolist(),
        "depth_stats": {
            "min": float(np.nanmin(obs.depth)),
            "median": float(np.nanmedian(obs.depth)),
            "max": float(np.nanmax(obs.depth)),
        },
        "bodies": {name: value.tolist() for name, value in positions.items()},
        "conventions": {},
    }
    out_dir = Path("/data/probes") / f"{suite}-{state}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs.rgb).save(out_dir / "agentview_rgb.png")
    np.save(out_dir / "agentview_depth.npy", obs.depth)

    probes = [name for name in positions if name.startswith(("akita_black_bowl", "plate_1", "basket_1", "alphabet_soup_1", "main_table"))]
    for convention in ("bottom_left", "top_left"):
        points = unproject_depth(obs.depth, obs.intrinsics, obs.camera_pose, convention=convention)
        flat = points.reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1)
        flat = flat[valid]
        distances = {}
        for name in probes:
            center = np.asarray(positions[name], dtype=np.float64)
            d = np.linalg.norm(flat - center, axis=1)
            distances[name] = float(d.min())
        report["conventions"][convention] = {
            "valid_points": int(flat.shape[0]),
            "nearest_point_distance_to_body_center": distances,
            "pointcloud_z_median": float(np.median(flat[:, 2])),
        }
        np.save(out_dir / f"pointcloud_{convention}.npy", points.astype(np.float32))

    if grasp_probe:
        from maestro_research.contracts import pose_from_pos_quat, quat_xyzw_to_matrix

        state0 = backend.robot_state()
        rot = quat_xyzw_to_matrix(state0.eef_quat_xyzw)
        report["eef_quat_xyzw"] = state0.eef_quat_xyzw.tolist()
        report["eef_axes_world"] = {
            "x": rot[:, 0].tolist(),
            "y": rot[:, 1].tolist(),
            "z": rot[:, 2].tolist(),
        }
        names = [n for n in positions if n.startswith("akita_black_bowl") or n.startswith("alphabet_soup_1")]
        target_id = sorted(names)[0] if names else None

        def _attempt_pose(label: str, grasp_pos, quat) -> dict:
            executor = StagedExecutor(ExecutorConfig(grasp_retries=0))
            backend.reset(spec, state)
            pose = pose_from_pos_quat(grasp_pos, quat)
            pregrasp = pose.copy()
            pregrasp[:3, 3] = np.asarray(grasp_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.12])
            positions_now = backend.oracle_positions()
            z_before = float(positions_now[target_id][2])
            ok_pre, pre_err, pre_reason = executor._move(
                backend, pregrasp[:3, 3], -1.0, None, "PREGRASP", 400, tolerance=0.014
            )
            eef_after_pre = backend.robot_state().eef_position.tolist()
            ok_app, app_err, app_reason = executor._move(
                backend, pose[:3, 3], -1.0, None, "APPROACH", 400, tolerance=0.014, speed_scale=0.6
            )
            eef_after_app = backend.robot_state().eef_position.tolist()
            sep_open = backend.robot_state().finger_separation()
            executor._hold(backend, 1.0, 10, 400)
            sep_closed = backend.robot_state().finger_separation()
            lift_target = np.asarray(grasp_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.12])
            lift_ok, lift_err, lift_reason = executor._move(
                backend, lift_target, 1.0, None, "LIFT", 400, tolerance=0.014, speed_scale=0.7
            )
            eef_after_lift = backend.robot_state().eef_position.tolist()
            z_after = float(backend.oracle_positions()[target_id][2])
            return {
                "label": label,
                "grasp_pos": np.asarray(grasp_pos, dtype=np.float64).tolist(),
                "eef_quat_xyzw": np.asarray(quat, dtype=np.float64).tolist(),
                "pregrasp_ok": ok_pre,
                "pregrasp_error": pre_err,
                "pregrasp_reason": pre_reason,
                "eef_after_pregrasp": eef_after_pre,
                "approach_ok": ok_app,
                "approach_error": app_err,
                "approach_reason": app_reason,
                "eef_after_approach": eef_after_app,
                "sep_open": sep_open,
                "sep_closed": sep_closed,
                "lift_ok": lift_ok,
                "lift_error": lift_err,
                "lift_reason": lift_reason,
                "eef_after_lift": eef_after_lift,
                "object_dz": z_after - z_before,
                "retained": bool(z_after - z_before > 0.02),
                "steps": backend.step_count,
            }

        attempts = []
        if target_id:
            extent = backend.body_extent(target_id)
            report["target_extent"] = _jsonable(extent)
            center = np.asarray(positions[target_id], dtype=np.float64)
            radius = float(extent["radius"])
            drop = 0.02
            identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
            attempts.append(_attempt_pose("center_identity", center + np.array([0.0, 0.0, -0.02]), identity_quat))
            for axis_name, axis in (("finger", rot[:, 1]), ("perp", rot[:, 0])):
                for sign in (1.0, -1.0):
                    grep = center.copy()
                    grep[:2] = center[:2] + sign * axis[:2] * radius
                    grep[2] = float(extent["top_z"]) - drop
                    attempts.append(_attempt_pose(f"rim_{axis_name}_{'pos' if sign > 0 else 'neg'}", grep, identity_quat))
        report["grasp_probe"] = attempts

    if action_probe:
        backend.reset(spec, state)
        start_eef = backend.robot_state().eef_position.copy()
        for _ in range(10):
            action = np.zeros(7)
            action[0] = 1.0
            action[6] = -1.0
            backend.step(action)
        plus_x = backend.robot_state().eef_position - start_eef
        backend.reset(spec, state)
        for _ in range(10):
            action = np.zeros(7)
            action[2] = 1.0
            action[6] = -1.0
            backend.step(action)
        plus_z = backend.robot_state().eef_position - start_eef
        backend.reset(spec, state)
        for _ in range(10):
            action = np.zeros(7)
            action[6] = -1.0
            backend.step(action)
        sep_open = backend.robot_state().finger_separation()
        for _ in range(10):
            action = np.zeros(7)
            action[6] = 1.0
            backend.step(action)
        sep_closed = backend.robot_state().finger_separation()
        report["action_probe"] = {
            "ten_steps_plus_x_displacement": plus_x.tolist(),
            "ten_steps_plus_z_displacement": plus_z.tolist(),
            "sep_after_open_command": sep_open,
            "sep_after_close_command": sep_closed,
        }

    report_path = out_dir / "probe.json"
    report_path.write_text(json.dumps(report, indent=2, default=_jsonable))
    volume.commit()
    backend.close()
    return report


@app.local_entrypoint()
def run_probe(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
    grasp_probe: bool = True,
    action_probe: bool = True,
):
    report = probe.remote(suite=suite, task=task, state=state, grasp_probe=grasp_probe, action_probe=action_probe)
    print(json.dumps(report, indent=2, default=_jsonable))


@app.local_entrypoint()
def run_pixel_probe(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
):
    print(json.dumps(pixel_probe.remote(suite=suite, task=task, state=state), indent=2, default=_jsonable))


@app.local_entrypoint()
def run_rgbd_probe(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
    convention: str = "bottom_left",
):
    print(json.dumps(rgbd_probe.remote(suite=suite, task=task, state=state, convention=convention), indent=2, default=_jsonable))


@app.local_entrypoint()
def run_smoke(
    settings: str = "oracle",
    limit: int = 5,
    resolution: int = 256,
    convention: str = "bottom_left",
    executor_config: str = "",
    run_name: str = "",
):
    result = smoke.remote(
        settings=settings,
        limit=limit,
        resolution=resolution,
        convention=convention,
        executor_config=executor_config,
        run_name=run_name,
    )
    print(json.dumps(result, indent=2, default=_jsonable))


@app.local_entrypoint()
def main():
    print("Use one of: modal run maestro_research/modal_libero.py::run_probe | ::run_smoke", file=sys.stderr)
    raise SystemExit(2)


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384, timeout=3600)
def detail(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
    side: float = 1.0,
    drop: float = 0.02,
) -> dict:
    from types import SimpleNamespace

    import mujoco
    import numpy as np

    from maestro_research.backends.libero import LiberoBackend
    from maestro_research.contracts import pose_from_pos_quat, quat_xyzw_to_matrix
    from maestro_research.skills import ExecutorConfig, StagedExecutor

    def _name(objtype, index: int) -> str:
        if int(objtype) == int(mujoco.mjtObj.mjOBJ_GEOM):
            value = sim.model.geom_id2name(int(index))
        elif int(objtype) == int(mujoco.mjtObj.mjOBJ_BODY):
            value = sim.model.body_id2name(int(index))
        else:
            value = None
        return value if value is not None else f"<unnamed:{index}>"

    spec = SimpleNamespace(suite=suite, task=task, task_id="detail", language="")
    backend = LiberoBackend(resolution=256, enable_depth=True)
    backend.reset(spec, state)
    sim = backend.env.sim
    model = sim.model
    positions = backend.oracle_positions()
    target_id = sorted(n for n in positions if n.startswith("akita_black_bowl") or n.startswith("alphabet_soup_1"))[0]
    extent = backend.body_extent(target_id)
    state0 = backend.robot_state()
    rot = quat_xyzw_to_matrix(state0.eef_quat_xyzw)
    center = np.asarray(positions[target_id], dtype=np.float64)
    axis = rot[:, 1] * side
    grasp = center.copy()
    grasp[:2] = center[:2] + axis[:2] * float(extent["radius_top"])
    grasp[2] = float(extent["top_z"]) - drop

    body_id = model.body_name2id(target_id)
    geoms = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        geoms.append(
            {
                "geom": _name(mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "type": int(model.geom_type[geom_id]),
                "size": np.asarray(model.geom_size[geom_id]).tolist(),
                "xpos": np.asarray(sim.data.geom_xpos[geom_id]).tolist(),
                "rbound": float(model.geom_rbound[geom_id]),
            }
        )

    executor = StagedExecutor(ExecutorConfig(grasp_retries=0))
    pose = pose_from_pos_quat(grasp, np.array([0.0, 0.0, 0.0, 1.0]))
    pregrasp = pose.copy()
    pregrasp[:3, 3] = grasp + np.array([0.0, 0.0, 0.12])
    ok_pre, pre_err, pre_reason = executor._move(backend, pregrasp[:3, 3], -1.0, None, "PREGRASP", 400, tolerance=0.014)
    ok_app, app_err, app_reason = executor._move(
        backend, grasp, -1.0, None, "APPROACH", 400, tolerance=0.014, speed_scale=0.6
    )

    contacts = []
    for i in range(int(sim.data.ncon)):
        contact = sim.data.contact[i]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        contacts.append(
            {
                "geom1": _name(mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                "body1": _name(mujoco.mjtObj.mjOBJ_BODY, body1),
                "geom2": _name(mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                "body2": _name(mujoco.mjtObj.mjOBJ_BODY, body2),
                "pos": np.asarray(contact.pos).tolist(),
                "dist": float(contact.dist),
            }
        )
    finger_geoms = []
    for geom_id in range(model.ngeom):
        body_name = _name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        if "finger" in body_name or "right_hand" in body_name:
            finger_geoms.append(
                {
                    "geom": _name(mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                    "body": body_name,
                    "type": int(model.geom_type[geom_id]),
                    "size": np.asarray(model.geom_size[geom_id]).tolist(),
                    "xpos": np.asarray(sim.data.geom_xpos[geom_id]).tolist(),
                    "rbound": float(model.geom_rbound[geom_id]),
                }
            )

    eef_after_approach = backend.robot_state().eef_position.tolist()
    sep_open = backend.robot_state().finger_separation()
    executor._hold(backend, 1.0, 10, 400)
    sep_closed = backend.robot_state().finger_separation()
    z_before = float(backend.oracle_positions()[target_id][2])
    executor._move(backend, grasp + np.array([0.0, 0.0, 0.12]), 1.0, None, "LIFT", 400, tolerance=0.014,
                   speed_scale=0.7)
    z_after = float(backend.oracle_positions()[target_id][2])
    volume.commit()
    return {
        "extent": _jsonable(extent),
        "radius_top": extent["radius_top"],
        "grasp": grasp.tolist(),
        "axis": axis.tolist(),
        "eef_after_approach": eef_after_approach,
        "pregrasp": {"ok": ok_pre, "error": pre_err, "reason": pre_reason},
        "approach": {"ok": ok_app, "error": app_err, "reason": app_reason},
        "sep_open": sep_open,
        "sep_closed": sep_closed,
        "object_dz": z_after - z_before,
        "geoms": geoms,
        "finger_geoms": finger_geoms,
        "contacts": contacts,
    }


@app.local_entrypoint()
def run_detail(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
    side: float = 1.0,
    drop: float = 0.02,
):
    result = detail.remote(suite=suite, task=task, state=state, side=side, drop=drop)
    print(json.dumps(result, indent=2, default=_jsonable))


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384, timeout=3600)
def grid(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
) -> dict:
    from types import SimpleNamespace

    import mujoco
    import numpy as np

    from maestro_research.backends.libero import LiberoBackend
    from maestro_research.contracts import pose_from_pos_quat, quat_xyzw_to_matrix
    from maestro_research.skills import ExecutorConfig, StagedExecutor

    spec = SimpleNamespace(suite=suite, task=task, task_id="grid", language="")
    backend = LiberoBackend(resolution=256, enable_depth=False)
    out = []
    cases = [
        ("center", 0.005, 0.0),
        ("center", 0.015, 0.0),
        ("center", 0.025, 0.0),
        ("center", 0.035, 0.0),
        ("rim", 0.002, 0.0),
        ("rim", 0.008, 0.0),
        ("rim", 0.015, 0.0),
        ("rim", 0.022, 0.0),
        ("rim", 0.008, 0.004),
        ("rim", 0.015, 0.004),
    ]
    for strategy, drop, radius_delta in cases:
        backend.reset(spec, state)
        positions = backend.oracle_positions()
        target_id = sorted(
            n for n in positions
            if n.startswith("akita_black_bowl") or n.startswith("alphabet_soup_1")
        )[0]
        extent = backend.body_extent(target_id)
        center = np.asarray(positions[target_id], dtype=np.float64)
        state0 = backend.robot_state()
        finger_axis = quat_xyzw_to_matrix(state0.eef_quat_xyzw)[:, 1]
        side = float(np.dot(state0.eef_position[:2] - center[:2], finger_axis[:2]))
        direction = finger_axis if side >= 0 else -finger_axis
        if strategy == "center":
            grasp_xy = center[:2]
        else:
            grasp_xy = center[:2] + direction[:2] * (float(extent["radius_top"]) + radius_delta)
        grasp = np.array([grasp_xy[0], grasp_xy[1], float(extent["top_z"]) - drop])
        executor = StagedExecutor(ExecutorConfig(grasp_retries=0))
        pose = pose_from_pos_quat(grasp, np.array([0.0, 0.0, 0.0, 1.0]))
        pregrasp = pose.copy()
        pregrasp[:3, 3] = grasp + np.array([0.0, 0.0, 0.12])
        executor._move(backend, pregrasp[:3, 3], -1.0, None, "PREGRASP", 400, tolerance=0.014)
        ok_app, app_err, app_reason = executor._move(
            backend, grasp, -1.0, None, "APPROACH", 400, tolerance=0.02, speed_scale=0.6
        )
        executor._hold(backend, 1.0, 12, 400)
        sep = backend.robot_state().finger_separation()
        z_before = float(backend.oracle_positions()[target_id][2])
        lift_ok, lift_err, lift_reason = executor._move(
            backend, grasp + np.array([0.0, 0.0, 0.12]), 1.0, None, "LIFT", 400, tolerance=0.02,
            speed_scale=0.7,
        )
        z_after = float(backend.oracle_positions()[target_id][2])
        model = backend.env.sim.model
        data = backend.env.sim.data
        target_body = int(model.body_name2id(target_id))
        touching = []
        for i in range(int(data.ncon)):
            contact = data.contact[i]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            if target_body not in (b1, b2):
                continue
            other = b2 if b1 == target_body else b1
            other_name = model.body_id2name(other) or ""
            if "finger" in other_name or "gripper" in other_name:
                touching.append(other_name)
        out.append(
            {
                "strategy": strategy,
                "drop": drop,
                "radius_delta": radius_delta,
                "grasp": grasp.tolist(),
                "extent": _jsonable(extent),
                "approach_ok": ok_app,
                "approach_error": app_err,
                "approach_reason": app_reason,
                "sep": sep,
                "lift_ok": lift_ok,
                "lift_error": lift_err,
                "lift_reason": lift_reason,
                "dz": z_after - z_before,
                "retained": bool(z_after - z_before > 0.02),
                "touching": sorted(set(touching)),
            }
        )
    volume.commit()
    return {"suite": suite, "task": task, "cases": out}


@app.local_entrypoint()
def run_grid(
    suite: str = "libero_spatial",
    task: str = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    state: int = 0,
):
    print(json.dumps(grid.remote(suite=suite, task=task, state=state), indent=2, default=_jsonable))
