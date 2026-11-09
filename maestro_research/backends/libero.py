"""LIBERO simulation backend.

Uses the official `OffScreenRenderEnv`, official initial states and the official
7-D OSC_POSE action interface. Physics owns contacts, grasp retention and
placement. No qpos teleporting, object welding or asset edits occur here.

This module imports LIBERO lazily so the rest of the package stays runnable on a
machine without the simulator.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..contracts import Observation, RobotState
from ..perception.cameras import (
    depth_buffer_to_meters,
    mujoco_camera_pose,
    mujoco_intrinsics,
    unproject_depth,
)
from .base import BackendInfo, StepResult

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CAMERA_NAME = "agentview"
LIBERO_ROOT = "/opt/LIBERO"


def _configure_libero_paths(root: str = LIBERO_ROOT) -> None:
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"LIBERO checkout not found at {root}; run inside the pinned Modal image")
    config = Path.home() / ".libero" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"benchmark_root: {root_path}/libero/libero\n"
        f"bddl_files: {root_path}/libero/libero/bddl_files\n"
        f"init_states: {root_path}/libero/libero/init_files\n"
        f"datasets: {root_path}/libero/datasets\n"
        f"assets: {root_path}/libero/libero/assets\n"
    )


def load_init_states(suite: str, task: str) -> list:
    from libero.libero import get_libero_path

    path = os.path.join(get_libero_path("init_states"), suite, task + ".pruned_init")
    if not os.path.exists(path):
        raise FileNotFoundError(f"official initial states not found: {path}")
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        import torch

        return torch.load(path, map_location="cpu")


class LiberoBackend:
    mode = "sim"
    info = BackendInfo(
        name="libero",
        mode="sim",
        robot="Panda",
        gripper="PandaGripper",
        control_hz=20.0,
        action_dim=7,
        action_semantics="OSC_POSE normalized deltas; gripper -1 open, +1 close",
        notes="official OffScreenRenderEnv, official initial states",
    )

    def __init__(
        self,
        resolution: int = 256,
        convention: str = "bottom_left",
        seed: int = 0,
        enable_depth: bool = True,
        num_steps_wait: int = 10,
    ):
        self.resolution = int(resolution)
        self.convention = convention
        self.seed = int(seed)
        self.enable_depth = bool(enable_depth)
        self.num_steps_wait = int(num_steps_wait)
        self.env = None
        self._env_key = None
        self._last_obs = None
        self._last_done = False
        self._frame = 0
        self.step_count = 0

    def _env_key_for(self, task_spec) -> tuple:
        return (task_spec.suite, task_spec.task)

    def _ensure_env(self, task_spec) -> None:
        key = self._env_key_for(task_spec)
        if self.env is not None and key == self._env_key:
            return
        _configure_libero_paths()
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bddl = os.path.join(get_libero_path("bddl_files"), task_spec.suite, task_spec.task + ".bddl")
        if not os.path.exists(bddl):
            raise FileNotFoundError(f"BDDL not found: {bddl}")
        self.close()
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self.resolution,
            camera_widths=self.resolution,
        )
        self.env.seed(self.seed)
        self._env_key = key
        self._init_states = None

    def reset(self, task_spec, initial_state_index: int) -> Observation:
        self._ensure_env(task_spec)
        if self._init_states is None:
            self._init_states = load_init_states(task_spec.suite, task_spec.task)
        if not (0 <= initial_state_index < len(self._init_states)):
            raise IndexError(
                f"initial state {initial_state_index} out of range for {task_spec.task} "
                f"({len(self._init_states)} available)"
            )
        self.env.reset()
        state = self._init_states[initial_state_index]
        self._last_obs = self.env.set_init_state(state)
        self._last_done = False
        self._frame = 0
        self.step_count = 0
        for _ in range(self.num_steps_wait):
            self._last_obs, _, done, _ = self.env.step(np.zeros(7, dtype=np.float64).tolist()[:-1] + [-1.0])
            if done:
                raise RuntimeError("task reported success during the protocol's stabilization wait")
        return self.observe()

    def _render_rgb_depth(self):
        if not self.enable_depth:
            raw = self.env.sim.render(camera_name=CAMERA_NAME, width=self.resolution, height=self.resolution)
            return raw, None
        rgb, raw_depth = self.env.sim.render(
            camera_name=CAMERA_NAME, width=self.resolution, height=self.resolution, depth=True
        )
        model = self.env.sim.model
        znear = float(model.vis.map.znear) * float(model.stat.extent)
        zfar = float(model.vis.map.zfar) * float(model.stat.extent)
        depth = depth_buffer_to_meters(raw_depth, znear, zfar)
        return rgb, depth

    def camera_pose(self) -> np.ndarray:
        model = self.env.sim.model
        data = self.env.sim.data
        cam_id = model.camera_name2id(CAMERA_NAME)
        return mujoco_camera_pose(data.cam_xpos[cam_id], data.cam_xmat[cam_id])

    def intrinsics(self) -> np.ndarray:
        model = self.env.sim.model
        cam_id = model.camera_name2id(CAMERA_NAME)
        return mujoco_intrinsics(float(model.cam_fovy[cam_id]), self.resolution, self.resolution)

    def robot_state(self) -> RobotState:
        if self._last_obs is None:
            self._last_obs = self.env.env._get_observations()
        obs = self._last_obs
        return RobotState(
            eef_position=np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            eef_quat_xyzw=np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
            gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            joints=np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float64),
            timestamp=time.monotonic(),
        )

    def observe(self) -> Observation:
        if self._last_obs is None:
            self._last_obs = self.env.env._get_observations()
        rgb, depth = self._render_rgb_depth()
        modalities = ("rgb", "depth") if depth is not None else ("rgb",)
        state = self.robot_state()
        return Observation(
            rgb=rgb,
            depth=depth,
            intrinsics=self.intrinsics(),
            camera_pose=self.camera_pose(),
            robot_state=state,
            calibration_id=f"libero-{LIBERO_COMMIT[:12]}-{CAMERA_NAME}",
            frame_id=f"libero-{self._env_key[0]}/{self._env_key[1]}@{self._frame}",
            acquisition_time=time.monotonic(),
            source="sim",
            permitted_modalities=modalities,
        )

    def step(self, action) -> StepResult:
        a = np.asarray(action, dtype=np.float64).reshape(7)
        if not np.isfinite(a).all():
            raise ValueError("action contains non-finite values")
        a = np.clip(a, -1.0, 1.0)
        obs, _, done, _ = self.env.step(a.tolist())
        self._last_obs = obs
        self._last_done = bool(done)
        self._frame += 1
        self.step_count += 1
        return StepResult(None, bool(done), {})

    def official_success(self) -> bool:
        return bool(self._last_done)

    def oracle_positions(self) -> Dict[str, np.ndarray]:
        model = self.env.sim.model
        data = self.env.sim.data
        out: Dict[str, np.ndarray] = {}
        for body_id in range(model.nbody):
            name = model.body_id2name(body_id)
            if not name or name == "world":
                continue
            out[name] = np.asarray(data.body_xpos[body_id], dtype=np.float64).copy()
        return out

    def table_center(self) -> Optional[np.ndarray]:
        positions = self.oracle_positions()
        for key in ("main_table", "table"):
            if key in positions:
                return positions[key]
        return None

    def body_extent(self, body_name: str) -> Dict[str, np.ndarray | float]:
        model = self.env.sim.model
        data = self.env.sim.data
        body_id = model.body_name2id(body_name)
        body_center = np.asarray(data.body_xpos[body_id], dtype=np.float64)
        import mujoco

        points = []
        body_geoms = [
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == int(body_id)
        ]
        has_primitive = any(
            int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)
            for geom_id in body_geoms
        )
        for geom_id in body_geoms:
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            geom_type = int(model.geom_type[geom_id])
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                if has_primitive:
                    continue
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
                points.append(vertices @ rotation.T + center)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                r = float(size[0])
                corners = np.array([[sx * r, sy * r, sz * r] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                points.append(corners @ rotation.T + center)
            elif geom_type in {
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            }:
                r, half = float(size[0]), float(size[1])
                corners = np.array(
                    [[sx * r, sy * r, sz * half] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                )
                points.append(corners @ rotation.T + center)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                corners = np.array(
                    [[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                )
                points.append(corners @ rotation.T + center)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                corners = np.array(
                    [[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                )
                points.append(corners @ rotation.T + center)
            else:
                r = float(model.geom_rbound[geom_id])
                corners = np.array([[sx * r, sy * r, sz * r] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                points.append(corners @ rotation.T + center)
        if not points:
            raise KeyError(f"body {body_name!r} has no geoms")
        all_points = np.concatenate(points, axis=0)
        radial = np.linalg.norm(all_points[:, :2] - body_center[None, :2], axis=1)
        top_z = float(all_points[:, 2].max())
        top_mask = all_points[:, 2] >= top_z - 0.01
        return {
            "top_z": top_z,
            "bottom_z": float(all_points[:, 2].min()),
            "radius": float(radial.max()),
            "radius_top": float(radial[top_mask].max()),
            "center": body_center,
        }

    def pointcloud(self, observation: Observation) -> np.ndarray:
        if observation.depth is None:
            raise RuntimeError("RGB-D observation required")
        return unproject_depth(
            observation.depth,
            observation.intrinsics,
            observation.camera_pose,
            convention=self.convention,
        )

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self._env_key = None
        self._last_obs = None
