"""Deterministic toy backend used to test contracts, the executor and the CLI.

It is not a robot simulation and must never be reported as benchmark evidence.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..contracts import Observation, RobotState
from ..selector import parse_instruction
from .base import BackendInfo, StepResult


class _ToyTask:
    def __init__(self, language: str = "Pick the alphabet soup and place it in the basket"):
        self.language = language
        self.name = "dryrun"
        self.suite = "dryrun"
        reference = parse_instruction(language)
        self.target_label = reference.target_type
        self.receptacle_label = reference.receptacle_type
        self.target_xy = np.array([0.30, 0.00])
        self.receptacle_xy = np.array([0.30, 0.25])


class DryRunBackend:
    mode = "sim"
    info = BackendInfo(
        name="dryrun",
        mode="sim",
        robot="toy",
        gripper="toy",
        control_hz=20.0,
        action_dim=7,
        action_semantics="normalized cartesian delta + gripper",
        notes="contracts/executor test double; not benchmark evidence",
    )

    def __init__(self, task: Optional[Any] = None, scale: float = 0.05):
        self.task = task if task is not None else _ToyTask()
        self.scale = scale
        self.eef = np.array([0.0, 0.0, 0.30])
        self.object_pose = np.array([*self.task.target_xy, 0.04])
        self.held = False
        self.success = False
        self._step_count = 0
        self._released_at = None

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self, task=None, initial_state_index: int = 0) -> Observation:
        if task is not None:
            self.task = task if isinstance(task, _ToyTask) else _ToyTask(task.language)
        self.eef = np.array([0.0, 0.0, 0.30])
        self.object_pose = np.array([*self.task.target_xy, 0.04])
        self.held = False
        self.success = False
        self._step_count = 0
        self._released_at = None
        return self.observe()

    def robot_state(self) -> RobotState:
        grip = np.array([0.04, 0.04]) if not self.held else np.array([0.005, 0.005])
        return RobotState(
            eef_position=self.eef.copy(),
            eef_quat_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_qpos=grip,
            timestamp=time.monotonic(),
        )

    def observe(self) -> Observation:
        return Observation(
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            intrinsics=np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]]),
            camera_pose=np.eye(4),
            robot_state=self.robot_state(),
            calibration_id="dryrun",
            frame_id=f"step{self._step_count}",
            acquisition_time=time.monotonic(),
            depth=None,
        )

    def step(self, action) -> StepResult:
        a = np.asarray(action, dtype=np.float64).reshape(7)
        if not np.isfinite(a).all():
            raise ValueError("action contains non-finite values")
        a = np.clip(a, -1.0, 1.0)
        self.eef = self.eef + a[:3] * self.scale
        self.eef[2] = max(self.eef[2], 0.02)
        if self.held:
            self.object_pose = self.eef + np.array([0.0, 0.0, -0.02])
        if a[6] > 0 and not self.held:
            if np.linalg.norm(self.eef - self.object_pose) < 0.06:
                self.held = True
                self.object_pose = self.eef + np.array([0.0, 0.0, -0.02])
        elif a[6] < 0 and self.held:
            self.held = False
            self._released_at = self.object_pose.copy()
            self.object_pose[2] = 0.04
        if self._released_at is not None and not self.held:
            receptacle = np.array([*self.task.receptacle_xy, 0.04])
            if np.linalg.norm(self.object_pose[:2] - receptacle[:2]) < 0.06:
                self.success = True
        self._step_count += 1
        return StepResult(None, self.success, {})

    def official_success(self) -> bool:
        return bool(self.success)

    def oracle_positions(self) -> Dict[str, np.ndarray]:
        return {
            f"{self.task.target_label}_1": self.object_pose.copy(),
            f"{self.task.receptacle_label}_1": np.array([*self.task.receptacle_xy, 0.0]),
        }

    def table_center(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def body_extent(self, body_name: str) -> Dict[str, Any]:
        if body_name.startswith(self.task.target_label + "_"):
            return {"top_z": 0.08, "bottom_z": 0.0, "radius": 0.03,
                    "center": self.object_pose.copy()}
        return {"top_z": 0.05, "bottom_z": 0.0, "radius": 0.06,
                "center": np.array([*self.task.receptacle_xy, 0.0])}

    def close(self) -> None:
        pass
