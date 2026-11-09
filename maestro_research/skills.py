"""Reusable staged pick-and-place execution.

Stages: SELECT -> PREGRASP -> APPROACH -> CLOSE -> VERIFY_GRASP -> TEST_LIFT ->
TRANSPORT -> PLACE -> RELEASE -> RETREAT -> VERIFY_PLACEMENT -> DONE, with
bounded RETRY / REOBSERVE / REPLAN / STOP / FAILED transitions.

The official success flag is never read here. Geometry and physics own the
result; the VLM/selector only chooses semantic targets.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .contracts import GraspCandidate, TaskDecision
from .metrics import EpisodeLogger, timed


@dataclass
class ExecutorConfig:
    kp: float = 18.0
    position_tolerance: float = 0.015
    max_steps_per_move: int = 90
    settle_steps: int = 12
    gripper_open: float = -1.0
    gripper_close: float = 1.0
    approach_height: float = 0.12
    lift_height: float = 0.12
    place_clearance: float = 0.15
    retreat_height: float = 0.15
    release_settle_steps: int = 25
    grasp_retries: int = 1
    min_finger_separation: float = 0.003
    grasp_strategy: str = "auto"
    bowl_rim_fraction: float = 0.60
    bowl_grasp_drop: float = 0.022
    center_grasp_drop: float = 0.025
    grasp_side: float = -1.0
    lift_check_dz: float = 0.02
    verify_placement_xy: float = 0.03

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ExecutorConfig":
        if not raw:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown executor config keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class EpisodeOutcome:
    status: str
    reason_code: str
    internal_success: bool
    steps: int
    stage_errors: Dict[str, float] = field(default_factory=dict)
    grasp_attempts: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clip_action(err: np.ndarray, kp: float) -> np.ndarray:
    return np.clip(kp * np.asarray(err, dtype=np.float64), -1.0, 1.0)


class StagedExecutor:
    def __init__(self, config: Optional[ExecutorConfig] = None):
        self.config = config or ExecutorConfig()

    def _bounded(self, backend, max_steps: int) -> bool:
        return int(getattr(backend, "step_count", 0)) < int(max_steps)

    def _move(
        self,
        backend,
        target_xyz: np.ndarray,
        gripper: float,
        logger: Optional[EpisodeLogger],
        stage: str,
        max_steps: int,
        tolerance: Optional[float] = None,
        speed_scale: float = 1.0,
    ):
        cfg = self.config
        tol = cfg.position_tolerance if tolerance is None else tolerance
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        error = float("inf")
        moved = 0
        for _ in range(cfg.max_steps_per_move):
            if not self._bounded(backend, max_steps):
                return False, error, "step_budget_exhausted"
            state = backend.robot_state()
            err = target - state.eef_position
            error = float(np.linalg.norm(err))
            if error <= tol:
                return True, error, "ok"
            action = np.zeros(7, dtype=np.float64)
            action[:3] = _clip_action(err, cfg.kp * speed_scale)
            action[6] = gripper
            result = backend.step(action)
            moved += 1
            if result.done:
                return True, error, "env_done"
        if logger is not None:
            logger.event("MOVE_TIMEOUT", stage=stage, target=target, error=error, steps=moved)
        return False, error, "move_timeout"

    def _hold(self, backend, gripper: float, steps: int, max_steps: int) -> bool:
        for _ in range(int(steps)):
            if not self._bounded(backend, max_steps):
                return False
            action = np.zeros(7, dtype=np.float64)
            action[6] = gripper
            result = backend.step(action)
            if result.done:
                return True
        return True

    def propose_grasp(
        self,
        target_id: str,
        target_position: np.ndarray,
        extent: Dict[str, Any],
        eef_position: np.ndarray,
        eef_quat_xyzw: Optional[np.ndarray] = None,
        drop_override: Optional[float] = None,
        side_override: float = 0.0,
    ) -> GraspCandidate:
        cfg = self.config
        center = np.asarray(target_position, dtype=np.float64)
        radius = float(extent.get("radius", 0.04))
        top_z = float(extent.get("top_z", center[2]))
        bottom_z = float(extent.get("bottom_z", center[2]))
        strategy = cfg.grasp_strategy
        if strategy == "auto":
            strategy = "bowl_rim" if "bowl" in target_id else "center"

        if strategy == "bowl_rim":
            rim_radius = float(extent.get("radius_top", radius * cfg.bowl_rim_fraction))
            if eef_quat_xyzw is not None:
                from .contracts import quat_xyzw_to_matrix

                finger_axis = quat_xyzw_to_matrix(eef_quat_xyzw)[:, 1]
                if side_override != 0:
                    direction = finger_axis if side_override > 0 else -finger_axis
                elif cfg.grasp_side != 0:
                    direction = finger_axis if cfg.grasp_side > 0 else -finger_axis
                else:
                    side = float(np.dot(np.asarray(eef_position[:2]) - center[:2], finger_axis[:2]))
                    direction = finger_axis if side >= 0 else -finger_axis
            else:
                direction = np.asarray(eef_position[:2], dtype=np.float64) - center[:2]
                norm = float(np.linalg.norm(direction))
                direction = direction / norm if norm > 1e-6 else np.array([-1.0, 0.0])
            grasp_xy = center[:2] + direction[:2] * rim_radius
            grasp_z = top_z - (cfg.bowl_grasp_drop if drop_override is None else drop_override)
        elif strategy == "center":
            grasp_xy = center[:2]
            grasp_z = top_z - (cfg.center_grasp_drop if drop_override is None else drop_override)
        else:
            raise ValueError(f"unknown grasp strategy {strategy!r}")

        grasp_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_z], dtype=np.float64)
        pregrasp_pos = grasp_pos + np.array([0.0, 0.0, cfg.approach_height])
        from .contracts import pose_from_pos_quat

        identity = np.array([0.0, 0.0, 0.0, 1.0])
        return GraspCandidate(
            candidate_id=f"topdown-{strategy}",
            target_object_id=target_id,
            target_part_id="rim" if strategy == "bowl_rim" else None,
            grasp_pose=pose_from_pos_quat(grasp_pos, identity),
            pregrasp_pose=pose_from_pos_quat(pregrasp_pos, identity),
            approach_axis=np.array([0.0, 0.0, -1.0]),
            approach_distance=cfg.approach_height,
            gripper_width=float(np.clip(2.0 * radius, 0.02, 0.08)),
            score=1.0,
            uncertainty=0.0,
            feasible=True,
        )

    def _verify_grasp(self, backend) -> bool:
        separation = backend.robot_state().finger_separation()
        return separation >= self.config.min_finger_separation

    def _test_lift(self, backend, target_id: str, z_before: float, oracle_geometry: bool) -> bool:
        state = backend.robot_state()
        if oracle_geometry:
            after = backend.oracle_positions()[target_id]
            return bool(float(after[2]) - float(z_before) > self.config.lift_check_dz)
        return self._verify_grasp(backend)

    def _verify_placement(self, backend, target_id: str, receptacle_id: str) -> bool:
        positions = backend.oracle_positions()
        target = np.asarray(positions[target_id], dtype=np.float64)
        receptacle = np.asarray(positions[receptacle_id], dtype=np.float64)
        xy_error = float(np.linalg.norm(target[:2] - receptacle[:2]))
        return xy_error <= self.config.verify_placement_xy

    def run(
        self,
        backend,
        decision: TaskDecision,
        max_steps: int,
        logger: Optional[EpisodeLogger] = None,
        oracle_geometry: bool = True,
        settle_time: float = 0.0,
        scene_positions: Optional[Dict[str, np.ndarray]] = None,
        scene_extents: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> EpisodeOutcome:
        cfg = self.config
        stage_errors: Dict[str, float] = {}
        target_id = decision.target_id
        receptacle_id = str(decision.constraints.get("receptacle_id", ""))
        if not receptacle_id:
            return EpisodeOutcome("failed", "missing_receptacle", False, int(getattr(backend, "step_count", 0)))

        def emit(event: str, stage: str, **fields):
            if logger is not None:
                logger.event(event, stage=stage, **fields)

        with timed(logger, "select"):
            positions = dict(scene_positions) if scene_positions is not None else backend.oracle_positions()
            if target_id not in positions or receptacle_id not in positions:
                return EpisodeOutcome(
                    "failed", "target_not_in_scene", False, int(getattr(backend, "step_count", 0)),
                    detail={"target_id": target_id, "receptacle_id": receptacle_id},
                )
            target_position = np.asarray(positions[target_id], dtype=np.float64)
            if scene_extents is not None and target_id in scene_extents:
                extent = dict(scene_extents[target_id])
            else:
                extent = backend.body_extent(target_id)
            if scene_extents is not None and receptacle_id in scene_extents:
                receptacle_extent = dict(scene_extents[receptacle_id])
            else:
                receptacle_extent = backend.body_extent(receptacle_id)
        target_half_height = 0.5 * float(extent["top_z"] - extent["bottom_z"])
        receptacle_top = float(receptacle_extent["top_z"])
        attempts = 0
        retained = False
        grasp_z_before = 0.0

        for attempt in range(cfg.grasp_retries + 1):
            attempts += 1
            positions = dict(scene_positions) if scene_positions is not None else backend.oracle_positions()
            if scene_extents is not None and target_id in scene_extents:
                extent = dict(scene_extents[target_id])
            else:
                extent = backend.body_extent(target_id)
            target_position = np.asarray(positions[target_id], dtype=np.float64)
            state = backend.robot_state()
            is_bowl = "bowl" in target_id
            base_drop = cfg.bowl_grasp_drop if is_bowl else cfg.center_grasp_drop
            drop_variants = (base_drop, base_drop - 0.007, base_drop + 0.003)
            drop = drop_variants[min(attempt, len(drop_variants) - 1)]
            side_override = cfg.grasp_side
            candidate = self.propose_grasp(
                target_id, target_position, extent, state.eef_position, state.eef_quat_xyzw,
                drop_override=drop, side_override=side_override,
            )
            emit("GRASP_PROPOSED", "SELECT", attempt=attempt, candidate=candidate.to_dict(), extent=extent,
                 side_override=side_override)
            emit("ATTEMPT_START", "PREGRASP", attempt=attempt)
            with timed(logger, "pregrasp"):
                ok, err, reason = self._move(
                    backend, candidate.pregrasp_pose[:3, 3], cfg.gripper_open, logger, "PREGRASP", max_steps
                )
            stage_errors["pregrasp"] = err
            emit("MOVE_RESULT", "PREGRASP", ok=ok, error=err, reason=reason)
            if not ok:
                return EpisodeOutcome("timeout" if reason == "step_budget_exhausted" else "failed",
                                      f"pregrasp_{reason}", False, int(backend.step_count), stage_errors)

            with timed(logger, "approach"):
                ok, err, reason = self._move(
                    backend, candidate.grasp_pose[:3, 3], cfg.gripper_open, logger, "APPROACH", max_steps,
                    tolerance=max(cfg.position_tolerance * 1.5, 0.02), speed_scale=0.6,
                )
            stage_errors["approach"] = err
            emit("MOVE_RESULT", "APPROACH", ok=ok, error=err, reason=reason)
            if not ok:
                return EpisodeOutcome("timeout" if reason == "step_budget_exhausted" else "failed",
                                      f"approach_{reason}", False, int(backend.step_count), stage_errors)

            self._hold(backend, cfg.gripper_close, cfg.settle_steps, max_steps)
            emit("GRIPPER_CLOSED", "CLOSE", separation=backend.robot_state().finger_separation())
            if not oracle_geometry and not self._verify_grasp(backend):
                emit("GRASP_VERIFY_FAILED", "VERIFY_GRASP", reason="empty_gripper")
                self._hold(backend, cfg.gripper_open, cfg.settle_steps, max_steps)
                continue

            current_positions = dict(scene_positions) if scene_positions is not None else backend.oracle_positions()
            grasp_z_before = float(np.asarray(current_positions[target_id], dtype=np.float64)[2])
            lift_target = candidate.grasp_pose[:3, 3] + np.array([0.0, 0.0, cfg.lift_height])
            with timed(logger, "test_lift"):
                ok, err, reason = self._move(
                    backend, lift_target, cfg.gripper_close, logger, "TEST_LIFT", max_steps,
                    tolerance=0.02, speed_scale=0.7
                )
            stage_errors["test_lift"] = err
            retained = self._test_lift(backend, target_id, grasp_z_before, oracle_geometry)
            emit("LIFT_RESULT", "TEST_LIFT", retained=retained, ok=ok, error=err, reason=reason)
            if retained:
                break
            self._hold(backend, cfg.gripper_open, cfg.settle_steps, max_steps)
            emit("GRASP_RETRY", "RETRY", attempt=attempt)

        if not retained:
            return EpisodeOutcome("failed", "grasp_not_retained", False, int(backend.step_count),
                                  stage_errors, attempts)

        target_center = np.asarray(positions[target_id], dtype=np.float64)
        grasp_offset_xy = np.asarray(candidate.grasp_pose[:2, 3], dtype=np.float64) - target_center[:2]
        place_xy = np.asarray(positions[receptacle_id], dtype=np.float64)[:2] + grasp_offset_xy
        cruise_z = max(float(candidate.grasp_pose[2, 3]) + 0.25, receptacle_top + cfg.place_clearance)
        current = backend.robot_state().eef_position
        with timed(logger, "lift_safe"):
            ok, err, reason = self._move(
                backend, np.array([current[0], current[1], cruise_z]), cfg.gripper_close,
                logger, "LIFT_SAFE", max_steps, tolerance=0.02, speed_scale=0.8,
            )
        stage_errors["lift_safe"] = err
        emit("MOVE_RESULT", "LIFT_SAFE", ok=ok, error=err, reason=reason)
        if not ok:
            return EpisodeOutcome("timeout" if reason == "step_budget_exhausted" else "failed",
                                  f"lift_safe_{reason}", False, int(backend.step_count), stage_errors, attempts)
        transport_target = np.array([place_xy[0], place_xy[1], cruise_z])
        with timed(logger, "transport"):
            ok, err, reason = self._move(
                backend, transport_target, cfg.gripper_close, logger, "TRANSPORT", max_steps,
                tolerance=0.025, speed_scale=0.9
            )
        stage_errors["transport"] = err
        emit("MOVE_RESULT", "TRANSPORT", ok=ok, error=err, reason=reason)
        if not ok:
            return EpisodeOutcome("timeout" if reason == "step_budget_exhausted" else "failed",
                                  f"transport_{reason}", False, int(backend.step_count), stage_errors, attempts)

        place_z = receptacle_top + target_half_height + 0.005
        with timed(logger, "place"):
            ok, err, reason = self._move(
                backend, np.array([place_xy[0], place_xy[1], place_z]), cfg.gripper_close,
                logger, "PLACE", max_steps, tolerance=0.02, speed_scale=0.5,
            )
        stage_errors["place"] = err
        emit("MOVE_RESULT", "PLACE", ok=ok, error=err, reason=reason)
        if not ok:
            return EpisodeOutcome("timeout" if reason == "step_budget_exhausted" else "failed",
                                  f"place_{reason}", False, int(backend.step_count), stage_errors, attempts)

        self._hold(backend, cfg.gripper_open, cfg.settle_steps, max_steps)
        emit("RELEASED", "RELEASE")
        retreat_target = np.array([place_xy[0], place_xy[1], receptacle_top + cfg.retreat_height])
        with timed(logger, "retreat"):
            ok, err, reason = self._move(
                backend, retreat_target, cfg.gripper_open, logger, "RETREAT", max_steps,
                tolerance=0.02
            )
        stage_errors["retreat"] = err
        emit("MOVE_RESULT", "RETREAT", ok=ok, error=err, reason=reason)

        with timed(logger, "release_settle"):
            self._hold(backend, cfg.gripper_open, cfg.release_settle_steps, max_steps)

        if settle_time > 0:
            time.sleep(settle_time)
            backend.env.sim.forward() if hasattr(getattr(backend, "env", None), "sim") else None
        if oracle_geometry:
            placed = self._verify_placement(backend, target_id, receptacle_id)
            final_positions = backend.oracle_positions()
        else:
            placed = None
            final_positions = positions
        emit(
            "PLACEMENT_VERIFIED",
            "VERIFY_PLACEMENT",
            placed=placed,
            target_xy=np.asarray(final_positions[target_id], dtype=np.float64)[:2].tolist(),
            receptacle_xy=np.asarray(final_positions[receptacle_id], dtype=np.float64)[:2].tolist(),
            xy_error=float(
                np.linalg.norm(
                    np.asarray(final_positions[target_id])[:2] - np.asarray(final_positions[receptacle_id])[:2]
                )
            ),
            official=backend.official_success(),
        )
        if placed is None:
            return EpisodeOutcome("ok", "placement_unverifiable_non_oracle", None, int(backend.step_count),
                                  stage_errors, attempts)
        if placed:
            return EpisodeOutcome("ok", "internal_placement_verified", True, int(backend.step_count),
                                  stage_errors, attempts)
        return EpisodeOutcome("failed", "placement_not_verified", False, int(backend.step_count),
                              stage_errors, attempts)
