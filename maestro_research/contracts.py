"""Typed observation, scene, grasp, decision and execution contracts.

Poses are 4x4 SE(3) matrices in metres. Quaternions are normalized (x, y, z, w),
matching robosuite. Euler angles are converted only at boundaries that require
them. No contract accepts or stores simulator success signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-9


def normalize_quat_xyzw(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < _EPS:
        raise ValueError("quaternion must be finite and non-zero")
    return q / norm


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return normalize_quat_xyzw(q)


def validate_pose(pose: np.ndarray, name: str = "pose") -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(p[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid last row")
    rot = p[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rot), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not 1")
    return p


def pose_from_pos_quat(position: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return validate_pose(pose)


def pose_inverse(pose: np.ndarray) -> np.ndarray:
    p = validate_pose(pose)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = p[:3, :3].T
    out[:3, 3] = -p[:3, :3].T @ p[:3, 3]
    return out


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = validate_pose(pose)
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 3:
        raise ValueError("points must have shape (..., 3)")
    out = pts @ p[:3, :3].T + p[:3, 3]
    return out[0] if single else out


def finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} contains non-finite values")
    return v


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class RobotState:
    eef_position: np.ndarray
    eef_quat_xyzw: np.ndarray
    gripper_qpos: np.ndarray
    joints: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame: str = "world"

    def __post_init__(self) -> None:
        self.eef_position = finite_vector(self.eef_position, 3, "eef_position")
        self.eef_quat_xyzw = normalize_quat_xyzw(self.eef_quat_xyzw)
        self.gripper_qpos = np.asarray(self.gripper_qpos, dtype=np.float64)
        if self.gripper_qpos.ndim != 1 or self.gripper_qpos.size < 2:
            raise ValueError("gripper_qpos must be a 1-D vector with at least two entries")
        if not np.isfinite(self.gripper_qpos).all():
            raise ValueError("gripper_qpos contains non-finite values")
        if self.joints is not None:
            self.joints = np.asarray(self.joints, dtype=np.float64)

    @property
    def eef_pose(self) -> np.ndarray:
        return pose_from_pos_quat(self.eef_position, self.eef_quat_xyzw)

    def finger_separation(self) -> float:
        # robosuite Panda exposes the two finger slide-joint positions as
        # non-negative distances from the closed centre: open ~[0.04, 0.04].
        return float(np.abs(self.gripper_qpos).sum())


@dataclass
class Observation:
    rgb: np.ndarray
    intrinsics: np.ndarray
    camera_pose: np.ndarray
    robot_state: RobotState
    calibration_id: str
    frame_id: str
    acquisition_time: float
    depth: Optional[np.ndarray] = None
    source: str = "sim"
    permitted_modalities: Tuple[str, ...] = ("rgb",)

    def __post_init__(self) -> None:
        self.rgb = np.asarray(self.rgb)
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        k = np.asarray(self.intrinsics, dtype=np.float64)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("intrinsics must be a finite 3x3 matrix with positive focal lengths")
        self.intrinsics = k
        self.camera_pose = validate_pose(self.camera_pose, "camera_pose")
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float64)
            if self.depth.shape != self.rgb.shape[:2]:
                raise ValueError("depth shape must match rgb")
        if not self.frame_id:
            raise ValueError("frame_id is required")
        if not self.calibration_id:
            raise ValueError("calibration_id is required")

    @property
    def depth_available(self) -> bool:
        return self.depth is not None and "depth" in self.permitted_modalities


@dataclass
class ObjectEstimate:
    object_id: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    extent: Optional[np.ndarray]
    source: str
    uncertainty: float
    part_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.position = finite_vector(self.position, 3, "position")
        self.quaternion_xyzw = normalize_quat_xyzw(self.quaternion_xyzw)
        if self.extent is not None:
            self.extent = np.asarray(self.extent, dtype=np.float64)
        if self.source not in ("oracle", "perception", "instruction"):
            raise ValueError(f"unknown object-estimate source {self.source!r}")
        if not np.isfinite(self.uncertainty) or self.uncertainty < 0:
            raise ValueError("uncertainty must be finite and non-negative")


@dataclass
class SceneSnapshot:
    version: int
    timestamp: float
    frame_id: str
    objects: Dict[str, ObjectEstimate]
    observation_age: float
    held_object_id: Optional[str] = None
    static_geometry: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("scene version must be non-negative")
        if self.observation_age < 0:
            raise ValueError("observation_age must be non-negative")

    def object(self, object_id: str) -> ObjectEstimate:
        if object_id not in self.objects:
            raise KeyError(f"object {object_id!r} is not in scene v{self.version}")
        return self.objects[object_id]

    def stale(self, max_age: float) -> bool:
        return self.observation_age > max_age

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "observation_age": self.observation_age,
            "held_object_id": self.held_object_id,
            "objects": {
                key: {
                    "position": obj.position.tolist(),
                    "quaternion_xyzw": obj.quaternion_xyzw.tolist(),
                    "source": obj.source,
                    "uncertainty": obj.uncertainty,
                    "part_ids": list(obj.part_ids),
                }
                for key, obj in self.objects.items()
            },
        }


@dataclass
class GraspCandidate:
    candidate_id: str
    target_object_id: str
    grasp_pose: np.ndarray
    pregrasp_pose: np.ndarray
    approach_axis: np.ndarray
    approach_distance: float
    gripper_width: float
    score: float
    uncertainty: float
    feasible: bool = True
    target_part_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.grasp_pose = validate_pose(self.grasp_pose, "grasp_pose")
        self.pregrasp_pose = validate_pose(self.pregrasp_pose, "pregrasp_pose")
        axis = np.asarray(self.approach_axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm < _EPS:
            raise ValueError("approach_axis must be non-zero")
        self.approach_axis = axis / norm
        if self.approach_distance < 0 or not np.isfinite(self.approach_distance):
            raise ValueError("approach_distance must be finite and non-negative")
        if self.gripper_width <= 0 or not np.isfinite(self.gripper_width):
            raise ValueError("gripper_width must be finite and positive")
        if not np.isfinite(self.uncertainty) or self.uncertainty < 0:
            raise ValueError("uncertainty must be finite and non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_object_id": self.target_object_id,
            "target_part_id": self.target_part_id,
            "grasp_pose": self.grasp_pose.tolist(),
            "pregrasp_pose": self.pregrasp_pose.tolist(),
            "approach_axis": self.approach_axis.tolist(),
            "approach_distance": self.approach_distance,
            "gripper_width": self.gripper_width,
            "score": self.score,
            "uncertainty": self.uncertainty,
            "feasible": self.feasible,
        }


@dataclass
class TaskDecision:
    target_id: str
    constraints: Dict[str, Any]
    stage: str
    candidate_id: Optional[str] = None
    recovery_action: Optional[str] = None
    viewpoint_request: Optional[str] = None

    ALLOWED_STAGES = ("SELECT", "PREGRASP", "APPROACH", "CLOSE", "TRANSPORT", "PLACE", "RELEASE", "RETREAT", "REOBSERVE", "REPLAN", "STOP")
    ALLOWED_RECOVERY = (None, "RETRY", "REOBSERVE", "REPLAN", "STOP")

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id is required")
        if self.stage not in self.ALLOWED_STAGES:
            raise ValueError(f"stage {self.stage!r} is not allowed")
        if self.recovery_action not in self.ALLOWED_RECOVERY:
            raise ValueError(f"recovery_action {self.recovery_action!r} is not allowed")
        if not isinstance(self.constraints, dict):
            raise ValueError("constraints must be a dict")


@dataclass
class ExecutionResult:
    stage: str
    status: str
    reason_code: str
    scene_version: int
    timestamps: Dict[str, float]
    measured_error: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    ALLOWED_STATUS = ("ok", "retry", "failed", "stopped", "timeout")

    def __post_init__(self) -> None:
        if self.status not in self.ALLOWED_STATUS:
            raise ValueError(f"status {self.status!r} is not allowed")
        if not isinstance(self.timestamps, dict):
            raise ValueError("timestamps must be a dict")

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(
            {
                "stage": self.stage,
                "status": self.status,
                "reason_code": self.reason_code,
                "scene_version": self.scene_version,
                "timestamps": self.timestamps,
                "measured_error": self.measured_error,
                "evidence": self.evidence,
            }
        )
