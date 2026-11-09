"""Camera geometry for MuJoCo/LIBERO observations.

MuJoCo camera frames: +x right, +y up, -z forward. The offscreen depth buffer is
OpenGL-normalized [0, 1] and must be converted with the model's near/far planes.
The vertical image convention is configurable and is validated against a known
scene on the target machine before use.
"""
from __future__ import annotations

import numpy as np


def mujoco_intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    if not (0.0 < fovy_deg < 180.0):
        raise ValueError("fovy must be in (0, 180) degrees")
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def depth_buffer_to_meters(raw_depth: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    raw = np.asarray(raw_depth, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("depth buffer contains non-finite values")
    if not (0.0 < znear < zfar):
        raise ValueError("expected 0 < znear < zfar")
    depth = znear * zfar / (zfar - raw * (zfar - znear))
    return depth


def mujoco_camera_pose(cam_xpos: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    pos = np.asarray(cam_xpos, dtype=np.float64).reshape(3)
    rot = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def unproject_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    convention: str = "bottom_left",
    min_depth: float = 0.02,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Return an (H, W, 3) world-frame point cloud.

    `bottom_left` treats row 0 as the bottom of the image (OpenGL/MuJoCo buffer);
    `top_left` treats row 0 as the top (display orientation).
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be 2-D")
    if convention not in ("bottom_left", "top_left"):
        raise ValueError("convention must be 'bottom_left' or 'top_left'")
    k = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(camera_pose, dtype=np.float64)
    if k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid intrinsics or camera pose")
    height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    x_cam = (us - k[0, 2]) / k[0, 0]
    if convention == "bottom_left":
        y_cam = (vs - k[1, 2]) / k[1, 1]
    else:
        y_cam = -(vs - k[1, 2]) / k[1, 1]
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    d = np.where(valid, depth, 0.0)
    points_cam = np.stack([x_cam * d, y_cam * d, -d], axis=-1)
    world = points_cam @ pose[:3, :3].T + pose[:3, 3]
    world[~valid] = np.nan
    return world
