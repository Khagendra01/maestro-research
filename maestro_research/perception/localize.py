"""RGB-D scene estimation for declared Setting B.

Colour/shape priors produce candidate object clusters. This is intentionally
weak: it cannot distinguish same-colour object classes (for example alphabet
soup vs tomato sauce). Ambiguity is surfaced, never resolved with simulator
poses. A real open-vocabulary detector or VLM is required for robust grounding.
"""
from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Tuple

import numpy as np

from ..selector import ScriptedSelector


def _rgb_to_hsv(rgb: np.ndarray) -> Tuple[float, float, float]:
    r, g, b = [float(max(0.0, min(1.0, c / 255.0))) for c in rgb[:3]]
    return colorsys.rgb_to_hsv(r, g, b)


def classify_color(mean_rgb: np.ndarray) -> str:
    h, s, v = _rgb_to_hsv(mean_rgb)
    hue = h * 360.0
    if v < 0.22:
        return "akita_black_bowl"
    if v > 0.78 and s < 0.18:
        return "plate"
    if s > 0.35 and (hue < 18 or hue > 345) and v > 0.25:
        return "alphabet_soup"
    if 10 <= hue <= 45 and 0.25 <= s <= 0.85 and 0.18 <= v <= 0.75:
        return "basket"
    return "unknown"


def _table_height(points: np.ndarray) -> float:
    z = points[:, 2]
    z = z[np.isfinite(z)]
    hist, edges = np.histogram(z, bins=200)
    mode_index = int(np.argmax(hist))
    return float(0.5 * (edges[mode_index] + edges[mode_index + 1]))


def _cluster_xy(points: np.ndarray, colors: np.ndarray, voxel: float = 0.012):
    from scipy import ndimage

    xy = points[:, :2]
    origin = xy.min(axis=0) - voxel
    grid = np.floor((xy - origin) / voxel).astype(np.int64)
    shape = grid.max(axis=0) + 3
    mask = np.zeros(shape, dtype=bool)
    mask[grid[:, 0], grid[:, 1]] = True
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    clusters: List[Tuple[np.ndarray, np.ndarray]] = []
    if count == 0:
        return clusters
    point_labels = labels[grid[:, 0], grid[:, 1]]
    for label in range(1, count + 1):
        idx = np.nonzero(point_labels == label)[0]
        if idx.size < 30:
            continue
        pts = points[idx]
        if np.ptp(pts[:, 2]) < 0.008:
            continue
        clusters.append((pts, colors[idx]))
    return clusters


def estimate_objects(observation, backend) -> List[Dict[str, Any]]:
    points = backend.pointcloud(observation)
    rgb = np.asarray(observation.rgb)
    if points.shape[:2] != rgb.shape[:2]:
        raise ValueError("depth and RGB resolution differ")
    valid = np.isfinite(points).all(axis=-1)
    flat_points = points[valid]
    flat_rgb = rgb[valid]
    if flat_points.shape[0] < 100:
        return []
    table_z = _table_height(flat_points)
    above = flat_points[:, 2] > table_z + 0.012
    object_points = flat_points[above]
    object_rgb = flat_rgb[above]
    if object_points.shape[0] < 50:
        return []
    clusters = _cluster_xy(object_points, object_rgb)
    objects = []
    for cluster, cluster_rgb in clusters:
        center = np.median(cluster, axis=0)
        radius = float(np.max(np.linalg.norm(cluster[:, :2] - center[:2], axis=1)))
        if radius < 0.01 or radius > 0.25:
            continue
        mean_rgb = cluster_rgb.mean(axis=0)
        object_type = classify_color(mean_rgb)
        if object_type == "unknown":
            continue
        objects.append(
            {
                "type": object_type,
                "position": np.asarray(center, dtype=np.float64),
                "top_z": float(cluster[:, 2].max()),
                "bottom_z": float(cluster[:, 2].min()),
                "radius": radius,
                "mean_rgb": np.asarray(mean_rgb, dtype=np.float64),
                "n_points": int(cluster.shape[0]),
            }
        )
    return objects


def build_scene_from_rgbd(observation, backend) -> Dict[str, Any]:
    objects = estimate_objects(observation, backend)
    positions: Dict[str, np.ndarray] = {}
    extents: Dict[str, Dict[str, Any]] = {}
    counters: Dict[str, int] = {}
    for obj in objects:
        counters[obj["type"]] = counters.get(obj["type"], 0) + 1
        name = f"{obj['type']}_{counters[obj['type']]}"
        positions[name] = obj["position"]
        extents[name] = {
            "top_z": obj["top_z"],
            "bottom_z": obj["bottom_z"],
            "radius": obj["radius"],
            "center": obj["position"],
        }
    selector = ScriptedSelector(positions, table_center=None)
    return {"objects": objects, "positions": positions, "extents": extents, "selector": selector}
