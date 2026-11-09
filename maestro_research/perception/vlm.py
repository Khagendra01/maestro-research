"""VLM grounding through the Meta Model API (Muse Spark).

The VLM sees the task instruction and the agentview RGB image and returns
normalized image points for the target and receptacle. Geometry then works in
3-D: the point is back-projected through the depth buffer and a local point-cloud
patch supplies the object extent. The VLM never supplies metric coordinates and
never sees simulator poses, segmentation IDs, goal facts, or the success signal.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

import numpy as np

from ..contracts import TaskDecision

MODEL = "muse-spark-1.3-contributor"
BASE_URL = "https://api.meta.ai/v1/chat/completions"
MAX_TOKENS = 5000


class VLMError(RuntimeError):
    pass


def _encode_png(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _prompt(instruction: str, height: int, width: int) -> str:
    return (
        f"Image: {width}x{height} robot table view. Task: {instruction}.\n"
        "Return JSON only: "
        '{"target":{"x":<0-1>,"y":<0-1>,"label":"<short name>"},'
        '"receptacle":{"x":<0-1>,"y":<0-1>,"label":"<short name>"}}\n'
        "Origin top-left; x right, y down. Give the object centers. "
        "Use null for a missing object."
    )


def _parse_content(content: str) -> Dict[str, Any]:
    if not content:
        raise VLMError("empty VLM content")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise VLMError(f"no JSON object in VLM response: {content[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VLMError(f"invalid VLM JSON: {exc}") from exc


def _point(data: Any, name: str) -> Tuple[float, float, str]:
    if not isinstance(data, dict):
        raise VLMError(f"missing {name}")
    try:
        x = float(data["x"])
        y = float(data["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VLMError(f"invalid {name} point: {data!r}") from exc
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise VLMError(f"{name} point out of range: {x}, {y}")
    return x, y, str(data.get("label") or name)


def _slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "object"


def _seed_3d(points: np.ndarray, x_norm: float, y_norm: float) -> np.ndarray:
    height, width = points.shape[:2]
    col = int(round(x_norm * (width - 1)))
    row_top = int(round(y_norm * (height - 1)))
    row_depth = row_top
    for radius in (2, 5, 9):
        rows = slice(max(0, row_depth - radius), min(height, row_depth + radius + 1))
        cols = slice(max(0, col - radius), min(width, col + radius + 1))
        patch = points[rows, cols].reshape(-1, 3)
        valid = patch[np.isfinite(patch).all(axis=1)]
        if valid.shape[0]:
            return np.median(valid, axis=0)
    raise VLMError("no valid depth at VLM point")


def _local_extent(points: np.ndarray, seed: np.ndarray, radius: float) -> Dict[str, Any]:
    flat = points.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    for r in (radius, radius * 1.5, radius * 2.0):
        near = finite & (np.linalg.norm(flat[:, :2] - seed[:2], axis=1) < r)
        near &= np.abs(flat[:, 2] - seed[2]) < 0.16
        cloud = flat[near]
        if cloud.shape[0] >= 30:
            break
    if cloud.shape[0] < 30:
        raise VLMError("too few points near VLM point for extent")
    center = np.median(cloud, axis=0)
    radial = np.linalg.norm(cloud[:, :2] - center[:2], axis=1)
    top_z = float(cloud[:, 2].max())
    top_mask = cloud[:, 2] >= top_z - 0.01
    return {
        "top_z": top_z,
        "bottom_z": float(cloud[:, 2].min()),
        "radius": float(radial.max()),
        "radius_top": float(radial[top_mask].max()),
        "center": center,
        "n_points": int(cloud.shape[0]),
    }


def call_vlm(instruction: str, rgb: np.ndarray) -> Dict[str, Any]:
    last_error = None
    for attempt in range(2):
        try:
            return _call_vlm_once(instruction, rgb)
        except VLMError as exc:
            last_error = exc
            if "empty VLM content" not in str(exc):
                raise
    raise last_error


def _call_vlm_once(instruction: str, rgb: np.ndarray) -> Dict[str, Any]:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise VLMError("MODEL_API_KEY is not set")
    height, width = np.asarray(rgb).shape[:2]
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(instruction, height, width)},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _encode_png(rgb)},
                    },
                ],
            }
        ],
    }
    started = time.monotonic()
    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise VLMError(f"HTTP {exc.code}: {exc.read().decode()[:400]}") from exc
    latency = time.monotonic() - started
    choice = data["choices"][0]
    content = choice["message"].get("content")
    parsed = _parse_content(content)
    return {
        "content": content,
        "parsed": parsed,
        "latency_seconds": latency,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "model": data.get("model", MODEL),
    }


def build_scene_from_vlm(observation, backend, instruction: str) -> Dict[str, Any]:
    response = call_vlm(instruction, observation.rgb)
    parsed = response["parsed"]
    points = backend.pointcloud(observation)
    target_x, target_y, target_label = _point(parsed.get("target"), "target")
    receptacle_x, receptacle_y, receptacle_label = _point(parsed.get("receptacle"), "receptacle")

    target_seed = _seed_3d(points, target_x, target_y)
    receptacle_seed = _seed_3d(points, receptacle_x, receptacle_y)
    target_extent = _local_extent(points, target_seed, 0.07)
    receptacle_extent = _local_extent(points, receptacle_seed, 0.10)

    target_id = "vlm_target_" + _slug(target_label)
    receptacle_id = "vlm_receptacle_" + _slug(receptacle_label)
    positions = {target_id: target_extent["center"], receptacle_id: receptacle_extent["center"]}
    extents = {target_id: target_extent, receptacle_id: receptacle_extent}
    decision = TaskDecision(
        target_id=target_id,
        constraints={"receptacle_id": receptacle_id},
        stage="SELECT",
    )
    metadata = {
        "model": response["model"],
        "latency_seconds": response["latency_seconds"],
        "finish_reason": response["finish_reason"],
        "usage": response["usage"],
        "content": response["content"],
        "target_point": [target_x, target_y],
        "receptacle_point": [receptacle_x, receptacle_y],
        "target_label": target_label,
        "receptacle_label": receptacle_label,
        "target_seed_3d": np.asarray(target_seed).tolist(),
        "receptacle_seed_3d": np.asarray(receptacle_seed).tolist(),
        "target_extent": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in target_extent.items()
        },
        "receptacle_extent": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in receptacle_extent.items()
        },
    }
    return {
        "positions": positions,
        "extents": extents,
        "decision": decision,
        "metadata": metadata,
    }
