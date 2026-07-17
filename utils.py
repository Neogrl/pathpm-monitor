import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-8
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(32)


def clip_points(points: np.ndarray, map_size: float) -> np.ndarray:
    return np.clip(points, 0.0, map_size)


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def circle_overlap_area(distance: float, radius: float) -> float:
    d = float(distance)
    r = float(radius)
    if d >= 2 * r:
        return 0.0
    if d <= EPS:
        return math.pi * r * r
    return 2 * r * r * math.acos(d / (2 * r)) - 0.5 * d * math.sqrt(max(4 * r * r - d * d, 0.0))


def clipped_circle_area(center: np.ndarray, radius: float, map_size: float) -> float:
    """Area of a circular FOV intersected with the square map."""
    cx, cy = np.asarray(center, dtype=np.float64)
    r = float(radius)
    size = float(map_size)
    if r <= 0.0 or size <= 0.0:
        return 0.0
    if r <= cx <= size - r and r <= cy <= size - r:
        return math.pi * r * r

    x0 = max(0.0, cx - r)
    x1 = min(size, cx + r)
    if x1 <= x0:
        return 0.0
    xs = 0.5 * (x1 - x0) * _GAUSS_NODES + 0.5 * (x1 + x0)
    half_height = np.sqrt(np.maximum(r * r - (xs - cx) ** 2, 0.0))
    lower = np.maximum(0.0, cy - half_height)
    upper = np.minimum(size, cy + half_height)
    heights = np.maximum(upper - lower, 0.0)
    return float(0.5 * (x1 - x0) * np.dot(_GAUSS_WEIGHTS, heights))


def non_max_suppression(points: np.ndarray, scores: np.ndarray, min_distance: float, max_count: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: list[int] = []
    for idx in order:
        p = points[idx]
        if all(np.linalg.norm(p - points[j]) >= min_distance for j in keep):
            keep.append(int(idx))
        if len(keep) >= max_count:
            break
    return np.asarray(keep, dtype=np.int64)


def local_maxima_2d(grid: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    scores = []
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            val = grid[y, x]
            if val < threshold:
                continue
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            if val >= np.max(grid[y0:y1, x0:x1]):
                coords.append((x, y))
                scores.append(float(val))
    if not coords:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def running_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0
