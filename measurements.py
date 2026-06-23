from dataclasses import dataclass

import numpy as np

from config import Config


@dataclass
class MeasurementBatch:
    points: np.ndarray
    detected_target_ids: list[int]
    clutter_count: int


def generate_measurements(
    cfg: Config,
    rng: np.random.Generator,
    uav_positions: np.ndarray,
    target_states: np.ndarray,
) -> MeasurementBatch:
    points: list[np.ndarray] = []
    detected: list[int] = []
    for tid, state in enumerate(target_states):
        pos = state[:2]
        in_fov = np.any(np.linalg.norm(uav_positions - pos, axis=1) <= cfg.fov_radius)
        if in_fov and rng.random() < cfg.p_detection:
            points.append(pos + rng.normal(0.0, cfg.meas_std, size=2))
            detected.append(tid)

    clutter_count = int(rng.poisson(cfg.clutter_mean * len(uav_positions)))
    if clutter_count:
        owners = rng.integers(0, len(uav_positions), size=clutter_count)
        angles = rng.uniform(0, 2 * np.pi, size=clutter_count)
        radii = cfg.fov_radius * np.sqrt(rng.uniform(0, 1, size=clutter_count))
        clutter = uav_positions[owners] + np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
        clutter = np.clip(clutter, 0.0, cfg.map_size)
        points.extend(list(clutter))

    if points:
        arr = np.asarray(points, dtype=np.float32)
        arr = np.clip(arr, 0.0, cfg.map_size)
    else:
        arr = np.zeros((0, 2), dtype=np.float32)
    return MeasurementBatch(points=arr, detected_target_ids=detected, clutter_count=clutter_count)

