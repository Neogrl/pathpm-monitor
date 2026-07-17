from dataclasses import dataclass

import numpy as np

from config import Config


@dataclass
class MeasurementBatch:
    """Measurements from the team-level union-FOV sensor model.

    Each target produces at most one detection per step when it lies inside at
    least one UAV FOV. Clutter processes remain per-UAV and are superimposed in
    the returned point set.
    """

    points: np.ndarray
    detected_target_ids: list[int]
    clutter_count: int


def _sample_clutter_point(
    cfg: Config,
    rng: np.random.Generator,
    center: np.ndarray,
) -> np.ndarray:
    """Sample uniformly from a circular FOV conditioned on lying inside the map."""
    while True:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = cfg.fov_radius * np.sqrt(rng.uniform(0.0, 1.0))
        point = center + np.asarray([np.cos(angle), np.sin(angle)]) * radius
        if np.all((point >= 0.0) & (point <= cfg.map_size)):
            return point


def generate_measurements(
    cfg: Config,
    rng: np.random.Generator,
    uav_positions: np.ndarray,
    target_states: np.ndarray,
) -> MeasurementBatch:
    """Generate one fused measurement set for the complete UAV team."""
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
        points.extend(_sample_clutter_point(cfg, rng, uav_positions[owner]) for owner in owners)

    if points:
        arr = np.asarray(points, dtype=np.float32)
    else:
        arr = np.zeros((0, 2), dtype=np.float32)
    return MeasurementBatch(points=arr, detected_target_ids=detected, clutter_count=clutter_count)
