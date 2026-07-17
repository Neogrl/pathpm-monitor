from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import Config
from measurements import MeasurementBatch, generate_measurements


@dataclass
class DiscoveredTruthMemory:
    is_discovered: np.ndarray
    first_detection_step: np.ndarray
    last_observed_step: np.ndarray
    observation_count: np.ndarray
    current_gap: np.ndarray
    max_observation_gap: np.ndarray
    observed_prev: np.ndarray
    current_unobserved_time: np.ndarray
    max_unobserved_time: np.ndarray

    @classmethod
    def create(cls, n_targets: int) -> "DiscoveredTruthMemory":
        return cls(
            is_discovered=np.zeros(n_targets, dtype=bool),
            first_detection_step=-np.ones(n_targets, dtype=np.int32),
            last_observed_step=-np.ones(n_targets, dtype=np.int32),
            observation_count=np.zeros(n_targets, dtype=np.int32),
            current_gap=np.zeros(n_targets, dtype=np.int32),
            max_observation_gap=np.zeros(n_targets, dtype=np.int32),
            observed_prev=np.zeros(n_targets, dtype=bool),
            current_unobserved_time=np.zeros(n_targets, dtype=np.float32),
            max_unobserved_time=np.zeros(n_targets, dtype=np.float32),
        )

    def update(
        self,
        step: int,
        detected_ids: list[int],
        visible_ids: Optional[list[int]] = None,
        duration: float = 1.0,
    ) -> dict:
        detected_set = set(detected_ids)
        visible_set = detected_set if visible_ids is None else set(visible_ids)
        duration = max(float(duration), 0.0)
        newly = 0
        continuous = 0
        prev_observed = self.observed_prev.copy()
        self.observed_prev[:] = False
        for tid in range(len(self.is_discovered)):
            if tid in detected_set:
                if not self.is_discovered[tid]:
                    self.is_discovered[tid] = True
                    self.first_detection_step[tid] = step
                    newly += 1
                if prev_observed[tid]:
                    continuous += 1
                self.last_observed_step[tid] = step
                self.observation_count[tid] += 1
                self.current_gap[tid] = 0
                self.observed_prev[tid] = True
            elif self.is_discovered[tid]:
                self.current_gap[tid] += 1
                self.max_observation_gap[tid] = max(self.max_observation_gap[tid], self.current_gap[tid])
            if self.is_discovered[tid]:
                if tid in visible_set:
                    self.current_unobserved_time[tid] = 0.0
                else:
                    self.current_unobserved_time[tid] += duration
                    self.max_unobserved_time[tid] = max(
                        self.max_unobserved_time[tid],
                        self.current_unobserved_time[tid],
                    )
        return {"newly_discovered": newly, "continuous": continuous}


@dataclass
class StepInfo:
    measurements: MeasurementBatch
    detected_ids: list[int]
    newly_discovered: int
    continuous_observed: int
    step_distance: np.ndarray
    step_duration: float
    visible_target_ids: list[int]
    target_coverage_counts: np.ndarray


class CMUOMMTEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.uav_positions = np.zeros((cfg.n_uavs, 2), dtype=np.float32)
        self.target_states = np.zeros((cfg.n_targets_true, 4), dtype=np.float32)
        self.memory = DiscoveredTruthMemory.create(cfg.n_targets_true)

    def reset(
        self,
        seed: Optional[int] = None,
        n_targets: Optional[int] = None,
        uav_positions: Optional[np.ndarray] = None,
    ) -> dict:
        self.rng = np.random.default_rng(seed)
        n = int(n_targets or self.cfg.n_targets_true)
        self.step_count = 0
        if uav_positions is None:
            self.uav_positions = self._initial_uav_positions()
        else:
            uav_positions = np.asarray(uav_positions, dtype=np.float32)
            if uav_positions.shape != (self.cfg.n_uavs, 2):
                raise ValueError(f"uav_positions must have shape {(self.cfg.n_uavs, 2)}, got {uav_positions.shape}")
            self.uav_positions = np.clip(uav_positions, 0.0, self.cfg.map_size).astype(np.float32)
        positions = self.rng.uniform(
            self.cfg.target_init_margin,
            self.cfg.map_size - self.cfg.target_init_margin,
            size=(n, 2),
        )
        angles = self.rng.uniform(0, 2 * np.pi, size=n)
        velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.cfg.target_speed
        self.target_states = np.concatenate([positions, velocities], axis=1).astype(np.float32)
        self.memory = DiscoveredTruthMemory.create(n)
        return self.state_dict()

    def _initial_uav_positions(self) -> np.ndarray:
        if not self.cfg.randomize_uav_start:
            y_positions = np.linspace(20.0, 80.0, self.cfg.n_uavs)
            return np.stack([np.full(self.cfg.n_uavs, 8.0), y_positions], axis=1).astype(np.float32)

        candidate_nodes = self._uav_start_node_candidates()
        margin = float(np.clip(self.cfg.uav_init_margin, 0.0, self.cfg.map_size * 0.49))
        low = margin
        high = self.cfg.map_size - margin
        if high <= low:
            low, high = 0.0, self.cfg.map_size
        min_sep = max(float(self.cfg.uav_init_min_separation), 0.0)
        positions: list[np.ndarray] = []
        max_attempts = max(200, self.cfg.n_uavs * 200)
        for _ in range(max_attempts):
            candidate = candidate_nodes[int(self.rng.integers(0, len(candidate_nodes)))].astype(np.float32)
            if all(np.linalg.norm(candidate - pos) >= min_sep for pos in positions):
                positions.append(candidate)
                if len(positions) == self.cfg.n_uavs:
                    return np.stack(positions, axis=0).astype(np.float32)

        while len(positions) < self.cfg.n_uavs:
            positions.append(candidate_nodes[int(self.rng.integers(0, len(candidate_nodes)))].astype(np.float32))
        return np.stack(positions, axis=0).astype(np.float32)

    def _uav_start_node_candidates(self) -> np.ndarray:
        spacing = max(float(self.cfg.graph_node_spacing), 1e-6)
        xs = np.arange(0.0, self.cfg.map_size + 1e-6, spacing, dtype=np.float32)
        ys = np.arange(0.0, self.cfg.map_size + 1e-6, spacing, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        nodes = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
        margin = float(np.clip(self.cfg.uav_init_margin, 0.0, self.cfg.map_size * 0.49))
        inside = (
            (nodes[:, 0] >= margin)
            & (nodes[:, 0] <= self.cfg.map_size - margin)
            & (nodes[:, 1] >= margin)
            & (nodes[:, 1] <= self.cfg.map_size - margin)
        )
        candidates = nodes[inside]
        return candidates if len(candidates) > 0 else nodes

    def state_dict(self) -> dict:
        return {
            "uav_positions": self.uav_positions.copy(),
            "target_states": self.target_states.copy(),
            "step": self.step_count,
        }

    def _move_uavs(self, waypoints: np.ndarray) -> tuple[np.ndarray, float]:
        delta = waypoints - self.uav_positions
        dist = np.linalg.norm(delta, axis=1)
        if self.cfg.graph_type.lower() == "prm":
            self.uav_positions = np.clip(waypoints.astype(np.float32), 0.0, self.cfg.map_size)
            duration = float(np.max(dist / max(self.cfg.uav_speed, 1e-8))) if len(dist) else 0.0
            return dist.astype(np.float32), duration
        scale = np.minimum(1.0, self.cfg.uav_speed / np.maximum(dist, 1e-8))
        movement = delta * scale[:, None]
        self.uav_positions = np.clip(self.uav_positions + movement, 0.0, self.cfg.map_size)
        return np.linalg.norm(movement, axis=1).astype(np.float32), self.cfg.dt

    def _move_targets(self, duration: float) -> None:
        duration = max(float(duration), 1e-6)
        noise = self.rng.normal(0.0, self.cfg.target_velocity_noise_std * np.sqrt(duration), size=(len(self.target_states), 2))
        self.target_states[:, 2:4] += noise
        speed = np.linalg.norm(self.target_states[:, 2:4], axis=1, keepdims=True)
        self.target_states[:, 2:4] = self.target_states[:, 2:4] / np.maximum(speed, 1e-8) * self.cfg.target_speed
        self.target_states[:, 0:2] += self.target_states[:, 2:4] * duration
        for axis in (0, 1):
            low = self.target_states[:, axis] < 0
            high = self.target_states[:, axis] > self.cfg.map_size
            self.target_states[low, axis] = -self.target_states[low, axis]
            self.target_states[high, axis] = 2 * self.cfg.map_size - self.target_states[high, axis]
            self.target_states[low | high, axis + 2] *= -1
        self.target_states[:, 0:2] = np.clip(self.target_states[:, 0:2], 0.0, self.cfg.map_size)

    def step(self, waypoints: np.ndarray) -> StepInfo:
        step_distance, step_duration = self._move_uavs(waypoints.astype(np.float32))
        self._move_targets(step_duration)
        target_distances = np.linalg.norm(
            self.target_states[:, None, 0:2] - self.uav_positions[None, :, :],
            axis=2,
        )
        target_coverage_counts = np.sum(
            target_distances <= self.cfg.fov_radius,
            axis=1,
        ).astype(np.int32)
        visible_target_ids = np.flatnonzero(target_coverage_counts > 0).tolist()
        measurements = generate_measurements(self.cfg, self.rng, self.uav_positions, self.target_states)
        memory_update = self.memory.update(
            self.step_count,
            measurements.detected_target_ids,
            visible_ids=visible_target_ids,
            duration=step_duration,
        )
        self.step_count += 1
        return StepInfo(
            measurements=measurements,
            detected_ids=measurements.detected_target_ids,
            newly_discovered=memory_update["newly_discovered"],
            continuous_observed=memory_update["continuous"],
            step_distance=step_distance,
            step_duration=step_duration,
            visible_target_ids=visible_target_ids,
            target_coverage_counts=target_coverage_counts,
        )

    def done(self) -> bool:
        return self.step_count >= self.cfg.episode_steps
