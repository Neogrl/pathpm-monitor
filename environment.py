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
        )

    def update(self, step: int, detected_ids: list[int]) -> dict:
        detected_set = set(detected_ids)
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
        return {"newly_discovered": newly, "continuous": continuous}


@dataclass
class StepInfo:
    measurements: MeasurementBatch
    detected_ids: list[int]
    newly_discovered: int
    continuous_observed: int
    step_distance: np.ndarray


class CMUOMMTEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.uav_positions = np.zeros((cfg.n_uavs, 2), dtype=np.float32)
        self.target_states = np.zeros((cfg.n_targets_true, 4), dtype=np.float32)
        self.memory = DiscoveredTruthMemory.create(cfg.n_targets_true)

    def reset(self, seed: Optional[int] = None, n_targets: Optional[int] = None) -> dict:
        self.rng = np.random.default_rng(seed)
        n = int(n_targets or self.cfg.n_targets_true)
        self.step_count = 0
        y_positions = np.linspace(20.0, 80.0, self.cfg.n_uavs)
        self.uav_positions = np.stack([np.full(self.cfg.n_uavs, 8.0), y_positions], axis=1).astype(np.float32)
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

    def state_dict(self) -> dict:
        return {
            "uav_positions": self.uav_positions.copy(),
            "target_states": self.target_states.copy(),
            "step": self.step_count,
        }

    def _move_uavs(self, waypoints: np.ndarray) -> np.ndarray:
        delta = waypoints - self.uav_positions
        dist = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, self.cfg.uav_speed / np.maximum(dist, 1e-8))
        movement = delta * scale[:, None]
        self.uav_positions = np.clip(self.uav_positions + movement, 0.0, self.cfg.map_size)
        return np.linalg.norm(movement, axis=1)

    def _move_targets(self) -> None:
        noise = self.rng.normal(0.0, self.cfg.target_velocity_noise_std, size=(len(self.target_states), 2))
        self.target_states[:, 2:4] += noise
        speed = np.linalg.norm(self.target_states[:, 2:4], axis=1, keepdims=True)
        self.target_states[:, 2:4] = self.target_states[:, 2:4] / np.maximum(speed, 1e-8) * self.cfg.target_speed
        self.target_states[:, 0:2] += self.target_states[:, 2:4] * self.cfg.dt
        for axis in (0, 1):
            low = self.target_states[:, axis] < 0
            high = self.target_states[:, axis] > self.cfg.map_size
            self.target_states[low, axis] = -self.target_states[low, axis]
            self.target_states[high, axis] = 2 * self.cfg.map_size - self.target_states[high, axis]
            self.target_states[low | high, axis + 2] *= -1
        self.target_states[:, 0:2] = np.clip(self.target_states[:, 0:2], 0.0, self.cfg.map_size)

    def step(self, waypoints: np.ndarray) -> StepInfo:
        step_distance = self._move_uavs(waypoints.astype(np.float32))
        self._move_targets()
        measurements = generate_measurements(self.cfg, self.rng, self.uav_positions, self.target_states)
        memory_update = self.memory.update(self.step_count, measurements.detected_target_ids)
        self.step_count += 1
        return StepInfo(
            measurements=measurements,
            detected_ids=measurements.detected_target_ids,
            newly_discovered=memory_update["newly_discovered"],
            continuous_observed=memory_update["continuous"],
            step_distance=step_distance,
        )

    def done(self) -> bool:
        return self.step_count >= self.cfg.episode_steps
