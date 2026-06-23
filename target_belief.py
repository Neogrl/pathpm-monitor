from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import Config
from utils import local_maxima_2d, non_max_suppression


@dataclass
class Peak:
    pos: np.ndarray
    weight: float


class TargetBelief:
    def __init__(self, cfg: Config, eval_mode: bool = False):
        self.cfg = cfg
        self.n_particles = cfg.n_particles_eval if eval_mode else cfg.n_particles_train
        self.rng = np.random.default_rng(0)
        self.particles = np.zeros((self.n_particles, 4), dtype=np.float32)
        self.weights = np.zeros((self.n_particles,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.particles[:, 0:2] = self.rng.uniform(0, self.cfg.map_size, size=(self.n_particles, 2))
        angles = self.rng.uniform(0, 2 * np.pi, size=self.n_particles)
        self.particles[:, 2] = np.cos(angles) * self.cfg.target_speed
        self.particles[:, 3] = np.sin(angles) * self.cfg.target_speed
        self.weights[:] = self.cfg.phd_prior_count / self.n_particles

    def predict(self) -> None:
        self.particles[:, 0:2] += self.particles[:, 2:4] * self.cfg.dt
        self.particles[:, 2:4] += self.rng.normal(0.0, self.cfg.transition_noise, size=(self.n_particles, 2))
        speed = np.linalg.norm(self.particles[:, 2:4], axis=1, keepdims=True)
        self.particles[:, 2:4] = self.particles[:, 2:4] / np.maximum(speed, 1e-8) * self.cfg.target_speed
        for axis in (0, 1):
            low = self.particles[:, axis] < 0
            high = self.particles[:, axis] > self.cfg.map_size
            self.particles[low, axis] = -self.particles[low, axis]
            self.particles[high, axis] = 2 * self.cfg.map_size - self.particles[high, axis]
            self.particles[low | high, axis + 2] *= -1
        self.particles[:, 0:2] = np.clip(self.particles[:, 0:2], 0.0, self.cfg.map_size)
        self.weights *= 1.0 - self.cfg.death_probability
        self._birth_particles()

    def _birth_particles(self) -> None:
        n_birth = max(1, self.n_particles // 20)
        idx = self.rng.choice(self.n_particles, size=n_birth, replace=False)
        self.particles[idx, 0:2] = self.rng.uniform(0, self.cfg.map_size, size=(n_birth, 2))
        angles = self.rng.uniform(0, 2 * np.pi, size=n_birth)
        self.particles[idx, 2] = np.cos(angles) * self.cfg.target_speed
        self.particles[idx, 3] = np.sin(angles) * self.cfg.target_speed
        self.weights[idx] += self.cfg.birth_rate / n_birth

    def update(self, measurements: np.ndarray, uav_positions: np.ndarray) -> None:
        if len(measurements) == 0:
            self.weights *= 0.97
            self._renormalize(max_total=max(self.cfg.phd_prior_count * 1.5, 1.0))
            return
        likelihood = np.zeros_like(self.weights)
        for z in measurements:
            dist2 = np.sum((self.particles[:, 0:2] - z[None, :]) ** 2, axis=1)
            likelihood += np.exp(-0.5 * dist2 / (self.cfg.meas_std ** 2))
        in_any_fov = np.any(
            np.linalg.norm(self.particles[:, None, 0:2] - uav_positions[None, :, :], axis=2) <= self.cfg.fov_radius,
            axis=1,
        )
        self.weights *= np.where(in_any_fov, 0.55 + self.cfg.p_detection * likelihood, 1.0)
        self._measurement_birth(measurements)
        self._renormalize(max_total=max(len(measurements) + self.cfg.phd_prior_count, 1.0))
        self._resample_if_needed()

    def _measurement_birth(self, measurements: np.ndarray) -> None:
        if len(measurements) == 0:
            return
        n_birth = min(len(measurements) * 20, self.n_particles // 4)
        idx = self.rng.choice(self.n_particles, size=n_birth, replace=False)
        src = measurements[self.rng.integers(0, len(measurements), size=n_birth)]
        self.particles[idx, 0:2] = np.clip(src + self.rng.normal(0, self.cfg.meas_std, size=(n_birth, 2)), 0, self.cfg.map_size)
        angles = self.rng.uniform(0, 2 * np.pi, size=n_birth)
        self.particles[idx, 2] = np.cos(angles) * self.cfg.target_speed
        self.particles[idx, 3] = np.sin(angles) * self.cfg.target_speed
        self.weights[idx] += self.cfg.birth_rate / max(n_birth, 1)

    def _renormalize(self, max_total: float) -> None:
        self.weights = np.maximum(self.weights, 1e-12)
        total = float(np.sum(self.weights))
        if total > max_total:
            self.weights *= max_total / total

    def _resample_if_needed(self) -> None:
        total = float(np.sum(self.weights))
        if total <= 0:
            self.reset()
            return
        normalized = self.weights / total
        ess = 1.0 / np.sum(normalized ** 2)
        if ess > self.n_particles * 0.35:
            return
        indices = self.rng.choice(self.n_particles, size=self.n_particles, replace=True, p=normalized)
        self.particles = self.particles[indices].copy()
        self.weights[:] = total / self.n_particles

    def grid(self) -> np.ndarray:
        grid = np.zeros((self.cfg.search_bins, self.cfg.search_bins), dtype=np.float32)
        cells = np.clip((self.particles[:, 0:2] / self.cfg.cell_size).astype(int), 0, self.cfg.search_bins - 1)
        for (x, y), w in zip(cells, self.weights):
            grid[y, x] += w
        return grid

    def peaks(self, max_peaks: Optional[int] = None) -> list[Peak]:
        max_peaks = max_peaks or self.cfg.max_target_candidates
        grid = self.grid()
        coords, scores = local_maxima_2d(grid, self.cfg.target_peak_min_weight)
        if len(coords) == 0:
            return []
        points = (coords + 0.5) * self.cfg.cell_size
        keep = non_max_suppression(points, scores, self.cfg.target_candidate_min_separation, max_peaks)
        return [Peak(pos=points[i].astype(np.float32), weight=float(scores[i])) for i in keep]

    def particles_in_fov(self, center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        dist = np.linalg.norm(self.particles[:, 0:2] - center[None, :], axis=1)
        mask = dist <= radius
        return self.particles[mask], self.weights[mask]

    def summary(self) -> np.ndarray:
        peaks = self.peaks()
        weights = np.asarray([p.weight for p in peaks], dtype=np.float32)
        estimated = float(np.sum(self.weights))
        if len(weights) == 0:
            return np.asarray([estimated / max(self.cfg.phd_prior_count, 1.0), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(
            [
                estimated / max(self.cfg.phd_prior_count, 1.0),
                float(np.var(self.weights) * self.n_particles),
                float(np.max(weights)),
                float(np.mean(weights)),
                len(weights) / max(self.cfg.max_target_candidates, 1),
            ],
            dtype=np.float32,
        )
