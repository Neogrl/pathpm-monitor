from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from config import Config
from utils import clipped_circle_area


@dataclass
class Peak:
    pos: np.ndarray
    weight: float


@dataclass(frozen=True)
class PHDUpdateDiagnostics:
    mass_before_update: float
    mass_after_measurement_update: float
    mass_after_resampling: float
    ess_before_resampling: float
    ess_after_resampling: float
    resampled: bool
    unique_particle_ratio: float
    regularized: bool
    birth_particle_count: int
    birth_mass: float
    proposal_particle_count: int
    proposal_measurement_count: int
    proposal_redistributed_mass: float
    component_masses: tuple[float, ...]
    component_particle_counts: tuple[int, ...]
    measurement_support_counts: tuple[int, ...]


class TargetBelief:
    def __init__(self, cfg: Config, eval_mode: bool = False):
        self.cfg = cfg
        self.n_particles = cfg.n_particles_eval if eval_mode else cfg.n_particles_train
        self.rng = np.random.default_rng(0)
        self.particles = np.zeros((self.n_particles, 4), dtype=np.float32)
        self.weights = np.zeros((self.n_particles,), dtype=np.float64)
        self._peak_cache: dict[int, list[Peak]] = {}
        self.last_update_diagnostics: Optional[PHDUpdateDiagnostics] = None
        self._birth_particle_count = 0
        self._birth_mass = 0.0
        self._proposal_particle_count = 0
        self._proposal_measurement_count = 0
        self._proposal_redistributed_mass = 0.0
        self._force_resample = False
        self._last_component_masses: tuple[float, ...] = ()
        self._last_component_counts: tuple[int, ...] = ()
        self._last_measurement_support_counts: tuple[int, ...] = ()
        self._last_regularized = False

    def _invalidate_cache(self) -> None:
        self._peak_cache.clear()

    def _sample_joint_particles(self, count: int) -> np.ndarray:
        count = int(count)
        if count <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        directions = max(1, min(int(self.cfg.phd_initial_velocity_directions), count))
        spatial_count = int(np.ceil(count / directions))
        columns = int(np.ceil(np.sqrt(spatial_count)))
        rows = int(np.ceil(spatial_count / columns))
        cells = self.rng.permutation(rows * columns)[:spatial_count]
        particles = np.zeros((count, 4), dtype=np.float32)
        write_index = 0
        for spatial_index, cell in enumerate(cells):
            group_count = min(directions, count - write_index)
            jitter = self.rng.uniform(0.0, 1.0, size=(group_count, 2))
            particles[write_index : write_index + group_count, 0] = (
                (cell % columns) + jitter[:, 0]
            ) / columns * self.cfg.map_size
            particles[write_index : write_index + group_count, 1] = (
                (cell // columns) + jitter[:, 1]
            ) / rows * self.cfg.map_size
            angle_offset = self.rng.uniform(0.0, 2.0 * np.pi / directions)
            direction_indices = np.arange(group_count) + (spatial_index % directions)
            angles = angle_offset + 2.0 * np.pi * direction_indices / directions
            particles[write_index : write_index + group_count, 2] = (
                np.cos(angles) * self.cfg.target_speed
            )
            particles[write_index : write_index + group_count, 3] = (
                np.sin(angles) * self.cfg.target_speed
            )
            write_index += group_count
            if write_index == count:
                break
        return particles

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.particles = self._sample_joint_particles(self.n_particles)
        self.weights = np.zeros((self.n_particles,), dtype=np.float64)
        self.weights[:] = self.cfg.phd_prior_count / self.n_particles
        self.last_update_diagnostics = None
        self._birth_particle_count = 0
        self._birth_mass = 0.0
        self._proposal_particle_count = 0
        self._proposal_measurement_count = 0
        self._proposal_redistributed_mass = 0.0
        self._force_resample = False
        self._last_component_masses = ()
        self._last_component_counts = ()
        self._last_measurement_support_counts = ()
        self._last_regularized = False
        self._invalidate_cache()

    def _sample_measurement_proposal(
        self,
        measurement: np.ndarray,
        count: int,
    ) -> np.ndarray:
        count = max(int(count), 1)
        particles = np.zeros((count, 4), dtype=np.float32)
        position_std = max(
            float(self.cfg.phd_measurement_proposal_position_std),
            1e-6,
        )
        particles[:, 0:2] = self.rng.normal(
            np.asarray(measurement, dtype=np.float64),
            position_std,
            size=(count, 2),
        )
        directions = max(
            1,
            min(int(self.cfg.phd_initial_velocity_directions), count),
        )
        angle_offset = self.rng.uniform(0.0, 2.0 * np.pi / directions)
        angles = (
            angle_offset
            + 2.0 * np.pi * np.arange(count) / directions
            + self.rng.normal(0.0, 0.04, size=count)
        )
        particles[:, 2] = np.cos(angles) * self.cfg.target_speed
        particles[:, 3] = np.sin(angles) * self.cfg.target_speed
        self._reflect_particles(particles)
        return particles

    def _inject_measurement_proposals(
        self,
        component_array: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        self._proposal_particle_count = 0
        self._proposal_measurement_count = 0
        self._proposal_redistributed_mass = 0.0
        if not self.cfg.phd_measurement_proposal_enabled or len(measurements) == 0:
            return component_array

        proposal_count = max(int(self.cfg.phd_measurement_proposal_particles), 1)
        mass_fraction = float(
            np.clip(self.cfg.phd_measurement_proposal_mass_fraction, 0.0, 1.0)
        )
        minimum_mass = max(
            float(self.cfg.phd_measurement_proposal_min_component_mass),
            0.0,
        )
        if mass_fraction <= 0.0:
            return component_array

        proposals: list[tuple[int, np.ndarray, float]] = []
        for measurement_index, measurement in enumerate(measurements):
            component_index = measurement_index + 1
            component_mass = float(np.sum(component_array[component_index]))
            if component_mass < minimum_mass:
                continue
            redistributed_mass = component_mass * mass_fraction
            proposal_particles = self._sample_measurement_proposal(
                measurement,
                proposal_count,
            )
            proposals.append(
                (component_index, proposal_particles, redistributed_mass)
            )
        if not proposals:
            return component_array

        old_count = len(self.particles)
        total_proposals = sum(len(item[1]) for item in proposals)
        expanded = np.zeros(
            (len(component_array), old_count + total_proposals),
            dtype=np.float64,
        )
        expanded[:, :old_count] = component_array
        new_particle_blocks = [self.particles]
        cursor = old_count
        for component_index, proposal_particles, redistributed_mass in proposals:
            expanded[component_index, :old_count] *= 1.0 - mass_fraction
            count = len(proposal_particles)
            expanded[
                component_index,
                cursor : cursor + count,
            ] = redistributed_mass / count
            new_particle_blocks.append(proposal_particles)
            cursor += count
            self._proposal_redistributed_mass += redistributed_mass
        self.particles = np.concatenate(new_particle_blocks, axis=0)
        self._proposal_particle_count = total_proposals
        self._proposal_measurement_count = len(proposals)
        self._force_resample = True
        return expanded

    def _reflect_particles(self, particles: np.ndarray) -> None:
        for axis in (0, 1):
            low = particles[:, axis] < 0
            high = particles[:, axis] > self.cfg.map_size
            particles[low, axis] = -particles[low, axis]
            particles[high, axis] = 2 * self.cfg.map_size - particles[high, axis]
            particles[low | high, axis + 2] *= -1
        particles[:, 0:2] = np.clip(particles[:, 0:2], 0.0, self.cfg.map_size)

    def predict(self, duration: float) -> None:
        duration = max(float(duration), 1e-6)
        noise_std = self.cfg.transition_noise * np.sqrt(duration)
        self.particles[:, 2:4] += self.rng.normal(
            0.0, noise_std, size=(len(self.particles), 2)
        )
        speed = np.linalg.norm(self.particles[:, 2:4], axis=1, keepdims=True)
        self.particles[:, 2:4] = self.particles[:, 2:4] / np.maximum(speed, 1e-8) * self.cfg.target_speed
        self.particles[:, 0:2] += self.particles[:, 2:4] * duration
        self._reflect_particles(self.particles)
        survival_probability = np.exp(-self.cfg.death_probability * duration)
        self.weights *= survival_probability
        self._birth_particle_count = 0
        self._birth_mass = 0.0
        self._force_resample = False
        if (
            self.cfg.phd_birth_scheme.lower() == "expansion"
            and self.cfg.birth_rate > 0.0
            and self.cfg.phd_birth_probability > 0.0
        ):
            birth_count = max(
                1, int(round(self.cfg.phd_birth_probability * self.n_particles))
            )
            birth_particles = self._sample_joint_particles(birth_count)
            birth_weights = np.full(
                birth_count,
                self.cfg.birth_rate / birth_count,
                dtype=np.float64,
            )
            self.particles = np.concatenate([self.particles, birth_particles], axis=0)
            self.weights = np.concatenate([self.weights, birth_weights], axis=0)
            self._birth_particle_count = birth_count
            self._birth_mass = float(self.cfg.birth_rate)
            self._force_resample = True
        self._invalidate_cache()

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if axis is None:
            maximum = float(np.max(values))
            if not np.isfinite(maximum):
                return np.asarray(-np.inf)
            return np.asarray(
                maximum + np.log(np.sum(np.exp(values - maximum)))
            )
        maximum = np.max(values, axis=axis, keepdims=True)
        finite = np.isfinite(maximum)
        shifted = np.full_like(values, -np.inf)
        np.subtract(values, maximum, out=shifted, where=np.broadcast_to(finite, values.shape))
        summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        result = np.where(finite, maximum + np.log(np.maximum(summed, 1e-300)), -np.inf)
        return np.squeeze(result, axis=axis)

    def clutter_intensity(
        self,
        measurement: np.ndarray,
        uav_positions: np.ndarray,
        fov_areas: Optional[np.ndarray] = None,
    ) -> float:
        if fov_areas is None:
            fov_areas = np.asarray(
                [clipped_circle_area(pos, self.cfg.fov_radius, self.cfg.map_size) for pos in uav_positions],
                dtype=np.float64,
            )
        sensors_covering = (
            np.linalg.norm(uav_positions - np.asarray(measurement)[None, :], axis=1)
            <= self.cfg.fov_radius
        )
        return float(
            np.sum(self.cfg.clutter_mean / np.maximum(fov_areas[sensors_covering], 1e-12))
        )

    def update(self, measurements: np.ndarray, uav_positions: np.ndarray) -> None:
        mass_before_update = float(np.sum(self.weights))
        in_any_fov = np.any(
            np.linalg.norm(self.particles[:, None, 0:2] - uav_positions[None, :, :], axis=2) <= self.cfg.fov_radius,
            axis=1,
        )
        p_detect = np.where(in_any_fov, self.cfg.filter_p_detection, 0.0).astype(np.float64)
        log_weights = np.full(len(self.weights), -np.inf, dtype=np.float64)
        positive_weights = self.weights > 0.0
        log_weights[positive_weights] = np.log(self.weights[positive_weights])
        log_missed_probability = np.full(len(p_detect), -np.inf, dtype=np.float64)
        missed_probability = 1.0 - p_detect
        positive_missed = missed_probability > 0.0
        log_missed_probability[positive_missed] = np.log(missed_probability[positive_missed])
        missed_log_weights = log_missed_probability + log_weights
        missed_component = np.exp(missed_log_weights)
        components = [missed_component]
        component_kinds = ["missed"]
        measurement_support_counts = []
        log_norm_const = -np.log(2.0 * np.pi * self.cfg.meas_std ** 2)
        log_p_detect = np.full(len(p_detect), -np.inf, dtype=np.float64)
        positive_detect = p_detect > 0.0
        log_p_detect[positive_detect] = np.log(p_detect[positive_detect])
        fov_areas = np.asarray(
            [clipped_circle_area(pos, self.cfg.fov_radius, self.cfg.map_size) for pos in uav_positions],
            dtype=np.float64,
        )
        for z in measurements:
            dist2 = np.sum((self.particles[:, 0:2] - z[None, :]) ** 2, axis=1)
            log_likelihood = log_norm_const - 0.5 * dist2 / (self.cfg.meas_std ** 2)
            log_numerator = log_p_detect + log_weights + log_likelihood
            log_target_intensity = float(self._logsumexp(log_numerator))
            clutter_intensity = self.clutter_intensity(z, uav_positions, fov_areas=fov_areas)
            log_clutter = (
                np.log(clutter_intensity) if clutter_intensity > 0.0 else -np.inf
            )
            log_denominator = float(np.logaddexp(log_clutter, log_target_intensity))
            if np.isfinite(log_denominator):
                component = np.exp(log_numerator - log_denominator)
            else:
                component = np.zeros_like(self.weights)
            components.append(component)
            component_kinds.append("measurement")
            measurement_support_counts.append(
                int(
                    np.sum(
                        positive_detect
                        & (dist2 <= (3.0 * self.cfg.meas_std) ** 2)
                    )
                )
            )
        component_array = np.stack(components, axis=0)
        component_array = self._inject_measurement_proposals(
            component_array,
            measurements,
        )
        updated = np.sum(component_array, axis=0)
        self.weights = np.where(np.isfinite(updated) & (updated > 0.0), updated, 0.0)
        mass_after_measurement_update = float(np.sum(self.weights))
        resampled, ess_before, ess_after, unique_ratio = self._resample_if_needed(
            components=component_array,
            component_kinds=component_kinds,
            force=self._force_resample,
        )
        self._last_measurement_support_counts = tuple(measurement_support_counts)
        self.last_update_diagnostics = PHDUpdateDiagnostics(
            mass_before_update=mass_before_update,
            mass_after_measurement_update=mass_after_measurement_update,
            mass_after_resampling=float(np.sum(self.weights)),
            ess_before_resampling=ess_before,
            ess_after_resampling=ess_after,
            resampled=resampled,
            unique_particle_ratio=unique_ratio,
            regularized=self._last_regularized,
            birth_particle_count=self._birth_particle_count,
            birth_mass=self._birth_mass,
            proposal_particle_count=self._proposal_particle_count,
            proposal_measurement_count=self._proposal_measurement_count,
            proposal_redistributed_mass=self._proposal_redistributed_mass,
            component_masses=self._last_component_masses,
            component_particle_counts=self._last_component_counts,
            measurement_support_counts=self._last_measurement_support_counts,
        )
        self._invalidate_cache()

    def effective_sample_size(self) -> float:
        total = float(np.sum(self.weights))
        if total <= 0:
            return 0.0
        normalized = self.weights / total
        return float(1.0 / np.sum(normalized ** 2))

    def _systematic_indices(self, probabilities: np.ndarray, count: int) -> np.ndarray:
        positions = (self.rng.random() + np.arange(count)) / count
        indices = np.searchsorted(np.cumsum(probabilities), positions, side="right")
        return np.minimum(indices, len(probabilities) - 1)

    def _component_allocations(
        self,
        masses: np.ndarray,
        component_kinds: list[str],
    ) -> np.ndarray:
        allocations = np.zeros(len(masses), dtype=np.int64)
        valid = masses > 1e-15
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            return allocations
        lower = np.ones(len(valid_indices), dtype=np.int64)
        for local_index, component_index in enumerate(valid_indices):
            if (
                component_kinds[component_index] == "measurement"
                and masses[component_index] >= self.cfg.phd_component_min_mass
            ):
                lower[local_index] = min(
                    int(self.cfg.phd_component_min_particles),
                    self.n_particles,
                )
        if int(np.sum(lower)) > self.n_particles:
            lower[:] = 1
        remaining = self.n_particles - int(np.sum(lower))
        probabilities = masses[valid_indices] / np.sum(masses[valid_indices])
        raw_extra = probabilities * remaining
        extra = np.floor(raw_extra).astype(np.int64)
        residual = remaining - int(np.sum(extra))
        if residual > 0:
            order = np.argsort(-(raw_extra - extra))
            extra[order[:residual]] += 1
        allocations[valid_indices] = lower + extra
        return allocations

    def _regularize_blocks(self, blocks: list[slice]) -> None:
        if not self.cfg.phd_regularization_enabled:
            self._last_regularized = False
            return
        min_scale = max(float(self.cfg.phd_regularization_min_scale), 0.0)
        max_scale = max(float(self.cfg.phd_regularization_max_scale), min_scale)
        for block_slice in blocks:
            block = self.particles[block_slice]
            count = len(block)
            if count <= 1:
                continue
            dimension = block.shape[1]
            bandwidth = (4.0 / (dimension + 2.0)) ** (1.0 / (dimension + 4.0))
            bandwidth *= count ** (-1.0 / (dimension + 4.0))
            local_std = np.std(block.astype(np.float64), axis=0)
            minimum_std = np.asarray(
                [
                    self.cfg.meas_std * min_scale,
                    self.cfg.meas_std * min_scale,
                    self.cfg.transition_noise * min_scale,
                    self.cfg.transition_noise * min_scale,
                ],
                dtype=np.float64,
            )
            maximum_std = np.asarray(
                [
                    self.cfg.meas_std * max_scale,
                    self.cfg.meas_std * max_scale,
                    self.cfg.transition_noise * max_scale,
                    self.cfg.transition_noise * max_scale,
                ],
                dtype=np.float64,
            )
            jitter_std = np.clip(bandwidth * local_std, minimum_std, maximum_std)
            block += self.rng.normal(0.0, jitter_std, size=block.shape).astype(np.float32)
            speed = np.linalg.norm(block[:, 2:4], axis=1, keepdims=True)
            block[:, 2:4] = (
                block[:, 2:4] / np.maximum(speed, 1e-8) * self.cfg.target_speed
            )
            self._reflect_particles(block)
            self.particles[block_slice] = block
        self._last_regularized = True

    def _resample_if_needed(
        self,
        components: Optional[np.ndarray] = None,
        component_kinds: Optional[list[str]] = None,
        force: bool = False,
    ) -> tuple[bool, float, float, float]:
        total = float(np.sum(self.weights))
        self._last_component_masses = ()
        self._last_component_counts = ()
        self._last_regularized = False
        if total <= 0:
            return False, 0.0, 0.0, float("nan")
        normalized = self.weights / total
        ess = float(1.0 / np.sum(normalized ** 2))
        if not force and len(self.particles) == self.n_particles and ess > self.n_particles * 0.35:
            return False, ess, ess, float("nan")
        mode = self.cfg.phd_resampling_mode.lower()
        blocks: list[slice] = []
        if mode == "component" and components is not None and component_kinds is not None:
            masses = np.sum(components, axis=1)
            allocations = self._component_allocations(masses, component_kinds)
            new_particles = []
            new_weights = []
            cursor = 0
            for component, mass, count in zip(components, masses, allocations):
                if count <= 0 or mass <= 0.0:
                    continue
                probabilities = component / mass
                indices = self._systematic_indices(probabilities, int(count))
                new_particles.append(self.particles[indices].copy())
                new_weights.append(np.full(int(count), mass / count, dtype=np.float64))
                blocks.append(slice(cursor, cursor + int(count)))
                cursor += int(count)
            self.particles = np.concatenate(new_particles, axis=0)
            self.weights = np.concatenate(new_weights, axis=0)
            self._last_component_masses = tuple(float(value) for value in masses)
            self._last_component_counts = tuple(int(value) for value in allocations)
        else:
            indices = self._systematic_indices(normalized, self.n_particles)
            self.particles = self.particles[indices].copy()
            self.weights = np.full(
                self.n_particles, total / self.n_particles, dtype=np.float64
            )
            blocks = [slice(0, self.n_particles)]
            if components is not None:
                self._last_component_masses = tuple(
                    float(value) for value in np.sum(components, axis=1)
                )
        unique_ratio = float(len(np.unique(self.particles, axis=0)) / self.n_particles)
        self._regularize_blocks(blocks)
        self._force_resample = False
        ess_after = self.effective_sample_size()
        return True, ess, ess_after, unique_ratio

    def diagnostics(self, max_peaks: Optional[int] = None) -> dict[str, object]:
        peaks = self.peaks(max_peaks)
        peak_positions = np.asarray([peak.pos for peak in peaks], dtype=np.float64).reshape(-1, 2)
        if len(peak_positions) >= 2:
            distances = np.linalg.norm(
                peak_positions[:, None, :] - peak_positions[None, :, :], axis=2
            )
            distances[np.eye(len(peak_positions), dtype=bool)] = np.inf
            minimum_cluster_distance = float(np.min(distances))
        else:
            minimum_cluster_distance = 0.0
        result: dict[str, object] = {
            "phd_total_mass": float(np.sum(self.weights)),
            "estimated_count_continuous": float(np.sum(self.weights)),
            "estimated_count_rounded": int(np.floor(np.sum(self.weights) + 0.5)),
            "num_extracted_candidates": len(peaks),
            "effective_sample_size": self.effective_sample_size(),
            "cluster_mass_list": [float(peak.weight) for peak in peaks],
            "minimum_cluster_distance": minimum_cluster_distance,
        }
        if self.last_update_diagnostics is not None:
            result.update(asdict(self.last_update_diagnostics))
            if not self.last_update_diagnostics.resampled:
                result["unique_particle_ratio"] = float(
                    len(np.unique(self.particles, axis=0)) / self.n_particles
                )
        return result

    def grid(self) -> np.ndarray:
        grid = np.zeros((self.cfg.search_bins, self.cfg.search_bins), dtype=np.float32)
        cells = np.clip((self.particles[:, 0:2] / self.cfg.cell_size).astype(int), 0, self.cfg.search_bins - 1)
        for (x, y), w in zip(cells, self.weights):
            grid[y, x] += w
        return grid

    @staticmethod
    def smooth_grid(grid: np.ndarray) -> np.ndarray:
        padded = np.pad(grid, 1, mode="edge")
        kernel = np.asarray(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        out = np.zeros_like(grid, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                out += kernel[dy, dx] * padded[dy : dy + grid.shape[0], dx : dx + grid.shape[1]]
        return out / float(np.sum(kernel))

    def peaks(self, max_peaks: Optional[int] = None) -> list[Peak]:
        max_peaks = max_peaks or self.cfg.max_target_candidates
        if max_peaks in self._peak_cache:
            return self._peak_cache[max_peaks]

        total = float(np.sum(self.weights))
        n_clusters = min(int(np.floor(total + 0.5)), int(max_peaks), self.n_particles)
        valid = np.isfinite(self.weights) & (self.weights > 0.0)
        if n_clusters <= 0 or not np.any(valid):
            self._peak_cache[max_peaks] = []
            return []

        points = self.particles[valid, 0:2].astype(np.float64)
        weights = self.weights[valid].astype(np.float64)
        n_clusters = min(n_clusters, len(points))
        density = self.smooth_grid(self.grid())
        centers: list[np.ndarray] = []
        min_seed_distance = max(float(self.cfg.target_candidate_min_separation), self.cfg.cell_size)
        for flat_index in np.argsort(density, axis=None)[::-1]:
            y, x = np.unravel_index(flat_index, density.shape)
            if density[y, x] <= 0.0:
                break
            candidate = np.asarray([(x + 0.5) * self.cfg.cell_size, (y + 0.5) * self.cfg.cell_size])
            if all(np.linalg.norm(candidate - center) >= min_seed_distance for center in centers):
                centers.append(candidate)
            if len(centers) == n_clusters:
                break

        if not centers:
            centers.append(points[int(np.argmax(weights))].copy())
        min_dist2 = np.min(
            np.sum((points[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        for _ in range(len(centers), n_clusters):
            score = weights * min_dist2
            next_index = int(np.argmax(score))
            centers.append(points[next_index].copy())
            min_dist2 = np.minimum(min_dist2, np.sum((points - centers[-1]) ** 2, axis=1))
        centers_array = np.asarray(centers, dtype=np.float64)

        labels = np.zeros(len(points), dtype=np.int64)
        for _ in range(30):
            dist2 = np.sum((points[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1)
            new_centers = centers_array.copy()
            for cluster_index in range(n_clusters):
                members = labels == cluster_index
                if np.any(members):
                    new_centers[cluster_index] = np.average(
                        points[members], axis=0, weights=weights[members]
                    )
            if np.max(np.linalg.norm(new_centers - centers_array, axis=1)) < 1e-4:
                centers_array = new_centers
                break
            centers_array = new_centers

        cluster_weights = np.bincount(labels, weights=weights, minlength=n_clusters)
        order = np.argsort(-cluster_weights)
        result = [
            Peak(pos=centers_array[index].astype(np.float32), weight=float(cluster_weights[index]))
            for index in order
            if cluster_weights[index] > 0.0
        ]
        self._peak_cache[max_peaks] = result
        return result

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
