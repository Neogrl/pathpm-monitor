import math
import unittest

import numpy as np

from config import Config
from environment import CMUOMMTEnv
from measurements import generate_measurements
from metrics import phd_tracking_errors
from target_belief import Peak, TargetBelief
from utils import clipped_circle_area


class TargetBeliefTests(unittest.TestCase):
    def test_reset_preserves_prior_cardinality(self) -> None:
        cfg = Config(n_particles_train=120, phd_prior_count=7.0)
        belief = TargetBelief(cfg)
        belief.reset(seed=4)
        self.assertAlmostEqual(float(np.sum(belief.weights)), 7.0, places=10)

    def test_reset_stratifies_velocity_directions_within_spatial_groups(self) -> None:
        cfg = Config(
            n_particles_train=80,
            phd_initial_velocity_directions=8,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=4)
        angles = np.mod(
            np.arctan2(belief.particles[:8, 3], belief.particles[:8, 2]),
            2.0 * np.pi,
        )
        bins = np.floor(angles / (2.0 * np.pi / 8)).astype(int)
        self.assertGreaterEqual(len(np.unique(bins)), 7)

    def test_predict_uses_actual_duration_and_preserves_mass(self) -> None:
        cfg = Config(
            n_particles_train=1,
            transition_noise=0.0,
            death_probability=0.0,
            phd_birth_scheme="none",
            target_speed=1.1,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=2)
        belief.particles[0] = np.asarray([50.0, 50.0, 1.1, 0.0], dtype=np.float32)
        initial_mass = float(np.sum(belief.weights))
        belief.predict(2.5)
        np.testing.assert_allclose(belief.particles[0, :2], [52.75, 50.0], atol=1e-6)
        self.assertAlmostEqual(float(np.sum(belief.weights)), initial_mass, places=10)

    def test_missed_detection_only_reduces_visible_particle(self) -> None:
        cfg = Config(n_particles_train=2, fov_radius=5.0, filter_p_detection=0.92)
        belief = TargetBelief(cfg)
        belief.reset(seed=1)
        belief.particles[:, :2] = np.asarray([[1.0, 1.0], [90.0, 90.0]], dtype=np.float32)
        belief.weights[:] = [1.0, 1.0]
        belief.update(np.zeros((0, 2), dtype=np.float32), np.asarray([[0.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(belief.weights, [0.08, 1.0], atol=1e-12)

    def test_systematic_resampling_preserves_phd_mass(self) -> None:
        cfg = Config(n_particles_train=4)
        belief = TargetBelief(cfg)
        belief.reset(seed=3)
        belief.weights[:] = [3.7, 0.1, 0.1, 0.1]
        belief._resample_if_needed()
        self.assertAlmostEqual(float(np.sum(belief.weights)), 4.0, places=10)
        np.testing.assert_allclose(belief.weights, np.ones(4), atol=1e-12)

    def test_ideal_measurements_recover_measurement_cardinality(self) -> None:
        cfg = Config(
            n_particles_train=200,
            fov_radius=100.0,
            filter_p_detection=1.0,
            clutter_mean=0.0,
            meas_std=1.3,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=5)
        belief.particles[:100, :2] = np.asarray([25.0, 25.0])
        belief.particles[100:, :2] = np.asarray([75.0, 75.0])
        belief.weights[:] = 2.0 / len(belief.weights)
        measurements = np.asarray([[25.0, 25.0], [75.0, 75.0]], dtype=np.float32)
        belief.update(measurements, np.asarray([[50.0, 50.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.sum(belief.weights)), 2.0, places=8)

    def test_log_space_update_keeps_far_supported_measurement_mass(self) -> None:
        cfg = Config(
            n_particles_train=20,
            fov_radius=200.0,
            filter_p_detection=1.0,
            clutter_mean=0.0,
            meas_std=1.3,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=5)
        belief.particles[:, :2] = 0.0
        belief.weights[:] = 1.0 / len(belief.weights)
        belief.update(
            np.asarray([[100.0, 100.0]], dtype=np.float32),
            np.asarray([[50.0, 50.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(float(np.sum(belief.weights)), 1.0, places=8)

    def test_expansion_birth_and_death_preserve_expected_prediction_mass(self) -> None:
        cfg = Config(
            n_particles_train=100,
            transition_noise=0.0,
            death_probability=0.2,
            birth_rate=1.0,
            phd_birth_probability=0.1,
            phd_birth_scheme="expansion",
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=8)
        belief.predict(2.0)
        self.assertEqual(len(belief.particles), 110)
        expected = cfg.phd_prior_count * math.exp(-0.4) + 1.0
        self.assertAlmostEqual(float(np.sum(belief.weights)), expected, places=8)
        belief.update(
            np.zeros((0, 2), dtype=np.float32),
            np.asarray([[200.0, 200.0]], dtype=np.float32),
        )
        self.assertEqual(len(belief.particles), 100)
        self.assertAlmostEqual(float(np.sum(belief.weights)), expected, places=8)
        diagnostics = belief.diagnostics()
        self.assertEqual(diagnostics["birth_particle_count"], 10)
        self.assertAlmostEqual(diagnostics["birth_mass"], 1.0, places=8)

    def test_measurement_proposal_restores_support_without_adding_mass(self) -> None:
        cfg = Config(
            n_particles_train=100,
            fov_radius=200.0,
            filter_p_detection=1.0,
            clutter_mean=0.0,
            meas_std=1.0,
            phd_measurement_proposal_enabled=True,
            phd_measurement_proposal_particles=20,
            phd_measurement_proposal_mass_fraction=0.5,
            phd_measurement_proposal_min_component_mass=0.1,
            phd_measurement_proposal_position_std=0.5,
            phd_regularization_enabled=False,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=19)
        belief.particles[:, :2] = 0.0
        belief.weights[:] = 1.0 / len(belief.weights)
        measurement = np.asarray([[50.0, 50.0]], dtype=np.float32)
        belief.update(
            measurement,
            np.asarray([[50.0, 50.0]], dtype=np.float32),
        )
        support = np.linalg.norm(
            belief.particles[:, :2] - measurement[0],
            axis=1,
        ) <= 2.0
        diagnostics = belief.diagnostics()
        self.assertAlmostEqual(float(np.sum(belief.weights)), 1.0, places=8)
        self.assertGreaterEqual(int(np.sum(support)), 40)
        self.assertEqual(diagnostics["proposal_particle_count"], 20)
        self.assertEqual(diagnostics["proposal_measurement_count"], 1)
        self.assertAlmostEqual(
            diagnostics["proposal_redistributed_mass"],
            0.5,
            places=8,
        )

    def test_component_resampling_preserves_component_masses(self) -> None:
        cfg = Config(
            n_particles_train=100,
            phd_resampling_mode="component",
            phd_component_min_mass=0.25,
            phd_component_min_particles=10,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=9)
        first = np.zeros(100, dtype=np.float64)
        second = np.zeros(100, dtype=np.float64)
        first[:10] = 0.1
        second[-10:] = 0.1
        components = np.stack([np.zeros(100), first, second], axis=0)
        belief.weights = np.sum(components, axis=0)
        belief._resample_if_needed(
            components=components,
            component_kinds=["missed", "measurement", "measurement"],
            force=True,
        )
        self.assertEqual(len(belief.particles), 100)
        self.assertAlmostEqual(float(np.sum(belief.weights)), 2.0, places=8)
        self.assertEqual(sum(belief._last_component_counts), 100)
        self.assertGreaterEqual(belief._last_component_counts[1], 10)
        self.assertGreaterEqual(belief._last_component_counts[2], 10)

    def test_regularization_moves_duplicate_particles_without_changing_mass(self) -> None:
        cfg = Config(
            n_particles_train=20,
            phd_regularization_enabled=True,
        )
        belief = TargetBelief(cfg)
        belief.reset(seed=10)
        belief.particles[:] = belief.particles[0]
        belief.weights[:] = 1.0 / len(belief.weights)
        belief._resample_if_needed(force=True)
        self.assertAlmostEqual(float(np.sum(belief.weights)), 1.0, places=8)
        self.assertGreater(len(np.unique(belief.particles, axis=0)), 1)
        self.assertTrue(belief._last_regularized)

    def test_clutter_intensity_adds_overlapping_sensor_intensities(self) -> None:
        cfg = Config(n_particles_train=4, map_size=100.0, fov_radius=12.0, clutter_mean=0.7)
        belief = TargetBelief(cfg)
        uav_positions = np.asarray([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
        expected = 2.0 * cfg.clutter_mean / (math.pi * cfg.fov_radius ** 2)
        self.assertAlmostEqual(
            belief.clutter_intensity(np.asarray([50.0, 50.0]), uav_positions),
            expected,
            places=12,
        )
        self.assertAlmostEqual(
            clipped_circle_area(np.asarray([50.0, 50.0]), cfg.fov_radius, cfg.map_size),
            math.pi * cfg.fov_radius ** 2,
            places=10,
        )

    def test_clutter_samples_stay_inside_effective_fov(self) -> None:
        cfg = Config(p_detection=0.0, clutter_mean=50.0, fov_radius=12.0)
        rng = np.random.default_rng(9)
        uav_positions = np.asarray([[0.0, 0.0]], dtype=np.float32)
        batch = generate_measurements(
            cfg,
            rng,
            uav_positions,
            np.zeros((0, 4), dtype=np.float32),
        )
        self.assertGreater(batch.clutter_count, 0)
        self.assertTrue(np.all(batch.points >= 0.0))
        self.assertTrue(np.all(batch.points <= cfg.map_size))
        self.assertTrue(np.all(np.linalg.norm(batch.points - uav_positions[0], axis=1) <= cfg.fov_radius))

    def test_union_fov_generates_at_most_one_detection_per_target(self) -> None:
        cfg = Config(p_detection=1.0, clutter_mean=0.0, fov_radius=20.0)
        batch = generate_measurements(
            cfg,
            np.random.default_rng(12),
            np.asarray([[45.0, 50.0], [55.0, 50.0]], dtype=np.float32),
            np.asarray([[50.0, 50.0, 0.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(batch.detected_target_ids, [0])
        self.assertEqual(len(batch.points), 1)

    def test_true_measurements_are_not_clipped_to_map_boundary(self) -> None:
        cfg = Config(
            p_detection=1.0,
            clutter_mean=0.0,
            fov_radius=200.0,
            meas_std=10.0,
        )
        targets = np.repeat(
            np.asarray([[0.0, 50.0, 0.0, 0.0]], dtype=np.float32),
            128,
            axis=0,
        )
        batch = generate_measurements(
            cfg,
            np.random.default_rng(15),
            np.asarray([[0.0, 50.0]], dtype=np.float32),
            targets,
        )
        self.assertEqual(len(batch.points), len(targets))
        self.assertTrue(np.any(batch.points[:, 0] < 0.0))

    def test_environment_and_phd_motion_updates_match(self) -> None:
        count = 64
        cfg = Config(
            n_targets_true=count,
            n_particles_train=count,
            target_velocity_noise_std=0.06,
            transition_noise=0.06,
            death_probability=0.0,
            phd_birth_scheme="none",
        )
        state_rng = np.random.default_rng(88)
        positions = state_rng.uniform(5.0, 95.0, size=(count, 2))
        angles = state_rng.uniform(0.0, 2.0 * np.pi, size=count)
        velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * cfg.target_speed
        states = np.concatenate([positions, velocities], axis=1).astype(np.float32)

        env = CMUOMMTEnv(cfg)
        env.target_states = states.copy()
        belief = TargetBelief(cfg)
        belief.particles = states.copy()
        belief.weights[:] = cfg.phd_prior_count / count
        env.rng = np.random.default_rng(44)
        belief.rng = np.random.default_rng(44)

        env._move_targets(2.75)
        belief.predict(2.75)
        np.testing.assert_allclose(belief.particles, env.target_states, atol=1e-6)

    def test_environment_and_phd_boundary_reflection_match(self) -> None:
        cfg = Config(
            n_targets_true=1,
            n_particles_train=1,
            target_speed=1.1,
            target_velocity_noise_std=0.0,
            transition_noise=0.0,
            death_probability=0.0,
            phd_birth_scheme="none",
        )
        state = np.asarray([[99.0, 50.0, 1.1, 0.0]], dtype=np.float32)
        env = CMUOMMTEnv(cfg)
        env.target_states = state.copy()
        belief = TargetBelief(cfg)
        belief.particles = state.copy()
        belief.weights[:] = 1.0
        env._move_targets(2.0)
        belief.predict(2.0)
        np.testing.assert_allclose(belief.particles, env.target_states, atol=1e-6)

    def test_weighted_clustering_extracts_two_centres(self) -> None:
        cfg = Config(n_particles_train=6, max_target_candidates=8)
        belief = TargetBelief(cfg)
        belief.particles[:, :2] = np.asarray(
            [[9.5, 10.0], [10.0, 10.5], [10.5, 9.5], [79.5, 80.0], [80.0, 80.5], [80.5, 79.5]],
            dtype=np.float32,
        )
        belief.weights[:] = 1.0 / 3.0
        peaks = belief.peaks()
        self.assertEqual(len(peaks), 2)
        centres = np.asarray([peak.pos for peak in peaks])
        expected = np.asarray([[10.0, 10.0], [80.0, 80.0]])
        distances = np.linalg.norm(centres[:, None, :] - expected[None, :, :], axis=2)
        self.assertLess(float(np.min(distances[:, 0])), 0.4)
        self.assertLess(float(np.min(distances[:, 1])), 0.4)

    def test_cardinality_error_uses_continuous_weight_sum(self) -> None:
        peaks = [Peak(np.asarray([0.0, 0.0]), 1.0), Peak(np.asarray([1.0, 1.0]), 1.0)]
        truths = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, number_error = phd_tracking_errors(peaks, truths, estimated_count=2.4)
        self.assertAlmostEqual(number_error, 0.6, places=10)

    def test_diagnostics_report_resampling_without_changing_mass(self) -> None:
        cfg = Config(n_particles_train=4)
        belief = TargetBelief(cfg)
        belief.reset(seed=6)
        belief.weights[:] = [3.7, 0.1, 0.1, 0.1]
        belief.update(
            np.zeros((0, 2), dtype=np.float32),
            np.asarray([[200.0, 200.0]], dtype=np.float32),
        )
        diagnostics = belief.diagnostics()
        self.assertTrue(diagnostics["resampled"])
        self.assertAlmostEqual(diagnostics["mass_after_measurement_update"], 4.0, places=10)
        self.assertAlmostEqual(diagnostics["mass_after_resampling"], 4.0, places=10)
        self.assertEqual(diagnostics["ess_after_resampling"], 4.0)

    def test_half_up_cardinality_rounding_is_used_for_extraction(self) -> None:
        cfg = Config(n_particles_train=14, max_target_candidates=8)
        belief = TargetBelief(cfg)
        belief.reset(seed=7)
        belief.weights[:] = 6.5 / len(belief.weights)
        diagnostics = belief.diagnostics()
        self.assertEqual(diagnostics["estimated_count_rounded"], 7)
        self.assertEqual(diagnostics["num_extracted_candidates"], 7)


if __name__ == "__main__":
    unittest.main()
