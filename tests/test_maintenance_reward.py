import unittest

import numpy as np

from config import Config
from environment import DiscoveredTruthMemory
from metrics import reward_terms, weighted_reward


class MaintenanceRewardTests(unittest.TestCase):
    def test_unobserved_time_only_accumulates_for_discovered_targets(self) -> None:
        memory = DiscoveredTruthMemory.create(3)
        memory.update(0, [0], visible_ids=[0], duration=2.0)
        memory.update(1, [], visible_ids=[], duration=2.5)
        memory.update(2, [1], visible_ids=[1], duration=1.0)

        np.testing.assert_allclose(
            memory.current_unobserved_time,
            [3.5, 0.0, 0.0],
            atol=1e-7,
        )
        self.assertTrue(memory.is_discovered[0])
        self.assertTrue(memory.is_discovered[1])
        self.assertFalse(memory.is_discovered[2])

    def test_reward_combines_visibility_discovered_age_and_duplicate_coverage(self) -> None:
        cfg = Config(
            n_uavs=2,
            maintenance_age_horizon=8.0,
            reward_visibility_weight=1.0,
            reward_maintenance_age_weight=-0.5,
            reward_duplicate_coverage_weight=-0.1,
        )
        memory = DiscoveredTruthMemory.create(3)
        memory.is_discovered[:] = [True, True, False]
        memory.current_unobserved_time[:] = [4.0, 0.0, 0.0]
        terms = reward_terms(
            cfg,
            memory,
            detected_count=0,
            newly_discovered=0,
            continuous_observed=0,
            estimated_peaks=[],
            estimated_count=3.0,
            true_positions=np.zeros((3, 2), dtype=np.float32),
            target_coverage_counts=np.asarray([2, 1, 0], dtype=np.int32),
            previous_coverage_age=np.zeros(
                (cfg.search_bins, cfg.search_bins),
                dtype=np.float32,
            ),
            uav_positions=np.zeros((2, 2), dtype=np.float32),
            step_distance=np.zeros(2, dtype=np.float32),
            option_switched=np.zeros(2, dtype=np.float32),
        )

        self.assertAlmostEqual(terms["visibility"], 2.0 / 3.0)
        self.assertAlmostEqual(terms["maintenance_age"], 0.25)
        self.assertAlmostEqual(terms["duplicate_coverage"], 0.5)
        self.assertAlmostEqual(
            weighted_reward(terms, cfg),
            2.0 / 3.0 - 0.5 * 0.25 - 0.1 * 0.5,
        )


if __name__ == "__main__":
    unittest.main()
