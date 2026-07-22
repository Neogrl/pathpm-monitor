import unittest

import torch

from config import Config
from model import OptionActor
from train import TrainingLogger


class TrainingInterfaceTests(unittest.TestCase):
    def test_tensorboard_uses_small_explicit_metric_set(self) -> None:
        row = {
            "episode_reward": 10.0,
            "observation_rate": 0.4,
            "reward_visibility": 0.5,
            "val_observation_rate": 0.45,
            "policy_loss": 0.1,
            "phd_total_weight": 4.2,
            "rollout_seconds": 3.0,
            "candidate_distance_norm_mean": 0.25,
            "seed": 123,
        }
        scalars = TrainingLogger.tensorboard_scalars(row)
        self.assertEqual(scalars["01_Train/episode_return"], 10.0)
        self.assertEqual(scalars["01_Train/observation_rate"], 0.4)
        self.assertEqual(scalars["02_Validation/observation_rate"], 0.45)
        self.assertEqual(scalars["03_Optimization/policy_loss"], 0.1)
        self.assertEqual(scalars["04_PHD/train_estimated_target_count"], 4.2)
        self.assertEqual(scalars["05_Performance/rollout_seconds"], 3.0)
        self.assertNotIn("candidate_distance_norm_mean", scalars)
        self.assertNotIn("seed", scalars)

    def test_agent_embedding_breaks_identical_actor_queries(self) -> None:
        torch.manual_seed(7)
        cfg = Config(n_uavs=2, graph_laplacian_pe_enabled=False, disable_options=True)
        actor = OptionActor(cfg).eval()
        b, n, g, m = 1, cfg.n_uavs, 4, 3
        shared_nodes = torch.randn(b, 1, g, cfg.node_input_dim).expand(b, n, g, cfg.node_input_dim).clone()
        candidate_indices = torch.tensor([[[1, 2, 3], [1, 2, 3]]])

        with torch.no_grad():
            _, logits, _ = actor(
                shared_nodes,
                torch.zeros(b, n, g, cfg.graph_laplacian_pe_dim),
                torch.zeros(b, g, g, dtype=torch.bool),
                torch.zeros(b, g, dtype=torch.bool),
                torch.zeros(b, n, dtype=torch.long),
                candidate_indices,
                torch.zeros(b, n, m, dtype=torch.bool),
                torch.zeros(b, n, m, dtype=torch.bool),
                torch.zeros(b, n, cfg.uav_state_dim),
                torch.zeros(b, n, dtype=torch.long),
            )

        self.assertFalse(torch.allclose(logits[0, 0], logits[0, 1]))

    def test_new_training_defaults_match_maintenance_task(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.best_metric, "val_observation_rate")
        self.assertEqual(cfg.reward_visibility_weight, 1.0)
        self.assertEqual(cfg.reward_maintenance_age_weight, -0.5)
        self.assertEqual(cfg.reward_search_weight, 8.0)
        self.assertEqual(cfg.reward_duplicate_coverage_weight, -0.5)
        self.assertGreater(cfg.eval_interval, 0)
        self.assertGreater(cfg.eval_episodes, 0)


if __name__ == "__main__":
    unittest.main()
