import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config
from ppo_buffer import PPORolloutBuffer
from trainer import Trainer
from worker import RolloutWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    cfg = Config()
    cfg.episode_steps = args.steps
    if args.device is not None:
        cfg.device = args.device
    cfg.ppo_minibatch_size = min(cfg.ppo_minibatch_size, 8)
    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    rollout = PPORolloutBuffer()
    worker = RolloutWorker(cfg, actor=trainer.actor, device=device)
    metrics = []
    for ep in range(args.episodes):
        metrics.append(worker.run_episode(args.seed + ep, replay=rollout, greedy=False))
    assert len(rollout) > 0, "PPO rollout buffer is empty"
    stats = trainer.update(rollout)
    batch = rollout.tensors(cfg.gamma, cfg.gae_lambda, device)
    assert batch["node_inputs"].shape[-1] == 16
    assert batch["action_mask"].shape[-1] == cfg.max_node_candidates
    assert np.isfinite(stats.policy_loss)
    print(
        {
            "episodes": args.episodes,
            "rollout_size": len(rollout),
            "node_inputs": tuple(batch["node_inputs"].shape),
            "action_mask": tuple(batch["action_mask"].shape),
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
        }
    )


if __name__ == "__main__":
    main()
