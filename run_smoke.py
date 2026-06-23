import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config
from replay_buffer import ReplayBuffer
from trainer import Trainer
from worker import RolloutWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    cfg = Config()
    cfg.episode_steps = args.steps
    cfg.batch_size = min(cfg.batch_size, 8)
    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    replay = ReplayBuffer(cfg.replay_size)
    worker = RolloutWorker(cfg, actor=trainer.actor, device=device)
    metrics = []
    for ep in range(args.episodes):
        metrics.append(worker.run_episode(args.seed + ep, replay=replay, greedy=False))
    assert len(replay) > 0, "replay buffer is empty"
    while len(replay) < cfg.batch_size:
        worker.run_episode(args.seed + 100 + len(replay), replay=replay, greedy=False)
    stats = trainer.update(replay)
    batch = replay.sample(cfg.batch_size, device)
    assert batch["node_inputs"].shape[-1] == 16
    assert batch["action_mask"].shape[-1] == cfg.max_node_candidates
    assert np.isfinite(stats.actor_loss)
    print(
        {
            "episodes": args.episodes,
            "replay_size": len(replay),
            "node_inputs": tuple(batch["node_inputs"].shape),
            "action_mask": tuple(batch["action_mask"].shape),
            "actor_loss": stats.actor_loss,
            "q1_loss": stats.q1_loss,
        }
    )


if __name__ == "__main__":
    main()

