import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config
from nodes import NODE_INPUT_DIM
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
    cfg.ppo_num_minibatches = 1
    cfg.ppo_minibatch_size = min(cfg.ppo_minibatch_size, 8)
    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    rollout = PPORolloutBuffer()
    worker = RolloutWorker(cfg, actor=trainer.actor, device=device)
    env, target, search, tracks, _ = worker.reset_stack(args.seed)
    graph_batch = worker.node_builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
    global_batch = worker.node_builder.global_batch_from_candidates(
        env.uav_positions,
        target,
        search,
        tracks,
        graph_batch.candidate_node_indices,
        graph_batch.node_padding_mask,
        graph_batch.action_mask,
        step=env.step_count,
    )
    g = global_batch.global_node_inputs.shape[1]
    present = ~graph_batch.node_padding_mask
    assert global_batch.global_node_inputs.shape == (cfg.n_uavs, g, NODE_INPUT_DIM)
    assert global_batch.global_edge_mask.shape == (g, g)
    assert global_batch.global_node_padding_mask.shape == (g,)
    assert np.all((global_batch.current_node_indices >= 0) & (global_batch.current_node_indices < g))
    assert np.all(graph_batch.candidate_node_indices[graph_batch.node_padding_mask] == -1)
    assert np.all((graph_batch.candidate_node_indices[present] >= 0) & (graph_batch.candidate_node_indices[present] < g))
    assert np.max(np.linalg.norm(global_batch.node_positions[graph_batch.candidate_node_indices[present]] - graph_batch.waypoints[present], axis=1)) <= 1e-5
    for uav_id, current_idx in enumerate(global_batch.current_node_indices):
        candidates = graph_batch.candidate_node_indices[uav_id, ~graph_batch.node_padding_mask[uav_id]]
        allowed = set(np.flatnonzero(~global_batch.global_edge_mask[int(current_idx)]).tolist())
        assert all(int(idx) in allowed for idx in candidates)
    worker.node_builder.reset()
    metrics = []
    for ep in range(args.episodes):
        metrics.append(worker.run_episode(args.seed + ep, replay=rollout, greedy=False))
    assert len(rollout) > 0, "PPO rollout buffer is empty"
    stats = trainer.update(rollout)
    batch = rollout.tensors(cfg.gamma, cfg.gae_lambda, device)
    assert batch["node_inputs"].shape[-1] == NODE_INPUT_DIM
    assert batch["global_node_inputs"].shape[-1] == NODE_INPUT_DIM
    assert batch["candidate_node_indices"].shape[-1] == cfg.max_node_candidates
    assert batch["action_mask"].shape[-1] == cfg.max_node_candidates
    assert np.isfinite(stats.policy_loss)
    print(
        {
            "episodes": args.episodes,
            "rollout_size": len(rollout),
            "node_inputs": tuple(batch["node_inputs"].shape),
            "global_node_inputs": tuple(batch["global_node_inputs"].shape),
            "candidate_node_indices": tuple(batch["candidate_node_indices"].shape),
            "action_mask": tuple(batch["action_mask"].shape),
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
        }
    )


if __name__ == "__main__":
    main()
