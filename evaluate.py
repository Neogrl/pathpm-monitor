import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from baselines import CoverageBaseline, HeuristicBaseline, PHDGreedyBaseline, RandomBaseline, SearchGreedyBaseline
from config import Config
from model import OptionActor
from nodes import NodeBuilder
from target_belief import TargetBelief
from search_belief import SearchBelief
from pseudo_tracks import PseudoTrackMemory
from environment import CMUOMMTEnv
from metrics import final_metrics, reward_terms, weighted_reward
from trainer import Trainer
from utils import write_json
from worker import RolloutWorker


def apply_ablation(cfg: Config, ablation: Optional[str]) -> None:
    if ablation is None:
        return
    if ablation == "no_search":
        cfg.disable_search_belief = True
        cfg.reward_search_weight = 0.0
    elif ablation == "no_phd":
        cfg.disable_phd_belief = True
    elif ablation == "no_option":
        cfg.disable_options = True
        cfg.disable_termination = True
    elif ablation == "no_termination":
        cfg.disable_termination = True
    elif ablation == "no_discover_reward":
        cfg.reward_discover_weight = 0.0
    elif ablation == "no_miss_penalty":
        cfg.reward_miss_weight = 0.0


def make_baseline(name: str):
    if name == "random":
        return RandomBaseline()
    if name == "heuristic":
        return HeuristicBaseline()
    if name == "coverage":
        return CoverageBaseline()
    if name == "search":
        return SearchGreedyBaseline()
    if name == "phd":
        return PHDGreedyBaseline()
    raise ValueError(f"Unknown baseline: {name}")


def evaluate_baseline_episode(cfg: Config, seed: int, baseline_name: str) -> dict:
    env = CMUOMMTEnv(cfg)
    env.reset(seed=seed)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
    node_builder = NodeBuilder(cfg)
    node_builder.reset()
    baseline = make_baseline(baseline_name)
    rng = np.random.default_rng(seed + 303)
    rewards = []
    overlaps = []
    prev_option = np.zeros(cfg.n_uavs, dtype=np.int64)
    for _ in range(cfg.episode_steps):
        target.predict()
        batch = node_builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        actions = baseline.select(cfg, batch, rng)
        selected_waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions]
        prev_search = float(np.mean(search.search_belief))
        info = env.step(selected_waypoints)
        target.update(info.measurements.points, env.uav_positions)
        tracks.update(env.step_count, info.measurements.points, [] if cfg.disable_phd_belief else target.peaks())
        search.update(env.uav_positions, info.measurements.points)
        cur_search = float(np.mean(search.search_belief))
        terms = reward_terms(
            cfg,
            env.memory,
            len(info.detected_ids),
            info.newly_discovered,
            info.continuous_observed,
            prev_search,
            cur_search,
            env.uav_positions,
            info.step_distance,
            np.zeros(cfg.n_uavs, dtype=np.float32),
        )
        rewards.append(weighted_reward(terms, cfg))
        overlaps.append(terms["overlap"])
        prev_option[:] = 0
        if env.done():
            break
    metrics = final_metrics(
        cfg,
        env.memory,
        rewards,
        overlaps,
        float(np.sum(target.weights)),
        env.target_states[:, 0:2],
        target.peaks(),
    )
    metrics["episode_reward"] = float(np.sum(rewards))
    metrics["steps"] = env.step_count
    return metrics


def evaluate_policy(cfg: Config, episodes: int, seed: int, checkpoint: Optional[str] = None, baseline: Optional[str] = None) -> dict:
    device = torch.device(cfg.device)
    actor = None
    if checkpoint:
        trainer = Trainer(cfg, device)
        trainer.load(Path(checkpoint))
        actor = trainer.actor
        actor.eval()
    worker = RolloutWorker(cfg, actor=actor, device=device)
    metrics = []
    for ep in range(episodes):
        if baseline:
            metrics.append(evaluate_baseline_episode(cfg, seed + ep, baseline))
        else:
            metrics.append(worker.run_episode(seed + ep, greedy=True, eval_mode=True))
    keys = metrics[0].keys()
    summary = {}
    for key in keys:
        values = np.asarray([m[key] for m in metrics], dtype=np.float32)
        if np.all(np.isnan(values)):
            summary[key] = float(np.nan)
            summary[f"{key}_std"] = float(np.nan)
        else:
            summary[key] = float(np.nanmean(values))
            summary[f"{key}_std"] = float(np.nanstd(values))
    summary["episodes"] = episodes
    summary["seed"] = seed
    summary["seed_start"] = seed
    summary["seed_end"] = seed + episodes - 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline", type=str, choices=["random", "coverage", "search", "phd", "heuristic"], default=None)
    parser.add_argument("--out-dir", type=str, default="evaluation_runs/eval")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["no_search", "no_phd", "no_option", "no_termination", "no_discover_reward", "no_miss_penalty"],
        default=None,
    )
    args = parser.parse_args()
    cfg = Config()
    if args.steps is not None:
        cfg.episode_steps = args.steps
    apply_ablation(cfg, args.ablation)
    summary = evaluate_policy(cfg, args.episodes, args.seed, checkpoint=args.checkpoint, baseline=args.baseline)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
