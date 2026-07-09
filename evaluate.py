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
    node_builder = NodeBuilder(cfg)
    node_builder.reset(seed=seed)
    start_rng = np.random.default_rng(seed + 909)
    uav_positions = node_builder.graph.sample_start_positions(cfg.n_uavs, start_rng)
    node_builder.reset(seed=seed, start_positions=uav_positions)
    env = CMUOMMTEnv(cfg)
    env.reset(seed=seed, uav_positions=uav_positions)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
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
        previous_coverage_age = search.coverage_age.copy()
        info = env.step(selected_waypoints)
        target.update(info.measurements.points, env.uav_positions)
        tracks.update(env.step_count, info.measurements.points, [] if cfg.disable_phd_belief else target.peaks())
        search.update(env.uav_positions, info.measurements.points)
        terms = reward_terms(
            cfg,
            env.memory,
            len(info.detected_ids),
            info.newly_discovered,
            info.continuous_observed,
            previous_coverage_age,
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


def evaluate_policy(
    cfg: Config,
    episodes: int,
    seed: int,
    checkpoint: Optional[str] = None,
    baseline: Optional[str] = None,
    deterministic: bool = False,
) -> dict:
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
            metrics.append(worker.run_episode(seed + ep, greedy=deterministic, eval_mode=True))
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
    summary["policy_mode"] = "baseline" if baseline else ("deterministic" if deterministic else "stochastic")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline", type=str, choices=["random", "coverage", "search", "phd", "heuristic"], default=None)
    parser.add_argument("--out-dir", type=str, default="evaluation_runs/eval")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--graph-type", type=str, choices=["grid", "prm"], default=None)
    parser.add_argument("--prm-random-nodes", type=int, default=None)
    parser.add_argument("--prm-sampling", type=str, choices=["stratified", "uniform"], default=None)
    parser.add_argument("--prm-jitter-ratio", type=float, default=None)
    parser.add_argument("--prm-boundary-points-per-side", type=int, default=None)
    parser.add_argument("--prm-edge-radius", type=float, default=None)
    parser.add_argument("--prm-min-node-distance", type=float, default=None)
    parser.add_argument("--no-prm-boundary", action="store_true")
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--obstacle-count", type=int, default=None)
    parser.add_argument("--obstacle-radius-min", type=float, default=None)
    parser.add_argument("--obstacle-radius-max", type=float, default=None)
    parser.add_argument("--obstacle-margin", type=float, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions and beta > 0.5 terminations for checkpoint policy evaluation. Default samples from the trained PPO policy.",
    )
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
    if args.device is not None:
        cfg.device = args.device
    for key in [
        "graph_type",
        "prm_random_nodes",
        "prm_sampling",
        "prm_jitter_ratio",
        "prm_boundary_points_per_side",
        "prm_edge_radius",
        "prm_min_node_distance",
        "obstacle_count",
        "obstacle_radius_min",
        "obstacle_radius_max",
        "obstacle_margin",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    if args.no_prm_boundary:
        cfg.prm_include_boundary = False
    if args.obstacles:
        cfg.obstacles_enabled = True
    apply_ablation(cfg, args.ablation)
    summary = evaluate_policy(
        cfg,
        args.episodes,
        args.seed,
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        deterministic=args.deterministic,
    )
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
