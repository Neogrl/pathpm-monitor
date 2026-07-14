import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from config import Config
from metrics import final_metrics, reward_terms, weighted_reward
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from trainer import Trainer
from utils import write_json
from worker import RolloutWorker

from tools.visualize_planning_signals import (
    candidate_stats,
    draw_discrete_graph_overview,
    draw_frame,
    draw_signal_counts,
    graph_stats,
    make_gif,
    phd_stats,
)


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


def safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float("nan") if len(arr) == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary = {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))
        if values:
            arr = np.asarray(values, dtype=np.float32)
            if np.all(np.isnan(arr)):
                summary[key] = float("nan")
                summary[f"{key}_std"] = float("nan")
            else:
                summary[key] = float(np.nanmean(arr))
                summary[f"{key}_std"] = float(np.nanstd(arr))
    return summary


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_mp4(frame_paths: list[Path], mp4_path: Path, fps: float) -> tuple[bool, str]:
    if not frame_paths:
        return False, "no frames"
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for path in frame_paths:
                writer.append_data(imageio.imread(path))
        return True, ""
    except Exception as imageio_exc:
        try:
            import cv2

            first = cv2.imread(str(frame_paths[0]))
            if first is None:
                raise RuntimeError(f"failed to read {frame_paths[0]}")
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            for path in frame_paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise RuntimeError(f"failed to read {path}")
                writer.write(frame)
            writer.release()
            return True, ""
        except Exception as cv2_exc:
            if mp4_path.exists():
                mp4_path.unlink()
            return False, f"imageio: {imageio_exc}; cv2: {cv2_exc}"


def select_policy_actions(
    worker: RolloutWorker,
    env,
    target: TargetBelief,
    search: SearchBelief,
    tracks: PseudoTrackMemory,
    prev_option: np.ndarray,
    greedy: bool = True,
):
    n = worker.cfg.n_uavs
    batch = worker.node_builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
    global_batch = worker.node_builder.global_batch_from_candidates(
        env.uav_positions,
        target,
        search,
        tracks,
        batch.candidate_node_indices,
        batch.node_padding_mask,
        batch.action_mask,
        step=env.step_count,
    )
    obs = worker._obs_dict_from_arrays(
        batch.node_inputs,
        batch.node_padding_mask,
        batch.action_mask,
        env,
        target,
        search,
        prev_option,
        global_batch=global_batch,
    )
    if worker.actor is None:
        actions = np.zeros(n, dtype=np.int64)
        options = prev_option.copy()
        terminations = np.zeros(n, dtype=bool)
        log_probs = np.zeros(n, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        betas = np.zeros(n, dtype=np.float32)
        for i in range(n):
            valid = np.flatnonzero(~batch.action_mask[i] & ~batch.node_padding_mask[i])
            actions[i] = int(valid[0])
    else:
        torch_obs = worker._to_torch(obs, batch_dim=True)
        with torch.no_grad():
            action_t, option_t, term_t, logp_t, value_t, beta_t = worker.actor.act_with_info(**torch_obs, greedy=greedy)
        actions = action_t[0].detach().cpu().numpy().astype(np.int64)
        options = option_t[0].detach().cpu().numpy().astype(np.int64)
        terminations = term_t[0].detach().cpu().numpy().astype(bool)
        log_probs = logp_t[0].detach().cpu().numpy().astype(np.float32)
        values = value_t[0].detach().cpu().numpy().astype(np.float32)
        betas = beta_t[0].detach().cpu().numpy().astype(np.float32)
    selected = batch.waypoints[np.arange(n), actions].copy()
    batch_for_viz = SimpleNamespace(
        node_inputs=batch.node_inputs,
        node_padding_mask=batch.node_padding_mask,
        action_mask=batch.action_mask,
        waypoints=batch.waypoints,
        candidate_node_indices=batch.candidate_node_indices,
    )
    return obs, batch_for_viz, actions, options, terminations, selected.astype(np.float32), log_probs, values, betas


def run_policy_episode(
    cfg: Config,
    trainer: Trainer,
    seed: int,
    n_targets: Optional[int] = None,
    greedy: bool = False,
) -> dict:
    worker = RolloutWorker(cfg, actor=trainer.actor, device=trainer.device)
    return worker.run_episode(seed, greedy=greedy, eval_mode=True, n_targets=n_targets)


def run_visualized_episode(
    cfg: Config,
    trainer: Trainer,
    seed: int,
    out_dir: Path,
    n_targets: Optional[int],
    frame_stride: int,
    duration_ms: int,
    mp4_fps: float,
    greedy: bool = False,
) -> dict:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    worker = RolloutWorker(cfg, actor=trainer.actor, device=trainer.device)
    env, target, search, tracks, prev_option = worker.reset_stack(seed, n_targets=n_targets, eval_mode=True)
    rewards: list[float] = []
    overlaps: list[float] = []
    rows: list[dict] = []
    frame_paths: list[Path] = []
    measurements = np.zeros((0, 2), dtype=np.float32)
    detected_count = 0

    for step in range(cfg.episode_steps):
        target.predict()
        obs, batch, actions, options, terminations, selected_waypoints, log_probs, values, betas = select_policy_actions(
            worker, env, target, search, tracks, prev_option, greedy=greedy
        )
        if step % max(frame_stride, 1) == 0:
            frame_path = frames_dir / f"frame_{step:04d}.png"
            draw_frame(frame_path, cfg, env, target, search, worker.node_builder, batch, actions, measurements, detected_count, step)
            frame_paths.append(frame_path)

        previous_coverage_age = search.coverage_age.copy()
        info = env.step(selected_waypoints)
        target.update(info.measurements.points, env.uav_positions)
        peaks = [] if cfg.disable_phd_belief else target.peaks()
        tracks.update(env.step_count, info.measurements.points, peaks)
        search.update(env.uav_positions, info.measurements.points)
        terms = reward_terms(
            cfg,
            env.memory,
            len(info.detected_ids),
            info.newly_discovered,
            info.continuous_observed,
            peaks,
            env.target_states[:, 0:2],
            previous_coverage_age,
            env.uav_positions,
            info.step_distance,
            terminations.astype(np.float32),
        )
        reward = weighted_reward(terms, cfg)
        rewards.append(reward)
        overlaps.append(terms["overlap"])

        row = {
            "step": step,
            "reward": float(reward),
            "detected_count": float(len(info.detected_ids)),
            "newly_discovered": float(info.newly_discovered),
            "continuous_observed": float(info.continuous_observed),
            "mean_beta": float(np.mean(betas)),
            "switch_rate": float(np.mean(terminations.astype(np.float32))),
            "option_0_ratio": float(np.mean(options == 0)),
            "option_1_ratio": float(np.mean(options == 1)),
            "mean_value": float(np.mean(values)),
            "mean_log_prob": float(np.mean(log_probs)),
            "search_belief_mean": float(np.mean(search.search_belief)),
            "coverage_age_mean": float(np.mean(search.coverage_age)),
            "target_estimated_count": float(np.sum(target.weights)),
            "track_count": float(len(tracks.tracks)),
            "reward_observe": terms["observe"],
            "reward_discover": terms["discover"],
            "reward_continuity": terms["continuity"],
            "reward_search": terms["search"],
            "reward_coverage": terms["coverage"],
            "reward_miss": terms["miss"],
            "reward_fairness_metric": terms["fairness"],
            "reward_overlap_metric": terms["overlap"],
            "reward_cost_metric": terms["cost"],
            "reward_switch_metric": terms["switch"],
            "phd_position_error": terms["phd_position_error"],
            "phd_number_error": terms["phd_number_error"],
            "phd_total_weight": float(np.sum(target.weights)),
            "phd_peak_count": float(len(peaks)),
            **worker._node_diagnostics(obs, actions),
            **phd_stats(target),
            **graph_stats(worker.node_builder),
            **candidate_stats(batch, actions),
        }
        rows.append(row)
        measurements = info.measurements.points
        detected_count = len(info.detected_ids)
        prev_option = options.copy()
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
    metrics["seed"] = float(seed)
    metrics["n_targets"] = float(len(env.target_states))
    for key in [
        "valid_candidates_mean",
        "candidate_distance_norm_mean",
        "candidate_coverage_age_value_mean",
        "candidate_overlap_mean",
        "candidate_target_belief_value_mean",
        "selected_candidate_distance_norm",
        "selected_coverage_age_value",
        "selected_overlap",
        "selected_target_belief_value",
        "switch_rate",
        "mean_beta",
        "option_0_ratio",
        "option_1_ratio",
        "search_belief_mean",
        "coverage_age_mean",
        "target_estimated_count",
        "track_count",
        "reward_observe",
        "reward_discover",
        "reward_continuity",
        "reward_search",
        "reward_coverage",
        "reward_miss",
        "reward_fairness_metric",
        "reward_overlap_metric",
        "reward_cost_metric",
        "reward_switch_metric",
        "phd_position_error",
        "phd_number_error",
        "phd_total_weight",
        "phd_peak_count",
    ]:
        metrics[key] = safe_mean([float(row[key]) for row in rows if key in row])

    write_table(out_dir / "viz_trace.csv", rows)
    draw_signal_counts(out_dir / "signal_counts.png", rows)
    draw_discrete_graph_overview(out_dir / "discrete_graph_overview.png", cfg, env, worker.node_builder)
    gif_ok = make_gif(frame_paths, out_dir / "policy_rollout.gif", duration_ms)
    mp4_ok, mp4_error = make_mp4(frame_paths, out_dir / "policy_rollout.mp4", mp4_fps)
    metrics["gif_created"] = bool(gif_ok)
    metrics["mp4_created"] = bool(mp4_ok)
    metrics["mp4_error"] = mp4_error
    write_json(
        out_dir / "viz_summary.json",
        {
            **metrics,
            "frame_count": len(frame_paths),
            "frame_stride": frame_stride,
            "frames_dir": str(frames_dir),
            "gif": str(out_dir / "policy_rollout.gif"),
            "mp4": str(out_dir / "policy_rollout.mp4") if mp4_ok else None,
        },
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and visualize the trained CMUOMMT planToGo policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="evaluation_runs/test_viz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--viz-seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
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
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=180)
    parser.add_argument("--mp4-fps", type=float, default=6.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions and beta > 0.5 terminations. By default, sample from the trained PPO policy.",
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    trainer.load(Path(args.checkpoint))
    trainer.actor.eval()

    rows = []
    greedy = bool(args.deterministic)
    policy_mode = "deterministic" if greedy else "stochastic"
    for ep in range(max(args.episodes, 1)):
        ep_seed = args.seed + ep
        print(f"[test_viz] evaluate episode {ep + 1}/{args.episodes} seed={ep_seed} mode={policy_mode}", flush=True)
        row = run_policy_episode(cfg, trainer, ep_seed, n_targets=args.n_targets, greedy=greedy)
        row["eval_episode"] = ep
        rows.append(row)

    summary = summarize_rows(rows)
    summary.update(
        {
            "episodes": int(max(args.episodes, 1)),
            "seed_start": int(args.seed),
            "seed_end": int(args.seed + max(args.episodes, 1) - 1),
            "checkpoint": str(Path(args.checkpoint)),
            "device": str(device),
            "steps": int(cfg.episode_steps),
            "n_targets": int(args.n_targets or cfg.n_targets_true),
            "greedy": greedy,
            "policy_mode": policy_mode,
        }
    )
    write_table(out / "episode_metrics.csv", rows)
    write_json(out / "metrics_summary.json", summary)

    viz_metrics = {}
    if not args.no_video:
        viz_seed = args.viz_seed if args.viz_seed is not None else args.seed
        print(f"[test_viz] visualize seed={viz_seed} mode={policy_mode}", flush=True)
        viz_metrics = run_visualized_episode(
            cfg,
            trainer,
            viz_seed,
            out,
            n_targets=args.n_targets,
            frame_stride=args.frame_stride,
            duration_ms=args.duration_ms,
            mp4_fps=args.mp4_fps,
            greedy=greedy,
        )

    result = {
        "out_dir": str(out.resolve()),
        "metrics_summary": str(out / "metrics_summary.json"),
        "episode_metrics": str(out / "episode_metrics.csv"),
        "viz_summary": str(out / "viz_summary.json") if not args.no_video else None,
        "gif": str(out / "policy_rollout.gif") if not args.no_video else None,
        "mp4": str(out / "policy_rollout.mp4") if viz_metrics.get("mp4_created") else None,
        "episode_reward": summary.get("episode_reward"),
        "discovery_rate": summary.get("discovery_rate"),
        "observation_rate": summary.get("observation_rate"),
        "OSPA": summary.get("OSPA"),
        "policy_mode": policy_mode,
        "viz_episode_reward": viz_metrics.get("episode_reward") if viz_metrics else None,
    }
    write_json(out / "run_summary.json", result)
    print(result)


if __name__ == "__main__":
    main()
