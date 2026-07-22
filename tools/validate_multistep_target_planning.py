from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from environment import CMUOMMTEnv
from metrics import reward_terms, weighted_reward
from nodes import NODE_INPUT_INDEX
from ppo_buffer import PPORolloutBuffer
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from trainer import Trainer
from worker import RolloutWorker


UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#17becf"]


class EmptyTargetBelief:
    """Cheap placeholder; perfect target values are injected after graph construction."""

    def __init__(self) -> None:
        self.particles = np.zeros((0, 4), dtype=np.float32)
        self.weights = np.zeros(0, dtype=np.float64)

    def peaks(self, max_peaks: int | None = None) -> list:
        del max_peaks
        return []

    def summary(self) -> np.ndarray:
        return np.zeros(5, dtype=np.float32)


@dataclass
class EpisodeState:
    worker: RolloutWorker
    env: CMUOMMTEnv
    target_belief: EmptyTargetBelief
    search_belief: SearchBelief
    tracks: PseudoTrackMemory
    prev_option: np.ndarray
    transitions: list[dict[str, Any]] = field(default_factory=list)
    pending_transition: dict[str, Any] | None = None
    rewards: list[float] = field(default_factory=list)
    visibility: list[float] = field(default_factory=list)
    maintenance_age: list[float] = field(default_factory=list)
    duplicate_coverage: list[float] = field(default_factory=list)
    coverage_progress: list[float] = field(default_factory=list)
    node_history: list[np.ndarray] = field(default_factory=list)
    uav_path: list[np.ndarray] = field(default_factory=list)
    initial_team_target_distance: float = 0.0
    elapsed_time: float = 0.0


@dataclass
class EpisodeResult:
    metrics: dict[str, float]
    uav_path: np.ndarray
    target_positions: np.ndarray


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def perfect_target_values(cfg: Config, node_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        node_positions[:, None, :] - target_positions[None, :, :],
        axis=2,
    )
    mass = np.sum(distances <= cfg.fov_radius, axis=1, dtype=np.float32)
    return np.clip(mass / max(cfg.phd_prior_count, 1.0), 0.0, 1.0).astype(np.float32)


def inject_perfect_target(obs: dict[str, np.ndarray], values: np.ndarray) -> None:
    target_index = NODE_INPUT_INDEX["target_belief_value"]
    obs["global_node_inputs"][:, :, target_index] = values[None, :]
    candidate_indices = obs["candidate_node_indices"]
    valid = candidate_indices >= 0
    local_values = np.zeros(candidate_indices.shape, dtype=np.float32)
    local_values[valid] = values[candidate_indices[valid]]
    obs["node_inputs"][:, :, target_index] = local_values


def team_target_graph_distance(state: EpisodeState) -> float:
    graph = state.worker.node_builder.graph
    uav_nodes = np.asarray(
        [graph.nearest_node_index(position) for position in state.env.uav_positions],
        dtype=np.int64,
    )
    target_nodes = np.asarray(
        [graph.nearest_node_index(position) for position in state.env.target_states[:, 0:2]],
        dtype=np.int64,
    )
    nearest_distances = []
    for target_node in target_nodes:
        distances, _ = graph.shortest_tree_from(int(target_node))
        nearest_distances.append(float(np.min(distances[uav_nodes])))
    return float(np.mean(nearest_distances)) if nearest_distances else 0.0


def create_episode(cfg: Config, seed: int, n_targets: int) -> EpisodeState:
    worker = RolloutWorker(cfg, actor=None, device="cpu")
    worker.node_builder.reset(seed=seed)
    start_rng = np.random.default_rng(seed + 909)
    starts = worker.node_builder.graph.sample_start_positions(cfg.n_uavs, start_rng)
    worker.node_builder.reset(seed=seed, start_positions=starts)
    env = CMUOMMTEnv(cfg)
    env.reset(seed=seed, n_targets=n_targets, uav_positions=starts)
    env.target_states[:, 2:4] = 0.0
    state = EpisodeState(
        worker=worker,
        env=env,
        target_belief=EmptyTargetBelief(),
        search_belief=SearchBelief(cfg),
        tracks=PseudoTrackMemory(cfg),
        prev_option=np.zeros(cfg.n_uavs, dtype=np.int64),
    )
    state.initial_team_target_distance = team_target_graph_distance(state)
    state.uav_path.append(env.uav_positions.copy())
    state.node_history.append(
        np.asarray(
            [worker.node_builder.graph.nearest_node_index(position) for position in env.uav_positions],
            dtype=np.int64,
        )
    )
    return state


def build_observation(state: EpisodeState) -> tuple[dict[str, np.ndarray], Any]:
    env = state.env
    builder = state.worker.node_builder
    batch = builder.build(
        env.uav_positions,
        state.target_belief,
        state.search_belief,
        state.tracks,
        step=env.step_count,
    )
    global_batch = builder.global_batch_from_candidates(
        env.uav_positions,
        state.target_belief,
        state.search_belief,
        state.tracks,
        batch.candidate_node_indices,
        batch.node_padding_mask,
        batch.action_mask,
        step=env.step_count,
    )
    obs = state.worker._obs_dict_from_arrays(
        batch.node_inputs,
        batch.node_padding_mask,
        batch.action_mask,
        env,
        state.target_belief,
        state.search_belief,
        state.prev_option,
        global_batch=global_batch,
    )
    values = perfect_target_values(
        state.worker.cfg,
        global_batch.node_positions,
        env.target_states[:, 0:2],
    )
    inject_perfect_target(obs, values)
    return obs, batch


def stack_actor_observations(
    observations: list[dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    keys = [
        "global_node_inputs",
        "spatio_pos_encoding",
        "global_edge_mask",
        "global_node_padding_mask",
        "current_node_indices",
        "candidate_node_indices",
        "candidate_padding_mask",
        "action_mask",
        "uav_state",
        "prev_option",
    ]
    boolean_keys = {
        "global_edge_mask",
        "global_node_padding_mask",
        "candidate_padding_mask",
        "action_mask",
    }
    integer_keys = {"current_node_indices", "candidate_node_indices", "prev_option"}
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        array = np.stack([obs[key] for obs in observations], axis=0)
        tensor = torch.as_tensor(array, device=device)
        if key in boolean_keys:
            tensor = tensor.bool()
        elif key in integer_keys:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        result[key] = tensor
    return result


def random_actions(observations: list[dict[str, np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros((len(observations), observations[0]["action_mask"].shape[0]), dtype=np.int64)
    for episode, obs in enumerate(observations):
        valid = ~obs["action_mask"] & ~obs["node_padding_mask"]
        for uav_id in range(valid.shape[0]):
            slots = np.flatnonzero(valid[uav_id])
            actions[episode, uav_id] = int(rng.choice(slots)) if len(slots) else 0
    return actions


def transition_from_step(
    obs: dict[str, np.ndarray],
    actions: np.ndarray,
    terminations: np.ndarray,
    log_probs: np.ndarray,
    values: np.ndarray,
    reward: float,
    done: bool,
) -> dict[str, Any]:
    return {
        "node_inputs": obs["node_inputs"].astype(np.float32),
        "node_padding_mask": obs["node_padding_mask"].astype(bool),
        "action_mask": obs["action_mask"].astype(bool),
        "global_node_inputs": obs["global_node_inputs"].astype(np.float32),
        "spatio_pos_encoding": obs["spatio_pos_encoding"].astype(np.float32),
        "global_edge_mask": obs["global_edge_mask"].astype(bool),
        "global_node_padding_mask": obs["global_node_padding_mask"].astype(bool),
        "current_node_indices": obs["current_node_indices"].astype(np.int64),
        "candidate_node_indices": obs["candidate_node_indices"].astype(np.int64),
        "candidate_padding_mask": obs["candidate_padding_mask"].astype(bool),
        "uav_state": obs["uav_state"].astype(np.float32),
        "prev_option": obs["prev_option"].astype(np.int64),
        "actions": actions.astype(np.int64),
        "terminations": terminations.astype(np.float32),
        "log_probs": log_probs.astype(np.float32),
        "values": values.astype(np.float32),
        "reward": np.asarray(reward, dtype=np.float32),
        "done": np.asarray(float(done), dtype=np.float32),
    }


def finish_episode(state: EpisodeState) -> EpisodeResult:
    if state.pending_transition is not None:
        state.pending_transition["next_values"] = np.zeros_like(
            state.pending_transition["values"],
            dtype=np.float32,
        )
        state.transitions.append(state.pending_transition)
        state.pending_transition = None
    history = np.asarray(state.node_history, dtype=np.int64)
    revisit_rates = []
    for uav_id in range(history.shape[1]):
        revisit_rates.append(1.0 - len(np.unique(history[:, uav_id])) / max(len(history), 1))
    two_cycle = float(np.mean(history[2:] == history[:-2])) if len(history) >= 3 else 0.0
    discovered = state.env.memory.is_discovered
    first_detection = state.env.memory.first_detection_step[discovered]
    final_distance = team_target_graph_distance(state)
    initial_distance = state.initial_team_target_distance
    distance_reduction = initial_distance - final_distance
    distance_reduction_ratio = (
        distance_reduction / initial_distance if initial_distance > 1e-8 else float("nan")
    )
    target_max_unobserved = state.env.memory.max_unobserved_time.copy()
    target_max_unobserved[~discovered] = state.elapsed_time
    metrics = {
        "episode_reward": float(np.sum(state.rewards)),
        "mean_reward": float(np.mean(state.rewards)),
        "mean_visibility": float(np.mean(state.visibility)),
        "mean_maintenance_age": float(np.mean(state.maintenance_age)),
        "mean_duplicate_coverage": float(np.mean(state.duplicate_coverage)),
        "mean_coverage_progress": float(np.mean(state.coverage_progress)),
        "discovery_rate": float(np.mean(discovered)),
        "mean_first_detection_step": float(np.mean(first_detection)) if len(first_detection) else float(state.worker.cfg.episode_steps),
        "initial_team_target_graph_distance": initial_distance,
        "final_team_target_graph_distance": final_distance,
        "team_target_distance_reduction": distance_reduction,
        "team_target_distance_reduction_ratio": distance_reduction_ratio,
        "mean_revisit_rate": float(np.mean(revisit_rates)),
        "two_cycle_rate": two_cycle,
        "max_unobserved_time": float(np.max(target_max_unobserved)) if len(discovered) else 0.0,
    }
    return EpisodeResult(
        metrics=metrics,
        uav_path=np.asarray(state.uav_path, dtype=np.float32),
        target_positions=state.env.target_states[:, 0:2].copy(),
    )


def run_episode_batch(
    cfg: Config,
    actor,
    device: torch.device,
    seeds: list[int],
    target_counts: list[int],
    mode: str,
    collect_rollout: bool,
) -> tuple[list[EpisodeResult], PPORolloutBuffer | None]:
    states = [create_episode(cfg, seed, target_count) for seed, target_count in zip(seeds, target_counts)]
    rng = np.random.default_rng(seeds[0] + 5003)
    for step in range(cfg.episode_steps):
        observations_and_batches = [build_observation(state) for state in states]
        observations = [item[0] for item in observations_and_batches]
        batches = [item[1] for item in observations_and_batches]
        torch_obs = stack_actor_observations(observations, device)
        if mode == "random":
            actions_np = random_actions(observations, rng)
            terminations_np = np.zeros_like(actions_np, dtype=np.float32)
            log_probs_np = np.zeros_like(actions_np, dtype=np.float32)
            values_np = np.zeros_like(actions_np, dtype=np.float32)
            options_np = np.zeros_like(actions_np, dtype=np.int64)
        else:
            with torch.no_grad():
                actions, options, terminations, log_probs, values, _ = actor.act_with_info(
                    **torch_obs,
                    greedy=mode == "greedy",
                )
            actions_np = actions.cpu().numpy().astype(np.int64)
            options_np = options.cpu().numpy().astype(np.int64)
            terminations_np = terminations.cpu().numpy().astype(np.float32)
            log_probs_np = log_probs.cpu().numpy().astype(np.float32)
            values_np = values.cpu().numpy().astype(np.float32)

        for index, state in enumerate(states):
            if collect_rollout and state.pending_transition is not None:
                state.pending_transition["next_values"] = values_np[index].copy()
                state.transitions.append(state.pending_transition)
                state.pending_transition = None
            batch = batches[index]
            selected_waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions_np[index]]
            previous_coverage_age = state.search_belief.coverage_age.copy()
            info = state.env.step(selected_waypoints)
            state.elapsed_time += float(info.step_duration)
            state.search_belief.update(state.env.uav_positions, info.measurements.points)
            state.tracks.update(state.env.step_count, info.measurements.points, [])
            terms = reward_terms(
                cfg,
                state.env.memory,
                len(info.detected_ids),
                info.newly_discovered,
                info.continuous_observed,
                [],
                float(len(state.env.target_states)),
                state.env.target_states[:, 0:2],
                info.target_coverage_counts,
                previous_coverage_age,
                state.env.uav_positions,
                info.step_distance,
                terminations_np[index],
            )
            reward = weighted_reward(terms, cfg)
            state.rewards.append(reward)
            state.visibility.append(terms["visibility"])
            state.maintenance_age.append(terms["maintenance_age"])
            state.duplicate_coverage.append(terms["duplicate_coverage"])
            state.coverage_progress.append(terms["coverage"])
            state.prev_option = options_np[index].copy()
            state.uav_path.append(state.env.uav_positions.copy())
            state.node_history.append(
                np.asarray(
                    [state.worker.node_builder.graph.nearest_node_index(pos) for pos in state.env.uav_positions],
                    dtype=np.int64,
                )
            )
            if collect_rollout:
                state.pending_transition = transition_from_step(
                    observations[index],
                    actions_np[index],
                    terminations_np[index],
                    log_probs_np[index],
                    values_np[index],
                    reward,
                    done=step == cfg.episode_steps - 1,
                )

    results = [finish_episode(state) for state in states]
    if not collect_rollout:
        return results, None
    rollout = PPORolloutBuffer()
    for state in states:
        rollout.extend(state.transitions)
    return results, rollout


def mean_metrics(results: list[EpisodeResult]) -> dict[str, float]:
    keys = results[0].metrics.keys()
    return {
        key: float(np.nanmean([result.metrics[key] for result in results]))
        for key in keys
    }


def target_counts_for_phase(phase: str, seeds: list[int]) -> list[int]:
    if phase == "single":
        return [1] * len(seeds)
    return [int(np.random.default_rng(seed + 17).integers(6, 9)) for seed in seeds]


def draw_training_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    updates = [row["update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=145)
    axes[0, 0].plot(updates, [row["train_mean_reward"] for row in rows], label="train")
    axes[0, 0].plot(updates, [row["eval_mean_reward"] for row in rows], label="eval")
    axes[0, 0].set_title("Mean step reward")
    axes[0, 0].legend()
    axes[0, 1].plot(updates, [row["eval_mean_visibility"] for row in rows])
    axes[0, 1].set_title("Evaluation visibility")
    axes[1, 0].plot(updates, [row["eval_mean_maintenance_age"] for row in rows], label="maintenance age")
    axes[1, 0].plot(updates, [row["eval_mean_duplicate_coverage"] for row in rows], label="duplicate")
    axes[1, 0].plot(updates, [row["eval_mean_coverage_progress"] for row in rows], label="coverage progress")
    axes[1, 0].set_title("Evaluation reward terms")
    axes[1, 0].legend()
    axes[1, 1].plot(updates, [row["policy_loss"] for row in rows], label="policy loss")
    axes[1, 1].plot(updates, [row["value_loss"] for row in rows], label="value loss")
    axes[1, 1].set_title("PPO losses")
    axes[1, 1].legend()
    for ax in axes.reshape(-1):
        ax.set_xlabel("MAPPO update")
        ax.grid(True, color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_trajectory(path: Path, cfg: Config, result: EpisodeResult, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=150)
    for target_id, target in enumerate(result.target_positions):
        ax.scatter(target[0], target[1], marker="X", color="#ff2b7a", edgecolor="black", s=120)
        ax.text(target[0] + 1.0, target[1] + 1.0, f"T{target_id}", fontsize=8)
    for uav_id in range(cfg.n_uavs):
        path_values = result.uav_path[:, uav_id]
        ax.plot(path_values[:, 0], path_values[:, 1], color=UAV_COLORS[uav_id], linewidth=1.7, label=f"U{uav_id}")
        ax.scatter(path_values[0, 0], path_values[0, 1], marker="o", color=UAV_COLORS[uav_id], s=45)
        ax.scatter(path_values[-1, 0], path_values[-1, 1], marker="*", color=UAV_COLORS[uav_id], edgecolor="black", s=100)
    ax.set_title(title)
    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gpu_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "cuda_device_name": properties.name,
                "cuda_total_memory_gib": properties.total_memory / (1024**3),
            }
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled 128-step MAPPO target-maintenance validation.")
    parser.add_argument("--phase", choices=["single", "multi"], default="single")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/multistep_single_128")
    parser.add_argument("--prm-random-nodes", type=int, default=240)
    parser.add_argument("--prm-boundary-points-per-side", type=int, default=11)
    parser.add_argument("--ppo-update-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-search-weight", type=float, default=0.0)
    parser.add_argument("--eval-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config()
    cfg.episode_steps = args.steps
    cfg.target_speed = 0.0
    cfg.target_velocity_noise_std = 0.0
    cfg.prm_random_nodes = args.prm_random_nodes
    cfg.prm_boundary_points_per_side = args.prm_boundary_points_per_side
    cfg.ppo_update_epochs = args.ppo_update_epochs
    cfg.ppo_minibatch_size = args.ppo_minibatch_size
    cfg.ppo_num_minibatches = 0
    cfg.actor_lr = args.learning_rate
    cfg.critic_lr = args.learning_rate
    cfg.reward_search_weight = args.reward_search_weight
    cfg.disable_options = True
    cfg.disable_termination = True
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg, device=device)
    if args.checkpoint:
        trainer.load(Path(args.checkpoint))
    eval_seeds = [args.seed + 9000 + index for index in range(args.eval_episodes)]
    random_seeds = eval_seeds.copy()
    random_counts = target_counts_for_phase(args.phase, random_seeds)
    eval_counts = target_counts_for_phase(args.phase, eval_seeds)
    random_results, _ = run_episode_batch(
        cfg, trainer.actor, device, random_seeds, random_counts, mode="random", collect_rollout=False
    )
    initial_results, _ = run_episode_batch(
        cfg, trainer.actor, device, eval_seeds, eval_counts, mode=args.eval_mode, collect_rollout=False
    )
    random_summary = mean_metrics(random_results)
    initial_summary = mean_metrics(initial_results)
    write_json(
        out / "runtime.json",
        {
            "config": asdict(cfg),
            "gpu": gpu_metadata(device),
            "checkpoint": args.checkpoint,
            "eval_only": args.eval_only,
            "eval_mode": args.eval_mode,
        },
    )
    initial_label = "Checkpoint policy" if args.checkpoint else "Initial policy"
    draw_trajectory(
        out / "initial_trajectory.png",
        cfg,
        initial_results[0],
        f"{initial_label}, {args.steps}-step trajectory",
    )

    if args.eval_only:
        summary = {
            "phase": args.phase,
            "mode": "checkpoint_evaluation",
            "checkpoint": args.checkpoint,
            "eval_mode": args.eval_mode,
            "perfect_target_input": "hard FOV mass / phd_prior_count, clipped to [0, 1]",
            "steps_per_episode": args.steps,
            "eval_episodes": args.eval_episodes,
            "random": random_summary,
            "checkpoint_policy": initial_summary,
        }
        write_json(out / "summary.json", summary)
        write_csv(
            out / "checkpoint_evaluation_metrics.csv",
            [result.metrics for result in initial_results],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    curve_rows: list[dict[str, Any]] = []
    best_reward = -float("inf")
    final_results = initial_results
    final_summary = initial_summary
    for update in range(1, args.updates + 1):
        train_seeds = [args.seed + update * 1000 + index for index in range(args.episodes_per_update)]
        train_counts = target_counts_for_phase(args.phase, train_seeds)
        train_results, rollout = run_episode_batch(
            cfg,
            trainer.actor,
            device,
            train_seeds,
            train_counts,
            mode="sample",
            collect_rollout=True,
        )
        train_summary = mean_metrics(train_results)
        stats = trainer.update(rollout)
        if update % args.eval_interval == 0 or update == args.updates:
            final_results, _ = run_episode_batch(
                cfg,
                trainer.actor,
                device,
                eval_seeds,
                eval_counts,
                mode=args.eval_mode,
                collect_rollout=False,
            )
            final_summary = mean_metrics(final_results)
        row = {
            "update": update,
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"eval_{key}": value for key, value in final_summary.items()},
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "clipfrac": stats.clipfrac,
            "grad_norm": stats.grad_norm,
        }
        curve_rows.append(row)
        trainer.save(out / "latest.pt")
        if final_summary["mean_reward"] > best_reward:
            best_reward = final_summary["mean_reward"]
            trainer.save(out / "best.pt")
        write_csv(out / "training_curve.csv", curve_rows)
        print(
            f"[multistep] phase={args.phase} update={update}/{args.updates} "
            f"train_reward={train_summary['mean_reward']:.4f} "
            f"eval_reward={final_summary['mean_reward']:.4f} "
            f"visibility={final_summary['mean_visibility']:.4f} "
            f"maintenance_age={final_summary['mean_maintenance_age']:.4f} "
            f"duplicate={final_summary['mean_duplicate_coverage']:.4f} "
            f"coverage={final_summary['mean_coverage_progress']:.4f} "
            f"value_loss={stats.value_loss:.4f} entropy={stats.entropy:.4f}",
            flush=True,
        )

    probability_baseline = max(random_summary["mean_visibility"], initial_summary["mean_visibility"])
    summary = {
        "phase": args.phase,
        "reward": (
            "visibility - 0.5 * maintenance_age - 0.1 * duplicate_coverage "
            f"+ {cfg.reward_search_weight:g} * coverage_progress"
        ),
        "eval_mode": args.eval_mode,
        "perfect_target_input": "hard FOV mass / phd_prior_count, clipped to [0, 1]",
        "steps_per_episode": args.steps,
        "updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "joint_transitions": args.updates * args.episodes_per_update * args.steps,
        "agent_actions": args.updates * args.episodes_per_update * args.steps * cfg.n_uavs,
        "random": random_summary,
        "initial": initial_summary,
        "final": final_summary,
        "passed": bool(
            np.isfinite(list(final_summary.values())).all()
            and final_summary["mean_visibility"] >= probability_baseline + 0.05
            and final_summary["mean_reward"] >= initial_summary["mean_reward"] + 0.02
        ),
    }
    write_json(out / "summary.json", summary)
    write_csv(
        out / "final_evaluation_metrics.csv",
        [result.metrics for result in final_results],
    )
    draw_training_curve(out / "training_curve.png", curve_rows)
    draw_trajectory(out / "final_trajectory.png", cfg, final_results[0], "Trained policy, 128-step trajectory")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
