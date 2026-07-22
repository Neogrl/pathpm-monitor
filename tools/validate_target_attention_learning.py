from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from model import OptionActor
from nodes import NODE_INPUT_INDEX
from ppo_buffer import PPORolloutBuffer
from tools.visualize_attention import build_attention_state, compute_attention
from trainer import Trainer
from utils import write_json


UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#17becf"]


@dataclass(frozen=True)
class TargetScenario:
    name: str
    centers: tuple[tuple[float, float], ...]
    amplitudes: tuple[float, ...]
    sigmas: tuple[float, ...]


def scenarios(map_size: float, fov_radius: float) -> list[TargetScenario]:
    low = 0.20 * map_size
    high = 0.80 * map_size
    center = 0.50 * map_size
    return [
        TargetScenario("no_target", (), (), ()),
        TargetScenario("single_nw", ((low, high),), (1.0,), (fov_radius,)),
        TargetScenario("single_se", ((high, low),), (1.0,), (fov_radius,)),
        TargetScenario("single_center", ((center, center),), (1.0,), (fov_radius,)),
        TargetScenario("single_weak", ((low, high),), (0.35,), (fov_radius,)),
        TargetScenario("single_wide", ((high, low),), (0.70,), (1.75 * fov_radius,)),
        TargetScenario(
            "double_diag",
            ((low, high), (high, low)),
            (1.0, 0.75),
            (fov_radius, fov_radius),
        ),
    ]


def target_distribution(positions: np.ndarray, scenario: TargetScenario) -> np.ndarray:
    values = np.zeros(len(positions), dtype=np.float32)
    for center, amplitude, sigma in zip(scenario.centers, scenario.amplitudes, scenario.sigmas):
        distance_sq = np.sum((positions - np.asarray(center, dtype=np.float32)[None, :]) ** 2, axis=1)
        values += float(amplitude) * np.exp(-distance_sq / (2.0 * float(sigma) ** 2)).astype(np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def inject_target(
    torch_obs: dict[str, torch.Tensor],
    values: np.ndarray,
) -> dict[str, torch.Tensor]:
    modified = {key: value.clone() for key, value in torch_obs.items()}
    target_index = NODE_INPUT_INDEX["target_belief_value"]
    target = torch.as_tensor(values, device=modified["global_node_inputs"].device).float()
    modified["global_node_inputs"][0, :, :, target_index] = target.unsqueeze(0)
    return modified


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_target_distributions(
    path: Path,
    cfg: Config,
    positions: np.ndarray,
    scenario_values: dict[str, np.ndarray],
    scenario_defs: list[TargetScenario],
) -> None:
    cols = 4
    rows = int(np.ceil(len(scenario_defs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), dpi=145)
    axes = np.asarray(axes).reshape(-1)
    for ax, scenario in zip(axes, scenario_defs):
        values = scenario_values[scenario.name]
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=values,
            cmap="magma",
            s=24,
            vmin=0.0,
            vmax=1.0,
        )
        for center in scenario.centers:
            ax.scatter(center[0], center[1], marker="x", color="#00d5ff", s=90, linewidth=2.0)
        ax.set_title(scenario.name)
        ax.set_xlim(0, cfg.map_size)
        ax.set_ylim(0, cfg.map_size)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#dddddd", linewidth=0.4)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes[len(scenario_defs) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def nearest_goal_index(positions: np.ndarray, centers: tuple[tuple[float, float], ...], current: int, graph) -> int:
    target_indices = [int(np.argmin(np.linalg.norm(positions - np.asarray(center)[None, :], axis=1))) for center in centers]
    distances, _ = graph.shortest_tree_from(current)
    finite = [(float(distances[index]), index) for index in target_indices if np.isfinite(distances[index])]
    if finite:
        return min(finite)[1]
    return min(target_indices, key=lambda index: float(np.linalg.norm(positions[index] - positions[current])))


def oracle_for_uav(graph, global_batch, uav_id: int, scenario: TargetScenario) -> tuple[int, int, float]:
    if not scenario.centers:
        return -1, -1, float("nan")
    current = int(global_batch.current_node_indices[uav_id])
    valid_slots = np.flatnonzero(
        ~global_batch.candidate_padding_mask[uav_id]
        & ~global_batch.action_mask[uav_id]
        & (global_batch.candidate_node_indices[uav_id] >= 0)
    )
    candidates = global_batch.candidate_node_indices[uav_id, valid_slots].astype(np.int64)
    goal = nearest_goal_index(graph.positions, scenario.centers, current, graph)
    compact_slot = graph.project_path_to_candidates(current, goal, candidates)
    oracle_slot = int(valid_slots[compact_slot]) if compact_slot >= 0 else -1
    distances, _ = graph.shortest_tree_from(goal)
    improvement = float(distances[current] - distances[int(global_batch.candidate_node_indices[uav_id, oracle_slot])]) if oracle_slot >= 0 else float("nan")
    return oracle_slot, goal, improvement


def target_region(positions: np.ndarray, scenario: TargetScenario) -> np.ndarray:
    region = np.zeros(len(positions), dtype=bool)
    for center, sigma in zip(scenario.centers, scenario.sigmas):
        region |= np.linalg.norm(positions - np.asarray(center)[None, :], axis=1) <= float(sigma)
    return region


def step1_validate(
    cfg: Config,
    out: Path,
    torch_obs: dict[str, torch.Tensor],
    positions: np.ndarray,
    scenario_defs: list[TargetScenario],
) -> tuple[dict[str, np.ndarray], dict]:
    scenario_values = {scenario.name: target_distribution(positions, scenario) for scenario in scenario_defs}
    target_index = NODE_INPUT_INDEX["target_belief_value"]
    base = torch_obs["global_node_inputs"].detach().cpu().numpy()
    input_rows = []
    peak_distances = []
    max_non_target_delta = 0.0
    all_finite = True
    all_in_range = True
    for scenario in scenario_defs:
        values = scenario_values[scenario.name]
        modified = inject_target(torch_obs, values)["global_node_inputs"].detach().cpu().numpy()
        keep = np.ones(base.shape[-1], dtype=bool)
        keep[target_index] = False
        max_non_target_delta = max(max_non_target_delta, float(np.max(np.abs(modified[..., keep] - base[..., keep]))))
        all_finite = all_finite and bool(np.isfinite(values).all())
        all_in_range = all_in_range and bool(np.all((values >= 0.0) & (values <= 1.0)))
        if scenario.centers:
            peak = positions[int(np.argmax(values))]
            peak_distances.append(min(float(np.linalg.norm(peak - np.asarray(center))) for center in scenario.centers))
        for node_index, (position, value) in enumerate(zip(positions, values)):
            input_rows.append(
                {
                    "scenario": scenario.name,
                    "node_index": node_index,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "target_belief_value": float(value),
                }
            )
    same_seed_delta = max(
        float(np.max(np.abs(values - target_distribution(positions, scenario))))
        for scenario, values in ((item, scenario_values[item.name]) for item in scenario_defs)
    )
    summary = {
        "n_global_nodes": int(len(positions)),
        "scenario_count": len(scenario_defs),
        "all_finite": all_finite,
        "all_in_range_0_1": all_in_range,
        "same_input_rebuild_max_abs_delta": same_seed_delta,
        "non_target_fields_max_abs_delta": max_non_target_delta,
        "max_peak_to_requested_center_distance": max(peak_distances) if peak_distances else 0.0,
    }
    summary["passed"] = bool(
        all_finite
        and all_in_range
        and same_seed_delta == 0.0
        and max_non_target_delta == 0.0
        and summary["max_peak_to_requested_center_distance"] <= 15.0
    )
    write_csv(out / "step1_controlled_target_inputs.csv", input_rows)
    write_json(out / "step1_summary.json", summary)
    draw_target_distributions(
        out / "step1_target_distributions.png",
        cfg,
        positions,
        scenario_values,
        scenario_defs,
    )
    return scenario_values, summary


def categorical_kl(p: np.ndarray, q: np.ndarray, valid: np.ndarray) -> float:
    pp = np.clip(p[valid], 1e-12, 1.0)
    qq = np.clip(q[valid], 1e-12, 1.0)
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def step2_baseline(
    cfg: Config,
    out: Path,
    actor: OptionActor,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    scenario_defs: list[TargetScenario],
    scenario_values: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    positions = global_batch.node_positions
    outputs = {}
    for scenario in scenario_defs:
        modified = inject_target(torch_obs, scenario_values[scenario.name])
        outputs[scenario.name] = compute_attention(actor, modified)

    no_target_probs = outputs["no_target"]["pointer_probs"][0].numpy()
    rows = []
    all_finite = True
    for scenario in scenario_defs:
        attention = outputs[scenario.name]["encoder_current_attention"][-1][0].numpy()
        probs = outputs[scenario.name]["pointer_probs"][0].numpy()
        actions = outputs[scenario.name]["greedy_actions"][0].numpy()
        region = target_region(positions, scenario)
        for uav_id in range(cfg.n_uavs):
            valid = (
                ~global_batch.candidate_padding_mask[uav_id]
                & ~global_batch.action_mask[uav_id]
                & (global_batch.candidate_node_indices[uav_id] >= 0)
            )
            oracle_slot, goal_index, oracle_improvement = oracle_for_uav(graph, global_batch, uav_id, scenario)
            if np.any(region):
                mass = float(np.sum(attention[uav_id, region]))
                uniform_mass = float(np.mean(region))
                lift = mass / max(uniform_mass, 1e-12)
                goal_rank = int(np.where(np.argsort(attention[uav_id])[::-1] == goal_index)[0][0] + 1)
                top10_hit = bool(np.any(region[np.argsort(attention[uav_id])[::-1][:10]]))
                max_node = int(np.argmax(attention[uav_id]))
                max_to_target = min(
                    float(np.linalg.norm(positions[max_node] - np.asarray(center))) for center in scenario.centers
                )
            else:
                mass = uniform_mass = lift = max_to_target = float("nan")
                goal_rank = -1
                top10_hit = False
            greedy_slot = int(actions[uav_id])
            greedy_node = int(global_batch.candidate_node_indices[uav_id, greedy_slot])
            if goal_index >= 0:
                distance_to_goal, _ = graph.shortest_tree_from(goal_index)
                current = int(global_batch.current_node_indices[uav_id])
                greedy_improvement = float(distance_to_goal[current] - distance_to_goal[greedy_node])
            else:
                greedy_improvement = float("nan")
            row = {
                "scenario": scenario.name,
                "uav": uav_id,
                "attention_target_mass": mass,
                "attention_uniform_mass": uniform_mass,
                "attention_lift": lift,
                "target_node_attention_rank": goal_rank,
                "attention_top10_hits_target_region": top10_hit,
                "max_attention_node_to_target_distance": max_to_target,
                "oracle_slot": oracle_slot,
                "oracle_action_probability": float(probs[uav_id, oracle_slot]) if oracle_slot >= 0 else float("nan"),
                "greedy_slot": greedy_slot,
                "greedy_action_is_oracle": bool(greedy_slot == oracle_slot) if oracle_slot >= 0 else False,
                "oracle_graph_distance_improvement": oracle_improvement,
                "greedy_graph_distance_improvement": greedy_improvement,
                "action_kl_from_no_target": categorical_kl(probs[uav_id], no_target_probs[uav_id], valid),
                "pointer_entropy": float(-np.sum(np.clip(probs[uav_id, valid], 1e-12, 1.0) * np.log(np.clip(probs[uav_id, valid], 1e-12, 1.0)))),
            }
            all_finite = all_finite and bool(np.isfinite(attention[uav_id]).all()) and bool(np.isfinite(probs[uav_id]).all())
            rows.append(row)

    selected = ["no_target", "single_nw", "single_se", "double_diag"]
    fig, axes = plt.subplots(len(selected), cfg.n_uavs, figsize=(4.0 * cfg.n_uavs, 3.9 * len(selected)), dpi=145)
    for row_index, scenario_name in enumerate(selected):
        scenario = next(item for item in scenario_defs if item.name == scenario_name)
        attention = outputs[scenario_name]["encoder_current_attention"][-1][0].numpy()
        actions = outputs[scenario_name]["greedy_actions"][0].numpy()
        for uav_id in range(cfg.n_uavs):
            ax = axes[row_index, uav_id]
            weights = attention[uav_id]
            ax.scatter(
                positions[:, 0], positions[:, 1], c=weights, cmap="viridis", s=18,
                vmin=0.0, vmax=max(float(weights.max()), 1e-9),
            )
            for center in scenario.centers:
                ax.scatter(center[0], center[1], marker="x", color="#ff2b7a", s=90, linewidth=2.2)
            current_idx = int(global_batch.current_node_indices[uav_id])
            current = positions[current_idx]
            ax.scatter(current[0], current[1], marker="*", color="white", edgecolor="black", s=115)
            chosen_slot = int(actions[uav_id])
            chosen_idx = int(global_batch.candidate_node_indices[uav_id, chosen_slot])
            chosen = positions[chosen_idx]
            ax.plot([current[0], chosen[0]], [current[1], chosen[1]], color=UAV_COLORS[uav_id], linewidth=2.0)
            oracle_slot, _, _ = oracle_for_uav(graph, global_batch, uav_id, scenario)
            if oracle_slot >= 0:
                oracle_idx = int(global_batch.candidate_node_indices[uav_id, oracle_slot])
                oracle = positions[oracle_idx]
                ax.scatter(oracle[0], oracle[1], marker="s", facecolors="none", edgecolors="#ff2b7a", s=80, linewidth=1.5)
            metric = next(item for item in rows if item["scenario"] == scenario_name and item["uav"] == uav_id)
            lift_text = "-" if not np.isfinite(metric["attention_lift"]) else f"{metric['attention_lift']:.2f}"
            ax.set_title(f"{scenario_name} | U{uav_id} | lift={lift_text}", fontsize=8)
            ax.set_xlim(0, cfg.map_size)
            ax.set_ylim(0, cfg.map_size)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#dddddd", linewidth=0.35)
    fig.tight_layout()
    fig.savefig(out / "step2_random_baseline_agents.png")
    plt.close(fig)

    target_rows = [row for row in rows if row["scenario"] != "no_target"]
    summary = {
        "network": "random_initialization",
        "all_forward_outputs_finite": all_finite,
        "mean_attention_lift": float(np.nanmean([row["attention_lift"] for row in target_rows])),
        "mean_oracle_action_probability": float(np.nanmean([row["oracle_action_probability"] for row in target_rows])),
        "greedy_oracle_accuracy": float(np.mean([row["greedy_action_is_oracle"] for row in target_rows])),
        "mean_action_kl_from_no_target": float(np.mean([row["action_kl_from_no_target"] for row in target_rows])),
    }
    summary["passed"] = bool(all_finite and len(rows) == len(scenario_defs) * cfg.n_uavs)
    write_csv(out / "step2_random_baseline_metrics.csv", rows)
    write_json(out / "step2_summary.json", summary)
    return rows, summary


def batched_target_observations(
    torch_obs: dict[str, torch.Tensor],
    target_values: np.ndarray,
) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(
        target_values,
        device=torch_obs["global_node_inputs"].device,
        dtype=torch.float32,
    )
    batch_size = values.shape[0]
    batched = {
        key: value.expand(batch_size, *value.shape[1:]).clone()
        for key, value in torch_obs.items()
    }
    target_index = NODE_INPUT_INDEX["target_belief_value"]
    batched["global_node_inputs"][:, :, :, target_index] = values[:, None, :]
    return batched


def local_inputs_from_global(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    global_inputs = obs["global_node_inputs"]
    indices = obs["candidate_node_indices"].clamp(min=0)
    feature_dim = global_inputs.shape[-1]
    gathered = torch.gather(
        global_inputs,
        dim=2,
        index=indices.unsqueeze(-1).expand(-1, -1, -1, feature_dim),
    )
    invalid = obs["candidate_padding_mask"] | (obs["candidate_node_indices"] < 0)
    return gathered.masked_fill(invalid.unsqueeze(-1), 0.0)


def target_values_for_nodes(
    positions: np.ndarray,
    target_nodes: np.ndarray,
    sigma: float,
    profile: str = "gaussian",
    amplitude: float = 1.0,
) -> np.ndarray:
    centers = positions[target_nodes]
    distance_sq = np.sum((positions[None, :, :] - centers[:, None, :]) ** 2, axis=2)
    if profile == "hard-fov":
        values = distance_sq <= sigma**2
    elif profile == "gaussian":
        values = np.exp(-distance_sq / (2.0 * sigma**2))
    else:
        raise ValueError(f"Unsupported target profile: {profile}")
    return (np.asarray(values, dtype=np.float32) * float(amplitude)).astype(np.float32)


def oracle_slots_for_nodes(graph, global_batch, target_nodes: np.ndarray) -> np.ndarray:
    slots = -np.ones((len(target_nodes), global_batch.current_node_indices.shape[0]), dtype=np.int64)
    for row, target_node in enumerate(target_nodes):
        for uav_id in range(global_batch.current_node_indices.shape[0]):
            valid_slots = np.flatnonzero(
                ~global_batch.candidate_padding_mask[uav_id]
                & ~global_batch.action_mask[uav_id]
                & (global_batch.candidate_node_indices[uav_id] >= 0)
            )
            candidates = global_batch.candidate_node_indices[uav_id, valid_slots].astype(np.int64)
            current = int(global_batch.current_node_indices[uav_id])
            compact_slot = graph.project_path_to_candidates(current, int(target_node), candidates)
            if compact_slot >= 0:
                slots[row, uav_id] = int(valid_slots[compact_slot])
    return slots


@torch.no_grad()
def pointer_raw_logits(actor: OptionActor, obs: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    encoded, _, _ = actor._encode_actor_global(
        obs["global_node_inputs"],
        obs["spatio_pos_encoding"],
        obs["global_edge_mask"],
        obs["global_node_padding_mask"],
    )
    current = actor._gather_current(encoded, obs["current_node_indices"])
    candidates = actor._gather_candidates(encoded, obs["candidate_node_indices"])
    b, n, m, dim = candidates.shape
    mask = (
        obs["action_mask"]
        | obs["candidate_padding_mask"]
        | (obs["candidate_node_indices"] < 0)
    ).reshape(b * n, 1, m)
    query = current.reshape(b * n, 1, dim)
    keys = candidates.reshape(b * n, m, dim)
    enhanced = actor.actor_decoder(query, keys, mask)
    q = torch.matmul(enhanced.reshape(-1, dim), actor.pointer.w_query).view(b * n, 1, dim)
    k = torch.matmul(keys.reshape(-1, dim), actor.pointer.w_key).view(b * n, m, dim)
    raw = actor.pointer.norm_factor * torch.matmul(q, k.transpose(1, 2))
    return raw.reshape(b, n, m).cpu().numpy(), (~mask.reshape(b, n, m)).cpu().numpy()


@torch.no_grad()
def evaluate_target_nodes(
    actor: OptionActor,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    target_nodes: np.ndarray,
    sigma: float,
) -> tuple[list[dict], dict]:
    was_training = actor.training
    actor.eval()
    values = target_values_for_nodes(global_batch.node_positions, target_nodes, sigma)
    obs = batched_target_observations(torch_obs, values)
    outputs = compute_attention(actor, obs)
    probs = outputs["pointer_probs"].numpy()
    actions = outputs["greedy_actions"].numpy()
    attention = outputs["encoder_current_attention"][-1].numpy()
    oracle_slots = oracle_slots_for_nodes(graph, global_batch, target_nodes)
    raw_logits, raw_valid = pointer_raw_logits(actor, obs)
    rows: list[dict] = []
    finite = bool(np.isfinite(probs).all() and np.isfinite(attention).all() and np.isfinite(raw_logits).all())
    for row_index, target_node in enumerate(target_nodes):
        region = values[row_index] >= np.exp(-0.5)
        uniform_mass = float(np.mean(region))
        for uav_id in range(global_batch.current_node_indices.shape[0]):
            oracle = int(oracle_slots[row_index, uav_id])
            valid = raw_valid[row_index, uav_id]
            rows.append(
                {
                    "target_node": int(target_node),
                    "uav": uav_id,
                    "valid_candidates": int(np.sum(valid)),
                    "random_oracle_probability": 1.0 / max(int(np.sum(valid)), 1),
                    "oracle_slot": oracle,
                    "oracle_action_probability": float(probs[row_index, uav_id, oracle]) if oracle >= 0 else float("nan"),
                    "greedy_action_is_oracle": bool(actions[row_index, uav_id] == oracle) if oracle >= 0 else False,
                    "attention_lift": float(np.sum(attention[row_index, uav_id, region]) / max(uniform_mass, 1e-12)),
                    "target_node_attention_rank": int(
                        np.where(np.argsort(attention[row_index, uav_id])[::-1] == target_node)[0][0] + 1
                    ),
                    "pointer_raw_abs_max": float(np.max(np.abs(raw_logits[row_index, uav_id, valid]))),
                    "pointer_raw_saturation_fraction": float(np.mean(np.abs(raw_logits[row_index, uav_id, valid]) > 3.0)),
                }
            )
    valid_rows = [row for row in rows if row["oracle_slot"] >= 0]
    summary = {
        "contexts": int(len(target_nodes)),
        "samples": int(len(valid_rows)),
        "all_finite": finite,
        "mean_random_oracle_probability": float(np.mean([row["random_oracle_probability"] for row in valid_rows])),
        "mean_oracle_action_probability": float(np.mean([row["oracle_action_probability"] for row in valid_rows])),
        "greedy_oracle_accuracy": float(np.mean([row["greedy_action_is_oracle"] for row in valid_rows])),
        "mean_attention_lift": float(np.mean([row["attention_lift"] for row in valid_rows])),
        "mean_target_node_attention_rank": float(np.mean([row["target_node_attention_rank"] for row in valid_rows])),
        "pointer_raw_abs_max": float(np.max([row["pointer_raw_abs_max"] for row in valid_rows])),
        "pointer_raw_saturation_fraction": float(np.mean([row["pointer_raw_saturation_fraction"] for row in valid_rows])),
    }
    if was_training:
        actor.train()
    return rows, summary


def add_context_batch(
    rollout: PPORolloutBuffer,
    actor: OptionActor,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    target_nodes: np.ndarray,
    sigma: float,
) -> float:
    target_values = target_values_for_nodes(global_batch.node_positions, target_nodes, sigma)
    obs = batched_target_observations(torch_obs, target_values)
    with torch.no_grad():
        actions, _, terminations, log_probs, values, _ = actor.act_with_info(**obs, greedy=False)
    oracle_slots = oracle_slots_for_nodes(graph, global_batch, target_nodes)
    action_np = actions.cpu().numpy().astype(np.int64)
    rewards = np.mean(action_np == oracle_slots, axis=1).astype(np.float32)
    local_inputs = local_inputs_from_global(obs).cpu().numpy().astype(np.float32)
    tensor_arrays = {key: value.detach().cpu().numpy() for key, value in obs.items()}
    for index in range(len(target_nodes)):
        rollout.add(
            {
                "global_node_inputs": tensor_arrays["global_node_inputs"][index],
                "spatio_pos_encoding": tensor_arrays["spatio_pos_encoding"][index],
                "global_edge_mask": tensor_arrays["global_edge_mask"][index],
                "global_node_padding_mask": tensor_arrays["global_node_padding_mask"][index],
                "current_node_indices": tensor_arrays["current_node_indices"][index],
                "candidate_node_indices": tensor_arrays["candidate_node_indices"][index],
                "candidate_padding_mask": tensor_arrays["candidate_padding_mask"][index],
                "action_mask": tensor_arrays["action_mask"][index],
                "node_inputs": local_inputs[index],
                "node_padding_mask": tensor_arrays["candidate_padding_mask"][index],
                "uav_state": tensor_arrays["uav_state"][index],
                "prev_option": tensor_arrays["prev_option"][index],
                "actions": action_np[index],
                "terminations": terminations[index].cpu().numpy().astype(np.float32),
                "log_probs": log_probs[index].cpu().numpy().astype(np.float32),
                "values": values[index].cpu().numpy().astype(np.float32),
                "reward": np.asarray(rewards[index], dtype=np.float32),
                "done": np.asarray(1.0, dtype=np.float32),
                "next_values": np.zeros(global_batch.current_node_indices.shape[0], dtype=np.float32),
            }
        )
    return float(np.mean(rewards))


def draw_step3_curves(path: Path, rows: list[dict]) -> None:
    updates = [row["update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=145)
    axes[0, 0].plot(updates, [row["oracle_action_probability"] for row in rows], label="held-out")
    axes[0, 0].plot(updates, [row["random_oracle_probability"] for row in rows], linestyle="--", label="random")
    axes[0, 0].set_title("Oracle action probability")
    axes[0, 0].legend()
    axes[0, 1].plot(updates, [row["greedy_oracle_accuracy"] for row in rows])
    axes[0, 1].plot(updates, [row["random_oracle_probability"] for row in rows], linestyle="--")
    axes[0, 1].set_title("Greedy oracle accuracy")
    axes[1, 0].plot(updates, [row["attention_lift"] for row in rows])
    axes[1, 0].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Target-region attention lift")
    axes[1, 1].plot(updates, [row["train_reward"] for row in rows], label="train reward")
    axes[1, 1].plot(updates, [row["entropy"] for row in rows], label="entropy")
    axes[1, 1].set_title("Training diagnostics")
    axes[1, 1].legend()
    for ax in axes.reshape(-1):
        ax.set_xlabel("MAPPO update")
        ax.grid(True, color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def held_out_target_nodes(graph, global_batch, seed: int, eval_contexts: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 701)
    current_nodes = set(int(index) for index in global_batch.current_node_indices)
    eligible = np.asarray(
        [index for index in range(graph.n_nodes) if index not in current_nodes],
        dtype=np.int64,
    )
    rng.shuffle(eligible)
    eval_count = min(eval_contexts, max(8, len(eligible) // 4))
    return eligible[eval_count:], eligible[:eval_count]


def step3_train_minimal_mappo(
    cfg: Config,
    out: Path,
    trainer: Trainer,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    seed: int,
    updates: int,
    contexts_per_update: int,
    eval_contexts: int,
    eval_interval: int,
    sigma: float,
) -> dict:
    rng = np.random.default_rng(seed + 701)
    train_nodes, eval_nodes = held_out_target_nodes(graph, global_batch, seed, eval_contexts)
    # Keep the sampling stream identical to the original split implementation.
    rng.shuffle(np.arange(len(train_nodes) + len(eval_nodes)))
    if len(train_nodes) < 8 or len(eval_nodes) < 4:
        raise RuntimeError("The diagnostic graph is too small for separate train/evaluation target nodes.")

    evaluation_rows: list[dict] = []
    metric_rows: list[dict] = []
    initial_rows, initial = evaluate_target_nodes(
        trainer.actor, torch_obs, graph, global_batch, eval_nodes, sigma
    )
    for row in initial_rows:
        evaluation_rows.append({"update": 0, **row})
    metric_rows.append(
        {
            "update": 0,
            "train_reward": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": float("nan"),
            "approx_kl": float("nan"),
            "oracle_action_probability": initial["mean_oracle_action_probability"],
            "random_oracle_probability": initial["mean_random_oracle_probability"],
            "greedy_oracle_accuracy": initial["greedy_oracle_accuracy"],
            "attention_lift": initial["mean_attention_lift"],
            "pointer_raw_saturation_fraction": initial["pointer_raw_saturation_fraction"],
        }
    )

    last_stats = None
    for update in range(1, updates + 1):
        sampled_nodes = rng.choice(train_nodes, size=contexts_per_update, replace=True)
        rollout = PPORolloutBuffer()
        train_reward = add_context_batch(
            rollout,
            trainer.actor,
            torch_obs,
            graph,
            global_batch,
            sampled_nodes,
            sigma,
        )
        trainer.actor.train()
        last_stats = trainer.update(rollout)
        if update % eval_interval != 0 and update != updates:
            continue
        rows, evaluation = evaluate_target_nodes(
            trainer.actor, torch_obs, graph, global_batch, eval_nodes, sigma
        )
        for row in rows:
            evaluation_rows.append({"update": update, **row})
        metric_rows.append(
            {
                "update": update,
                "train_reward": train_reward,
                "policy_loss": last_stats.policy_loss,
                "value_loss": last_stats.value_loss,
                "entropy": last_stats.entropy,
                "approx_kl": last_stats.approx_kl,
                "oracle_action_probability": evaluation["mean_oracle_action_probability"],
                "random_oracle_probability": evaluation["mean_random_oracle_probability"],
                "greedy_oracle_accuracy": evaluation["greedy_oracle_accuracy"],
                "attention_lift": evaluation["mean_attention_lift"],
                "pointer_raw_saturation_fraction": evaluation["pointer_raw_saturation_fraction"],
            }
        )
        print(
            f"[step3] update={update}/{updates} reward={train_reward:.3f} "
            f"oracle_p={evaluation['mean_oracle_action_probability']:.3f} "
            f"greedy_acc={evaluation['greedy_oracle_accuracy']:.3f} "
            f"attention_lift={evaluation['mean_attention_lift']:.3f}"
        )

    final_rows, final = evaluate_target_nodes(
        trainer.actor, torch_obs, graph, global_batch, eval_nodes, sigma
    )
    probability_gain = final["mean_oracle_action_probability"] - initial["mean_oracle_action_probability"]
    accuracy_gain = final["greedy_oracle_accuracy"] - initial["greedy_oracle_accuracy"]
    summary = {
        "training_task": "one_step_contextual_mappo",
        "reward": "mean_uav(action == shortest_path_first_step)",
        "graph_nodes": int(graph.n_nodes),
        "train_target_nodes": int(len(train_nodes)),
        "held_out_target_nodes": int(len(eval_nodes)),
        "updates": int(updates),
        "contexts_per_update": int(contexts_per_update),
        "initial": initial,
        "final": final,
        "oracle_probability_gain": float(probability_gain),
        "greedy_accuracy_gain": float(accuracy_gain),
        "final_train_reward": float(metric_rows[-1]["train_reward"]),
        "last_policy_loss": float(last_stats.policy_loss) if last_stats is not None else float("nan"),
        "last_value_loss": float(last_stats.value_loss) if last_stats is not None else float("nan"),
    }
    summary["passed"] = bool(
        final["all_finite"]
        and final["mean_oracle_action_probability"] >= final["mean_random_oracle_probability"] + 0.05
        and probability_gain >= 0.05
        and final["greedy_oracle_accuracy"] >= final["mean_random_oracle_probability"] + 0.10
        and final["pointer_raw_saturation_fraction"] < 0.50
    )
    write_csv(out / "step3_training_curve.csv", metric_rows)
    write_csv(out / "step3_heldout_metrics.csv", evaluation_rows)
    write_csv(out / "step3_final_heldout_metrics.csv", final_rows)
    write_json(out / "step3_summary.json", summary)
    draw_step3_curves(out / "step3_learning_curves.png", metric_rows)
    trainer.save(out / "step3_checkpoint.pt")
    return summary


@torch.no_grad()
def evaluate_intervention(
    actor: OptionActor,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    input_values: np.ndarray,
    goal_nodes: np.ndarray,
    sigma: float,
    target_profile: str = "gaussian",
    target_amplitude: float = 1.0,
    reference_probs: np.ndarray | None = None,
) -> tuple[list[dict], dict, np.ndarray]:
    actor.eval()
    obs = batched_target_observations(torch_obs, input_values)
    outputs = compute_attention(actor, obs)
    probs = outputs["pointer_probs"].numpy()
    actions = outputs["greedy_actions"].numpy()
    attention = outputs["encoder_current_attention"][-1].numpy()
    oracle_slots = oracle_slots_for_nodes(graph, global_batch, goal_nodes)
    goal_values = target_values_for_nodes(
        global_batch.node_positions,
        goal_nodes,
        sigma,
        profile=target_profile,
        amplitude=target_amplitude,
    )
    rows: list[dict] = []
    for context_index, goal_node in enumerate(goal_nodes):
        if target_profile == "hard-fov":
            region = goal_values[context_index] > 0.0
        else:
            region = goal_values[context_index] >= target_amplitude * np.exp(-0.5)
        uniform_mass = float(np.mean(region))
        distances, _ = graph.shortest_tree_from(int(goal_node))
        for uav_id in range(global_batch.current_node_indices.shape[0]):
            valid = (
                ~global_batch.candidate_padding_mask[uav_id]
                & ~global_batch.action_mask[uav_id]
                & (global_batch.candidate_node_indices[uav_id] >= 0)
            )
            oracle = int(oracle_slots[context_index, uav_id])
            chosen = int(actions[context_index, uav_id])
            current_node = int(global_batch.current_node_indices[uav_id])
            chosen_node = int(global_batch.candidate_node_indices[uav_id, chosen])
            row = {
                "context": context_index,
                "goal_node": int(goal_node),
                "uav": uav_id,
                "oracle_action_probability": float(probs[context_index, uav_id, oracle]),
                "greedy_action_is_oracle": bool(chosen == oracle),
                "greedy_graph_distance_improvement": float(distances[current_node] - distances[chosen_node]),
                "attention_lift_at_scored_goal": float(
                    np.sum(attention[context_index, uav_id, region]) / max(uniform_mass, 1e-12)
                ),
                "action_kl_from_correct": (
                    categorical_kl(probs[context_index, uav_id], reference_probs[context_index, uav_id], valid)
                    if reference_probs is not None
                    else 0.0
                ),
            }
            rows.append(row)
    summary = {
        "mean_oracle_action_probability": float(np.mean([row["oracle_action_probability"] for row in rows])),
        "greedy_oracle_accuracy": float(np.mean([row["greedy_action_is_oracle"] for row in rows])),
        "mean_graph_distance_improvement": float(np.mean([row["greedy_graph_distance_improvement"] for row in rows])),
        "mean_attention_lift_at_scored_goal": float(np.mean([row["attention_lift_at_scored_goal"] for row in rows])),
        "mean_action_kl_from_correct": float(np.mean([row["action_kl_from_correct"] for row in rows])),
        "all_finite": bool(np.isfinite(probs).all() and np.isfinite(attention).all()),
    }
    return rows, summary, probs


def draw_step4_comparison(path: Path, summaries: dict[str, dict]) -> None:
    labels = list(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), dpi=145)
    metrics = [
        ("mean_oracle_action_probability", "Oracle action probability"),
        ("greedy_oracle_accuracy", "Greedy oracle accuracy"),
        ("mean_graph_distance_improvement", "Graph-distance improvement"),
    ]
    colors = ["#2a9d8f", "#8d99ae", "#e9c46a", "#e76f51", "#457b9d"]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(np.arange(len(labels)), [summaries[label][metric] for label in labels], color=colors)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def step4_causal_ablation(
    out: Path,
    actor: OptionActor,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    seed: int,
    eval_contexts: int,
    sigma: float,
    target_profile: str = "gaussian",
    target_amplitude: float = 1.0,
) -> dict:
    _, eval_nodes = held_out_target_nodes(graph, global_batch, seed, eval_contexts)
    positions = global_batch.node_positions
    correct_values = target_values_for_nodes(
        positions,
        eval_nodes,
        sigma,
        profile=target_profile,
        amplitude=target_amplitude,
    )
    zero_values = np.zeros_like(correct_values)
    shuffle_rng = np.random.default_rng(seed + 1701)
    shuffled_values = np.stack(
        [values[shuffle_rng.permutation(graph.n_nodes)] for values in correct_values],
        axis=0,
    )
    moved_nodes = np.asarray(
        [eval_nodes[int(np.argmax(np.linalg.norm(positions[eval_nodes] - positions[node], axis=1)))] for node in eval_nodes],
        dtype=np.int64,
    )
    moved_values = target_values_for_nodes(
        positions,
        moved_nodes,
        sigma,
        profile=target_profile,
        amplitude=target_amplitude,
    )

    rows: list[dict] = []
    correct_rows, correct, correct_probs = evaluate_intervention(
        actor,
        torch_obs,
        graph,
        global_batch,
        correct_values,
        eval_nodes,
        sigma,
        target_profile=target_profile,
        target_amplitude=target_amplitude,
    )
    cases = {"correct": correct}
    for row in correct_rows:
        rows.append({"intervention": "correct", **row})
    definitions = [
        ("zero", zero_values, eval_nodes),
        ("shuffled", shuffled_values, eval_nodes),
        ("moved_scored_original", moved_values, eval_nodes),
        ("moved_scored_moved", moved_values, moved_nodes),
    ]
    for name, input_values, goal_nodes in definitions:
        case_rows, summary, _ = evaluate_intervention(
            actor,
            torch_obs,
            graph,
            global_batch,
            input_values,
            goal_nodes,
            sigma,
            target_profile=target_profile,
            target_amplitude=target_amplitude,
            reference_probs=correct_probs,
        )
        cases[name] = summary
        for row in case_rows:
            rows.append({"intervention": name, **row})

    summary = {
        "held_out_target_nodes": int(len(eval_nodes)),
        "target_profile": target_profile,
        "target_amplitude": float(target_amplitude),
        "cases": cases,
        "correct_minus_zero_probability": float(
            correct["mean_oracle_action_probability"] - cases["zero"]["mean_oracle_action_probability"]
        ),
        "correct_minus_shuffled_probability": float(
            correct["mean_oracle_action_probability"] - cases["shuffled"]["mean_oracle_action_probability"]
        ),
        "moved_goal_following_gain": float(
            cases["moved_scored_moved"]["mean_oracle_action_probability"]
            - cases["moved_scored_original"]["mean_oracle_action_probability"]
        ),
    }
    summary["passed"] = bool(
        all(case["all_finite"] for case in cases.values())
        and summary["correct_minus_zero_probability"] >= 0.10
        and summary["correct_minus_shuffled_probability"] >= 0.10
        and summary["moved_goal_following_gain"] >= 0.10
        and cases["moved_scored_moved"]["mean_oracle_action_probability"] >= 0.20
    )
    write_csv(out / "step4_causal_metrics.csv", rows)
    write_json(out / "step4_summary.json", summary)
    draw_step4_comparison(out / "step4_causal_comparison.png", cases)
    return summary


def separated_target_sets(
    positions: np.ndarray,
    pool: np.ndarray,
    contexts: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 2501)
    result: list[np.ndarray] = []
    for context in range(contexts):
        count = 2 + context % 3
        selected = [int(rng.choice(pool))]
        while len(selected) < count:
            remaining = np.asarray([node for node in pool if int(node) not in selected], dtype=np.int64)
            minimum_separation = np.min(
                np.linalg.norm(
                    positions[remaining, None, :] - positions[np.asarray(selected)][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            selected.append(int(remaining[int(np.argmax(minimum_separation))]))
        result.append(np.asarray(selected, dtype=np.int64))
    return result


def multi_target_values(positions: np.ndarray, target_sets: list[np.ndarray], sigma: float) -> np.ndarray:
    rows = []
    for target_nodes in target_sets:
        centers = positions[target_nodes]
        distance_sq = np.sum((positions[None, :, :] - centers[:, None, :]) ** 2, axis=2)
        rows.append(np.clip(np.sum(np.exp(-distance_sq / (2.0 * sigma**2)), axis=0), 0.0, 1.0))
    return np.asarray(rows, dtype=np.float32)


def nearest_target_oracles(graph, global_batch, target_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    contexts = len(target_sets)
    n_uavs = global_batch.current_node_indices.shape[0]
    oracle_slots = -np.ones((contexts, n_uavs), dtype=np.int64)
    nearest_targets = -np.ones((contexts, n_uavs), dtype=np.int64)
    for context, target_nodes in enumerate(target_sets):
        distance_maps = [graph.shortest_tree_from(int(node))[0] for node in target_nodes]
        for uav_id in range(n_uavs):
            current = int(global_batch.current_node_indices[uav_id])
            nearest_local = int(np.argmin([distances[current] for distances in distance_maps]))
            nearest_targets[context, uav_id] = nearest_local
            valid_slots = np.flatnonzero(
                ~global_batch.candidate_padding_mask[uav_id]
                & ~global_batch.action_mask[uav_id]
                & (global_batch.candidate_node_indices[uav_id] >= 0)
            )
            candidates = global_batch.candidate_node_indices[uav_id, valid_slots].astype(np.int64)
            compact = graph.project_path_to_candidates(current, int(target_nodes[nearest_local]), candidates)
            if compact >= 0:
                oracle_slots[context, uav_id] = int(valid_slots[compact])
    return oracle_slots, nearest_targets


def draw_step5_agents(
    path: Path,
    cfg: Config,
    positions: np.ndarray,
    global_batch,
    target_sets: list[np.ndarray],
    target_values: np.ndarray,
    actions: np.ndarray,
    contexts: int = 4,
) -> None:
    shown = min(contexts, len(target_sets))
    fig, axes = plt.subplots(1, shown, figsize=(4.6 * shown, 4.4), dpi=145)
    axes = np.atleast_1d(axes)
    for context, ax in enumerate(axes):
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=target_values[context],
            cmap="magma",
            s=22,
            vmin=0.0,
            vmax=1.0,
        )
        for target_id, node in enumerate(target_sets[context]):
            center = positions[node]
            ax.scatter(center[0], center[1], marker="X", color="#00d5ff", edgecolor="black", s=105)
            ax.text(center[0] + 1.0, center[1] + 1.0, f"T{target_id}", fontsize=8)
        for uav_id in range(cfg.n_uavs):
            current = positions[int(global_batch.current_node_indices[uav_id])]
            slot = int(actions[context, uav_id])
            chosen = positions[int(global_batch.candidate_node_indices[uav_id, slot])]
            ax.scatter(current[0], current[1], marker="*", color=UAV_COLORS[uav_id], edgecolor="black", s=120)
            ax.plot(
                [current[0], chosen[0]],
                [current[1], chosen[1]],
                color=UAV_COLORS[uav_id],
                linewidth=2.0,
                label=f"U{uav_id}" if context == 0 else None,
            )
        ax.set_title(f"{len(target_sets[context])} targets | context {context}")
        ax.set_xlim(0, cfg.map_size)
        ax.set_ylim(0, cfg.map_size)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#dddddd", linewidth=0.4)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_step5_global_attention(
    path: Path,
    cfg: Config,
    positions: np.ndarray,
    global_batch,
    target_sets: list[np.ndarray],
    attention: np.ndarray,
    actions: np.ndarray,
    contexts: int = 3,
) -> None:
    shown = min(contexts, len(target_sets))
    fig, axes = plt.subplots(
        shown,
        cfg.n_uavs,
        figsize=(3.7 * cfg.n_uavs, 3.6 * shown),
        dpi=145,
        squeeze=False,
    )
    for context in range(shown):
        row_vmax = max(float(np.max(attention[context])), 1e-9)
        for uav_id in range(cfg.n_uavs):
            ax = axes[context, uav_id]
            weights = attention[context, uav_id]
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=weights,
                cmap="viridis",
                s=23,
                vmin=0.0,
                vmax=row_vmax,
            )
            for target_id, node in enumerate(target_sets[context]):
                center = positions[node]
                ax.scatter(
                    center[0],
                    center[1],
                    marker="X",
                    color="#ff2b7a",
                    edgecolor="white",
                    linewidth=0.7,
                    s=95,
                )
                ax.text(center[0] + 1.0, center[1] + 1.0, f"T{target_id}", fontsize=7)
            current = positions[int(global_batch.current_node_indices[uav_id])]
            slot = int(actions[context, uav_id])
            chosen = positions[int(global_batch.candidate_node_indices[uav_id, slot])]
            ax.scatter(
                current[0],
                current[1],
                marker="*",
                color="white",
                edgecolor="black",
                s=115,
                zorder=5,
            )
            ax.plot(
                [current[0], chosen[0]],
                [current[1], chosen[1]],
                color=UAV_COLORS[uav_id],
                linewidth=2.0,
                zorder=4,
            )
            ax.set_title(f"{len(target_sets[context])} targets | U{uav_id}", fontsize=8)
            ax.set_xlim(0, cfg.map_size)
            ax.set_ylim(0, cfg.map_size)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#dddddd", linewidth=0.35)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    fig.suptitle("Global encoder attention from each UAV current node (8-head mean)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def step5_multi_target_behavior(
    cfg: Config,
    out: Path,
    actor: OptionActor,
    torch_obs: dict[str, torch.Tensor],
    graph,
    global_batch,
    seed: int,
    eval_contexts: int,
    sigma: float,
) -> dict:
    train_nodes, eval_nodes = held_out_target_nodes(graph, global_batch, seed, max(eval_contexts, 16))
    pool = np.concatenate([eval_nodes, train_nodes[: min(16, len(train_nodes))]])
    target_sets = separated_target_sets(global_batch.node_positions, pool, eval_contexts, seed)
    target_values = multi_target_values(global_batch.node_positions, target_sets, sigma)
    obs = batched_target_observations(torch_obs, target_values)
    outputs = compute_attention(actor, obs)
    probs = outputs["pointer_probs"].numpy()
    actions = outputs["greedy_actions"].numpy()
    attention = outputs["encoder_current_attention"][-1].numpy()
    oracle_slots, nearest_targets = nearest_target_oracles(graph, global_batch, target_sets)
    rows: list[dict] = []
    context_rows: list[dict] = []
    for context, target_nodes in enumerate(target_sets):
        distance_maps = [graph.shortest_tree_from(int(node))[0] for node in target_nodes]
        assigned_targets = []
        region = np.zeros(graph.n_nodes, dtype=bool)
        for node in target_nodes:
            region |= np.linalg.norm(global_batch.node_positions - global_batch.node_positions[node], axis=1) <= sigma
        uniform_mass = float(np.mean(region))
        for uav_id in range(cfg.n_uavs):
            oracle = int(oracle_slots[context, uav_id])
            chosen_slot = int(actions[context, uav_id])
            current = int(global_batch.current_node_indices[uav_id])
            chosen_node = int(global_batch.candidate_node_indices[uav_id, chosen_slot])
            improvements = np.asarray(
                [distances[current] - distances[chosen_node] for distances in distance_maps],
                dtype=np.float32,
            )
            assigned = int(np.argmax(improvements))
            assigned_targets.append(assigned)
            valid_count = int(
                np.sum(~global_batch.candidate_padding_mask[uav_id] & ~global_batch.action_mask[uav_id])
            )
            rows.append(
                {
                    "context": context,
                    "target_count": len(target_nodes),
                    "uav": uav_id,
                    "nearest_target": int(nearest_targets[context, uav_id]),
                    "assigned_target": assigned,
                    "oracle_action_probability": float(probs[context, uav_id, oracle]),
                    "greedy_action_is_oracle": bool(chosen_slot == oracle),
                    "nearest_target_graph_distance_improvement": float(
                        improvements[int(nearest_targets[context, uav_id])]
                    ),
                    "best_target_graph_distance_improvement": float(np.max(improvements)),
                    "attention_lift_all_target_regions": float(
                        np.sum(attention[context, uav_id, region]) / max(uniform_mass, 1e-12)
                    ),
                    "random_oracle_probability": 1.0 / max(valid_count, 1),
                }
            )
        unique_assigned = len(set(assigned_targets))
        context_rows.append(
            {
                "context": context,
                "target_count": len(target_nodes),
                "unique_targets_assigned": unique_assigned,
                "target_allocation_coverage": unique_assigned / min(len(target_nodes), cfg.n_uavs),
            }
        )

    summary = {
        "contexts": len(target_sets),
        "target_counts": [int(len(nodes)) for nodes in target_sets],
        "mean_random_oracle_probability": float(np.mean([row["random_oracle_probability"] for row in rows])),
        "mean_oracle_action_probability": float(np.mean([row["oracle_action_probability"] for row in rows])),
        "greedy_oracle_accuracy": float(np.mean([row["greedy_action_is_oracle"] for row in rows])),
        "mean_nearest_target_graph_distance_improvement": float(
            np.mean([row["nearest_target_graph_distance_improvement"] for row in rows])
        ),
        "mean_attention_lift_all_target_regions": float(
            np.mean([row["attention_lift_all_target_regions"] for row in rows])
        ),
        "mean_target_allocation_coverage": float(
            np.mean([row["target_allocation_coverage"] for row in context_rows])
        ),
        "all_finite": bool(np.isfinite(probs).all() and np.isfinite(attention).all()),
    }
    summary["passed"] = bool(
        summary["all_finite"]
        and summary["mean_oracle_action_probability"] >= summary["mean_random_oracle_probability"] + 0.05
        and summary["greedy_oracle_accuracy"] >= summary["mean_random_oracle_probability"] + 0.10
        and summary["mean_nearest_target_graph_distance_improvement"] > 0.0
        and summary["mean_target_allocation_coverage"] >= 0.60
    )
    write_csv(out / "step5_agent_metrics.csv", rows)
    write_csv(out / "step5_context_metrics.csv", context_rows)
    write_json(out / "step5_summary.json", summary)
    draw_step5_agents(
        out / "step5_multi_target_agents.png",
        cfg,
        global_batch.node_positions,
        global_batch,
        target_sets,
        target_values,
        actions,
    )
    draw_step5_global_attention(
        out / "step5_global_attention_all_uavs.png",
        cfg,
        global_batch.node_positions,
        global_batch,
        target_sets,
        attention,
        actions,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate target-conditioned attention and action behavior.")
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/target_attention_validation")
    parser.add_argument("--step3-updates", type=int, default=0)
    parser.add_argument("--contexts-per-update", type=int, default=32)
    parser.add_argument("--eval-contexts", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--prm-random-nodes", type=int, default=48)
    parser.add_argument("--prm-boundary-points-per-side", type=int, default=5)
    parser.add_argument("--ppo-update-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--step4-checkpoint", type=str, default=None)
    parser.add_argument("--target-profile", choices=["gaussian", "hard-fov"], default="gaussian")
    parser.add_argument("--target-amplitude", type=float, default=1.0)
    parser.add_argument("--run-step5", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config()
    diagnostic_training = args.step3_updates > 0 or args.step4_checkpoint is not None
    if diagnostic_training:
        cfg.prm_random_nodes = args.prm_random_nodes
        cfg.prm_boundary_points_per_side = args.prm_boundary_points_per_side
        cfg.ppo_update_epochs = args.ppo_update_epochs
        cfg.ppo_minibatch_size = args.ppo_minibatch_size
        cfg.ppo_num_minibatches = 0
        cfg.actor_lr = args.learning_rate
        cfg.critic_lr = args.learning_rate
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg, device=device) if diagnostic_training else None
    actor = trainer.actor if trainer is not None else OptionActor(cfg).to(device)
    actor.eval()
    worker, _, _, _, _, _, global_batch, _, torch_obs = build_attention_state(
        cfg, actor, device, args.seed, None, 0
    )
    scenario_defs = scenarios(cfg.map_size, cfg.fov_radius)
    scenario_values, step1 = step1_validate(
        cfg, out, torch_obs, global_batch.node_positions, scenario_defs
    )
    if not step1["passed"]:
        raise RuntimeError(f"Step 1 failed: {step1}")
    _, step2 = step2_baseline(
        cfg,
        out,
        actor,
        torch_obs,
        worker.node_builder.graph,
        global_batch,
        scenario_defs,
        scenario_values,
    )
    if not step2["passed"]:
        raise RuntimeError(f"Step 2 failed: {step2}")
    result = {"out_dir": str(out.resolve()), "step1": step1, "step2": step2}
    if args.step4_checkpoint is not None:
        trainer.load(Path(args.step4_checkpoint))
    if trainer is not None:
        if args.step3_updates > 0:
            step3 = step3_train_minimal_mappo(
                cfg,
                out,
                trainer,
                torch_obs,
                worker.node_builder.graph,
                global_batch,
                args.seed,
                args.step3_updates,
                args.contexts_per_update,
                args.eval_contexts,
                args.eval_interval,
                cfg.fov_radius,
            )
            result["step3"] = step3
        if args.step4_checkpoint is not None:
            step4 = step4_causal_ablation(
                out,
                trainer.actor,
                torch_obs,
                worker.node_builder.graph,
                global_batch,
                args.seed,
                args.eval_contexts,
                cfg.fov_radius,
                target_profile=args.target_profile,
                target_amplitude=args.target_amplitude,
            )
            result["step4"] = step4
            if args.run_step5 and step4["passed"]:
                step5 = step5_multi_target_behavior(
                    cfg,
                    out,
                    trainer.actor,
                    torch_obs,
                    worker.node_builder.graph,
                    global_batch,
                    args.seed,
                    args.eval_contexts,
                    cfg.fov_radius,
                )
                result["step5"] = step5
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
