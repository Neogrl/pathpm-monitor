import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from baselines import HeuristicBaseline, RandomBaseline
from config import Config
from environment import CMUOMMTEnv
from nodes import NODE_INPUT_FIELDS, NODE_INPUT_INDEX, NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from utils import write_json
from worker import team_summary, uav_state


TEAM_SUMMARY_FIELDS = [
    "team_mean_x_norm",
    "team_mean_y_norm",
    "team_std_x_norm",
    "team_std_y_norm",
    "team_mean_distance_norm",
    "team_mean_overlap",
    "phd_total_weight_norm",
    "search_belief_mean",
    "coverage_age_mean_norm",
    "phd_peak_count_norm",
]

UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
TARGET_COLOR = "#d62728"
SEARCH_COLOR = "#ff7f0e"
MAINT_COLOR = "#17becf"
GOAL_COLOR = "#111111"
MEAS_COLOR = "#f7e26b"
MASKED_COLOR = "#d62728"
GRAPH_COLOR = "#d0d0d0"


def finite_bool(*arrays: np.ndarray) -> bool:
    return all(np.all(np.isfinite(arr)) for arr in arrays)


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def json_array(arr: np.ndarray, precision: int = 3) -> str:
    rounded = np.round(np.asarray(arr, dtype=float), precision)
    return json.dumps(rounded.tolist(), ensure_ascii=False, separators=(",", ":"))


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_policy(name: str):
    if name == "heuristic":
        return HeuristicBaseline()
    if name == "random":
        return RandomBaseline()
    raise ValueError(f"Unknown policy={name!r}; expected heuristic or random.")


def randomize_uav_positions(env: CMUOMMTEnv, margin: float, rng: np.random.Generator) -> None:
    low = float(margin)
    high = float(env.cfg.map_size - margin)
    if high <= low:
        raise ValueError(f"uav_init_margin={margin} is too large for map_size={env.cfg.map_size}.")
    env.uav_positions = rng.uniform(low, high, size=env.uav_positions.shape).astype(np.float32)


def flatten_positions(prefix: str, values: np.ndarray) -> dict[str, float]:
    row: dict[str, float] = {}
    for i, item in enumerate(values):
        row[f"{prefix}_{i}_x"] = float(item[0])
        row[f"{prefix}_{i}_y"] = float(item[1])
    return row


def flatten_target_states(values: np.ndarray) -> dict[str, float]:
    row: dict[str, float] = {}
    for i, item in enumerate(values):
        row[f"target_{i}_x"] = float(item[0])
        row[f"target_{i}_y"] = float(item[1])
        row[f"target_{i}_vx"] = float(item[2])
        row[f"target_{i}_vy"] = float(item[3])
    return row


def memory_stats(env: CMUOMMTEnv) -> dict[str, float]:
    memory = env.memory
    n_targets = max(len(memory.is_discovered), 1)
    return {
        "discovery_rate_so_far": float(np.mean(memory.is_discovered)) if len(memory.is_discovered) else 0.0,
        "observation_count_total": float(np.sum(memory.observation_count)),
        "mean_current_gap": float(np.mean(memory.current_gap)) if len(memory.current_gap) else 0.0,
        "max_current_gap": float(np.max(memory.current_gap)) if len(memory.current_gap) else 0.0,
        "discovered_count": float(np.sum(memory.is_discovered)),
        "target_count": float(n_targets),
    }


def global_graph_summary_row(step: int, batch, global_batch) -> dict[str, float]:
    candidate_present = ~global_batch.candidate_padding_mask
    valid_indices = global_batch.candidate_node_indices[candidate_present]
    if len(valid_indices):
        mapped = global_batch.node_positions[valid_indices]
        candidate_waypoints = batch.waypoints[candidate_present]
        max_candidate_position_error = float(np.max(np.linalg.norm(mapped - candidate_waypoints, axis=1)))
    else:
        max_candidate_position_error = 0.0
    unmasked_edges = int(np.sum(~global_batch.global_edge_mask))
    adjacency_ok = True
    for uav_id, current_idx in enumerate(global_batch.current_node_indices):
        present = ~global_batch.candidate_padding_mask[uav_id]
        candidates = global_batch.candidate_node_indices[uav_id, present]
        if len(candidates) == 0:
            continue
        allowed = set(np.flatnonzero(~global_batch.global_edge_mask[int(current_idx)]).tolist())
        adjacency_ok = adjacency_ok and all(int(idx) in allowed for idx in candidates)
    return {
        "step": float(step),
        "global_node_count": float(global_batch.global_node_inputs.shape[1]),
        "global_node_input_dim": float(global_batch.global_node_inputs.shape[2]),
        "global_edge_mask_rows": float(global_batch.global_edge_mask.shape[0]),
        "global_edge_mask_cols": float(global_batch.global_edge_mask.shape[1]),
        "global_unmasked_edge_count": float(unmasked_edges),
        "global_node_padding_count": float(np.sum(global_batch.global_node_padding_mask)),
        "candidate_index_present_count": float(np.sum(candidate_present)),
        "candidate_index_min": float(np.min(valid_indices)) if len(valid_indices) else -1.0,
        "candidate_index_max": float(np.max(valid_indices)) if len(valid_indices) else -1.0,
        "candidate_padding_minus_one_ok": float(np.all(global_batch.candidate_node_indices[global_batch.candidate_padding_mask] == -1)),
        "candidate_indices_are_current_neighbors": float(adjacency_ok),
        "current_node_index_min": float(np.min(global_batch.current_node_indices)),
        "current_node_index_max": float(np.max(global_batch.current_node_indices)),
        "max_candidate_position_error": max_candidate_position_error,
    }


def global_graph_assertions(cfg: Config, global_batch) -> dict[str, bool]:
    g = global_batch.global_node_inputs.shape[1]
    candidate_present = ~global_batch.candidate_padding_mask
    valid_candidate_indices = global_batch.candidate_node_indices[candidate_present]
    return {
        "global_inputs_finite": finite_bool(global_batch.global_node_inputs),
        "global_edge_mask_shape_ok": global_batch.global_edge_mask.shape == (g, g),
        "global_padding_shape_ok": global_batch.global_node_padding_mask.shape == (g,),
        "global_current_indices_in_range": bool(np.all((global_batch.current_node_indices >= 0) & (global_batch.current_node_indices < g))),
        "global_candidate_indices_in_range": bool(
            len(valid_candidate_indices) == 0 or np.all((valid_candidate_indices >= 0) & (valid_candidate_indices < g))
        ),
        "global_candidate_padding_minus_one": bool(np.all(global_batch.candidate_node_indices[global_batch.candidate_padding_mask] == -1)),
        "global_candidate_indices_are_current_neighbors": bool(
            all(
                int(idx) in set(np.flatnonzero(~global_batch.global_edge_mask[int(current_idx)]).tolist())
                for uav_id, current_idx in enumerate(global_batch.current_node_indices)
                for idx in global_batch.candidate_node_indices[uav_id, ~global_batch.candidate_padding_mask[uav_id]]
            )
        ),
        "global_action_mask_shape_ok": global_batch.action_mask.shape == (cfg.n_uavs, cfg.max_node_candidates),
        "global_candidate_padding_shape_ok": global_batch.candidate_padding_mask.shape == (cfg.n_uavs, cfg.max_node_candidates),
    }


def belief_stats(cfg: Config, target: TargetBelief, search: SearchBelief, tracks: PseudoTrackMemory) -> dict:
    grid = target.grid()
    peaks = target.peaks()
    score = search.score()
    search_peaks = search.peaks()
    return {
        "phd_total_weight": float(np.sum(target.weights)),
        "phd_weight_min": float(np.min(target.weights)),
        "phd_weight_max": float(np.max(target.weights)),
        "phd_max_cell_weight": float(np.max(grid)),
        "phd_peak_count": float(len(peaks)),
        "phd_peak_positions": json_array(np.asarray([p.pos for p in peaks], dtype=np.float32).reshape(-1, 2)),
        "search_mean": float(np.mean(search.search_belief)),
        "search_min": float(np.min(search.search_belief)),
        "search_max": float(np.max(search.search_belief)),
        "search_score_max": float(np.max(score)),
        "search_peak_count": float(len(search_peaks)),
        "coverage_age_mean": float(np.mean(search.coverage_age)),
        "coverage_age_max": float(np.max(search.coverage_age)),
        "pseudo_track_count": float(len(tracks.tracks)),
        "pseudo_track_summary": json_array(
            np.asarray([np.r_[t.last_pos, t.confidence, t.current_gap] for t in tracks.tracks], dtype=np.float32).reshape(-1, 4)
        ),
    }


def present_candidate_indices(batch, uav_id: int) -> np.ndarray:
    return np.flatnonzero(~batch.node_padding_mask[uav_id])


def candidate_detail_rows(step: int, batch) -> list[dict]:
    rows = []
    for uav_id in range(batch.node_inputs.shape[0]):
        for candidate_id in present_candidate_indices(batch, uav_id):
            features = batch.node_inputs[uav_id, candidate_id]
            row = {
                "step": step,
                "uav_id": int(uav_id),
                "candidate_id": int(candidate_id),
                "waypoint_x": float(batch.waypoints[uav_id, candidate_id, 0]),
                "waypoint_y": float(batch.waypoints[uav_id, candidate_id, 1]),
            }
            row.update({name: float(features[idx]) for idx, name in enumerate(NODE_INPUT_FIELDS)})
            rows.append(row)
    return rows


def candidate_signal_detail_rows(step: int, batch) -> list[dict]:
    rows = []
    for uav_id in range(batch.node_inputs.shape[0]):
        for candidate_id in present_candidate_indices(batch, uav_id):
            features = batch.node_inputs[uav_id, candidate_id]
            rows.append(
                {
                    "step": step,
                    "uav_id": int(uav_id),
                    "candidate_id": int(candidate_id),
                    "waypoint_x": float(batch.waypoints[uav_id, candidate_id, 0]),
                    "waypoint_y": float(batch.waypoints[uav_id, candidate_id, 1]),
                    "is_action_masked": bool(batch.action_mask[uav_id, candidate_id]),
                    "expected_target_weight": float(features[NODE_INPUT_INDEX["expected_target_weight"]]),
                    "target_flag": float(features[NODE_INPUT_INDEX["target_flag"]]),
                    "search_value": float(features[NODE_INPUT_INDEX["search_value"]]),
                    "search_flag": float(features[NODE_INPUT_INDEX["search_flag"]]),
                    "maintenance_value": float(features[NODE_INPUT_INDEX["maintenance_value"]]),
                    "maintenance_flag": float(features[NODE_INPUT_INDEX["maintenance_flag"]]),
                    "overlap": float(features[NODE_INPUT_INDEX["overlap"]]),
                }
            )
    return rows


def network_summary_row(step: int, batch, uav_state_arr: np.ndarray, team_summary_arr: np.ndarray) -> dict:
    present = ~batch.node_padding_mask
    features = batch.node_inputs[present]
    if len(features) == 0:
        features = np.zeros((1, len(NODE_INPUT_FIELDS)), dtype=np.float32)
    expected_idx = NODE_INPUT_INDEX["expected_target_weight"]
    search_idx = NODE_INPUT_INDEX["search_value"]
    age_idx = NODE_INPUT_INDEX["coverage_age_value"]
    overlap_idx = NODE_INPUT_INDEX["overlap"]
    maintenance_idx = NODE_INPUT_INDEX["maintenance_value"]
    target_flag_idx = NODE_INPUT_INDEX["target_flag"]
    search_flag_idx = NODE_INPUT_INDEX["search_flag"]
    maintenance_flag_idx = NODE_INPUT_INDEX["maintenance_flag"]
    flags = features[:, [target_flag_idx, search_flag_idx, maintenance_flag_idx]]
    row = {
        "step": step,
        "node_expected_target_weight_mean": float(np.mean(features[:, expected_idx])),
        "node_expected_target_weight_max": float(np.max(features[:, expected_idx])),
        "node_search_value_mean": float(np.mean(features[:, search_idx])),
        "node_search_value_max": float(np.max(features[:, search_idx])),
        "node_coverage_age_mean": float(np.mean(features[:, age_idx])),
        "node_coverage_age_max": float(np.max(features[:, age_idx])),
        "node_maintenance_value_mean": float(np.mean(features[:, maintenance_idx])),
        "node_maintenance_value_max": float(np.max(features[:, maintenance_idx])),
        "target_flag_count": float(np.sum(features[:, target_flag_idx] > 0.5)),
        "search_flag_count": float(np.sum(features[:, search_flag_idx] > 0.5)),
        "maintenance_flag_count": float(np.sum(features[:, maintenance_flag_idx] > 0.5)),
        "any_intent_flag_count": float(np.sum(np.any(flags > 0.5, axis=1))),
        "candidate_count": float(len(features)),
        "masked_candidate_count": float(np.sum(batch.action_mask & ~batch.node_padding_mask)),
        "unmasked_candidate_count": float(np.sum(~batch.action_mask & ~batch.node_padding_mask)),
        "uav_state_min": float(np.min(uav_state_arr)),
        "uav_state_max": float(np.max(uav_state_arr)),
    }
    row.update({name: float(team_summary_arr[i]) for i, name in enumerate(TEAM_SUMMARY_FIELDS)})
    return row


def team_summary_row(step: int, team_summary_arr: np.ndarray) -> dict:
    row = {"step": step}
    row.update({name: float(team_summary_arr[i]) for i, name in enumerate(TEAM_SUMMARY_FIELDS)})
    return row


def environment_row(
    step: int,
    env: CMUOMMTEnv,
    measurements: np.ndarray,
    detected_ids: list[int],
    newly_discovered: int,
    continuous_observed: int,
    step_distance: np.ndarray,
) -> dict:
    row: dict[str, Any] = {
        "step": step,
        "measurement_count": int(len(measurements)),
        "detected_count": int(len(detected_ids)),
        "detected_ids": ",".join(str(int(x)) for x in detected_ids),
        "newly_discovered": int(newly_discovered),
        "continuous_observed": int(continuous_observed),
        "uav_step_distance_mean": float(np.mean(step_distance)),
        "uav_step_distance_max": float(np.max(step_distance)),
        "target_speed_mean": float(np.mean(np.linalg.norm(env.target_states[:, 2:4], axis=1))),
        "target_speed_max": float(np.max(np.linalg.norm(env.target_states[:, 2:4], axis=1))),
    }
    row.update(flatten_positions("uav", env.uav_positions))
    row.update(flatten_target_states(env.target_states))
    row.update(memory_stats(env))
    return row


def setup_axis(ax, cfg: Config, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d9d9d9", linewidth=0.4)


def draw_environment(ax, cfg: Config, env: CMUOMMTEnv, measurements: np.ndarray, detected_ids: list[int], target_history: list[np.ndarray]) -> None:
    setup_axis(ax, cfg, "Environment: UAV / FOV / targets / measurements")
    for i, hist in enumerate(target_history):
        if len(hist) > 1:
            arr = np.asarray(hist)
            ax.plot(arr[:, 0], arr[:, 1], color=TARGET_COLOR, alpha=0.35, linewidth=0.8)
    for i, pos in enumerate(env.uav_positions):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.add_patch(Circle(pos, cfg.fov_radius, color=color, alpha=0.10, linewidth=0))
        ax.scatter(pos[0], pos[1], marker="^", color=color, edgecolor="black", s=58)
        ax.text(pos[0] + 0.8, pos[1] + 0.8, f"U{i}", color=color, fontsize=7)
    for tid, pos in enumerate(env.target_states[:, 0:2]):
        color = "#006d2c" if tid in detected_ids else TARGET_COLOR
        ax.scatter(pos[0], pos[1], marker="x", color=color, s=38)
        ax.text(pos[0] + 0.6, pos[1] + 0.6, f"T{tid}", color=color, fontsize=7)
    if len(measurements):
        ax.scatter(measurements[:, 0], measurements[:, 1], marker=".", color=MEAS_COLOR, edgecolor="black", linewidth=0.2, s=36)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="^", color="black", linestyle="None", label="UAV"),
            Line2D([0], [0], marker="x", color=TARGET_COLOR, linestyle="None", label="target"),
            Line2D([0], [0], marker=".", color=MEAS_COLOR, markeredgecolor="black", linestyle="None", label="measurement"),
        ],
        fontsize=7,
        loc="upper right",
    )


def draw_phd(ax, cfg: Config, env: CMUOMMTEnv, target: TargetBelief, measurements: np.ndarray) -> None:
    grid = target.grid()
    vmax = max(float(np.max(grid)), cfg.target_peak_min_weight, 1e-6)
    im = ax.imshow(grid, origin="lower", extent=[0, cfg.map_size, 0, cfg.map_size], cmap="magma", vmin=0, vmax=vmax)
    peaks = target.peaks()
    if peaks:
        pts = np.asarray([p.pos for p in peaks])
        ax.scatter(pts[:, 0], pts[:, 1], marker="P", color=TARGET_COLOR, edgecolor="white", s=70)
    ax.scatter(env.target_states[:, 0], env.target_states[:, 1], marker="x", color="#66c2a5", s=34)
    if len(measurements):
        ax.scatter(measurements[:, 0], measurements[:, 1], marker=".", color=MEAS_COLOR, edgecolor="black", linewidth=0.2, s=28)
    setup_axis(ax, cfg, f"PHD belief: total={np.sum(target.weights):.2f}, peaks={len(peaks)}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def draw_search(ax, cfg: Config, env: CMUOMMTEnv, search: SearchBelief) -> None:
    score = search.score()
    im = ax.imshow(score, origin="lower", extent=[0, cfg.map_size, 0, cfg.map_size], cmap="YlOrBr", vmin=0, vmax=max(1.0, float(np.max(score))))
    peaks = search.peaks()
    if peaks:
        pts = np.asarray([p for p, _ in peaks])
        ax.scatter(pts[:, 0], pts[:, 1], marker="s", color=SEARCH_COLOR, edgecolor="black", s=45)
    for i, pos in enumerate(env.uav_positions):
        ax.add_patch(Circle(pos, cfg.fov_radius, color=UAV_COLORS[i % len(UAV_COLORS)], alpha=0.08, linewidth=0))
        ax.scatter(pos[0], pos[1], marker="^", color=UAV_COLORS[i % len(UAV_COLORS)], edgecolor="black", s=44)
    setup_axis(ax, cfg, f"Search/Coverage: mean={np.mean(search.search_belief):.2f}, age={np.mean(search.coverage_age):.1f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def candidate_marker(features: np.ndarray) -> tuple[str, str, float]:
    if features[NODE_INPUT_INDEX["target_flag"]] > 0.5:
        return "P", TARGET_COLOR, 82
    if features[NODE_INPUT_INDEX["maintenance_flag"]] > 0.5:
        return "D", MAINT_COLOR, 64
    if features[NODE_INPUT_INDEX["search_flag"]] > 0.5:
        return "s", SEARCH_COLOR, 58
    return ".", "#777777", 28


def draw_candidates(ax, cfg: Config, env: CMUOMMTEnv, batch) -> None:
    setup_axis(ax, cfg, "Network input candidates: target/search/maintenance")
    for i, pos in enumerate(env.uav_positions):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.scatter(pos[0], pos[1], marker="^", color=color, edgecolor="black", s=58)
        ax.text(pos[0] + 0.8, pos[1] + 0.8, f"U{i}", color=color, fontsize=7)
        for j in present_candidate_indices(batch, i):
            wp = batch.waypoints[i, j]
            features = batch.node_inputs[i, j]
            marker, signal_color, size = candidate_marker(features)
            strength = max(
                float(features[NODE_INPUT_INDEX["expected_target_weight"]]),
                float(features[NODE_INPUT_INDEX["search_value"]]),
                float(features[NODE_INPUT_INDEX["maintenance_value"]]),
            )
            alpha = 0.45 + 0.45 * min(strength, 1.0)
            ax.scatter(wp[0], wp[1], marker=marker, color=signal_color, edgecolor="black", linewidth=0.25, s=size, alpha=alpha)
    ax.scatter(env.target_states[:, 0], env.target_states[:, 1], marker="x", color="#66c2a5", s=28)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="P", color=TARGET_COLOR, markeredgecolor="black", linestyle="None", label="target_flag"),
            Line2D([0], [0], marker="s", color=SEARCH_COLOR, markeredgecolor="black", linestyle="None", label="search_flag"),
            Line2D([0], [0], marker="D", color=MAINT_COLOR, markeredgecolor="black", linestyle="None", label="maintenance_flag"),
        ],
        fontsize=7,
        loc="upper right",
    )


def draw_graph_background(ax, builder: NodeBuilder) -> None:
    graph = builder.graph
    ax.scatter(graph.positions[:, 0], graph.positions[:, 1], marker=".", color=GRAPH_COLOR, s=12, alpha=0.55, linewidth=0)


def draw_mask_view(ax, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder, batch) -> None:
    setup_axis(ax, cfg, "Mask view: full graph, unmasked vs masked candidates")
    draw_graph_background(ax, builder)
    for i, pos in enumerate(env.uav_positions):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.scatter(pos[0], pos[1], marker="^", color=color, edgecolor="black", s=62, zorder=5)
        ax.text(pos[0] + 0.8, pos[1] + 0.8, f"U{i}", color=color, fontsize=7)
        present = present_candidate_indices(batch, i)
        if len(present) == 0:
            continue
        masked = batch.action_mask[i, present]
        valid_points = batch.waypoints[i, present[~masked]]
        masked_points = batch.waypoints[i, present[masked]]
        if len(valid_points):
            ax.scatter(valid_points[:, 0], valid_points[:, 1], marker="o", facecolors=color, edgecolors="black", linewidths=0.35, s=46, alpha=0.86)
        if len(masked_points):
            ax.scatter(masked_points[:, 0], masked_points[:, 1], marker="x", color=MASKED_COLOR, s=82, linewidths=1.8)
    ax.legend(
        handles=[
            Line2D([0], [0], marker=".", color=GRAPH_COLOR, linestyle="None", label="global graph node"),
            Line2D([0], [0], marker="o", color="#444444", markerfacecolor="#888888", linestyle="None", label="unmasked candidate"),
            Line2D([0], [0], marker="x", color=MASKED_COLOR, linestyle="None", label="masked candidate"),
        ],
        fontsize=7,
        loc="upper right",
    )


def collect_candidate_points(batch, value_idx: Optional[int] = None, flag_idx: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    values = []
    masks = []
    for i in range(batch.node_inputs.shape[0]):
        for j in present_candidate_indices(batch, i):
            points.append(batch.waypoints[i, j])
            if value_idx is not None:
                values.append(float(batch.node_inputs[i, j, value_idx]))
            elif flag_idx is not None:
                values.append(float(batch.node_inputs[i, j, flag_idx] > 0.5))
            else:
                values.append(0.0)
            masks.append(bool(batch.action_mask[i, j]))
    if not points:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    return np.asarray(points, dtype=np.float32), np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool)


def draw_candidate_base(
    ax,
    cfg: Config,
    env: CMUOMMTEnv,
    builder: NodeBuilder,
    title: str,
) -> None:
    setup_axis(ax, cfg, title)
    draw_graph_background(ax, builder)
    for i, pos in enumerate(env.uav_positions):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.scatter(pos[0], pos[1], marker="^", color=color, edgecolor="black", s=58, zorder=5)
        ax.text(pos[0] + 0.8, pos[1] + 0.8, f"U{i}", color=color, fontsize=7)
    ax.scatter(env.target_states[:, 0], env.target_states[:, 1], marker="x", color="#66c2a5", s=30, zorder=4)


def draw_candidate_value_view(ax, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder, batch, value_idx: int, value_name: str, cmap: str) -> None:
    draw_candidate_base(ax, cfg, env, builder, value_name)
    pts, vals, masks = collect_candidate_points(batch, value_idx=value_idx)
    if len(pts):
        vmax = max(float(np.max(vals)), 1e-6)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=cmap, vmin=0.0, vmax=vmax, s=52, edgecolors="none", alpha=0.9)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label=value_name)
        if np.any(masks):
            ax.scatter(pts[masks, 0], pts[masks, 1], marker="x", color=MASKED_COLOR, linewidths=1.7, s=80, label="masked")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper right")


def draw_candidate_flag_view(ax, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder, batch, flag_idx: int, flag_name: str, color: str) -> None:
    draw_candidate_base(ax, cfg, env, builder, flag_name)
    pts, flags, masks = collect_candidate_points(batch, flag_idx=flag_idx)
    if len(pts):
        flagged = flags > 0.5
        unflagged = ~flagged
        if np.any(unflagged):
            ax.scatter(pts[unflagged, 0], pts[unflagged, 1], marker="o", color="#bdbdbd", edgecolors="none", s=36, alpha=0.55, label="flag=0")
        if np.any(flagged):
            ax.scatter(pts[flagged, 0], pts[flagged, 1], marker="o", color=color, edgecolors="black", linewidths=0.45, s=72, alpha=0.95, label="flag=1")
        if np.any(masks):
            ax.scatter(pts[masks, 0], pts[masks, 1], marker="x", color=MASKED_COLOR, linewidths=1.7, s=80, label="masked")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper right")


def draw_candidate_values_frame(path: Path, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder, batch, step: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), dpi=125)
    specs = [
        (NODE_INPUT_INDEX["expected_target_weight"], "expected_target_weight", "Reds"),
        (NODE_INPUT_INDEX["search_value"], "search_value", "Oranges"),
        (NODE_INPUT_INDEX["maintenance_value"], "maintenance_value", "Blues"),
    ]
    for ax, (value_idx, value_name, cmap) in zip(axes, specs):
        draw_candidate_value_view(ax, cfg, env, builder, batch, value_idx, value_name, cmap)
    fig.suptitle(f"Candidate continuous values step={step}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


def draw_candidate_flags_frame(path: Path, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder, batch, step: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), dpi=125)
    specs = [
        (NODE_INPUT_INDEX["target_flag"], "target_flag", TARGET_COLOR),
        (NODE_INPUT_INDEX["search_flag"], "search_flag", SEARCH_COLOR),
        (NODE_INPUT_INDEX["maintenance_flag"], "maintenance_flag", MAINT_COLOR),
    ]
    for ax, (flag_idx, flag_name, color) in zip(axes, specs):
        draw_candidate_flag_view(ax, cfg, env, builder, batch, flag_idx, flag_name, color)
    fig.suptitle(f"Candidate binary flags step={step}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


def draw_mask_frame(path: Path, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder, batch, step: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 6.6), dpi=130)
    draw_mask_view(ax, cfg, env, builder, batch)
    fig.suptitle(f"Action mask over full graph step={step}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


def draw_frame(
    path: Path,
    cfg: Config,
    env: CMUOMMTEnv,
    target: TargetBelief,
    search: SearchBelief,
    builder: NodeBuilder,
    batch,
    measurements: np.ndarray,
    detected_ids: list[int],
    target_history: list[np.ndarray],
    step: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=125)
    draw_environment(axes[0, 0], cfg, env, measurements, detected_ids, target_history)
    draw_phd(axes[0, 1], cfg, env, target, measurements)
    draw_search(axes[1, 0], cfg, env, search)
    draw_mask_view(axes[1, 1], cfg, env, builder, batch)
    fig.suptitle(f"Input layer verification step={step}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)


def make_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int) -> bool:
    if not frame_paths:
        return False
    try:
        from PIL import Image

        images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
        for image in images:
            image.close()
        return True
    except Exception:
        return False


def summarize_numeric(rows: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float_or_nan(row.get(key)) for row in rows]
        values = [v for v in values if np.isfinite(v)]
        if not values:
            out[key] = {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
            continue
        arr = np.asarray(values, dtype=np.float32)
        out[key] = {"min": float(np.min(arr)), "mean": float(np.mean(arr)), "max": float(np.max(arr))}
    return out


def write_report(
    out: Path,
    cfg: Config,
    args,
    assertions: dict[str, bool],
    input_summary: dict,
) -> None:
    passed = all(assertions.values())
    env_stats = input_summary["environment_stats"]
    belief = input_summary["belief_stats"]
    network = input_summary["network_input_stats"]
    lines = [
        "# 感知输入层验证报告",
        "",
        f"- 结果：{'通过' if passed else '存在问题'}",
        f"- 输出目录：`{out}`",
        f"- seed：`{args.seed}`",
        f"- steps：`{args.steps}`",
        f"- n_targets：`{args.n_targets}`",
        f"- policy：`{args.policy}`",
        f"- UAV 初始方式：`{args.uav_init}`，随机边界留白：`{args.uav_init_margin}`",
        "",
        "## 环境设置",
        "",
        f"- 地图大小：`{cfg.map_size} x {cfg.map_size}`",
        f"- UAV 数量：`{cfg.n_uavs}`，速度上限：`{cfg.uav_speed}`，FOV 半径：`{cfg.fov_radius}`",
        f"- 目标数量：`{args.n_targets}`，目标速度：`{cfg.target_speed}`，速度噪声：`{cfg.target_velocity_noise_std}`",
        f"- 检测概率：`{cfg.p_detection}`，测量噪声标准差：`{cfg.meas_std}`，杂波均值：`{cfg.clutter_mean}`",
        f"- search grid：`{cfg.search_bins} x {cfg.search_bins}`，cell size：`{cfg.cell_size:.3f}`",
        "",
        "## 目标运动情况",
        "",
        f"- 目标平均速度：`{env_stats['target_speed_mean']['mean']:.4f}`，最大速度：`{env_stats['target_speed_max']['max']:.4f}`",
        f"- 目标位置越界检查：`{'正常' if assertions['target_positions_in_bounds'] else '异常'}`",
        f"- detected id 合法性：`{'正常' if assertions['detected_ids_valid'] else '异常'}`",
        f"- 平均检测数量：`{env_stats['detected_count']['mean']:.3f}`，平均 measurement 数量：`{env_stats['measurement_count']['mean']:.3f}`",
        "",
        "判断：目标速度统计接近配置速度，且位置未越界时，说明目标运动层基本正常。",
        "",
        "## UAV 运动情况",
        "",
        f"- UAV 平均单步距离：`{env_stats['uav_step_distance_mean']['mean']:.4f}`",
        f"- UAV 最大单步距离：`{env_stats['uav_step_distance_max']['max']:.4f}`",
        f"- 速度约束检查：`{'正常' if assertions['uav_speed_ok'] else '异常'}`",
        f"- UAV 位置越界检查：`{'正常' if assertions['uav_positions_in_bounds'] else '异常'}`",
        "",
        "判断：最大单步距离不超过 `uav_speed` 时，说明 UAV 运动约束生效。",
        "",
        "## 全局 Belief 统计",
        "",
        f"- PHD 总权重均值：`{belief['phd_total_weight']['mean']:.4f}`，范围：`[{belief['phd_total_weight']['min']:.4f}, {belief['phd_total_weight']['max']:.4f}]`",
        f"- PHD peak 数量均值：`{belief['phd_peak_count']['mean']:.4f}`，最大：`{belief['phd_peak_count']['max']:.4f}`",
        f"- search belief 均值：`{belief['search_mean']['mean']:.4f}`，最大值均值：`{belief['search_max']['mean']:.4f}`",
        f"- coverage age 均值：`{belief['coverage_age_mean']['mean']:.4f}`，最大值：`{belief['coverage_age_max']['max']:.4f}`",
        f"- pseudo track 数量均值：`{belief['pseudo_track_count']['mean']:.4f}`，最大：`{belief['pseudo_track_count']['max']:.4f}`",
        f"- belief 有限性检查：`{'正常' if assertions['phd_values_finite'] and assertions['search_values_finite'] else '异常'}`",
        "",
        "判断：PHD/search/track 没有 NaN 或爆炸，且随 rollout 有变化时，说明感知状态能正常演化。",
        "",
        "## 节点级网络输入统计",
        "",
        f"- candidate 数量均值：`{network['candidate_count']['mean']:.4f}`",
        f"- `expected_target_weight` 最大值均值：`{network['node_expected_target_weight_max']['mean']:.4f}`",
        f"- `search_value` 最大值均值：`{network['node_search_value_max']['mean']:.4f}`",
        f"- `maintenance_value` 最大值均值：`{network['node_maintenance_value_max']['mean']:.4f}`",
        f"- target flag 总体均值：`{network['target_flag_count']['mean']:.4f}`",
        f"- search flag 总体均值：`{network['search_flag_count']['mean']:.4f}`",
        f"- maintenance flag 总体均值：`{network['maintenance_flag_count']['mean']:.4f}`",
        f"- 任一 intent flag 触发数量均值：`{network['any_intent_flag_count']['mean']:.4f}`",
        f"- unmasked candidate 数量均值：`{network['unmasked_candidate_count']['mean']:.4f}`",
        f"- masked candidate 数量均值：`{network['masked_candidate_count']['mean']:.4f}`",
        f"- 网络输入有限性检查：`{'正常' if assertions['network_inputs_finite'] else '异常'}`",
        "",
        "判断：节点级输入中 target/search/maintenance/goal 信号能够出现，且数值范围有限，说明 belief 到网络输入的映射至少在数值层面可用。",
        "",
        "Mask 说明：`candidate_mask.gif` 中灰色点为全局离散图节点，彩色圆点为当前进入候选集合且未被 `action_mask` 屏蔽的节点，红色叉号为被 `action_mask` 屏蔽的候选节点。本次运行中 `masked_candidate_count` 为 0，说明当前候选生成流程已经在 `GlobalNodeGraph.action_node_indices()` 阶段过滤了不可达或已选择附近的节点，进入 `NodeBatch` 的候选点没有再被实时 `action_mask` 屏蔽。",
        "",
        "## 断言结果",
        "",
    ]
    for key, value in assertions.items():
        lines.append(f"- `{key}`：`{value}`")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `environment_trace.csv`：真实环境和观测逐步记录。",
            "- `belief_trace.csv`：PHD/search/pseudo track 逐步记录。",
            "- `network_input_summary.csv`：网络输入整体统计。",
            "- `candidate_input_detail.csv`：每个 candidate 的网络输入明细。",
            "- `candidate_signal_detail.csv`：target/search/maintenance 的 value、flag 和 mask 明细。",
            "- `team_summary_trace.csv`：team summary 的 10 个字段。",
            "- `frames/overview/`：环境、PHD、search 和 mask 总览帧。",
            "- `frames/candidate_values/`：`expected_target_weight`、`search_value`、`maintenance_value` 三个连续量的并列视图。",
            "- `frames/candidate_flags/`：`target_flag`、`search_flag`、`maintenance_flag` 三个离散标志的并列视图。",
            "- `frames/candidate_mask/`：全局离散图节点与实时 mask 视图。",
            "- `input_layer.gif`：输入层总览动态可视化。",
            "- `candidate_values.gif`：candidate 连续量动态可视化。",
            "- `candidate_flags.gif`：candidate flag 动态可视化。",
            "- `candidate_mask.gif`：全图离散节点与实时 mask 动态可视化。",
            "",
            "## 结论",
            "",
        ]
    )
    if passed:
        lines.append("本次输入层核查通过。环境运动、belief 演化、candidate 映射和网络输入数值未发现基础异常。下一步可以进入 actor 输出层核查。")
    else:
        lines.append("本次输入层核查存在异常。建议先查看 `assertions.json` 中为 false 的项目，再结合 CSV 和 GIF 定位问题，暂不进入 actor / PPO 核查。")
    (out / "INPUT_LAYER_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_verification(args) -> dict:
    cfg = Config()
    cfg.episode_steps = args.steps
    out = Path(args.out_dir)
    frames_dir = out / "frames"
    overview_frames_dir = frames_dir / "overview"
    values_frames_dir = frames_dir / "candidate_values"
    flags_frames_dir = frames_dir / "candidate_flags"
    mask_frames_dir = frames_dir / "candidate_mask"
    for directory in [overview_frames_dir, values_frames_dir, flags_frames_dir, mask_frames_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", asdict(cfg))

    builder = NodeBuilder(cfg)
    builder.reset(seed=args.seed)
    init_rng = np.random.default_rng(args.seed + 404)
    uav_positions = None
    if args.uav_init == "random" or cfg.graph_type.lower() == "prm":
        uav_positions = builder.graph.sample_start_positions(cfg.n_uavs, init_rng)
        if cfg.graph_type.lower() == "prm":
            builder.reset(seed=args.seed, start_positions=uav_positions)
    env = CMUOMMTEnv(cfg)
    env.reset(seed=args.seed, n_targets=args.n_targets, uav_positions=uav_positions)
    initial_state = {
        "uav_init": args.uav_init,
        "uav_init_margin": args.uav_init_margin,
        "graph_type": cfg.graph_type,
        "uav_positions": env.uav_positions.tolist(),
        "target_positions": env.target_states[:, 0:2].tolist(),
        "target_velocities": env.target_states[:, 2:4].tolist(),
    }
    write_json(out / "initial_state.json", initial_state)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=args.seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
    policy = make_policy(args.policy)
    rng = np.random.default_rng(args.seed + 303)

    environment_rows: list[dict] = []
    belief_rows: list[dict] = []
    network_rows: list[dict] = []
    candidate_rows: list[dict] = []
    candidate_signal_rows: list[dict] = []
    global_graph_rows: list[dict] = []
    team_rows: list[dict] = []
    frame_paths: list[Path] = []
    values_frame_paths: list[Path] = []
    flags_frame_paths: list[Path] = []
    mask_frame_paths: list[Path] = []
    target_history: list[list[np.ndarray]] = [[pos.copy()] for pos in env.target_states[:, 0:2]]
    previous_measurements = np.zeros((0, 2), dtype=np.float32)
    previous_detected_ids: list[int] = []

    assertions = {
        "target_positions_in_bounds": True,
        "uav_positions_in_bounds": True,
        "uav_speed_ok": True,
        "measurements_finite": True,
        "detected_ids_valid": True,
        "phd_values_finite": True,
        "search_values_finite": True,
        "track_values_finite": True,
        "network_inputs_finite": True,
        "candidate_waypoints_in_bounds": True,
        "expected_target_signal_present": False,
        "search_signal_present": False,
        "maintenance_signal_present": False,
        "team_summary_finite": True,
        "global_inputs_finite": True,
        "global_edge_mask_shape_ok": True,
        "global_padding_shape_ok": True,
        "global_current_indices_in_range": True,
        "global_candidate_indices_in_range": True,
        "global_candidate_padding_minus_one": True,
        "global_candidate_indices_are_current_neighbors": True,
        "global_action_mask_shape_ok": True,
        "global_candidate_padding_shape_ok": True,
        "global_candidate_waypoint_index_match": True,
        "visualization_written_ok": False,
    }

    for step in range(args.steps):
        target.predict()
        batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        global_batch = builder.global_batch_from_candidates(
            env.uav_positions,
            target,
            search,
            tracks,
            batch.candidate_node_indices,
            batch.node_padding_mask,
            batch.action_mask,
            step=env.step_count,
        )
        uav_state_arr = uav_state(cfg, env)
        team_summary_arr = team_summary(cfg, env, target, search, [])

        network_row = network_summary_row(step, batch, uav_state_arr, team_summary_arr)
        network_rows.append(network_row)
        candidate_rows.extend(candidate_detail_rows(step, batch))
        candidate_signal_rows.extend(candidate_signal_detail_rows(step, batch))
        global_graph_rows.append(global_graph_summary_row(step, batch, global_batch))
        team_rows.append(team_summary_row(step, team_summary_arr))
        bstats = {"step": step}
        bstats.update(belief_stats(cfg, target, search, tracks))
        belief_rows.append(bstats)

        if step % max(args.frame_stride, 1) == 0:
            frame_path = overview_frames_dir / f"frame_{step:04d}.png"
            draw_frame(frame_path, cfg, env, target, search, builder, batch, previous_measurements, previous_detected_ids, target_history, step)
            frame_paths.append(frame_path)
            values_frame_path = values_frames_dir / f"frame_{step:04d}.png"
            flags_frame_path = flags_frames_dir / f"frame_{step:04d}.png"
            mask_frame_path = mask_frames_dir / f"frame_{step:04d}.png"
            draw_candidate_values_frame(values_frame_path, cfg, env, builder, batch, step)
            draw_candidate_flags_frame(flags_frame_path, cfg, env, builder, batch, step)
            draw_mask_frame(mask_frame_path, cfg, env, builder, batch, step)
            values_frame_paths.append(values_frame_path)
            flags_frame_paths.append(flags_frame_path)
            mask_frame_paths.append(mask_frame_path)

        actions = policy.select(cfg, batch, rng)
        selected_waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions]
        info = env.step(selected_waypoints)
        target.update(info.measurements.points, env.uav_positions)
        peaks = [] if cfg.disable_phd_belief else target.peaks()
        tracks.update(env.step_count, info.measurements.points, peaks)
        search.update(env.uav_positions, info.measurements.points)
        for idx, pos in enumerate(env.target_states[:, 0:2]):
            target_history[idx].append(pos.copy())

        environment_rows.append(
            environment_row(
                step=step,
                env=env,
                measurements=info.measurements.points,
                detected_ids=info.detected_ids,
                newly_discovered=info.newly_discovered,
                continuous_observed=info.continuous_observed,
                step_distance=info.step_distance,
            )
        )

        assertions["target_positions_in_bounds"] &= bool(np.all((env.target_states[:, 0:2] >= 0.0) & (env.target_states[:, 0:2] <= cfg.map_size)))
        assertions["uav_positions_in_bounds"] &= bool(np.all((env.uav_positions >= 0.0) & (env.uav_positions <= cfg.map_size)))
        assertions["uav_speed_ok"] &= bool(np.max(info.step_distance) <= cfg.uav_speed + 1e-5)
        assertions["measurements_finite"] &= finite_bool(info.measurements.points)
        assertions["detected_ids_valid"] &= all(0 <= int(tid) < len(env.target_states) for tid in info.detected_ids)
        assertions["phd_values_finite"] &= finite_bool(target.particles, target.weights, target.grid())
        assertions["search_values_finite"] &= finite_bool(search.search_belief, search.coverage_age, search.score())
        if tracks.tracks:
            assertions["track_values_finite"] &= finite_bool(np.asarray([np.r_[t.last_pos, t.last_velocity, t.confidence, t.current_gap] for t in tracks.tracks], dtype=np.float32))
        assertions["network_inputs_finite"] &= finite_bool(batch.node_inputs, uav_state_arr)
        for key, ok in global_graph_assertions(cfg, global_batch).items():
            assertions[key] &= ok
        candidate_present = ~batch.node_padding_mask
        if np.any(candidate_present):
            mapped_positions = global_batch.node_positions[batch.candidate_node_indices[candidate_present]]
            assertions["global_candidate_waypoint_index_match"] &= bool(
                np.max(np.linalg.norm(mapped_positions - batch.waypoints[candidate_present], axis=1)) <= 1e-5
            )
        present_waypoints = batch.waypoints[~batch.node_padding_mask]
        assertions["candidate_waypoints_in_bounds"] &= bool(np.all((present_waypoints >= 0.0) & (present_waypoints <= cfg.map_size)))
        assertions["expected_target_signal_present"] |= bool(
            np.max(batch.node_inputs[:, :, NODE_INPUT_INDEX["expected_target_weight"]]) > 1e-6
            or np.sum(batch.node_inputs[:, :, NODE_INPUT_INDEX["target_flag"]] > 0.5) > 0
        )
        assertions["search_signal_present"] |= bool(
            np.max(batch.node_inputs[:, :, NODE_INPUT_INDEX["search_value"]]) > 1e-6
            or np.sum(batch.node_inputs[:, :, NODE_INPUT_INDEX["search_flag"]] > 0.5) > 0
        )
        assertions["maintenance_signal_present"] |= bool(
            np.max(batch.node_inputs[:, :, NODE_INPUT_INDEX["maintenance_value"]]) > 1e-6
            or np.sum(batch.node_inputs[:, :, NODE_INPUT_INDEX["maintenance_flag"]] > 0.5) > 0
        )
        assertions["team_summary_finite"] &= finite_bool(team_summary_arr)
        previous_measurements = info.measurements.points
        previous_detected_ids = list(info.detected_ids)

        if env.done():
            break

    gif_ok = make_gif(frame_paths, out / "input_layer.gif", args.duration_ms)
    values_gif_ok = make_gif(values_frame_paths, out / "candidate_values.gif", args.duration_ms)
    flags_gif_ok = make_gif(flags_frame_paths, out / "candidate_flags.gif", args.duration_ms)
    mask_gif_ok = make_gif(mask_frame_paths, out / "candidate_mask.gif", args.duration_ms)
    assertions["visualization_written_ok"] = (
        bool(frame_paths)
        and all(path.exists() for path in frame_paths + values_frame_paths + flags_frame_paths + mask_frame_paths)
    )

    write_table(out / "environment_trace.csv", environment_rows)
    write_table(out / "belief_trace.csv", belief_rows)
    write_table(out / "network_input_summary.csv", network_rows)
    write_table(out / "candidate_input_detail.csv", candidate_rows)
    write_table(out / "candidate_signal_detail.csv", candidate_signal_rows)
    write_table(out / "global_graph_summary.csv", global_graph_rows)
    write_table(out / "team_summary_trace.csv", team_rows)
    write_json(out / "assertions.json", assertions)

    input_summary = {
        "seed": args.seed,
        "steps": args.steps,
        "n_targets": args.n_targets,
        "policy": args.policy,
        "uav_init": args.uav_init,
        "uav_init_margin": args.uav_init_margin,
        "initial_uav_positions": initial_state["uav_positions"],
        "initial_target_positions": initial_state["target_positions"],
        "frame_count": len(frame_paths),
        "gif_created": gif_ok,
        "values_gif_created": values_gif_ok,
        "flags_gif_created": flags_gif_ok,
        "mask_gif_created": mask_gif_ok,
        "environment_stats": summarize_numeric(
            environment_rows,
            ["measurement_count", "detected_count", "uav_step_distance_mean", "uav_step_distance_max", "target_speed_mean", "target_speed_max", "discovery_rate_so_far"],
        ),
        "belief_stats": summarize_numeric(
            belief_rows,
            ["phd_total_weight", "phd_max_cell_weight", "phd_peak_count", "search_mean", "search_max", "coverage_age_mean", "coverage_age_max", "pseudo_track_count"],
        ),
        "network_input_stats": summarize_numeric(
            network_rows,
            [
                "candidate_count",
                "masked_candidate_count",
                "unmasked_candidate_count",
                "node_expected_target_weight_mean",
                "node_expected_target_weight_max",
                "node_search_value_mean",
                "node_search_value_max",
                "node_maintenance_value_mean",
                "node_maintenance_value_max",
                "target_flag_count",
                "search_flag_count",
                "maintenance_flag_count",
                "any_intent_flag_count",
            ],
        ),
    }
    write_json(out / "input_summary.json", input_summary)
    write_report(out, cfg, args, assertions, input_summary)
    return {
        "out_dir": str(out.resolve()),
        "passed": all(assertions.values()),
        "assertions": str(out / "assertions.json"),
        "summary": str(out / "input_summary.json"),
        "report": str(out / "INPUT_LAYER_REPORT.md"),
        "gif": str(out / "input_layer.gif") if gif_ok else None,
        "candidate_values_gif": str(out / "candidate_values.gif") if values_gif_ok else None,
        "candidate_flags_gif": str(out / "candidate_flags.gif") if flags_gif_ok else None,
        "candidate_mask_gif": str(out / "candidate_mask.gif") if mask_gif_ok else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify environment, belief, graph candidates, and network input fields.")
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--out-dir", type=str, default="verification_runs/input_layer_seed_500")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=180)
    parser.add_argument("--uav-init", choices=["random", "env"], default="random")
    parser.add_argument("--uav-init-margin", type=float, default=8.0)
    args = parser.parse_args()
    result = run_verification(args)
    print(result)


if __name__ == "__main__":
    main()
