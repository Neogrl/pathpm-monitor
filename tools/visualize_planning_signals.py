import argparse
import csv
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from baselines import HeuristicBaseline, RandomBaseline
from config import Config
from environment import CMUOMMTEnv
from nodes import NODE_INPUT_INDEX, NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from utils import write_json


UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd"]
TARGET_COLOR = "#d62728"
SEARCH_COLOR = "#ff7f0e"
MAINT_COLOR = "#17becf"
TRUE_TARGET_COLOR = "#66c2a5"
MEAS_COLOR = "#ffffbf"


def setup_map_axis(ax, cfg: Config, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d8d8d8", linewidth=0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def entity_legend_handles() -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=6, label="UAV position"),
        Line2D([0], [0], marker="o", color="#999999", linestyle="None", markersize=8, alpha=0.25, label="UAV FOV"),
        Line2D([0], [0], marker="x", color=TRUE_TARGET_COLOR, linestyle="None", markersize=6, label="true target"),
        Line2D([0], [0], marker=".", color=MEAS_COLOR, markeredgecolor="black", linestyle="None", markersize=8, label="measurement"),
    ]
    return handles


def signal_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="P", color=TARGET_COLOR, markeredgecolor="black", linestyle="None", markersize=8, label="PHD target peak"),
        Line2D([0], [0], marker="s", color=SEARCH_COLOR, markeredgecolor="black", linestyle="None", markersize=7, label="search peak"),
        Line2D([0], [0], marker="D", color=MAINT_COLOR, markeredgecolor="black", linestyle="None", markersize=7, label="maintenance peak"),
        Line2D([0], [0], marker="*", color="#333333", markeredgecolor="black", linestyle="None", markersize=10, label="selected action"),
        Line2D([0], [0], marker=".", color="#555555", linestyle="None", markersize=8, label="valid action candidate"),
    ]


def draw_entities(ax, cfg: Config, env: CMUOMMTEnv, measurements: Optional[np.ndarray] = None, show_fov: bool = True) -> None:
    if show_fov:
        for i, pos in enumerate(env.uav_positions):
            ax.add_patch(Circle(pos, cfg.fov_radius, color=UAV_COLORS[i % len(UAV_COLORS)], alpha=0.12, linewidth=0))
    for i, pos in enumerate(env.uav_positions):
        ax.scatter(pos[0], pos[1], marker="^", color=UAV_COLORS[i % len(UAV_COLORS)], s=54, edgecolor="black", linewidth=0.25)
        ax.text(pos[0] + 0.8, pos[1] + 0.8, f"U{i}", color=UAV_COLORS[i % len(UAV_COLORS)], fontsize=7)
    ax.scatter(env.target_states[:, 0], env.target_states[:, 1], marker="x", color=TRUE_TARGET_COLOR, s=34, label="true target")
    if measurements is not None and len(measurements):
        ax.scatter(measurements[:, 0], measurements[:, 1], marker=".", color=MEAS_COLOR, edgecolor="black", linewidth=0.2, s=28, label="measurement")


def candidate_stats(batch, actions: np.ndarray) -> dict:
    valid = ~batch.node_padding_mask & ~batch.action_mask
    features = batch.node_inputs
    selected = features[np.arange(features.shape[0]), actions]
    distance_idx = NODE_INPUT_INDEX["candidate_distance_norm"]
    age_idx = NODE_INPUT_INDEX["coverage_age_value"]
    overlap_idx = NODE_INPUT_INDEX["overlap"]
    valid_distance = features[:, :, distance_idx][valid]
    valid_age = features[:, :, age_idx][valid]
    valid_overlap = features[:, :, overlap_idx][valid]
    return {
        "valid_candidates_mean": float(np.mean(np.sum(valid, axis=1))),
        "candidate_distance_norm_mean": float(np.mean(valid_distance)) if len(valid_distance) else 0.0,
        "candidate_coverage_age_value_mean": float(np.mean(valid_age)) if len(valid_age) else 0.0,
        "candidate_overlap_mean": float(np.mean(valid_overlap)) if len(valid_overlap) else 0.0,
        "selected_candidate_distance_norm": float(np.mean(selected[:, distance_idx])) if len(selected) else 0.0,
        "selected_coverage_age_value": float(np.mean(selected[:, age_idx])) if len(selected) else 0.0,
        "selected_overlap": float(np.mean(selected[:, overlap_idx])) if len(selected) else 0.0,
    }


def graph_stats(builder: NodeBuilder) -> dict:
    graph = builder.graph
    return {
        "graph_target_value_max": float(np.max(graph.target_value)),
        "graph_search_value_max": float(np.max(graph.search_value)),
        "graph_maintenance_value_max": float(np.max(graph.maintenance_value)),
        "graph_target_flag_count": int(np.sum(graph.target_flag)),
        "graph_search_flag_count": int(np.sum(graph.search_flag)),
        "graph_maintenance_flag_count": int(np.sum(graph.maintenance_flag)),
        "graph_target_value_node_count": int(np.sum(graph.target_value >= 0.05)),
    }


def phd_stats(target: TargetBelief) -> dict:
    grid = target.grid()
    peaks = target.peaks()
    return {
        "phd_total_weight": float(np.sum(target.weights)),
        "phd_max_cell_weight": float(np.max(grid)),
        "phd_peak_count": int(len(peaks)),
        "phd_peak_max_weight": float(max([p.weight for p in peaks])) if peaks else 0.0,
    }


def draw_phd(ax, cfg: Config, env: CMUOMMTEnv, target: TargetBelief, measurements: np.ndarray, step: int) -> None:
    grid = target.grid()
    smooth = target.smooth_grid(grid)
    peaks = target.peaks()
    vmax = max(float(np.max(grid)), cfg.target_peak_min_weight, 1e-6)
    im = ax.imshow(grid, origin="lower", extent=[0, cfg.map_size, 0, cfg.map_size], cmap="magma", vmin=0, vmax=vmax)
    draw_entities(ax, cfg, env, measurements=measurements)
    if peaks:
        pts = np.asarray([p.pos for p in peaks])
        weights = [p.weight for p in peaks]
        ax.scatter(pts[:, 0], pts[:, 1], marker="P", color=TARGET_COLOR, edgecolor="white", s=75, label="target hard flag")
        for pos, weight in zip(pts, weights):
            ax.text(pos[0] + 0.8, pos[1] + 0.8, f"{weight:.2f}", color="white", fontsize=7)
    setup_map_axis(ax, cfg, f"PHD target belief | step {step}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="PHD cell weight")
    ax.text(
        1.0,
        3.0,
        f"total={np.sum(target.weights):.2f}\nraw max={np.max(grid):.3f}\nsmooth max={np.max(smooth):.3f}\nflags={len(peaks)}\nth={cfg.target_peak_min_weight:.2f}",
        color="white",
        fontsize=7,
        bbox={"facecolor": "black", "alpha": 0.4, "pad": 3},
    )
    ax.legend(handles=entity_legend_handles()[:4] + [signal_legend_handles()[0]], loc="upper right", fontsize=6, framealpha=0.72)


def draw_search(ax, cfg: Config, env: CMUOMMTEnv, search: SearchBelief) -> None:
    score = search.score()
    im = ax.imshow(score, origin="lower", extent=[0, cfg.map_size, 0, cfg.map_size], cmap="YlOrBr", vmin=0, vmax=max(1.0, float(np.max(score))))
    peaks = search.peaks()
    if peaks:
        pts = np.asarray([p for p, _ in peaks])
        vals = [v for _, v in peaks]
        ax.scatter(pts[:, 0], pts[:, 1], marker="s", color=SEARCH_COLOR, edgecolor="black", s=46, label="search flag")
        for pos, val in zip(pts, vals):
            ax.text(pos[0] + 0.7, pos[1] + 0.7, f"{val:.2f}", color="#6b3d00", fontsize=7)
    draw_entities(ax, cfg, env, measurements=None)
    setup_map_axis(ax, cfg, "Search score = search_belief * (1 + age)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="search score")
    ax.text(
        1.0,
        3.0,
        f"mean={np.mean(score):.3f}\nmax={np.max(score):.3f}\npeaks={len(peaks)}",
        color="black",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.65, "pad": 3},
    )
    ax.legend(handles=entity_legend_handles()[:3] + [signal_legend_handles()[1]], loc="upper right", fontsize=6, framealpha=0.72)


def draw_graph_signals(ax, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder) -> None:
    graph = builder.graph
    ax.scatter(graph.positions[:, 0], graph.positions[:, 1], s=14, color="#9e9e9e", alpha=0.45, edgecolor="none", label="global graph node")
    if np.any(graph.target_flag):
        pts = graph.positions[graph.target_flag]
        ax.scatter(pts[:, 0], pts[:, 1], marker="P", color=TARGET_COLOR, edgecolor="black", s=90, label="target flag")
    if np.any(graph.search_flag):
        pts = graph.positions[graph.search_flag]
        ax.scatter(pts[:, 0], pts[:, 1], marker="s", color=SEARCH_COLOR, edgecolor="black", s=55, label="search flag")
    if np.any(graph.maintenance_flag):
        pts = graph.positions[graph.maintenance_flag]
        ax.scatter(pts[:, 0], pts[:, 1], marker="D", color=MAINT_COLOR, edgecolor="black", s=55, label="maintenance flag")
    draw_entities(ax, cfg, env, measurements=None)
    setup_map_axis(ax, cfg, "Global graph values and hard flags (diagnostic only)")
    legend = [
        Line2D([0], [0], marker="o", color="#9e9e9e", linestyle="None", markersize=6, label="global graph node"),
    ]
    ax.legend(handles=legend + signal_legend_handles()[:3] + entity_legend_handles()[:3], loc="upper right", fontsize=5.8, framealpha=0.75)


def draw_candidates(ax, cfg: Config, env: CMUOMMTEnv, batch, actions: np.ndarray) -> None:
    draw_entities(ax, cfg, env, measurements=None)
    age_idx = NODE_INPUT_INDEX["coverage_age_value"]
    overlap_idx = NODE_INPUT_INDEX["overlap"]
    for i in range(cfg.n_uavs):
        valid = ~batch.node_padding_mask[i] & ~batch.action_mask[i]
        points = batch.waypoints[i, valid]
        feats = batch.node_inputs[i, valid]
        color = UAV_COLORS[i % len(UAV_COLORS)]
        sizes = 28 + 70 * np.clip(feats[:, age_idx], 0.0, 1.0) if len(feats) else 28
        ax.scatter(points[:, 0], points[:, 1], marker=".", color=color, s=sizes, alpha=0.82)
        for point, feat in zip(points, feats):
            if feat[overlap_idx] >= 0.25:
                ax.scatter(point[0], point[1], marker="o", facecolor="none", edgecolor="#111111", s=72, linewidth=0.8)
        chosen = batch.waypoints[i, actions[i]]
        ax.plot([env.uav_positions[i, 0], chosen[0]], [env.uav_positions[i, 1], chosen[1]], color=color, linewidth=2.2)
        ax.scatter(chosen[0], chosen[1], marker="*", color=color, edgecolor="black", s=110)
    setup_map_axis(ax, cfg, "Actor candidates: size=coverage age, ring=overlap")
    ax.legend(handles=entity_legend_handles()[:3] + signal_legend_handles()[3:], loc="upper right", fontsize=6, framealpha=0.75)


def draw_frame(
    out_path: Path,
    cfg: Config,
    env: CMUOMMTEnv,
    target: TargetBelief,
    search: SearchBelief,
    builder: NodeBuilder,
    batch,
    actions: np.ndarray,
    measurements: np.ndarray,
    detected_count: int,
    step: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=130)
    draw_phd(axes[0, 0], cfg, env, target, measurements, step)
    draw_search(axes[0, 1], cfg, env, search)
    draw_graph_signals(axes[1, 0], cfg, env, builder)
    draw_candidates(axes[1, 1], cfg, env, batch, actions)
    stats = {**phd_stats(target), **graph_stats(builder), **candidate_stats(batch, actions)}
    fig.suptitle(
        f"Planning/RL observable signals, step={step} | measurements={len(measurements)} detected={detected_count} "
        f"| candidates={stats['valid_candidates_mean']:.1f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def draw_discrete_graph_overview(out_path: Path, cfg: Config, env: CMUOMMTEnv, builder: NodeBuilder) -> None:
    graph = builder.graph
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=140)
    ax = axes[0]
    for i, src in enumerate(graph.positions):
        for dst_idx in graph.edge_indices[i, 1:]:
            dst = graph.positions[int(dst_idx)]
            ax.plot([src[0], dst[0]], [src[1], dst[1]], color="#bbbbbb", linewidth=0.35, alpha=0.12)
    ax.scatter(graph.positions[:, 0], graph.positions[:, 1], s=12, color="#333333", alpha=0.75)
    draw_entities(ax, cfg, env, measurements=None)
    setup_map_axis(ax, cfg, "Discrete global graph: nodes and kNN edges")
    ax.text(
        1.0,
        3.0,
        f"nodes={graph.n_nodes}\nnode spacing={cfg.graph_node_spacing:.1f}\nedge k={cfg.k_neighbors}\naction k={cfg.action_k_neighbors}",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.76, "pad": 4},
    )
    ax.legend(handles=entity_legend_handles()[:3], loc="upper right", fontsize=7, framealpha=0.75)

    ax = axes[1]
    ax.scatter(graph.positions[:, 0], graph.positions[:, 1], s=14, color="#9e9e9e", alpha=0.45)
    for i, pos in enumerate(env.uav_positions):
        action_idx = graph.action_node_indices(pos, [])
        pts = graph.positions[action_idx]
        ax.scatter(pts[:, 0], pts[:, 1], marker=".", color=UAV_COLORS[i % len(UAV_COLORS)], s=36, alpha=0.88)
        ax.scatter(pos[0], pos[1], marker="^", color=UAV_COLORS[i % len(UAV_COLORS)], edgecolor="black", s=70)
    if np.any(graph.target_flag):
        pts = graph.positions[graph.target_flag]
        ax.scatter(pts[:, 0], pts[:, 1], marker="P", color=TARGET_COLOR, edgecolor="black", s=85)
    if np.any(graph.search_flag):
        pts = graph.positions[graph.search_flag]
        ax.scatter(pts[:, 0], pts[:, 1], marker="s", color=SEARCH_COLOR, edgecolor="black", s=55)
    if np.any(graph.maintenance_flag):
        pts = graph.positions[graph.maintenance_flag]
        ax.scatter(pts[:, 0], pts[:, 1], marker="D", color=MAINT_COLOR, edgecolor="black", s=55)
    ax.scatter(env.target_states[:, 0], env.target_states[:, 1], marker="x", color=TRUE_TARGET_COLOR, s=40)
    setup_map_axis(ax, cfg, "Final graph values and local action neighborhoods")
    graph_handles = [
        Line2D([0], [0], marker="o", color="#9e9e9e", linestyle="None", markersize=6, label="global graph node"),
        Line2D([0], [0], marker=".", color="#555555", linestyle="None", markersize=8, label="local action neighborhood"),
    ]
    ax.legend(handles=graph_handles + signal_legend_handles()[:3] + entity_legend_handles()[:3], loc="upper right", fontsize=6.4, framealpha=0.75)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_signal_counts(out_path: Path, rows: list[dict]) -> None:
    steps = np.asarray([row["step"] for row in rows], dtype=np.float32)
    def series(key: str) -> list[float]:
        return [float(row.get(key, 0.0)) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), dpi=140, sharex=True)

    ax = axes[0]
    ax.plot(steps, series("valid_candidates_mean"), color="#333333", label="valid candidates / UAV")
    ax.plot(steps, series("phd_peak_count"), color=TARGET_COLOR, label="PHD peak count")
    ax.set_ylabel("count")
    ax.set_title("Candidate availability and PHD peaks")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(steps, series("candidate_coverage_age_value_mean"), color=SEARCH_COLOR, label="candidate coverage age mean")
    ax.plot(steps, series("selected_coverage_age_value"), color="#111111", linewidth=1.7, label="selected coverage age")
    ax.set_ylabel("age value")
    ax.set_title("Coverage age signal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    ax.plot(steps, series("candidate_overlap_mean"), color="#777777", label="candidate overlap mean")
    ax.plot(steps, series("selected_overlap"), color="#111111", linewidth=1.7, label="selected overlap")
    ax.set_ylabel("overlap")
    ax.set_title("Overlap signal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[3]
    ax.plot(steps, series("candidate_distance_norm_mean"), color="#777777", label="candidate distance mean")
    ax.plot(steps, series("selected_candidate_distance_norm"), color="#111111", linewidth=1.7, label="selected distance")
    ax.set_xlabel("step")
    ax.set_ylabel("distance / map")
    ax.set_title("Movement distance signal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    if not frame_paths:
        return False
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    for img in images:
        img.close()
    return True


def run_visualization(cfg: Config, args) -> dict:
    out = Path(args.out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    builder = NodeBuilder(cfg)
    builder.reset(seed=args.seed)
    start_rng = np.random.default_rng(args.seed + 909)
    uav_positions = builder.graph.sample_start_positions(cfg.n_uavs, start_rng)
    builder.reset(seed=args.seed, start_positions=uav_positions)
    env = CMUOMMTEnv(cfg)
    env.reset(seed=args.seed, n_targets=args.n_targets, uav_positions=uav_positions)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=args.seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
    if args.policy == "random":
        policy = RandomBaseline()
    else:
        policy = HeuristicBaseline()
    rng = np.random.default_rng(args.seed + 303)

    rows = []
    frame_paths = []
    measurements = np.zeros((0, 2), dtype=np.float32)
    detected_count = 0
    for step in range(args.frames):
        target.predict()
        batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        actions = policy.select(cfg, batch, rng)
        frame_path = frames_dir / f"frame_{step:03d}.png"
        draw_frame(frame_path, cfg, env, target, search, builder, batch, actions, measurements, detected_count, step)
        frame_paths.append(frame_path)
        row = {
            "step": step,
            "selected_actions": ";".join(str(int(a)) for a in actions),
            "measurement_count_prev": int(len(measurements)),
            "detected_count_prev": int(detected_count),
            **phd_stats(target),
            **graph_stats(builder),
            **candidate_stats(batch, actions),
        }
        rows.append(row)

        waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions]
        info = env.step(waypoints)
        measurements = info.measurements.points
        detected_count = len(info.detected_ids)
        target.update(measurements, env.uav_positions)
        tracks.update(env.step_count, measurements, target.peaks())
        search.update(env.uav_positions, measurements)

    with (out / "planning_signal_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts_path = out / "signal_counts.png"
    graph_path = out / "discrete_graph_overview.png"
    draw_signal_counts(counts_path, rows)
    draw_discrete_graph_overview(graph_path, cfg, env, builder)
    summary = {
        "seed": args.seed,
        "n_targets": args.n_targets,
        "frames": args.frames,
        "policy": args.policy,
        "frame_dir": str(frames_dir),
        "gif": str(out / "planning_signals.gif"),
        "csv": str(out / "planning_signal_trace.csv"),
        "signal_counts": str(counts_path),
        "discrete_graph_overview": str(graph_path),
        "first": rows[0],
        "last": rows[-1],
        "max_phd_peak_count": max(row["phd_peak_count"] for row in rows),
        "max_graph_target_value_node_count": max(row["graph_target_value_node_count"] for row in rows),
        "max_graph_target_flags": max(row["graph_target_flag_count"] for row in rows),
        "max_graph_search_flags": max(row["graph_search_flag_count"] for row in rows),
        "max_graph_maintenance_flags": max(row["graph_maintenance_flag_count"] for row in rows),
    }
    gif_ok = make_gif(frame_paths, out / "planning_signals.gif", args.duration_ms)
    summary["gif_created"] = gif_ok
    write_json(out / "planning_signal_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/planning_signals")
    args = parser.parse_args()
    cfg = Config()
    cfg.episode_steps = max(args.frames, cfg.episode_steps)
    summary = run_visualization(cfg, args)
    print(summary)


if __name__ == "__main__":
    main()
