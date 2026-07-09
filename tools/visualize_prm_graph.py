import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from config import Config
from environment import CMUOMMTEnv
from nodes import NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from utils import write_json


UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#17becf", "#8c564b"]


def setup_axes(ax, cfg: Config, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(-1, cfg.map_size + 1)
    ax.set_ylim(-1, cfg.map_size + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.add_patch(Rectangle((0, 0), cfg.map_size, cfg.map_size, fill=False, linewidth=1.4, edgecolor="#222222"))


def edge_pairs(edge_indices: np.ndarray) -> list[tuple[int, int]]:
    pairs = set()
    for i, row in enumerate(edge_indices):
        for j in row:
            j = int(j)
            if i == j:
                continue
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def edge_length_stats(positions: np.ndarray, pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    lengths = np.asarray([np.linalg.norm(positions[a] - positions[b]) for a, b in pairs], dtype=np.float32)
    return {"min": float(np.min(lengths)), "mean": float(np.mean(lengths)), "max": float(np.max(lengths))}


def is_boundary_node(positions: np.ndarray, cfg: Config) -> np.ndarray:
    eps = 1e-5
    return (
        (np.abs(positions[:, 0]) <= eps)
        | (np.abs(positions[:, 0] - cfg.map_size) <= eps)
        | (np.abs(positions[:, 1]) <= eps)
        | (np.abs(positions[:, 1] - cfg.map_size) <= eps)
    )


def draw_obstacles(ax, obstacles: np.ndarray) -> None:
    for i, (x, y, radius) in enumerate(obstacles):
        ax.add_patch(Circle((x, y), radius, facecolor="#444444", edgecolor="#111111", alpha=0.20, linewidth=1.0, zorder=0))
        ax.text(x, y, f"O{i}", fontsize=8, ha="center", va="center", color="#111111", zorder=6)


def graph_connectivity(edge_indices: np.ndarray, passable: np.ndarray) -> dict:
    passable_indices = np.flatnonzero(passable)
    if len(passable_indices) == 0:
        return {"component_count": 0, "largest_component_ratio": 0.0, "isolated_count": 0}
    passable_set = set(int(i) for i in passable_indices)
    adj = {int(i): set() for i in passable_indices}
    for i in passable_indices:
        i = int(i)
        for j in edge_indices[i]:
            j = int(j)
            if j != i and j in passable_set:
                adj[i].add(j)
                adj[j].add(i)
    seen = set()
    sizes = []
    for node in passable_indices:
        node = int(node)
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        sizes.append(size)
    return {
        "component_count": int(len(sizes)),
        "largest_component_ratio": float(max(sizes) / max(len(passable_indices), 1)),
        "isolated_count": int(sum(1 for node in passable_indices if len(adj[int(node)]) == 0)),
    }


def draw_prm_graph(path: Path, cfg: Config, builder: NodeBuilder, env: CMUOMMTEnv, batch) -> None:
    graph = builder.graph
    positions = graph.positions
    pairs = edge_pairs(graph.edge_indices)
    boundary = is_boundary_node(positions, cfg)
    passable = graph.node_passable

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    setup_axes(ax, cfg, f"PRM graph, seed={builder.graph._geometry_seed}, nodes={graph.n_nodes}")
    draw_obstacles(ax, graph.obstacles)
    for a, b in pairs:
        pa, pb = positions[a], positions[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#9a9a9a", alpha=0.22, linewidth=0.55, zorder=1)
    regular = passable & ~boundary
    if np.any(regular):
        ax.scatter(positions[regular, 0], positions[regular, 1], s=10, color="#4d4d4d", alpha=0.75, zorder=2, label="PRM nodes")
    if np.any(boundary & passable):
        ax.scatter(positions[boundary & passable, 0], positions[boundary & passable, 1], s=18, color="#111111", alpha=0.9, zorder=3, label="boundary nodes")
    if np.any(~passable):
        ax.scatter(positions[~passable, 0], positions[~passable, 1], s=14, color="#cc6677", marker="x", alpha=0.85, zorder=4, label="masked obstacle nodes")

    for i, pos in enumerate(env.uav_positions):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.add_patch(Circle(pos, cfg.fov_radius, color=color, alpha=0.08, linewidth=0, zorder=0))
        ax.scatter(pos[0], pos[1], s=95, marker="^", color=color, edgecolor="black", linewidth=0.5, zorder=5)
        ax.text(pos[0] + 1.0, pos[1] + 1.0, f"UAV {i}", fontsize=8, color=color, zorder=6)
        valid = ~batch.node_padding_mask[i] & ~batch.action_mask[i]
        cand = batch.waypoints[i, valid]
        for point in cand:
            ax.plot([pos[0], point[0]], [pos[1], point[1]], color=color, alpha=0.65, linewidth=1.2, zorder=4)
        if len(cand):
            ax.scatter(cand[:, 0], cand[:, 1], s=40, facecolors="white", edgecolors=color, linewidth=1.2, zorder=5)

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_candidate_panels(path: Path, cfg: Config, builder: NodeBuilder, env: CMUOMMTEnv, batch) -> None:
    cols = min(cfg.n_uavs, 3)
    rows = int(np.ceil(cfg.n_uavs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    positions = builder.graph.positions
    pairs = edge_pairs(builder.graph.edge_indices)
    passable = builder.graph.node_passable
    for i, ax in enumerate(axes):
        if i >= cfg.n_uavs:
            ax.axis("off")
            continue
        color = UAV_COLORS[i % len(UAV_COLORS)]
        setup_axes(ax, cfg, f"UAV {i} current node and candidates")
        draw_obstacles(ax, builder.graph.obstacles)
        for a, b in pairs:
            pa, pb = positions[a], positions[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#b5b5b5", alpha=0.12, linewidth=0.45, zorder=1)
        ax.scatter(positions[passable, 0], positions[passable, 1], s=6, color="#666666", alpha=0.45, zorder=2)
        if np.any(~passable):
            ax.scatter(positions[~passable, 0], positions[~passable, 1], s=10, color="#cc6677", marker="x", alpha=0.65, zorder=3)
        ax.add_patch(Circle(env.uav_positions[i], cfg.fov_radius, color=color, alpha=0.08, linewidth=0, zorder=0))
        ax.scatter(env.uav_positions[:, 0], env.uav_positions[:, 1], s=30, marker="^", color="#888888", zorder=3)
        ax.scatter(env.uav_positions[i, 0], env.uav_positions[i, 1], s=90, marker="^", color=color, edgecolor="black", linewidth=0.5, zorder=5)
        valid = ~batch.node_padding_mask[i] & ~batch.action_mask[i]
        cand = batch.waypoints[i, valid]
        for j, point in enumerate(cand):
            ax.plot([env.uav_positions[i, 0], point[0]], [env.uav_positions[i, 1], point[1]], color=color, alpha=0.78, linewidth=1.5)
            ax.scatter(point[0], point[1], s=62, facecolors="white", edgecolors=color, linewidth=1.4, zorder=5)
            ax.text(point[0] + 0.6, point[1] + 0.6, str(j), fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_edge_histogram(path: Path, cfg: Config, builder: NodeBuilder) -> None:
    pairs = edge_pairs(builder.graph.edge_indices)
    lengths = np.asarray([np.linalg.norm(builder.graph.positions[a] - builder.graph.positions[b]) for a, b in pairs], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(lengths, bins=28, color="#4c78a8", edgecolor="white")
    ax.axvline(cfg.uav_speed, color="#d62728", linewidth=1.5, label=f"uav_speed={cfg.uav_speed:g}")
    ax.set_xlabel("edge length")
    ax.set_ylabel("count")
    ax.set_title("PRM edge length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_cfg(args) -> Config:
    cfg = Config()
    cfg.graph_type = "prm"
    cfg.n_uavs = args.n_uavs
    if args.prm_random_nodes is not None:
        cfg.prm_random_nodes = args.prm_random_nodes
    if args.prm_sampling is not None:
        cfg.prm_sampling = args.prm_sampling
    if args.prm_jitter_ratio is not None:
        cfg.prm_jitter_ratio = args.prm_jitter_ratio
    if args.prm_boundary_points_per_side is not None:
        cfg.prm_boundary_points_per_side = args.prm_boundary_points_per_side
    if args.prm_edge_radius is not None:
        cfg.prm_edge_radius = args.prm_edge_radius
    if args.prm_min_node_distance is not None:
        cfg.prm_min_node_distance = args.prm_min_node_distance
    if args.k_neighbors is not None:
        cfg.k_neighbors = args.k_neighbors
    if args.action_k_neighbors is not None:
        cfg.action_k_neighbors = args.action_k_neighbors
    if args.uav_speed is not None:
        cfg.uav_speed = args.uav_speed
    if args.no_boundary:
        cfg.prm_include_boundary = False
    if args.obstacles:
        cfg.obstacles_enabled = True
    if args.obstacle_count is not None:
        cfg.obstacle_count = args.obstacle_count
    if args.obstacle_radius_min is not None:
        cfg.obstacle_radius_min = args.obstacle_radius_min
    if args.obstacle_radius_max is not None:
        cfg.obstacle_radius_max = args.obstacle_radius_max
    if args.obstacle_margin is not None:
        cfg.obstacle_margin = args.obstacle_margin
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the episode-level PRM graph and local action candidates.")
    parser.add_argument("--out-dir", type=str, default="prm_visualizations/default")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--n-uavs", type=int, default=5)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--prm-random-nodes", type=int, default=None)
    parser.add_argument("--prm-sampling", type=str, choices=["stratified", "uniform"], default=None)
    parser.add_argument("--prm-jitter-ratio", type=float, default=None)
    parser.add_argument("--prm-boundary-points-per-side", type=int, default=None)
    parser.add_argument("--prm-edge-radius", type=float, default=None)
    parser.add_argument("--prm-min-node-distance", type=float, default=None)
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--action-k-neighbors", type=int, default=None)
    parser.add_argument("--uav-speed", type=float, default=None)
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--obstacle-count", type=int, default=None)
    parser.add_argument("--obstacle-radius-min", type=float, default=None)
    parser.add_argument("--obstacle-radius-max", type=float, default=None)
    parser.add_argument("--obstacle-margin", type=float, default=None)
    args = parser.parse_args()

    cfg = build_cfg(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    builder = NodeBuilder(cfg)
    builder.reset(seed=args.seed)
    start_rng = np.random.default_rng(args.seed + 909)
    start_positions = builder.graph.sample_start_positions(cfg.n_uavs, start_rng)
    builder.reset(seed=args.seed, start_positions=start_positions)

    env = CMUOMMTEnv(cfg)
    env.reset(seed=args.seed, n_targets=args.n_targets, uav_positions=start_positions)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=args.seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
    batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)

    graph_path = out / "prm_graph.png"
    candidates_path = out / "prm_candidates.png"
    edge_hist_path = out / "prm_edge_lengths.png"
    draw_prm_graph(graph_path, cfg, builder, env, batch)
    draw_candidate_panels(candidates_path, cfg, builder, env, batch)
    draw_edge_histogram(edge_hist_path, cfg, builder)

    pairs = edge_pairs(builder.graph.edge_indices)
    boundary = is_boundary_node(builder.graph.positions, cfg)
    connectivity = graph_connectivity(builder.graph.edge_indices, builder.graph.node_passable)
    summary = {
        "graph_type": cfg.graph_type,
        "prm_sampling": cfg.prm_sampling,
        "seed": args.seed,
        "node_count": int(builder.graph.n_nodes),
        "passable_node_count": int(np.sum(builder.graph.node_passable)),
        "masked_node_count": int(np.sum(~builder.graph.node_passable)),
        "random_node_count": int(cfg.prm_random_nodes),
        "boundary_node_count": int(np.sum(boundary)),
        "edge_count_undirected": int(len(pairs)),
        "edge_length": edge_length_stats(builder.graph.positions, pairs),
        "connectivity": connectivity,
        "obstacles_enabled": bool(cfg.obstacles_enabled),
        "obstacles": builder.graph.obstacles.tolist(),
        "uav_speed": float(cfg.uav_speed),
        "prm_edge_radius": float(cfg.prm_edge_radius),
        "k_neighbors": int(cfg.k_neighbors),
        "action_k_neighbors": int(cfg.action_k_neighbors),
        "valid_candidates_per_uav": batch.valid_counts.astype(int).tolist(),
        "start_positions": env.uav_positions.tolist(),
        "start_node_indices": [int(builder.graph.nearest_node_index(pos)) for pos in env.uav_positions],
        "images": {
            "graph": str(graph_path),
            "candidates": str(candidates_path),
            "edge_lengths": str(edge_hist_path),
        },
    }
    write_json(out / "prm_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
