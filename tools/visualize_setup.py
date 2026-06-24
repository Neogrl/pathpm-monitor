import argparse
from dataclasses import asdict
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

from baselines import HeuristicBaseline
from config import Config
from environment import CMUOMMTEnv
from nodes import NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from utils import write_json


COLORS = {
    "uav": ["#1f77b4", "#2ca02c", "#9467bd"],
    "target": "#d62728",
    "search": "#ff7f0e",
    "maintenance": "#17becf",
    "local": "#7f7f7f",
    "selected": "#111111",
}


def setup_axes(ax, cfg: Config, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.add_patch(Rectangle((0, 0), cfg.map_size, cfg.map_size, fill=False, linewidth=1.5, edgecolor="#222222"))


def plot_fov(ax, positions: np.ndarray, cfg: Config, alpha: float = 0.12) -> None:
    for i, pos in enumerate(positions):
        color = COLORS["uav"][i % len(COLORS["uav"])]
        ax.add_patch(Circle(pos, cfg.fov_radius, color=color, alpha=alpha, linewidth=0))


def init_stack(cfg: Config, seed: int, n_targets: int):
    env = CMUOMMTEnv(cfg)
    env.reset(seed=seed, n_targets=n_targets)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
    return env, target, search, tracks


def draw_initial_scene(cfg: Config, out: Path, seed: int, n_targets: int) -> dict:
    env, _, _, _ = init_stack(cfg, seed, n_targets)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    setup_axes(ax, cfg, f"Initial scene, seed={seed}")
    plot_fov(ax, env.uav_positions, cfg)
    for i, pos in enumerate(env.uav_positions):
        color = COLORS["uav"][i % len(COLORS["uav"])]
        ax.scatter(pos[0], pos[1], s=70, color=color, marker="^", label=f"UAV {i}")
        ax.text(pos[0] + 1.2, pos[1] + 1.2, f"U{i}", fontsize=9, color=color)
    for j, state in enumerate(env.target_states):
        ax.scatter(state[0], state[1], s=42, color=COLORS["target"], marker="x")
        ax.arrow(state[0], state[1], state[2] * 3.5, state[3] * 3.5, color=COLORS["target"], width=0.12, head_width=1.4, length_includes_head=True)
        ax.text(state[0] + 1.0, state[1] + 1.0, f"T{j}", fontsize=8, color=COLORS["target"])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "initial_scene.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "initial_uav_positions": env.uav_positions.tolist(),
        "initial_target_states_xyvxvy": env.target_states.tolist(),
        "image": str(path),
    }


def rollout_heuristic(cfg: Config, seed: int, n_targets: int, steps: int):
    env, target, search, tracks = init_stack(cfg, seed, n_targets)
    builder = NodeBuilder(cfg)
    builder.reset()
    baseline = HeuristicBaseline()
    rng = np.random.default_rng(seed + 303)
    uav_traj = [env.uav_positions.copy()]
    target_traj = [env.target_states[:, 0:2].copy()]
    detections = []
    selected_waypoints_history = []
    last_batch = None
    for _ in range(steps):
        target.predict()
        batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        actions = baseline.select(cfg, batch, rng)
        selected_waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions]
        info = env.step(selected_waypoints)
        target.update(info.measurements.points, env.uav_positions)
        tracks.update(env.step_count, info.measurements.points, target.peaks())
        search.update(env.uav_positions, info.measurements.points)
        uav_traj.append(env.uav_positions.copy())
        target_traj.append(env.target_states[:, 0:2].copy())
        detections.append(len(info.detected_ids))
        selected_waypoints_history.append(selected_waypoints.copy())
        last_batch = batch
    return {
        "env": env,
        "target": target,
        "search": search,
        "tracks": tracks,
        "last_batch": last_batch,
        "uav_traj": np.asarray(uav_traj),
        "target_traj": np.asarray(target_traj),
        "detections": detections,
        "selected_waypoints": np.asarray(selected_waypoints_history),
    }


def draw_motion_rollout(cfg: Config, out: Path, seed: int, n_targets: int, steps: int) -> dict:
    data = rollout_heuristic(cfg, seed, n_targets, steps)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    setup_axes(ax, cfg, f"Heuristic rollout: UAV and target motion, {steps} steps")
    uav_traj = data["uav_traj"]
    target_traj = data["target_traj"]
    for i in range(cfg.n_uavs):
        color = COLORS["uav"][i % len(COLORS["uav"])]
        ax.plot(uav_traj[:, i, 0], uav_traj[:, i, 1], color=color, linewidth=2.0, label=f"UAV {i}")
        ax.scatter(uav_traj[0, i, 0], uav_traj[0, i, 1], color=color, marker="^", s=55)
        ax.scatter(uav_traj[-1, i, 0], uav_traj[-1, i, 1], color=color, marker="o", s=45)
    for j in range(n_targets):
        ax.plot(target_traj[:, j, 0], target_traj[:, j, 1], color=COLORS["target"], alpha=0.55, linewidth=1.3)
        ax.scatter(target_traj[0, j, 0], target_traj[0, j, 1], color=COLORS["target"], marker="x", s=38)
        ax.text(target_traj[0, j, 0] + 0.8, target_traj[0, j, 1] + 0.8, f"T{j}", fontsize=8, color=COLORS["target"])
    plot_fov(ax, uav_traj[-1], cfg, alpha=0.10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "motion_rollout.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "image": str(path),
        "total_detections": int(np.sum(data["detections"])),
        "per_step_detection_counts": data["detections"],
        "final_uav_positions": data["env"].uav_positions.tolist(),
        "final_target_positions": data["env"].target_states[:, 0:2].tolist(),
    }


def draw_candidate_nodes(cfg: Config, out: Path, seed: int, n_targets: int, warmup_steps: int) -> dict:
    data = rollout_heuristic(cfg, seed, n_targets, warmup_steps)
    env = data["env"]
    target = data["target"]
    search = data["search"]
    tracks = data["tracks"]
    builder = NodeBuilder(cfg)
    builder.reset()
    batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
    fig, axes = plt.subplots(1, cfg.n_uavs, figsize=(5.4 * cfg.n_uavs, 5.2), dpi=150)
    if cfg.n_uavs == 1:
        axes = [axes]
    summary = []
    for i, ax in enumerate(axes):
        setup_axes(ax, cfg, f"Candidate waypoints for UAV {i}, after {warmup_steps} steps")
        ax.imshow(search.search_belief, origin="lower", extent=[0, cfg.map_size, 0, cfg.map_size], cmap="YlOrBr", alpha=0.35, vmin=0, vmax=1)
        plot_fov(ax, env.uav_positions, cfg, alpha=0.08)
        ax.scatter(env.uav_positions[:, 0], env.uav_positions[:, 1], color="#555555", marker="^", s=40)
        ax.scatter(env.uav_positions[i, 0], env.uav_positions[i, 1], color=COLORS["uav"][i % len(COLORS["uav"])], marker="^", s=85)
        ax.scatter(env.target_states[:, 0], env.target_states[:, 1], color="#cc6677", marker="x", s=32, alpha=0.65)
        valid = ~batch.node_padding_mask[i] & ~batch.action_mask[i]
        points = batch.waypoints[i, valid]
        features = batch.node_inputs[i, valid]
        type_counts = {"target": 0, "search": 0, "maintenance": 0, "graph_neighbor": 0}
        for point, feat in zip(points, features):
            if feat[12] > 0.5:
                color, marker, key = COLORS["target"], "P", "target"
            elif feat[14] > 0.5:
                color, marker, key = COLORS["maintenance"], "D", "maintenance"
            elif feat[13] > 0.5:
                color, marker, key = COLORS["search"], "s", "search"
            else:
                color, marker, key = COLORS["local"], ".", "graph_neighbor"
            type_counts[key] += 1
            ax.scatter(point[0], point[1], color=color, marker=marker, s=46, edgecolor="black", linewidth=0.25)
            ax.plot([env.uav_positions[i, 0], point[0]], [env.uav_positions[i, 1], point[1]], color=color, alpha=0.22, linewidth=0.8)
        summary.append({"uav": i, "valid_candidates": int(np.sum(valid)), **type_counts})
    handles = [
        plt.Line2D([0], [0], marker="x", color="#cc6677", label="true target", markersize=7, linestyle="None"),
        plt.Line2D([0], [0], marker="P", color="w", markerfacecolor=COLORS["target"], markeredgecolor="black", label="target signal", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["search"], markeredgecolor="black", label="search signal", markersize=8),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["maintenance"], markeredgecolor="black", label="maintenance signal", markersize=7),
        plt.Line2D([0], [0], marker=".", color="w", markerfacecolor=COLORS["local"], markeredgecolor="black", label="graph neighbor", markersize=11),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "candidate_waypoints.png"
    fig.savefig(path)
    plt.close(fig)
    return {"image": str(path), "candidate_summary": summary}


def write_profile(cfg: Config, out: Path, seed: int, n_targets: int, steps: int, initial: dict, motion: dict, candidates: dict) -> None:
    profile = {
        "seed": seed,
        "n_targets": n_targets,
        "rollout_steps": steps,
        "map": {
            "map_size": cfg.map_size,
            "search_bins": cfg.search_bins,
            "cell_size": cfg.cell_size,
        },
        "uav": {
            "n_uavs": cfg.n_uavs,
            "speed_per_step": cfg.uav_speed,
            "fov_radius": cfg.fov_radius,
            "initial_positions": initial["initial_uav_positions"],
        },
        "target": {
            "n_targets": n_targets,
            "speed_per_step": cfg.target_speed,
            "velocity_noise_std": cfg.target_velocity_noise_std,
            "init_margin": cfg.target_init_margin,
            "initial_states_xyvxvy": initial["initial_target_states_xyvxvy"],
        },
        "measurement": {
            "p_detection": cfg.p_detection,
            "meas_std": cfg.meas_std,
            "clutter_mean": cfg.clutter_mean,
        },
        "candidate_waypoints": {
            "graph_node_spacing": cfg.graph_node_spacing,
            "action_k_neighbors": cfg.action_k_neighbors,
            "max_node_candidates": cfg.max_node_candidates,
            "k_neighbors": cfg.k_neighbors,
            "candidate_summary": candidates["candidate_summary"],
        },
        "motion_rollout": motion,
        "config": asdict(cfg),
    }
    write_json(out / "setup_summary.json", profile)
    md = [
        "# CMUOMMT planToGo V1 Setup Visualization",
        "",
        f"- map_size: {cfg.map_size} x {cfg.map_size}",
        f"- search grid: {cfg.search_bins} x {cfg.search_bins}, cell_size={cfg.cell_size:.2f}",
        f"- UAVs: {cfg.n_uavs}, speed={cfg.uav_speed}/step, FOV radius={cfg.fov_radius}",
        f"- targets in this visualization: {n_targets}, speed={cfg.target_speed}/step, velocity_noise_std={cfg.target_velocity_noise_std}",
        f"- target initialization margin: [{cfg.target_init_margin}, {cfg.map_size - cfg.target_init_margin}] on both axes",
        f"- detection: p_detection={cfg.p_detection}, meas_std={cfg.meas_std}, clutter_mean={cfg.clutter_mean}",
        f"- candidate action: global graph spacing={cfg.graph_node_spacing}, action k-neighbors={cfg.action_k_neighbors}, max nodes={cfg.max_node_candidates}, model kNN={cfg.k_neighbors}",
        "",
        "Images:",
        "",
        "- initial_scene.png: initial UAV/target positions and FOV.",
        "- motion_rollout.png: heuristic rollout trajectories.",
        "- candidate_waypoints.png: k-nearest global graph action nodes and target/search/maintenance signals.",
    ]
    (out / "setup_profile.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--candidate-warmup-steps", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/setup_visualization")
    args = parser.parse_args()

    cfg = Config()
    cfg.episode_steps = args.steps
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    initial = draw_initial_scene(cfg, out, args.seed, args.n_targets)
    motion = draw_motion_rollout(cfg, out, args.seed, args.n_targets, args.steps)
    candidates = draw_candidate_nodes(cfg, out, args.seed, args.n_targets, args.candidate_warmup_steps)
    write_profile(cfg, out, args.seed, args.n_targets, args.steps, initial, motion, candidates)
    print(
        {
            "out_dir": str(out.resolve()),
            "images": ["initial_scene.png", "motion_rollout.png", "candidate_waypoints.png"],
            "summary": "setup_summary.json",
            "profile": "setup_profile.md",
        }
    )


if __name__ == "__main__":
    main()
