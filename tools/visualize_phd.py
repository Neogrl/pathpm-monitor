import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from baselines import HeuristicBaseline
from config import Config
from environment import CMUOMMTEnv
from nodes import NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from utils import write_json


def setup_axis(ax, cfg: Config, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def summarize_phd(cfg: Config, target: TargetBelief, measurements: np.ndarray, detected_count: int, step: int) -> dict:
    grid = target.grid()
    peaks = target.peaks()
    peak_weights = [p.weight for p in peaks]
    total = float(np.sum(target.weights))
    normalized = target.weights / max(total, 1e-12)
    ess = float(1.0 / np.sum(normalized ** 2))
    return {
        "step": step,
        "phd_total_weight": total,
        "phd_max_cell_weight": float(np.max(grid)),
        "phd_mean_cell_weight": float(np.mean(grid)),
        "phd_peak_count": len(peaks),
        "phd_peak_max_weight": float(max(peak_weights)) if peak_weights else 0.0,
        "phd_peak_mean_weight": float(np.mean(peak_weights)) if peak_weights else 0.0,
        "measurement_count": int(len(measurements)),
        "detected_true_target_count": int(detected_count),
        "effective_sample_size": ess,
    }


def record_snapshot(cfg: Config, env: CMUOMMTEnv, target: TargetBelief, measurements: np.ndarray, detected_count: int, step: int) -> dict:
    return {
        "step": step,
        "grid": target.grid().copy(),
        "peaks": [(p.pos.copy(), p.weight) for p in target.peaks()],
        "uav_positions": env.uav_positions.copy(),
        "target_positions": env.target_states[:, 0:2].copy(),
        "measurements": measurements.copy(),
        "summary": summarize_phd(cfg, target, measurements, detected_count, step),
    }


def run_rollout(cfg: Config, seed: int, n_targets: int, steps: int, frames: list[int]) -> tuple[list[dict], list[dict]]:
    env = CMUOMMTEnv(cfg)
    env.reset(seed=seed, n_targets=n_targets)
    target = TargetBelief(cfg, eval_mode=True)
    target.reset(seed=seed + 101)
    search = SearchBelief(cfg)
    tracks = PseudoTrackMemory(cfg)
    builder = NodeBuilder(cfg)
    builder.reset()
    baseline = HeuristicBaseline()
    rng = np.random.default_rng(seed + 303)

    frame_set = set(frames)
    snapshots = []
    trace = []
    empty_measurements = np.zeros((0, 2), dtype=np.float32)
    if 0 in frame_set:
        snapshots.append(record_snapshot(cfg, env, target, empty_measurements, 0, 0))
    trace.append(summarize_phd(cfg, target, empty_measurements, 0, 0))

    for step in range(1, steps + 1):
        target.predict()
        batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        actions = baseline.select(cfg, batch, rng)
        waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions]
        info = env.step(waypoints)
        target.update(info.measurements.points, env.uav_positions)
        peaks = target.peaks()
        tracks.update(env.step_count, info.measurements.points, peaks)
        search.update(env.uav_positions, info.measurements.points)
        summary = summarize_phd(cfg, target, info.measurements.points, len(info.detected_ids), step)
        trace.append(summary)
        if step in frame_set:
            snapshots.append(record_snapshot(cfg, env, target, info.measurements.points, len(info.detected_ids), step))
    return snapshots, trace


def draw_snapshots(cfg: Config, snapshots: list[dict], out: Path) -> None:
    if not snapshots:
        return
    cols = min(3, len(snapshots))
    rows = int(np.ceil(len(snapshots) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols + 0.8, 4.8 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(np.max(s["grid"])) for s in snapshots)
    vmax = max(vmax, cfg.target_peak_min_weight)
    for ax, snap in zip(axes, snapshots):
        summary = snap["summary"]
        setup_axis(
            ax,
            cfg,
            f"step {snap['step']} | total={summary['phd_total_weight']:.2f} max={summary['phd_max_cell_weight']:.3f} peaks={summary['phd_peak_count']}",
        )
        im = ax.imshow(
            snap["grid"],
            origin="lower",
            extent=[0, cfg.map_size, 0, cfg.map_size],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            alpha=0.88,
        )
        for pos in snap["uav_positions"]:
            ax.add_patch(Circle(pos, cfg.fov_radius, color="#66c2a5", alpha=0.13, linewidth=0))
        ax.scatter(snap["uav_positions"][:, 0], snap["uav_positions"][:, 1], marker="^", color="#1b9e77", s=48, label="UAV")
        ax.scatter(snap["target_positions"][:, 0], snap["target_positions"][:, 1], marker="x", color="#80cdc1", s=36, label="true target")
        if len(snap["measurements"]):
            ax.scatter(snap["measurements"][:, 0], snap["measurements"][:, 1], marker=".", color="#ffffbf", s=34, label="measurement")
        if snap["peaks"]:
            peak_pos = np.asarray([p for p, _ in snap["peaks"]])
            peak_w = np.asarray([w for _, w in snap["peaks"]])
            ax.scatter(peak_pos[:, 0], peak_pos[:, 1], marker="P", color="#3288bd", edgecolor="white", s=70, label="PHD peak")
            for pos, weight in zip(peak_pos, peak_w):
                ax.text(pos[0] + 0.8, pos[1] + 0.8, f"{weight:.2f}", color="white", fontsize=7)
        ax.text(
            1.5,
            3.0,
            f"threshold={cfg.target_peak_min_weight:.2f}\nmeas={summary['measurement_count']} detected={summary['detected_true_target_count']}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 3},
        )
    for ax in axes[len(snapshots):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.08, top=0.94, wspace=0.18, hspace=0.24)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    cbar_ax = fig.add_axes([0.905, 0.20, 0.018, 0.62])
    fig.colorbar(im, cax=cbar_ax, label="PHD cell weight")
    fig.savefig(out / "phd_snapshots.png")
    plt.close(fig)


def draw_trace(trace: list[dict], out: Path, cfg: Config) -> None:
    steps = np.asarray([row["step"] for row in trace])
    total = np.asarray([row["phd_total_weight"] for row in trace])
    max_cell = np.asarray([row["phd_max_cell_weight"] for row in trace])
    peak_count = np.asarray([row["phd_peak_count"] for row in trace])
    measurements = np.asarray([row["measurement_count"] for row in trace])
    detected = np.asarray([row["detected_true_target_count"] for row in trace])

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), dpi=150, sharex=True)
    axes[0].plot(steps, total, color="#762a83", linewidth=2)
    axes[0].axhline(cfg.phd_prior_count, color="#999999", linestyle="--", linewidth=1, label="prior count")
    axes[0].set_ylabel("total weight")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps, max_cell, color="#d7191c", linewidth=2)
    axes[1].axhline(cfg.target_peak_min_weight, color="#2b83ba", linestyle="--", linewidth=1, label="peak threshold")
    axes[1].set_ylabel("max cell")
    axes[1].legend(fontsize=8)

    axes[2].step(steps, peak_count, where="mid", color="#1b9e77", linewidth=2)
    axes[2].set_ylabel("peak count")

    axes[3].plot(steps, measurements, color="#fdae61", linewidth=1.8, label="measurements")
    axes[3].plot(steps, detected, color="#2c7bb6", linewidth=1.8, label="true detections")
    axes[3].set_ylabel("count")
    axes[3].set_xlabel("step")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle("PHD belief numeric trace")
    fig.tight_layout()
    fig.savefig(out / "phd_trace.png")
    plt.close(fig)


def write_trace(trace: list[dict], out: Path) -> None:
    if not trace:
        return
    with (out / "phd_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    write_json(out / "phd_trace_summary.json", {"first": trace[0], "last": trace[-1], "max_peak_count": max(row["phd_peak_count"] for row in trace)})


def parse_frames(text: str) -> list[int]:
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--frames", type=str, default="0,5,10,20,40,60")
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/phd_visualization")
    args = parser.parse_args()

    cfg = Config()
    cfg.episode_steps = args.steps
    frames = [f for f in parse_frames(args.frames) if 0 <= f <= args.steps]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshots, trace = run_rollout(cfg, args.seed, args.n_targets, args.steps, frames)
    draw_snapshots(cfg, snapshots, out)
    draw_trace(trace, out, cfg)
    write_trace(trace, out)
    write_json(
        out / "phd_visualization_config.json",
        {
            "seed": args.seed,
            "n_targets": args.n_targets,
            "steps": args.steps,
            "frames": frames,
            "target_peak_min_weight": cfg.target_peak_min_weight,
            "cell_size": cfg.cell_size,
            "search_bins": cfg.search_bins,
            "n_particles_eval": cfg.n_particles_eval,
        },
    )
    print(
        {
            "out_dir": str(out.resolve()),
            "images": ["phd_snapshots.png", "phd_trace.png"],
            "csv": "phd_trace.csv",
            "summary": "phd_trace_summary.json",
        }
    )


if __name__ == "__main__":
    main()
