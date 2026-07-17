from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from config import Config
from metrics import ospa_distance
from target_belief import TargetBelief


def load_replay(path: Path) -> list[dict[str, np.ndarray | float]]:
    data = np.load(path)
    offsets = data["measurement_offsets"]
    points = data["measurement_points"]
    frames = []
    for index in range(len(data["durations"])):
        frames.append(
            {
                "duration": float(data["durations"][index]),
                "uav_positions": data["uav_positions"][index].astype(np.float32),
                "true_positions": data["true_positions"][index].astype(np.float32),
                "measurements": points[offsets[index] : offsets[index + 1]].astype(np.float32),
            }
        )
    return frames


def load_config(path: Path, particles: int) -> Config:
    cfg = Config()
    if path.exists():
        values = json.loads(path.read_text(encoding="utf-8"))
        aliases = {"n_targets": "n_targets_true"}
        for key, value in values.items():
            field = aliases.get(key, key)
            if hasattr(cfg, field):
                setattr(cfg, field, value)
    cfg.n_particles_train = int(particles)
    cfg.n_particles_eval = int(particles)
    return cfg


def replay_belief(
    cfg: Config,
    frames: list[dict[str, np.ndarray | float]],
    belief_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    belief = TargetBelief(cfg, eval_mode=True)
    belief.reset(seed=belief_seed)
    snapshots: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    for step, frame in enumerate(frames, start=1):
        belief.predict(float(frame["duration"]))
        belief.update(
            np.asarray(frame["measurements"]),
            np.asarray(frame["uav_positions"]),
        )
        peaks = belief.peaks()
        diagnostics = belief.diagnostics()
        estimates = np.asarray([peak.pos for peak in peaks], dtype=np.float32).reshape(-1, 2)
        expected_count = float(np.sum(belief.weights))
        ospa = ospa_distance(
            estimates,
            np.asarray(frame["true_positions"]),
            cfg.ospa_cutoff,
            cfg.ospa_order,
        )
        snapshots.append(
            {
                "step": step,
                "particles": belief.particles[:, :2].copy(),
                "weights": belief.weights.copy(),
                "uav_positions": np.asarray(frame["uav_positions"]).copy(),
                "true_positions": np.asarray(frame["true_positions"]).copy(),
                "measurements": np.asarray(frame["measurements"]).copy(),
                "estimates": estimates,
                "expected_count": expected_count,
                "ospa": ospa,
                "birth_particle_count": int(
                    diagnostics.get("birth_particle_count", 0)
                ),
                "proposal_particle_count": int(
                    diagnostics.get("proposal_particle_count", 0)
                ),
                "proposal_measurement_count": int(
                    diagnostics.get("proposal_measurement_count", 0)
                ),
            }
        )
        trace.append(
            {
                "step": step,
                "true_count": len(np.asarray(frame["true_positions"])),
                "expected_count": expected_count,
                "estimated_peak_count": len(estimates),
                "measurement_count": len(np.asarray(frame["measurements"])),
                "ospa": ospa,
                "birth_particle_count": int(
                    diagnostics.get("birth_particle_count", 0)
                ),
                "proposal_particle_count": int(
                    diagnostics.get("proposal_particle_count", 0)
                ),
                "proposal_measurement_count": int(
                    diagnostics.get("proposal_measurement_count", 0)
                ),
                "estimated_positions": json.dumps(estimates.round(4).tolist()),
                "true_positions": json.dumps(
                    np.asarray(frame["true_positions"]).round(4).tolist()
                ),
            }
        )
    return snapshots, trace


def save_trace(path: Path, trace: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)


def animate(
    cfg: Config,
    snapshots: list[dict[str, object]],
    out_path: Path,
    fps: int,
    dpi: int,
) -> None:
    steps = np.asarray([int(item["step"]) for item in snapshots])
    expected_counts = np.asarray([float(item["expected_count"]) for item in snapshots])
    peak_counts = np.asarray([len(np.asarray(item["estimates"])) for item in snapshots])
    true_count = len(np.asarray(snapshots[0]["true_positions"]))

    fig = plt.figure(figsize=(14.0, 7.2), dpi=dpi)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_count = fig.add_subplot(grid[0, 1])
    ax_positions = fig.add_subplot(grid[1, 1])

    def update(frame_index: int) -> list[object]:
        snap = snapshots[frame_index]
        step = int(snap["step"])
        particles = np.asarray(snap["particles"])
        weights = np.asarray(snap["weights"])
        uavs = np.asarray(snap["uav_positions"])
        truths = np.asarray(snap["true_positions"])
        measurements = np.asarray(snap["measurements"])
        estimates = np.asarray(snap["estimates"])

        ax_map.clear()
        ax_map.set_xlim(0.0, cfg.map_size)
        ax_map.set_ylim(0.0, cfg.map_size)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("x")
        ax_map.set_ylabel("y")
        ax_map.grid(color="#d9d9d9", linewidth=0.45, alpha=0.65)
        ax_map.set_title(
            f"Step {step:03d}/{len(snapshots)}  |  E[K]={float(snap['expected_count']):.2f}  "
            f"|  peaks={len(estimates)}  |  OSPA={float(snap['ospa']):.2f}"
        )

        trail_start = max(0, frame_index - 14)
        for target_index in range(len(truths)):
            trail = np.asarray(
                [snapshots[i]["true_positions"][target_index] for i in range(trail_start, frame_index + 1)]
            )
            ax_map.plot(trail[:, 0], trail[:, 1], color="#d73027", alpha=0.35, linewidth=1.0)
        for uav_index in range(len(uavs)):
            trail = np.asarray(
                [snapshots[i]["uav_positions"][uav_index] for i in range(trail_start, frame_index + 1)]
            )
            ax_map.plot(trail[:, 0], trail[:, 1], color="#1b9e77", alpha=0.45, linewidth=1.0)

        positive = weights > 0.0
        if np.any(positive):
            log_weights = np.log10(np.maximum(weights[positive], 1e-12))
            low, high = np.percentile(log_weights, [5.0, 99.0])
            span = max(float(high - low), 1e-6)
            colors = np.clip((log_weights - low) / span, 0.0, 1.0)
            sizes = 3.0 + 13.0 * colors
            ax_map.scatter(
                particles[positive, 0],
                particles[positive, 1],
                c=colors,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=sizes,
                alpha=0.38,
                linewidths=0,
                rasterized=True,
                label="PHD particles",
            )

        for pos in uavs:
            ax_map.add_patch(
                Circle(pos, cfg.fov_radius, facecolor="#1b9e77", edgecolor="#1b9e77", alpha=0.08)
            )
        ax_map.scatter(uavs[:, 0], uavs[:, 1], marker="^", color="#1b9e77", s=58, label="UAV", zorder=6)
        ax_map.scatter(
            truths[:, 0], truths[:, 1], marker="x", color="#d73027", linewidths=2.2, s=68,
            label="true target", zorder=8,
        )
        for index, pos in enumerate(truths):
            ax_map.text(pos[0] + 0.8, pos[1] + 0.8, f"T{index}", color="#a50026", fontsize=7)
        if len(measurements):
            ax_map.scatter(
                measurements[:, 0], measurements[:, 1], marker="o", facecolors="none",
                edgecolors="#fdae61", linewidths=1.0, s=32, label="measurement", zorder=7,
            )
        if len(estimates):
            ax_map.scatter(
                estimates[:, 0], estimates[:, 1], marker="P", color="#4575b4", edgecolors="white",
                linewidths=0.6, s=78, label="PHD estimate", zorder=9,
            )
            for index, pos in enumerate(estimates):
                ax_map.text(pos[0] + 0.8, pos[1] - 2.0, f"E{index}", color="#313695", fontsize=7)
        ax_map.legend(loc="upper right", fontsize=8, framealpha=0.9)

        ax_count.clear()
        ax_count.plot(steps, expected_counts, color="#762a83", linewidth=1.8, label="sum(weights)")
        ax_count.step(steps, peak_counts, where="mid", color="#4575b4", linewidth=1.2, label="peak count")
        ax_count.axhline(true_count, color="#d73027", linestyle="--", linewidth=1.2, label="true count")
        ax_count.axvline(step, color="#555555", linewidth=1.0)
        ax_count.scatter([step], [expected_counts[frame_index]], color="#762a83", s=28, zorder=4)
        ax_count.set_xlim(1, len(snapshots))
        upper = max(true_count + 2.0, float(np.max(expected_counts)) + 0.5)
        ax_count.set_ylim(0.0, upper)
        ax_count.set_ylabel("target count")
        ax_count.set_xlabel("step")
        ax_count.grid(color="#dddddd", linewidth=0.5)
        ax_count.legend(loc="upper right", fontsize=8)
        ax_count.set_title("Cardinality over time")

        ax_positions.clear()
        ax_positions.axis("off")
        lines = ["True positions              Estimated positions"]
        row_count = max(len(truths), len(estimates))
        for index in range(row_count):
            if index < len(truths):
                true_text = f"T{index}: ({truths[index, 0]:6.2f}, {truths[index, 1]:6.2f})"
            else:
                true_text = " " * 21
            if index < len(estimates):
                estimate_text = f"E{index}: ({estimates[index, 0]:6.2f}, {estimates[index, 1]:6.2f})"
            else:
                estimate_text = ""
            lines.append(f"{true_text}    {estimate_text}")
        lines.append("")
        lines.append(f"Measurements: {len(measurements)}")
        lines.append(
            f"Birth particles: {int(snap['birth_particle_count'])}   "
            f"Proposal particles: {int(snap['proposal_particle_count'])}"
        )
        lines.append(f"OSPA(c={cfg.ospa_cutoff:g}): {float(snap['ospa']):.3f}")
        ax_positions.text(
            0.02, 0.98, "\n".join(lines), transform=ax_positions.transAxes,
            va="top", ha="left", family="monospace", fontsize=8.5,
        )
        ax_positions.set_title("Current target-set estimate", loc="left")
        fig.tight_layout()
        return []

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=True,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a saved validate_phd replay with the actual PHD filter.")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--belief-seed", type=int, default=1300)
    parser.add_argument("--birth-scheme", choices=["none", "expansion"], default=None)
    parser.add_argument("--birth-probability", type=float, default=None)
    parser.add_argument("--birth-rate", type=float, default=None)
    parser.add_argument("--death-rate", type=float, default=None)
    parser.add_argument("--measurement-proposal", action="store_true", default=None)
    parser.add_argument("--proposal-particles", type=int, default=None)
    parser.add_argument("--proposal-mass-fraction", type=float, default=None)
    parser.add_argument("--proposal-min-component-mass", type=float, default=None)
    parser.add_argument("--proposal-position-std", type=float, default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config if args.config is not None else args.replay.with_name("config.json")
    cfg = load_config(config_path, args.particles)
    overrides = {
        "phd_birth_scheme": args.birth_scheme,
        "phd_birth_probability": args.birth_probability,
        "birth_rate": args.birth_rate,
        "death_probability": args.death_rate,
        "phd_measurement_proposal_enabled": args.measurement_proposal,
        "phd_measurement_proposal_particles": args.proposal_particles,
        "phd_measurement_proposal_mass_fraction": args.proposal_mass_fraction,
        "phd_measurement_proposal_min_component_mass": args.proposal_min_component_mass,
        "phd_measurement_proposal_position_std": args.proposal_position_std,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    frames = load_replay(args.replay)[: max(1, args.steps)]
    snapshots, trace = replay_belief(cfg, frames, args.belief_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    animate(cfg, snapshots, args.out, args.fps, args.dpi)
    save_trace(args.out.with_suffix(".csv"), trace)
    print(
        {
            "video": str(args.out.resolve()),
            "trace": str(args.out.with_suffix('.csv').resolve()),
            "frames": len(snapshots),
            "belief_seed": args.belief_seed,
            "particles": args.particles,
        }
    )


if __name__ == "__main__":
    main()
