from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import Config
from environment import CMUOMMTEnv
from measurements import generate_measurements
from metrics import ospa_distance, phd_tracking_errors
from target_belief import TargetBelief
from utils import write_json


@dataclass(frozen=True)
class ReplayFrame:
    duration: float
    uav_positions: np.ndarray
    true_positions: np.ndarray
    measurements: np.ndarray
    true_detection_count: int
    clutter_count: int


def _uav_schedule(
    scenario: str,
    step: int,
    steps: int,
    cfg: Config,
    target_positions: np.ndarray,
) -> np.ndarray:
    center = np.asarray([cfg.map_size / 2.0, cfg.map_size / 2.0], dtype=np.float32)
    if scenario in {"single", "multi"}:
        return np.repeat(center[None, :], cfg.n_uavs, axis=0)
    if scenario == "overlap":
        return np.asarray(
            [[center[0] - 5.0, center[1]], [center[0] + 5.0, center[1]]],
            dtype=np.float32,
        )
    if scenario == "leave-reenter":
        third = max(steps // 3, 1)
        if step < third or step >= 2 * third:
            return target_positions[:1].copy()
        return np.zeros((1, 2), dtype=np.float32)
    if scenario == "coverage":
        phase = (step % max(2 * steps, 2)) / max(steps, 1)
        x = 5.0 + 90.0 * (phase if phase <= 1.0 else 2.0 - phase)
        ys = np.linspace(8.0, cfg.map_size - 8.0, cfg.n_uavs)
        return np.stack([np.full(cfg.n_uavs, x), ys], axis=1).astype(np.float32)
    raise ValueError(f"Unknown scenario: {scenario}")


def scenario_config(scenario: str, n_targets: int) -> Config:
    cfg = Config()
    cfg.n_targets_true = int(n_targets)
    cfg.n_targets_min = int(n_targets)
    cfg.n_targets_max = int(n_targets)
    cfg.death_probability = 0.0
    cfg.birth_rate = 0.0
    if scenario == "single":
        cfg.n_uavs = 1
        cfg.fov_radius = cfg.map_size * 2.0
        cfg.p_detection = 1.0
        cfg.filter_p_detection = 1.0
        cfg.clutter_mean = 0.0
    elif scenario == "multi":
        cfg.n_uavs = 1
        cfg.fov_radius = cfg.map_size * 2.0
        cfg.p_detection = 1.0
        cfg.filter_p_detection = 1.0
        cfg.clutter_mean = 0.0
    elif scenario == "overlap":
        cfg.n_uavs = 2
        cfg.fov_radius = 25.0
        cfg.p_detection = 1.0
        cfg.filter_p_detection = 1.0
        cfg.clutter_mean = 0.0
    elif scenario == "leave-reenter":
        cfg.n_uavs = 1
        cfg.fov_radius = 12.0
        cfg.p_detection = 1.0
        cfg.filter_p_detection = 1.0
        cfg.clutter_mean = 0.0
    elif scenario == "coverage":
        cfg.n_uavs = 5
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return cfg


def generate_replay_sequence(
    cfg: Config,
    scenario: str,
    seed: int,
    steps: int,
) -> list[ReplayFrame]:
    env = CMUOMMTEnv(cfg)
    initial_uavs = _uav_schedule(
        scenario,
        0,
        steps,
        cfg,
        np.full((cfg.n_targets_true, 2), cfg.map_size / 2.0, dtype=np.float32),
    )
    env.reset(seed=seed, n_targets=cfg.n_targets_true, uav_positions=initial_uavs)
    if scenario in {"overlap", "leave-reenter"}:
        env.target_states[0] = np.asarray(
            [cfg.map_size / 2.0, cfg.map_size / 2.0, cfg.target_speed, 0.0],
            dtype=np.float32,
        )

    frames: list[ReplayFrame] = []
    for step in range(steps):
        uav_positions = _uav_schedule(
            scenario,
            step,
            steps,
            cfg,
            env.target_states[:, :2],
        )
        env.uav_positions = uav_positions.copy()
        env._move_targets(cfg.dt)
        batch = generate_measurements(
            cfg,
            env.rng,
            env.uav_positions,
            env.target_states,
        )
        frames.append(
            ReplayFrame(
                duration=float(cfg.dt),
                uav_positions=env.uav_positions.copy(),
                true_positions=env.target_states[:, :2].copy(),
                measurements=batch.points.copy(),
                true_detection_count=len(batch.detected_target_ids),
                clutter_count=int(batch.clutter_count),
            )
        )
    return frames


def save_replay_sequence(path: Path, frames: list[ReplayFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([len(frame.measurements) for frame in frames], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    points = (
        np.concatenate([frame.measurements for frame in frames], axis=0)
        if int(np.sum(counts)) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    np.savez_compressed(
        path,
        durations=np.asarray([frame.duration for frame in frames], dtype=np.float32),
        uav_positions=np.stack([frame.uav_positions for frame in frames]),
        true_positions=np.stack([frame.true_positions for frame in frames]),
        measurement_points=points,
        measurement_offsets=offsets,
        true_detection_counts=np.asarray([frame.true_detection_count for frame in frames], dtype=np.int32),
        clutter_counts=np.asarray([frame.clutter_count for frame in frames], dtype=np.int32),
    )


def _particle_support(
    target: TargetBelief,
    true_positions: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(
        target.particles[:, None, :2] - true_positions[None, :, :],
        axis=2,
    )
    support = distances <= radius
    return np.sum(support, axis=0), np.sum(support * target.weights[:, None], axis=0)


def replay_phd(
    cfg: Config,
    frames: list[ReplayFrame],
    particle_count: int,
    belief_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    run_cfg = replace(
        cfg,
        n_particles_train=int(particle_count),
        n_particles_eval=int(particle_count),
    )
    target = TargetBelief(run_cfg, eval_mode=True)
    target.reset(seed=belief_seed)
    trace: list[dict[str, object]] = []
    support_trace: list[dict[str, object]] = []
    started = time.perf_counter()

    for step, frame in enumerate(frames, start=1):
        target.predict(frame.duration)
        target.update(frame.measurements, frame.uav_positions)
        peaks = target.peaks()
        estimated_count = float(np.sum(target.weights))
        position_error, number_error = phd_tracking_errors(
            peaks,
            frame.true_positions,
            estimated_count=estimated_count,
        )
        estimated_positions = np.asarray([peak.pos for peak in peaks], dtype=np.float32).reshape(-1, 2)
        ospa = ospa_distance(
            estimated_positions,
            frame.true_positions,
            run_cfg.ospa_cutoff,
            run_cfg.ospa_order,
        )
        diagnostics = target.diagnostics()
        covered = np.any(
            np.linalg.norm(
                frame.true_positions[:, None, :] - frame.uav_positions[None, :, :],
                axis=2,
            )
            <= run_cfg.fov_radius,
            axis=1,
        )
        support_counts, support_mass = _particle_support(
            target,
            frame.true_positions,
            radius=3.0 * run_cfg.meas_std,
        )
        trace.append(
            {
                "step": step,
                "particle_count": particle_count,
                "belief_seed": belief_seed,
                "resampling_mode": run_cfg.phd_resampling_mode,
                "regularization_enabled": int(run_cfg.phd_regularization_enabled),
                "true_target_count": len(frame.true_positions),
                "measurement_count": len(frame.measurements),
                "true_detection_count": frame.true_detection_count,
                "clutter_count": frame.clutter_count,
                "fov_covered_target_count": int(np.sum(covered)),
                "phd_total_mass": diagnostics["phd_total_mass"],
                "estimated_count_continuous": estimated_count,
                "estimated_count_rounded": diagnostics["estimated_count_rounded"],
                "num_extracted_candidates": diagnostics["num_extracted_candidates"],
                "number_error": number_error,
                "position_error": position_error,
                "ospa": ospa,
                "effective_sample_size": diagnostics["effective_sample_size"],
                "ess_before_resampling": diagnostics.get("ess_before_resampling", 0.0),
                "ess_after_resampling": diagnostics.get("ess_after_resampling", 0.0),
                "resampled_this_step": int(bool(diagnostics.get("resampled", False))),
                "regularized_this_step": int(bool(diagnostics.get("regularized", False))),
                "unique_particle_ratio": diagnostics.get("unique_particle_ratio", 1.0),
                "birth_particle_count": diagnostics.get("birth_particle_count", 0),
                "birth_mass": diagnostics.get("birth_mass", 0.0),
                "proposal_particle_count": diagnostics.get("proposal_particle_count", 0),
                "proposal_measurement_count": diagnostics.get("proposal_measurement_count", 0),
                "proposal_redistributed_mass": diagnostics.get("proposal_redistributed_mass", 0.0),
                "mass_before_update": diagnostics.get("mass_before_update", estimated_count),
                "mass_after_measurement_update": diagnostics.get(
                    "mass_after_measurement_update", estimated_count
                ),
                "mass_after_resampling": diagnostics.get("mass_after_resampling", estimated_count),
                "minimum_cluster_distance": diagnostics["minimum_cluster_distance"],
                "cluster_mass_list": json.dumps(diagnostics["cluster_mass_list"]),
                "phd_component_mass_list": json.dumps(
                    diagnostics.get("component_masses", [])
                ),
                "phd_component_particle_counts": json.dumps(
                    diagnostics.get("component_particle_counts", [])
                ),
                "measurement_support_counts": json.dumps(
                    diagnostics.get("measurement_support_counts", [])
                ),
                "minimum_truth_support_count": int(np.min(support_counts)),
                "minimum_truth_support_mass": float(np.min(support_mass)),
            }
        )
        for target_id, (count, mass, is_covered) in enumerate(
            zip(support_counts, support_mass, covered)
        ):
            support_trace.append(
                {
                    "step": step,
                    "particle_count": particle_count,
                    "belief_seed": belief_seed,
                    "target_id": target_id,
                    "inside_any_fov": int(is_covered),
                    "support_radius": 3.0 * run_cfg.meas_std,
                    "support_particle_count": int(count),
                    "support_phd_mass": float(mass),
                }
            )

    elapsed = time.perf_counter() - started
    number_errors = np.asarray([row["number_error"] for row in trace], dtype=np.float64)
    summary = {
        "particle_count": float(particle_count),
        "belief_seed": float(belief_seed),
        "resampling_mode": run_cfg.phd_resampling_mode,
        "regularization_enabled": bool(run_cfg.phd_regularization_enabled),
        "initial_velocity_directions": float(run_cfg.phd_initial_velocity_directions),
        "birth_scheme": run_cfg.phd_birth_scheme,
        "birth_rate": float(run_cfg.birth_rate),
        "death_probability": float(run_cfg.death_probability),
        "measurement_proposal_enabled": bool(
            run_cfg.phd_measurement_proposal_enabled
        ),
        "proposal_particles_per_measurement": float(
            run_cfg.phd_measurement_proposal_particles
        ),
        "mean_proposal_measurement_count": float(
            np.mean([row["proposal_measurement_count"] for row in trace])
        ),
        "steps": float(len(frames)),
        "count_rmse": float(np.sqrt(np.mean(number_errors ** 2))),
        "mean_position_error": float(np.mean([row["position_error"] for row in trace])),
        "mean_ospa": float(np.mean([row["ospa"] for row in trace])),
        "mean_ess": float(np.mean([row["effective_sample_size"] for row in trace])),
        "minimum_ess": float(np.min([row["effective_sample_size"] for row in trace])),
        "resample_count": float(np.sum([row["resampled_this_step"] for row in trace])),
        "regularization_count": float(
            np.sum([row["regularized_this_step"] for row in trace])
        ),
        "zero_support_steps": float(
            np.sum([row["minimum_truth_support_count"] == 0 for row in trace])
        ),
        "minimum_truth_support_count": float(
            np.min([row["minimum_truth_support_count"] for row in trace])
        ),
        "final_estimated_count": float(trace[-1]["estimated_count_continuous"]),
        "runtime_seconds": float(elapsed),
    }
    return trace, support_trace, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    trace: list[dict[str, object]],
    field: str,
    ylabel: str,
    path: Path,
    reference: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=150)
    ax.plot(
        [row["step"] for row in trace],
        [row[field] for row in trace],
        color="#2166ac",
        linewidth=1.8,
    )
    if reference is not None:
        ax.axhline(reference, color="#b2182b", linestyle="--", linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_validation_outputs(
    out_dir: Path,
    cfg: Config,
    frames: list[ReplayFrame],
    trace: list[dict[str, object]],
    support_trace: list[dict[str, object]],
    summary: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_step.csv", trace)
    _write_csv(out_dir / "particle_support.csv", support_trace)
    write_json(out_dir / "summary.json", summary)
    save_replay_sequence(out_dir / "replay_sequence.npz", frames)
    true_count = float(len(frames[0].true_positions))
    _plot_metric(
        trace,
        "estimated_count_continuous",
        "estimated count",
        out_dir / "count_curve.png",
        reference=true_count,
    )
    _plot_metric(trace, "position_error", "position error", out_dir / "position_error_curve.png")
    _plot_metric(trace, "effective_sample_size", "ESS", out_dir / "ess_curve.png")
    _plot_metric(
        trace,
        "phd_total_mass",
        "PHD total mass",
        out_dir / "phd_mass_curve.png",
        reference=true_count,
    )
    write_json(
        out_dir / "config.json",
        {
            "n_uavs": cfg.n_uavs,
            "n_targets": cfg.n_targets_true,
            "steps": len(frames),
            "p_detection": cfg.p_detection,
            "filter_p_detection": cfg.filter_p_detection,
            "clutter_mean": cfg.clutter_mean,
            "fov_radius": cfg.fov_radius,
            "phd_resampling_mode": cfg.phd_resampling_mode,
            "phd_regularization_enabled": cfg.phd_regularization_enabled,
            "phd_initial_velocity_directions": cfg.phd_initial_velocity_directions,
            "phd_birth_scheme": cfg.phd_birth_scheme,
            "phd_birth_probability": cfg.phd_birth_probability,
            "birth_rate": cfg.birth_rate,
            "death_probability": cfg.death_probability,
            "phd_measurement_proposal_enabled": cfg.phd_measurement_proposal_enabled,
            "phd_measurement_proposal_particles": cfg.phd_measurement_proposal_particles,
            "phd_measurement_proposal_mass_fraction": cfg.phd_measurement_proposal_mass_fraction,
            "phd_measurement_proposal_min_component_mass": cfg.phd_measurement_proposal_min_component_mass,
            "phd_measurement_proposal_position_std": cfg.phd_measurement_proposal_position_std,
        },
    )


def run_case(
    scenario: str,
    n_targets: int,
    steps: int,
    scenario_seed: int,
    belief_seeds: list[int],
    particle_counts: list[int],
    out_dir: Path,
    resampling_mode: str = "global",
    regularization_enabled: bool = False,
    velocity_directions: int = 8,
    birth_scheme: str = "none",
    birth_probability: float = 0.05,
    birth_rate: float = 0.0,
    death_probability: float = 0.0,
    measurement_proposal_enabled: bool = False,
    proposal_particles: int = 100,
    proposal_mass_fraction: float = 0.5,
    proposal_min_component_mass: float = 0.15,
    proposal_position_std: float = 1.3,
) -> list[dict[str, float]]:
    cfg = scenario_config(scenario, n_targets)
    cfg.phd_resampling_mode = resampling_mode
    cfg.phd_regularization_enabled = regularization_enabled
    cfg.phd_initial_velocity_directions = velocity_directions
    cfg.phd_birth_scheme = birth_scheme
    cfg.phd_birth_probability = birth_probability
    cfg.birth_rate = birth_rate
    cfg.death_probability = death_probability
    cfg.phd_measurement_proposal_enabled = measurement_proposal_enabled
    cfg.phd_measurement_proposal_particles = proposal_particles
    cfg.phd_measurement_proposal_mass_fraction = proposal_mass_fraction
    cfg.phd_measurement_proposal_min_component_mass = proposal_min_component_mass
    cfg.phd_measurement_proposal_position_std = proposal_position_std
    frames = generate_replay_sequence(cfg, scenario, scenario_seed, steps)
    summaries = []
    for particle_count in particle_counts:
        for belief_seed in belief_seeds:
            run_out = out_dir / f"particles_{particle_count}"
            if len(belief_seeds) > 1:
                run_out = run_out / f"belief_seed_{belief_seed}"
            trace, support, summary = replay_phd(
                cfg, frames, particle_count, belief_seed
            )
            write_validation_outputs(run_out, cfg, frames, trace, support, summary)
            summaries.append(summary)
    aggregates = []
    for particle_count in particle_counts:
        group = [
            summary
            for summary in summaries
            if int(summary["particle_count"]) == int(particle_count)
        ]
        aggregates.append(
            {
                "particle_count": float(particle_count),
                "belief_seed_count": float(len(group)),
                "count_rmse_mean": float(np.mean([row["count_rmse"] for row in group])),
                "count_rmse_std": float(np.std([row["count_rmse"] for row in group])),
                "mean_ospa_mean": float(np.mean([row["mean_ospa"] for row in group])),
                "mean_ospa_std": float(np.std([row["mean_ospa"] for row in group])),
                "zero_support_steps_mean": float(
                    np.mean([row["zero_support_steps"] for row in group])
                ),
                "runtime_seconds_mean": float(
                    np.mean([row["runtime_seconds"] for row in group])
                ),
            }
        )
    write_json(
        out_dir / "comparison.json",
        {
            "scenario_seed": scenario_seed,
            "belief_seeds": belief_seeds,
            "runs": summaries,
            "aggregates": aggregates,
        },
    )
    return summaries


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone particle-PHD validation without MAPPO.")
    parser.add_argument(
        "--scenario",
        choices=["single", "multi", "leave-reenter", "overlap", "coverage", "ablation", "all"],
        default="all",
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--targets", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument(
        "--belief-seeds",
        type=str,
        default="",
        help="Comma-separated PHD particle seeds. Defaults to --seed.",
    )
    parser.add_argument("--particles", type=int, default=2500)
    parser.add_argument("--particle-counts", type=str, default="2500,5000,10000")
    parser.add_argument(
        "--resampling-mode",
        choices=["global", "component"],
        default="global",
    )
    parser.add_argument("--regularization", action="store_true")
    parser.add_argument("--velocity-directions", type=int, default=8)
    parser.add_argument(
        "--birth-scheme",
        choices=["none", "expansion"],
        default="none",
    )
    parser.add_argument("--birth-probability", type=float, default=0.05)
    parser.add_argument("--birth-rate", type=float, default=0.0)
    parser.add_argument("--death-probability", type=float, default=0.0)
    parser.add_argument("--measurement-proposal", action="store_true")
    parser.add_argument("--proposal-particles", type=int, default=100)
    parser.add_argument("--proposal-mass-fraction", type=float, default=0.5)
    parser.add_argument("--proposal-min-component-mass", type=float, default=0.15)
    parser.add_argument("--proposal-position-std", type=float, default=1.3)
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/phd_validation")
    args = parser.parse_args()

    root = Path(args.out_dir)
    belief_seeds = (
        _parse_int_list(args.belief_seeds)
        if args.belief_seeds.strip()
        else [args.seed]
    )
    run_options = {
        "resampling_mode": args.resampling_mode,
        "regularization_enabled": args.regularization,
        "velocity_directions": args.velocity_directions,
        "birth_scheme": args.birth_scheme,
        "birth_probability": args.birth_probability,
        "birth_rate": args.birth_rate,
        "death_probability": args.death_probability,
        "measurement_proposal_enabled": args.measurement_proposal,
        "proposal_particles": args.proposal_particles,
        "proposal_mass_fraction": args.proposal_mass_fraction,
        "proposal_min_component_mass": args.proposal_min_component_mass,
        "proposal_position_std": args.proposal_position_std,
    }
    if args.scenario == "ablation":
        summaries = run_case(
            "coverage",
            args.targets,
            args.steps,
            args.seed,
            belief_seeds,
            _parse_int_list(args.particle_counts),
            root / "ablation",
            **run_options,
        )
    elif args.scenario == "all":
        summaries = []
        cases = [("single", 1), ("multi", 6), ("multi", 7), ("multi", 8), ("leave-reenter", 1), ("overlap", 1), ("coverage", args.targets)]
        for index, (scenario, n_targets) in enumerate(cases):
            name = f"{scenario}_{n_targets}targets"
            summaries.extend(
                run_case(
                    scenario,
                    n_targets,
                    args.steps,
                    args.seed + index,
                    belief_seeds,
                    [args.particles],
                    root / name,
                    **run_options,
                )
            )
        summaries.extend(
            run_case(
                "coverage",
                args.targets,
                args.steps,
                args.seed + 100,
                belief_seeds,
                _parse_int_list(args.particle_counts),
                root / "ablation",
                **run_options,
            )
        )
        write_json(root / "all_summaries.json", {"runs": summaries})
    else:
        summaries = run_case(
            args.scenario,
            args.targets if args.scenario not in {"single", "leave-reenter", "overlap"} else 1,
            args.steps,
            args.seed,
            belief_seeds,
            [args.particles],
            root / args.scenario,
            **run_options,
        )
    print(json.dumps({"out_dir": str(root.resolve()), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
