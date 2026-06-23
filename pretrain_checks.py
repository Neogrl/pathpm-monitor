import argparse
import csv
from pathlib import Path

import numpy as np

from config import Config
from evaluate import evaluate_baseline_episode
from replay_buffer import ReplayBuffer
from trainer import Trainer
from utils import write_json
from worker import RolloutWorker


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary = {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float, np.floating, np.integer))]
        if values:
            arr = np.asarray(values, dtype=np.float32)
            summary[key] = float(np.nan) if np.all(np.isnan(arr)) else float(np.nanmean(arr))
    return summary


def run_baseline(cfg: Config, baseline: str, episodes: int, seed: int, out: Path) -> dict:
    rows = []
    for ep in range(episodes):
        print(f"[{baseline}] episode {ep + 1}/{episodes}", flush=True)
        metrics = evaluate_baseline_episode(cfg, seed + ep, baseline)
        metrics["episode_index"] = ep
        metrics["mode"] = baseline
        rows.append(metrics)
        write_rows(out / f"{baseline}_episodes.csv", rows)
    summary = summarize_rows(rows)
    summary["episodes"] = episodes
    summary["seed"] = seed
    summary["mode"] = baseline
    write_json(out / f"{baseline}_summary.json", summary)
    return summary


def rollout_stability(cfg: Config, episodes: int, steps: int, seed: int, out: Path) -> dict:
    cfg.episode_steps = steps
    trainer = Trainer(cfg, cfg.device)
    replay = ReplayBuffer(cfg.replay_size)
    worker = RolloutWorker(cfg, actor=trainer.actor, device=cfg.device)
    rows = []
    failures = []
    for ep in range(episodes):
        print(f"[stability] episode {ep + 1}/{episodes}", flush=True)
        try:
            metrics = worker.run_episode(seed + ep, replay=replay, greedy=False, randomize_targets=True)
            metrics["episode_index"] = ep
            rows.append(metrics)
            numeric = [float(v) for v in metrics.values() if isinstance(v, (int, float, np.floating, np.integer))]
            if not np.all(np.isfinite([x for x in numeric if not np.isnan(x)])):
                failures.append({"episode": ep, "reason": "non_finite_metric"})
        except Exception as exc:
            failures.append({"episode": ep, "reason": repr(exc)})
        write_rows(out / "rollout_stability_episodes.csv", rows)
        write_json(out / "rollout_stability_failures.json", {"failures": failures[:10]})
    summary = {
        "episodes": episodes,
        "steps": steps,
        "replay_size": len(replay),
        "failure_count": len(failures),
        "failures": failures[:10],
        "mean_episode_reward": float(np.nanmean([r["episode_reward"] for r in rows])) if rows else float("nan"),
        "mean_discovery_rate": float(np.nanmean([r["discovery_rate"] for r in rows])) if rows else float("nan"),
        "mean_valid_candidates": float(np.nanmean([r["valid_candidates_mean"] for r in rows])) if rows else float("nan"),
        "mean_target_node_count": float(np.nanmean([r["target_node_count"] for r in rows])) if rows else float("nan"),
        "mean_search_node_count": float(np.nanmean([r["search_node_count"] for r in rows])) if rows else float("nan"),
        "mean_maintenance_node_count": float(np.nanmean([r["maintenance_node_count"] for r in rows])) if rows else float("nan"),
        "mean_selected_target_rate": float(np.nanmean([r["selected_target_rate"] for r in rows])) if rows else float("nan"),
        "mean_selected_search_rate": float(np.nanmean([r["selected_search_rate"] for r in rows])) if rows else float("nan"),
        "mean_selected_maintenance_rate": float(np.nanmean([r["selected_maintenance_rate"] for r in rows])) if rows else float("nan"),
    }
    write_json(out / "rollout_stability.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/pretrain_checks")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.episode_steps = args.steps
    if args.device is not None:
        cfg.device = args.device

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing diagnostics to {out.resolve()}", flush=True)
    print(f"Total work: {args.episodes * 3} episodes ({args.episodes} random + {args.episodes} heuristic + {args.episodes} stability)", flush=True)
    summaries = []
    for baseline in ("random", "heuristic"):
        summary = run_baseline(cfg, baseline, args.episodes, args.seed, out)
        summaries.append(summary)
    stability = rollout_stability(cfg, args.episodes, args.steps, args.seed + 1000, out)
    summaries.append({"mode": "rollout_stability", **{k: v for k, v in stability.items() if k != "failures"}})
    write_rows(out / "summary.csv", summaries)
    print({"baselines": summaries[:2], "stability": stability})


if __name__ == "__main__":
    main()
