import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import Config, ensure_output_dir
from replay_buffer import ReplayBuffer
from trainer import Trainer
from utils import write_json
from worker import RolloutWorker


def safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.nan) if len(arr) == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def aggregate_dicts(rows: list[dict], prefix: str = "") -> dict:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    out = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.floating, np.integer)):
                values.append(float(value))
        if values:
            out[prefix + key] = safe_mean(values)
    return out


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_actor(cfg: Config, trainer: Trainer, episodes: int, seed: int) -> dict:
    was_training = trainer.actor.training
    trainer.actor.eval()
    worker = RolloutWorker(cfg, actor=trainer.actor, device=trainer.device)
    metrics = [worker.run_episode(seed + i, greedy=True, eval_mode=True) for i in range(episodes)]
    if was_training:
        trainer.actor.train()
    return aggregate_dicts(metrics, prefix="val_")


def train(
    cfg: Config,
    updates: int,
    steps: int,
    seed: int,
    run_name: str,
    resume: bool = False,
    validation_metric: str = "val_discovery_rate",
) -> dict:
    cfg.episode_steps = steps
    out = ensure_output_dir(cfg, run_name)
    write_json(out / "config.json", asdict(cfg))

    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    latest_path = out / "latest.pt"
    if resume and latest_path.exists():
        trainer.load(latest_path)

    replay = ReplayBuffer(cfg.replay_size)
    worker = RolloutWorker(cfg, actor=trainer.actor, device=device)
    rows: list[dict] = []
    episode_seed = seed
    warmup_target = max(cfg.batch_size, cfg.minimum_buffer_size)
    while len(replay) < warmup_target:
        worker.run_episode(episode_seed, replay=replay, greedy=False, randomize_targets=True)
        episode_seed += 1

    best_value = -float("inf")
    best_summary = {}
    for update in range(updates):
        episode_metrics = worker.run_episode(episode_seed, replay=replay, greedy=False, randomize_targets=True)
        episode_seed += 1

        stats_rows = []
        for _ in range(cfg.updates_per_collection):
            stats_rows.append(trainer.update(replay).__dict__)
        train_stats = aggregate_dicts(stats_rows)

        row = {
            "update": trainer.update_count,
            "collection_index": update,
            "replay_size": len(replay),
            **train_stats,
            **episode_metrics,
        }

        should_eval = (
            update == 0
            or update == updates - 1
            or ((update + 1) % max(cfg.eval_interval, 1) == 0)
        )
        if should_eval:
            val_metrics = evaluate_actor(cfg, trainer, cfg.eval_episodes, cfg.validation_seed)
            row.update(val_metrics)
            metric_value = row.get(validation_metric, -float("inf"))
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value) and metric_value > best_value:
                best_value = float(metric_value)
                best_summary = row.copy()
                trainer.save(out / "best.pt")

        rows.append(row)
        if (update + 1) % max(cfg.save_interval, 1) == 0 or update == updates - 1:
            trainer.save(latest_path)
            trainer.save(out / f"checkpoint_update_{trainer.update_count}.pt")
        if (update + 1) % max(cfg.log_interval, 1) == 0 or update == updates - 1 or should_eval:
            write_table(out / "training_metrics.csv", rows)
            write_json(out / "training_summary.json", row)
            if best_summary:
                write_json(out / "best_summary.json", best_summary)

    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--minimum-buffer-size", type=int, default=None)
    parser.add_argument("--updates-per-collection", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-metric", type=str, default="val_discovery_rate")
    args = parser.parse_args()

    cfg = Config()
    overrides = {
        "batch_size": args.batch_size,
        "minimum_buffer_size": args.minimum_buffer_size,
        "updates_per_collection": args.updates_per_collection,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    summary = train(
        cfg,
        args.updates,
        args.steps,
        args.seed,
        args.run_name,
        resume=args.resume,
        validation_metric=args.validation_metric,
    )
    print(summary)


if __name__ == "__main__":
    main()
