import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import Config, ensure_output_dir
from ppo_buffer import PPORolloutBuffer
from trainer import Trainer
from utils import write_json
from worker import RolloutWorker


def apply_ablation(cfg: Config, ablation: Optional[str]) -> None:
    if ablation is None:
        return
    if ablation == "no_search":
        cfg.disable_search_belief = True
        cfg.reward_search_weight = 0.0
    elif ablation == "no_phd":
        cfg.disable_phd_belief = True
    elif ablation == "no_option":
        cfg.disable_options = True
        cfg.disable_termination = True
    elif ablation == "no_termination":
        cfg.disable_termination = True
    elif ablation == "no_discover_reward":
        cfg.reward_discover_weight = 0.0
    elif ablation == "no_miss_penalty":
        cfg.reward_miss_weight = 0.0


class TrainingLogger:
    def __init__(self, out: Path, run_name: str, use_tensorboard: bool = True, use_wandb: bool = False):
        self.writer = None
        self.wandb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(out / "tensorboard")
            except Exception as exc:
                print(f"TensorBoard disabled: {exc}")
        if use_wandb:
            try:
                import wandb

                wandb.init(project="CMUOMMT_planToGo_v1", name=run_name, config={"run_name": run_name})
                self.wandb = wandb
            except Exception as exc:
                print(f"W&B disabled: {exc}")

    @staticmethod
    def _group(key: str) -> str:
        if key.startswith("val_"):
            return "val"
        if key.startswith("reward_"):
            return "reward_terms"
        if key in {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "loss", "grad_norm", "switch_loss"}:
            return "ppo"
        if key in {"mean_beta", "switch_rate", "option_0_ratio", "option_1_ratio"}:
            return "options"
        if key in {"mean_reward", "episode_reward", "return_mean", "advantage_mean", "reward_mean"}:
            return "returns"
        if key in {"discovery_rate", "observation_rate", "continuity", "miss_violation_rate", "OSPA", "cardinality_error"}:
            return "task"
        return "metrics"

    def log(self, row: dict, step: int) -> None:
        numeric = {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating, np.integer)) and not np.isnan(float(v))}
        if self.writer is not None:
            for key, value in numeric.items():
                self.writer.add_scalar(f"{self._group(key)}/{key}", value, step)
            self.writer.flush()
        if self.wandb is not None:
            self.wandb.log(numeric, step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


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
    use_tensorboard: bool = True,
    use_wandb: bool = False,
) -> dict:
    cfg.episode_steps = steps
    out = ensure_output_dir(cfg, run_name)
    write_json(out / "config.json", asdict(cfg))
    logger = TrainingLogger(out, run_name, use_tensorboard=use_tensorboard, use_wandb=use_wandb)

    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    latest_path = out / "latest.pt"
    if resume and latest_path.exists():
        trainer.load(latest_path)

    worker = RolloutWorker(cfg, actor=trainer.actor, device=device)
    rows: list[dict] = []
    episode_seed = seed
    best_value = -float("inf")
    best_summary = {}
    try:
        for update in range(updates):
            rollout = PPORolloutBuffer()
            episode_rows = []
            for _ in range(max(cfg.updates_per_collection, 1)):
                episode_rows.append(worker.run_episode(episode_seed, replay=rollout, greedy=False, randomize_targets=True))
                episode_seed += 1
            episode_metrics = aggregate_dicts(episode_rows)
            train_stats = trainer.update(rollout).__dict__

            row = {
                "update": trainer.update_count,
                "collection_index": update,
                "rollout_size": len(rollout),
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
            logger.log(row, trainer.update_count)
            if (update + 1) % max(cfg.save_interval, 1) == 0 or update == updates - 1:
                trainer.save(latest_path)
                trainer.save(out / f"checkpoint_update_{trainer.update_count}.pt")
            if (update + 1) % max(cfg.log_interval, 1) == 0 or update == updates - 1 or should_eval:
                write_table(out / "training_metrics.csv", rows)
                write_json(out / "training_summary.json", row)
                if best_summary:
                    write_json(out / "best_summary.json", best_summary)
    finally:
        logger.close()

    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated alias for --ppo-minibatch-size.")
    parser.add_argument("--ppo-minibatch-size", type=int, default=None)
    parser.add_argument("--ppo-update-epochs", type=int, default=None)
    parser.add_argument("--updates-per-collection", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-metric", type=str, default="val_discovery_rate")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["no_search", "no_phd", "no_option", "no_termination", "no_discover_reward", "no_miss_penalty"],
        default=None,
    )
    args = parser.parse_args()

    cfg = Config()
    overrides = {
        "ppo_minibatch_size": args.ppo_minibatch_size or args.batch_size,
        "ppo_update_epochs": args.ppo_update_epochs,
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
    apply_ablation(cfg, args.ablation)

    summary = train(
        cfg,
        args.updates,
        args.steps,
        args.seed,
        args.run_name,
        resume=args.resume,
        validation_metric=args.validation_metric,
        use_tensorboard=not args.no_tensorboard,
        use_wandb=args.wandb,
    )
    print(summary)


if __name__ == "__main__":
    main()
