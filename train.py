import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import Config, ensure_output_dir
from model import OptionActor
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
        if key in {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "loss", "grad_norm", "switch_loss", "learning_rate"}:
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


def format_progress(row: dict) -> str:
    keys = [
        "episode_reward",
        "mean_reward",
        "discovery_rate",
        "observation_rate",
        "policy_loss",
        "value_loss",
        "entropy",
        "mean_beta",
        "switch_rate",
        "valid_candidates_mean",
    ]
    parts = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(float(value)):
            parts.append(f"{key}={float(value):.4f}")
    return " ".join(parts)


def rollout_state_dict(actor: OptionActor) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in actor.state_dict().items()}


def collect_rollout_episode(args: tuple[Config, dict[str, torch.Tensor], int, str]) -> tuple[int, dict, list[dict]]:
    cfg, state_dict, seed, rollout_device = args
    torch.set_num_threads(1)
    device = torch.device(rollout_device)
    actor = OptionActor(cfg).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    rollout = PPORolloutBuffer()
    worker = RolloutWorker(cfg, actor=actor, device=device)
    row = worker.run_episode(seed, replay=rollout, greedy=False, randomize_targets=True)
    return seed, row, rollout.data


def collect_rollouts(
    cfg: Config,
    trainer: Trainer,
    episode_seed: int,
    executor: Optional[ProcessPoolExecutor],
    update_label: str,
) -> tuple[PPORolloutBuffer, list[dict], int]:
    n_episodes = max(cfg.updates_per_collection, 1)
    seeds = [episode_seed + i for i in range(n_episodes)]
    rollout = PPORolloutBuffer()
    episode_rows: list[dict] = []
    state_dict = rollout_state_dict(trainer.actor)
    worker_count = min(max(cfg.rollout_workers, 1), n_episodes)

    if worker_count <= 1 or executor is None:
        device = torch.device(cfg.rollout_device)
        actor = OptionActor(cfg).to(device)
        actor.load_state_dict(state_dict)
        actor.eval()
        worker = RolloutWorker(cfg, actor=actor, device=device)
        for ep_i, seed in enumerate(seeds, start=1):
            print(f"[train] {update_label}: episode {ep_i}/{n_episodes} seed={seed}", flush=True)
            row = worker.run_episode(seed, replay=rollout, greedy=False, randomize_targets=True)
            episode_rows.append(row)
            print(f"[train] {update_label}: episode done {format_progress(row)}", flush=True)
        return rollout, episode_rows, episode_seed + n_episodes

    print(
        f"[train] {update_label}: parallel rollout episodes={n_episodes} workers={worker_count} device={cfg.rollout_device}",
        flush=True,
    )
    jobs = [(cfg, state_dict, seed, cfg.rollout_device) for seed in seeds]
    for ep_i, (seed, row, transitions) in enumerate(executor.map(collect_rollout_episode, jobs), start=1):
        rollout.extend(transitions)
        episode_rows.append(row)
        print(f"[train] {update_label}: episode {ep_i}/{n_episodes} seed={seed} done {format_progress(row)}", flush=True)
    return rollout, episode_rows, episode_seed + n_episodes


def evaluate_actor(cfg: Config, trainer: Trainer, episodes: int, seed: int) -> dict:
    was_training = trainer.actor.training
    trainer.actor.eval()
    worker = RolloutWorker(cfg, actor=trainer.actor, device=trainer.device)
    metrics = []
    print(f"[eval] start episodes={episodes} seed_start={seed}", flush=True)
    for i in range(episodes):
        ep_seed = seed + i
        print(f"[eval] episode {i + 1}/{episodes} seed={ep_seed}", flush=True)
        metrics.append(worker.run_episode(ep_seed, greedy=True, eval_mode=True))
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
    print(
        "[train] start "
        f"run_name={run_name} updates={updates} steps={steps} seed={seed} device={cfg.device} "
        f"updates_per_collection={cfg.updates_per_collection} ppo_epochs={cfg.ppo_update_epochs} "
        f"ppo_minibatch_size={cfg.ppo_minibatch_size} rollout_workers={cfg.rollout_workers} "
        f"rollout_device={cfg.rollout_device} eval_interval={cfg.eval_interval} "
        f"eval_episodes={cfg.eval_episodes} log_interval={cfg.log_interval} save_interval={cfg.save_interval} "
        f"tensorboard={use_tensorboard} wandb={use_wandb} out={out}",
        flush=True,
    )

    device = torch.device(cfg.device)
    trainer = Trainer(cfg, device)
    latest_path = out / "latest.pt"
    if resume and latest_path.exists():
        print(f"[train] resume checkpoint={latest_path}", flush=True)
        trainer.load(latest_path)

    rows: list[dict] = []
    episode_seed = seed
    best_value = -float("inf")
    best_summary = {}
    worker_count = min(max(cfg.rollout_workers, 1), max(cfg.updates_per_collection, 1))
    executor = ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        for update in range(updates):
            update_label = f"update {update + 1}/{updates}"
            print(f"[train] {update_label}: collect rollout", flush=True)
            rollout, episode_rows, episode_seed = collect_rollouts(cfg, trainer, episode_seed, executor, update_label)
            episode_metrics = aggregate_dicts(episode_rows)
            print(f"[train] update {update + 1}/{updates}: ppo update rollout_size={len(rollout)}", flush=True)
            train_stats = trainer.update(rollout).__dict__

            row = {
                "update": trainer.update_count,
                "collection_index": update,
                "rollout_size": len(rollout),
                **train_stats,
                **episode_metrics,
            }

            should_eval = (
                cfg.eval_interval > 0
                and cfg.eval_episodes > 0
                and (
                    update == 0
                    or update == updates - 1
                    or ((update + 1) % cfg.eval_interval == 0)
                )
            )
            if should_eval:
                print(f"[train] update {update + 1}/{updates}: eval", flush=True)
                val_metrics = evaluate_actor(cfg, trainer, cfg.eval_episodes, cfg.validation_seed)
                row.update(val_metrics)
                metric_value = row.get(validation_metric, -float("inf"))
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value) and metric_value > best_value:
                    best_value = float(metric_value)
                    best_summary = row.copy()
                    trainer.save(out / "best.pt")
                    print(f"[train] update {update + 1}/{updates}: new best {validation_metric}={best_value:.4f}", flush=True)

            rows.append(row)
            logger.log(row, trainer.update_count)
            print(f"[train] update {update + 1}/{updates}: done {format_progress(row)}", flush=True)
            if (update + 1) % max(cfg.save_interval, 1) == 0 or update == updates - 1:
                trainer.save(latest_path)
                trainer.save(out / f"checkpoint_update_{trainer.update_count}.pt")
                print(f"[train] update {update + 1}/{updates}: saved latest/checkpoint update_count={trainer.update_count}", flush=True)
            if (update + 1) % max(cfg.log_interval, 1) == 0 or update == updates - 1 or should_eval:
                write_table(out / "training_metrics.csv", rows)
                write_json(out / "training_summary.json", row)
                if best_summary:
                    write_json(out / "best_summary.json", best_summary)
                print(f"[train] update {update + 1}/{updates}: wrote metrics rows={len(rows)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        logger.close()

    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated alias for --ppo-minibatch-size.")
    parser.add_argument("--ppo-minibatch-size", type=int, default=None)
    parser.add_argument("--ppo-update-epochs", type=int, default=None)
    parser.add_argument("--updates-per-collection", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--rollout-device", type=str, default=None)
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
        "rollout_workers": args.rollout_workers,
        "rollout_device": args.rollout_device,
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
        args.steps if args.steps is not None else cfg.episode_steps,
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
