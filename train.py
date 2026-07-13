import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, TextIO


THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def configure_thread_env(num_threads: int, overwrite: bool = False) -> None:
    value = str(max(int(num_threads), 1))
    for env_name in THREAD_ENV_DEFAULTS:
        if overwrite:
            os.environ[env_name] = value
        else:
            os.environ.setdefault(env_name, value)


configure_thread_env(1)

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
        if key in {"rollout_seconds", "ppo_update_seconds", "collection_seconds", "rollout_steps_per_second"}:
            return "timing"
        if key.startswith("val_"):
            return "val"
        if key.startswith("reward_"):
            return "reward_terms"
        if key in {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "loss", "grad_norm", "ratio", "value_clipfrac", "switch_loss", "learning_rate"}:
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


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)

    def write(self, text: str) -> int:
        for stream in self.streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.write(text)
                stream.flush()
            except ValueError:
                continue
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.flush()
            except ValueError:
                continue

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self) -> int:
        return self.streams[0].fileno()


class ConsoleLogCapture:
    def __init__(self, path: Path, append: bool = False):
        self.path = path
        self.append = append
        self.file: Optional[TextIO] = None
        self.stdout: Optional[TextIO] = None
        self.stderr: Optional[TextIO] = None

    def __enter__(self) -> "ConsoleLogCapture":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a" if self.append else "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = TeeStream(self.stdout, self.file)  # type: ignore[assignment]
        sys.stderr = TeeStream(self.stderr, self.file)  # type: ignore[assignment]
        print(f"[train] console log file={self.path}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stdout is not None:
            sys.stdout = self.stdout
        if self.stderr is not None:
            sys.stderr = self.stderr
        if self.file is not None:
            self.file.close()


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


def read_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


def configure_torch_threads(num_threads: int) -> None:
    torch.set_num_threads(max(int(num_threads), 1))
    try:
        torch.set_num_interop_threads(max(int(num_threads), 1))
    except RuntimeError:
        pass


class StampStyleRolloutRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        configure_thread_env(cfg.worker_num_threads, overwrite=True)
        configure_torch_threads(cfg.worker_num_threads)
        self.device = torch.device(cfg.rollout_device)
        self.actor = OptionActor(cfg).to(self.device)
        self.actor.eval()
        self.worker = RolloutWorker(cfg, actor=self.actor, device=self.device)

    def run_episode(self, state_dict: dict[str, torch.Tensor], seed: int) -> tuple[int, dict, list[dict]]:
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        rollout = PPORolloutBuffer()
        row = self.worker.run_episode(seed, replay=rollout, greedy=False, randomize_targets=True)
        return seed, row, rollout.data


def collect_rollout_episode(args: tuple[Config, dict[str, torch.Tensor], int, str]) -> tuple[int, dict, list[dict]]:
    cfg, state_dict, seed, rollout_device = args
    configure_thread_env(cfg.worker_num_threads, overwrite=True)
    configure_torch_threads(cfg.worker_num_threads)
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
    rollout_pool: Optional[dict[str, Any]],
    update_label: str,
) -> tuple[PPORolloutBuffer, list[dict], int]:
    n_episodes = max(cfg.episodes_per_collection, 1)
    seeds = [episode_seed + i for i in range(n_episodes)]
    rollout = PPORolloutBuffer()
    episode_rows: list[dict] = []
    state_dict = rollout_state_dict(trainer.actor)
    worker_count = min(max(cfg.rollout_workers, 1), n_episodes)

    if worker_count <= 1 or rollout_pool is None:
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

    if rollout_pool["backend"] == "ray":
        ray = rollout_pool["ray"]
        handles = rollout_pool["handles"]
        state_ref = ray.put(state_dict)
        print(
            f"[train] {update_label}: ray rollout episodes={n_episodes} workers={len(handles)} "
            f"cpus_per_worker={cfg.rollout_cpus_per_worker} gpus_per_worker={cfg.rollout_gpus_per_worker} "
            f"threads_per_worker={cfg.worker_num_threads} device={cfg.rollout_device}",
            flush=True,
        )
        futures = [handles[i % len(handles)].run_episode.remote(state_ref, seed) for i, seed in enumerate(seeds)]
        for ep_i, (seed, row, transitions) in enumerate(ray.get(futures), start=1):
            rollout.extend(transitions)
            episode_rows.append(row)
            print(f"[train] {update_label}: episode {ep_i}/{n_episodes} seed={seed} done {format_progress(row)}", flush=True)
        return rollout, episode_rows, episode_seed + n_episodes

    executor = rollout_pool["executor"]
    print(
        f"[train] {update_label}: process rollout episodes={n_episodes} workers={worker_count} "
        f"threads_per_worker={cfg.worker_num_threads} device={cfg.rollout_device}",
        flush=True,
    )
    jobs = [(cfg, state_dict, seed, cfg.rollout_device) for seed in seeds]
    for ep_i, (seed, row, transitions) in enumerate(executor.map(collect_rollout_episode, jobs), start=1):
        rollout.extend(transitions)
        episode_rows.append(row)
        print(f"[train] {update_label}: episode {ep_i}/{n_episodes} seed={seed} done {format_progress(row)}", flush=True)
    return rollout, episode_rows, episode_seed + n_episodes


def create_rollout_pool(cfg: Config, worker_count: int) -> Optional[dict[str, Any]]:
    if worker_count <= 1:
        return None
    backend = cfg.rollout_backend.lower()
    configure_thread_env(cfg.worker_num_threads, overwrite=True)
    if backend == "ray":
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError("rollout_backend='ray' requires ray. Install ray or use --rollout-backend process.") from exc
        thread_env = {env_name: str(max(int(cfg.worker_num_threads), 1)) for env_name in THREAD_ENV_DEFAULTS}
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            started_ray = True
        else:
            started_ray = False
        runner_cls = ray.remote(
            num_cpus=cfg.rollout_cpus_per_worker,
            num_gpus=cfg.rollout_gpus_per_worker,
            runtime_env={"env_vars": thread_env},
        )(StampStyleRolloutRunner)
        handles = [runner_cls.remote(cfg) for _ in range(worker_count)]
        return {"backend": "ray", "ray": ray, "handles": handles, "started_ray": started_ray}
    if backend == "process":
        return {"backend": "process", "executor": ProcessPoolExecutor(max_workers=worker_count)}
    raise ValueError(f"Unknown rollout_backend={cfg.rollout_backend!r}; expected 'ray' or 'process'.")


def close_rollout_pool(pool: Optional[dict[str, Any]]) -> None:
    if pool is None:
        return
    if pool["backend"] == "ray":
        ray = pool["ray"]
        for handle in pool["handles"]:
            ray.kill(handle, no_restart=True)
        if pool.get("started_ray"):
            ray.shutdown()
        return
    pool["executor"].shutdown(wait=True, cancel_futures=True)


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
    best_metric: Optional[str] = None,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    log_file: str = "train.log",
) -> dict:
    cfg.episode_steps = steps
    configure_thread_env(cfg.worker_num_threads, overwrite=True)
    configure_torch_threads(cfg.worker_num_threads)
    out = ensure_output_dir(cfg, run_name)
    with ConsoleLogCapture(out / log_file, append=resume):
        write_json(out / "config.json", asdict(cfg))
        logger = TrainingLogger(out, run_name, use_tensorboard=use_tensorboard, use_wandb=use_wandb)
        print(
            "[train] start "
            f"run_name={run_name} updates={updates} steps={steps} seed={seed} device={cfg.device} "
            f"episodes_per_collection={cfg.episodes_per_collection} ppo_epochs={cfg.ppo_update_epochs} "
            f"ppo_num_minibatches={cfg.ppo_num_minibatches} ppo_minibatch_size={cfg.ppo_minibatch_size} "
            f"use_clipped_value_loss={cfg.use_clipped_value_loss} use_huber_loss={cfg.use_huber_loss} "
            f"huber_delta={cfg.huber_delta} actor_lr={cfg.actor_lr} critic_lr={cfg.critic_lr} "
            f"adam_eps={cfg.adam_eps} reward_overlap_weight={cfg.reward_overlap_weight} "
            f"reward_cost_weight={cfg.reward_cost_weight} "
            f"graph_type={cfg.graph_type} prm_random_nodes={cfg.prm_random_nodes} "
            f"prm_sampling={cfg.prm_sampling} prm_jitter_ratio={cfg.prm_jitter_ratio} "
            f"prm_boundary_points_per_side={cfg.prm_boundary_points_per_side} prm_edge_radius={cfg.prm_edge_radius} "
            f"obstacles_enabled={cfg.obstacles_enabled} obstacle_count={cfg.obstacle_count} "
            f"rollout_backend={cfg.rollout_backend} "
            f"rollout_workers={cfg.rollout_workers} rollout_cpus_per_worker={cfg.rollout_cpus_per_worker} "
            f"rollout_gpus_per_worker={cfg.rollout_gpus_per_worker} worker_num_threads={cfg.worker_num_threads} "
            f"rollout_device={cfg.rollout_device} eval_interval={cfg.eval_interval} "
            f"eval_episodes={cfg.eval_episodes} log_interval={cfg.log_interval} "
            f"checkpoint_episode_interval={cfg.checkpoint_episode_interval} best_metric={best_metric or cfg.best_metric} "
            f"disable_options={cfg.disable_options} disable_termination={cfg.disable_termination} "
            f"tensorboard={use_tensorboard} wandb={use_wandb} out={out}",
            flush=True,
        )

        device = torch.device(cfg.device)
        trainer = Trainer(cfg, device)
        latest_path = out / "latest.pt"
        rows: list[dict] = []
        episode_seed = seed
        completed_episodes = 0
        start_update = 0
        best_value = -float("inf")
        best_summary: dict = {}
        if resume and latest_path.exists():
            print(f"[train] resume checkpoint={latest_path}", flush=True)
            trainer.load(latest_path)
            rows = read_table(out / "training_metrics.csv")
            run_state = read_json(out / "run_state.json")
            if int(run_state.get("trainer_update_count", -1)) == trainer.update_count:
                start_update = int(run_state.get("completed_collections", 0))
                completed_episodes = int(run_state.get("completed_episodes", 0))
                episode_seed = int(run_state.get("episode_seed", seed + completed_episodes))
            else:
                rollout_size = max(cfg.episodes_per_collection, 1) * max(cfg.episode_steps, 1)
                if cfg.ppo_num_minibatches > 0:
                    minibatches_per_epoch = min(cfg.ppo_num_minibatches, rollout_size)
                else:
                    minibatch_size = max(min(cfg.ppo_minibatch_size, rollout_size), 1)
                    minibatches_per_epoch = (rollout_size + minibatch_size - 1) // minibatch_size
                optimizer_steps_per_collection = max(cfg.ppo_update_epochs, 1) * minibatches_per_epoch
                if trainer.update_count % optimizer_steps_per_collection != 0:
                    raise RuntimeError(
                        "Cannot infer resume collection from checkpoint: "
                        f"update_count={trainer.update_count} optimizer_steps_per_collection={optimizer_steps_per_collection}"
                    )
                start_update = trainer.update_count // optimizer_steps_per_collection
                completed_episodes = start_update * max(cfg.episodes_per_collection, 1)
                episode_seed = seed + completed_episodes
            best_summary = read_json(out / "best_summary.json")
            best_metric_name = validation_metric if validation_metric in best_summary else (best_metric or cfg.best_metric)
            best_candidate = best_summary.get(best_metric_name)
            if isinstance(best_candidate, (int, float)) and not np.isnan(float(best_candidate)):
                best_value = float(best_candidate)
            print(
                f"[train] resume state start_update={start_update} completed_episodes={completed_episodes} "
                f"next_seed={episode_seed} history_rows={len(rows)} best_value={best_value:.4f}",
                flush=True,
            )

        checkpoint_interval = max(cfg.checkpoint_episode_interval, 1)
        next_checkpoint_episode = ((completed_episodes // checkpoint_interval) + 1) * checkpoint_interval
        worker_count = min(max(cfg.rollout_workers, 1), max(cfg.episodes_per_collection, 1))
        rollout_pool = create_rollout_pool(cfg, worker_count)
        try:
            for update in range(start_update, updates):
                update_label = f"update {update + 1}/{updates}"
                print(f"[train] {update_label}: collect rollout", flush=True)
                rollout_started = time.perf_counter()
                rollout, episode_rows, episode_seed = collect_rollouts(cfg, trainer, episode_seed, rollout_pool, update_label)
                rollout_seconds = time.perf_counter() - rollout_started
                completed_episodes += len(episode_rows)
                episode_metrics = aggregate_dicts(episode_rows)
                print(f"[train] update {update + 1}/{updates}: ppo update rollout_size={len(rollout)}", flush=True)
                ppo_started = time.perf_counter()
                train_stats = trainer.update(rollout).__dict__
                ppo_update_seconds = time.perf_counter() - ppo_started

                row = {
                    "update": trainer.update_count,
                    "collection_index": update,
                    "episode_count": completed_episodes,
                    "rollout_size": len(rollout),
                    "rollout_seconds": rollout_seconds,
                    "ppo_update_seconds": ppo_update_seconds,
                    "collection_seconds": rollout_seconds + ppo_update_seconds,
                    "rollout_steps_per_second": len(rollout) / max(rollout_seconds, 1e-8),
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

                metric_name = validation_metric if validation_metric in row else (best_metric or cfg.best_metric)
                metric_value = row.get(metric_name, -float("inf"))
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value) and metric_value > best_value:
                    best_value = float(metric_value)
                    best_summary = row.copy()
                    trainer.save(out / "best.pt")
                    print(f"[train] update {update + 1}/{updates}: saved best {metric_name}={best_value:.4f}", flush=True)

                rows.append(row)
                logger.log(row, completed_episodes)
                print(
                    f"[train] update {update + 1}/{updates}: timing "
                    f"rollout_seconds={rollout_seconds:.2f} ppo_update_seconds={ppo_update_seconds:.2f} "
                    f"rollout_steps_per_second={row['rollout_steps_per_second']:.2f}",
                    flush=True,
                )
                print(f"[train] update {update + 1}/{updates}: done {format_progress(row)}", flush=True)
                trainer.save(latest_path)
                write_json(
                    out / "run_state.json",
                    {
                        "completed_collections": update + 1,
                        "completed_episodes": completed_episodes,
                        "episode_seed": episode_seed,
                        "trainer_update_count": trainer.update_count,
                        "target_updates": updates,
                        "episode_steps": cfg.episode_steps,
                        "episodes_per_collection": cfg.episodes_per_collection,
                    },
                )
                print(f"[train] update {update + 1}/{updates}: saved latest update_count={trainer.update_count}", flush=True)
                checkpoint_due = False
                while completed_episodes >= next_checkpoint_episode:
                    checkpoint_due = True
                    next_checkpoint_episode += checkpoint_interval
                if checkpoint_due or update == updates - 1:
                    checkpoint_path = out / f"checkpoint_episode_{completed_episodes}_update_{trainer.update_count}.pt"
                    trainer.save(checkpoint_path)
                    print(
                        f"[train] update {update + 1}/{updates}: saved checkpoint "
                        f"episode_count={completed_episodes} update_count={trainer.update_count} path={checkpoint_path}",
                        flush=True,
                    )
                if (update + 1) % max(cfg.log_interval, 1) == 0 or update == updates - 1 or should_eval:
                    write_table(out / "training_metrics.csv", rows)
                    write_json(out / "training_summary.json", row)
                    if best_summary:
                        write_json(out / "best_summary.json", best_summary)
                    print(f"[train] update {update + 1}/{updates}: wrote metrics rows={len(rows)}", flush=True)
        finally:
            close_rollout_pool(rollout_pool)
            logger.close()

        summary = rows[-1] if rows else {}
        print(f"[train] final summary {summary}", flush=True)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated alias for --ppo-minibatch-size.")
    parser.add_argument("--ppo-minibatch-size", type=int, default=None)
    parser.add_argument("--ppo-num-minibatches", type=int, default=None)
    parser.add_argument("--ppo-update-epochs", type=int, default=None)
    parser.add_argument("--ppo-entropy-coef", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--reward-overlap-weight", type=float, default=None)
    parser.add_argument("--reward-cost-weight", type=float, default=None)
    parser.add_argument("--graph-type", type=str, choices=["grid", "prm"], default=None)
    parser.add_argument("--prm-random-nodes", type=int, default=None)
    parser.add_argument("--prm-sampling", type=str, choices=["stratified", "uniform"], default=None)
    parser.add_argument("--prm-jitter-ratio", type=float, default=None)
    parser.add_argument("--prm-boundary-points-per-side", type=int, default=None)
    parser.add_argument("--prm-edge-radius", type=float, default=None)
    parser.add_argument("--prm-min-node-distance", type=float, default=None)
    parser.add_argument("--no-prm-boundary", action="store_true")
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--obstacle-count", type=int, default=None)
    parser.add_argument("--obstacle-radius-min", type=float, default=None)
    parser.add_argument("--obstacle-radius-max", type=float, default=None)
    parser.add_argument("--obstacle-margin", type=float, default=None)
    parser.add_argument("--no-clipped-value-loss", action="store_true")
    parser.add_argument("--no-huber-loss", action="store_true")
    parser.add_argument("--episodes-per-collection", type=int, default=None)
    parser.add_argument("--updates-per-collection", type=int, default=None, help="Deprecated alias for --episodes-per-collection.")
    parser.add_argument("--rollout-backend", type=str, choices=["ray", "process"], default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--rollout-device", type=str, default=None)
    parser.add_argument("--rollout-cpus-per-worker", type=float, default=None)
    parser.add_argument("--rollout-gpus-per-worker", type=float, default=None)
    parser.add_argument("--worker-num-threads", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-episode-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None, help="Deprecated alias for --checkpoint-episode-interval.")
    parser.add_argument("--best-metric", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="train.log")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-metric", type=str, default="val_discovery_rate")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use-option-critic", action="store_true", help="Enable learned option and termination heads. Disabled by default for stage 1.")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["no_search", "no_phd", "no_option", "no_termination", "no_discover_reward", "no_miss_penalty"],
        default=None,
    )
    args = parser.parse_args()

    cfg = Config()
    episodes_per_collection = (
        args.episodes_per_collection
        if args.episodes_per_collection is not None
        else args.updates_per_collection
    )
    checkpoint_episode_interval = (
        args.checkpoint_episode_interval
        if args.checkpoint_episode_interval is not None
        else args.save_interval
    )
    requested_ppo_minibatch_size = args.ppo_minibatch_size or args.batch_size
    requested_ppo_num_minibatches = args.ppo_num_minibatches
    if requested_ppo_minibatch_size is not None and requested_ppo_num_minibatches is None:
        requested_ppo_num_minibatches = 0
    overrides = {
        "ppo_minibatch_size": requested_ppo_minibatch_size,
        "ppo_num_minibatches": requested_ppo_num_minibatches,
        "ppo_update_epochs": args.ppo_update_epochs,
        "ppo_entropy_coef": args.ppo_entropy_coef,
        "critic_lr": args.critic_lr,
        "adam_eps": args.adam_eps,
        "huber_delta": args.huber_delta,
        "reward_overlap_weight": args.reward_overlap_weight,
        "reward_cost_weight": args.reward_cost_weight,
        "graph_type": args.graph_type,
        "prm_random_nodes": args.prm_random_nodes,
        "prm_sampling": args.prm_sampling,
        "prm_jitter_ratio": args.prm_jitter_ratio,
        "prm_boundary_points_per_side": args.prm_boundary_points_per_side,
        "prm_edge_radius": args.prm_edge_radius,
        "prm_min_node_distance": args.prm_min_node_distance,
        "obstacle_count": args.obstacle_count,
        "obstacle_radius_min": args.obstacle_radius_min,
        "obstacle_radius_max": args.obstacle_radius_max,
        "obstacle_margin": args.obstacle_margin,
        "episodes_per_collection": episodes_per_collection,
        "rollout_backend": args.rollout_backend,
        "rollout_workers": args.rollout_workers,
        "rollout_device": args.rollout_device,
        "rollout_cpus_per_worker": args.rollout_cpus_per_worker,
        "rollout_gpus_per_worker": args.rollout_gpus_per_worker,
        "worker_num_threads": args.worker_num_threads,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
        "checkpoint_episode_interval": checkpoint_episode_interval,
        "best_metric": args.best_metric,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.use_option_critic:
        cfg.disable_options = False
        cfg.disable_termination = False
    if args.no_clipped_value_loss:
        cfg.use_clipped_value_loss = False
    if args.no_huber_loss:
        cfg.use_huber_loss = False
    if args.no_prm_boundary:
        cfg.prm_include_boundary = False
    if args.obstacles:
        cfg.obstacles_enabled = True
    apply_ablation(cfg, args.ablation)

    summary = train(
        cfg,
        args.updates,
        args.steps if args.steps is not None else cfg.episode_steps,
        args.seed,
        args.run_name,
        resume=args.resume,
        validation_metric=args.validation_metric,
        best_metric=args.best_metric,
        use_tensorboard=not args.no_tensorboard,
        use_wandb=args.wandb,
        log_file=args.log_file,
    )
    print(summary)


if __name__ == "__main__":
    main()
