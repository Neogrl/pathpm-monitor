"""Reproduce and isolate policy collapse during the first PPO collection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


VARIANTS = ("full", "no_entropy", "policy_only", "separate_clip", "entropy_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("diagnostic_runs/first_ppo_update"))
    parser.add_argument("--network-seed", type=int, default=0)
    parser.add_argument("--episode-seed", type=int, default=10)
    parser.add_argument("--minibatch-seed", type=int, default=20260719)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rollout-device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--probe-size", type=int, default=8)
    parser.add_argument("--no-graph-laplacian-pe", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:4]))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_cls, path: Path | None):
    cfg = config_cls()
    if path is None:
        return cfg
    values = json.loads(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def parameter_groups(actor: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    policy_params: list[torch.nn.Parameter] = []
    critic_params: list[torch.nn.Parameter] = []
    for name, param in actor.named_parameters():
        if name.startswith("critic_") or name.startswith("value_head"):
            critic_params.append(param)
        else:
            policy_params.append(param)
    return policy_params, critic_params


def make_optimizer(actor: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    policy_params, critic_params = parameter_groups(actor)
    return torch.optim.Adam(
        [
            {"params": policy_params, "lr": cfg.actor_lr},
            {"params": critic_params, "lr": cfg.critic_lr},
        ],
        eps=cfg.adam_eps,
    )


def gradient_norm(params: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for param in params:
        if param.grad is not None:
            squared += float(torch.sum(param.grad.detach().double() ** 2))
    return math.sqrt(squared)


def fixed_minibatches(size: int, epochs: int, minibatch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    step = max(1, min(int(minibatch_size), size))
    for _ in range(max(int(epochs), 1)):
        indices = rng.permutation(size)
        batches.extend(indices[start : start + step] for start in range(0, size, step))
    return batches


def select_batch(tensors: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    idx = torch.as_tensor(indices, device=tensors["actions"].device).long()
    return {key: value.index_select(0, idx) for key, value in tensors.items()}


@torch.no_grad()
def pointer_probe(actor, batch: dict[str, torch.Tensor], limit: int) -> dict[str, float]:
    sample = {key: value[:limit] for key, value in batch.items()}
    encoded, _, _ = actor._encode_actor_global(
        sample["global_node_inputs"].float(),
        sample["spatio_pos_encoding"].float(),
        sample["global_edge_mask"].bool(),
        sample["global_node_padding_mask"].bool(),
    )
    current = actor._gather_current(encoded, sample["current_node_indices"].long())
    candidates = actor._gather_candidates(encoded, sample["candidate_node_indices"].long())
    b, n, m, dim = candidates.shape
    query = current.reshape(b * n, 1, dim)
    keys = candidates.reshape(b * n, m, dim)
    mask = (
        sample["action_mask"].bool()
        | sample["candidate_padding_mask"].bool()
        | (sample["candidate_node_indices"] < 0)
    ).reshape(b * n, 1, m)
    enhanced = actor.actor_decoder(query, keys, mask)
    pointer_query = (enhanced.reshape(-1, dim) @ actor.pointer.w_query).reshape(b * n, 1, dim)
    pointer_key = (keys.reshape(-1, dim) @ actor.pointer.w_key).reshape(b * n, m, dim)
    raw = (actor.pointer.norm_factor * torch.matmul(pointer_query, pointer_key.transpose(1, 2))).squeeze(1)
    clipped = actor.pointer.tanh_clipping * torch.tanh(raw)
    logits = clipped.masked_fill(mask.squeeze(1), -1e9)
    distribution = torch.distributions.Categorical(logits=logits)
    valid = ~mask.squeeze(1)
    valid_raw = raw[valid]
    row_stds = []
    row_probability_max = []
    probabilities = distribution.probs
    for row in range(raw.shape[0]):
        row_valid = valid[row]
        row_stds.append(clipped[row, row_valid].std(unbiased=False))
        row_probability_max.append(probabilities[row, row_valid].max())
    return {
        "raw_min": float(valid_raw.min()),
        "raw_max": float(valid_raw.max()),
        "raw_abs_mean": float(valid_raw.abs().mean()),
        "raw_saturated_fraction_abs_gt_3": float((valid_raw.abs() > 3.0).float().mean()),
        "clipped_logit_std": float(torch.stack(row_stds).mean()),
        "entropy": float(distribution.entropy().mean()),
        "probability_max": float(torch.stack(row_probability_max).mean()),
        "encoded_std": float(encoded.std(unbiased=False)),
        "candidate_slot_std": float(candidates.std(dim=2, unbiased=False).mean()),
        "enhanced_query_abs_mean": float(enhanced.abs().mean()),
        "pointer_query_abs_mean": float(pointer_query.abs().mean()),
        "pointer_key_abs_mean": float(pointer_key.abs().mean()),
    }


def losses(actor, trainer, cfg, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_logp, entropy, values, beta = actor.evaluate_actions(
        *trainer.actor_args(batch),
        actions=batch["actions"].long(),
        terminations=batch["terminations"].float(),
    )
    old_logp = batch["log_probs"].float()
    logratio = new_logp - old_logp.detach()
    ratio = torch.exp(logratio)
    advantage = batch["advantages"].detach()
    pg_loss1 = -advantage * ratio
    pg_loss2 = -advantage * torch.clamp(ratio, 1.0 - cfg.ppo_clip_coef, 1.0 + cfg.ppo_clip_coef)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    returns = batch["returns"].float()
    old_values = batch["values"].float()
    value_pred_clipped = old_values + (values - old_values).clamp(-cfg.ppo_clip_coef, cfg.ppo_clip_coef)
    if cfg.use_huber_loss:
        value_original = F.huber_loss(values, returns, delta=cfg.huber_delta, reduction="none")
        value_clipped = F.huber_loss(value_pred_clipped, returns, delta=cfg.huber_delta, reduction="none")
    else:
        value_original = F.mse_loss(values, returns, reduction="none")
        value_clipped = F.mse_loss(value_pred_clipped, returns, reduction="none")
    value_loss = torch.max(value_original, value_clipped).mean() if cfg.use_clipped_value_loss else value_original.mean()
    return {
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy.mean(),
        "switch": beta.mean(),
        "ratio": ratio.mean(),
        "approx_kl": ((ratio - 1.0) - logratio).mean(),
        "clipfrac": ((ratio - 1.0).abs().gt(cfg.ppo_clip_coef)).float().mean(),
    }


def total_loss(parts: dict[str, torch.Tensor], cfg, variant: str) -> torch.Tensor:
    if variant == "policy_only":
        return parts["policy"]
    if variant == "entropy_only":
        return -cfg.ppo_entropy_coef * parts["entropy"]
    result = parts["policy"] + cfg.ppo_value_coef * parts["value"]
    if variant in {"full", "separate_clip"}:
        result = result - cfg.ppo_entropy_coef * parts["entropy"]
    return result


def run_variant(
    variant: str,
    cfg,
    trainer_cls,
    initial_state: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    minibatches: list[np.ndarray],
    probe_size: int,
) -> dict:
    trainer = trainer_cls(cfg, tensors["actions"].device)
    trainer.actor.load_state_dict(initial_state)
    actor = trainer.actor
    actor.train()
    optimizer = make_optimizer(actor, cfg)
    policy_params, critic_params = parameter_groups(actor)
    milestones = {0, 1, 2, 4, 8, 16, 32, len(minibatches)}
    probes = [{"optimizer_step": 0, **pointer_probe(actor, tensors, probe_size)}]
    step_rows = []

    for optimizer_step, indices in enumerate(minibatches, start=1):
        batch = select_batch(tensors, indices)
        parts = losses(actor, trainer, cfg, batch)
        loss = total_loss(parts, cfg, variant)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        policy_grad = gradient_norm(policy_params)
        critic_grad = gradient_norm(critic_params)
        if variant == "separate_clip":
            torch.nn.utils.clip_grad_norm_(policy_params, cfg.ppo_max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic_params, cfg.ppo_max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.ppo_max_grad_norm)
        optimizer.step()
        step_rows.append(
            {
                "variant": variant,
                "optimizer_step": optimizer_step,
                "loss": float(loss.detach()),
                "policy_loss": float(parts["policy"].detach()),
                "value_loss": float(parts["value"].detach()),
                "entropy_loss": float(parts["entropy"].detach()),
                "ratio": float(parts["ratio"].detach()),
                "approx_kl": float(parts["approx_kl"].detach()),
                "clipfrac": float(parts["clipfrac"].detach()),
                "policy_grad_norm": policy_grad,
                "critic_grad_norm": critic_grad,
            }
        )
        if optimizer_step in milestones:
            probes.append({"optimizer_step": optimizer_step, **pointer_probe(actor, tensors, probe_size)})

    return {"variant": variant, "probes": probes, "steps": step_rows, "final": probes[-1]}


def advantage_diagnostics(tensors: dict[str, torch.Tensor]) -> dict:
    advantages = tensors["advantages"].detach().cpu()
    actions = tensors["actions"].detach().cpu()
    by_action = {}
    for action in range(int(actions.max()) + 1):
        selected = advantages[actions == action]
        by_action[str(action)] = {
            "count": int(selected.numel()),
            "mean": float(selected.mean()) if selected.numel() else None,
        }
    within_step_std = advantages.std(dim=1, unbiased=False)
    return {
        "shape": list(advantages.shape),
        "mean": float(advantages.mean()),
        "std": float(advantages.std(unbiased=False)),
        "within_step_uav_std_mean": float(within_step_std.mean()),
        "within_step_uav_std_max": float(within_step_std.max()),
        "mean_by_action_slot": by_action,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.project_root is not None:
        root = args.project_root.resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from config import Config
    from ppo_buffer import PPORolloutBuffer
    from train import close_rollout_pool, collect_rollouts, create_rollout_pool
    from trainer import Trainer

    cfg = load_config(Config, args.config_json)
    if args.no_graph_laplacian_pe:
        cfg.graph_laplacian_pe_enabled = False
    if args.device is not None:
        cfg.device = args.device
    cfg.rollout_device = args.rollout_device
    if args.episodes is not None:
        cfg.episodes_per_collection = args.episodes
    if args.steps is not None:
        cfg.episode_steps = args.steps
    if args.rollout_workers is not None:
        cfg.rollout_workers = args.rollout_workers
    if args.ppo_epochs is not None:
        cfg.ppo_update_epochs = args.ppo_epochs
    if args.minibatch_size is not None:
        cfg.ppo_minibatch_size = args.minibatch_size
        cfg.ppo_num_minibatches = 0

    device = torch.device(cfg.device)
    seed_everything(args.network_seed)
    base_trainer = Trainer(cfg, device)
    initial_state = deepcopy(base_trainer.actor.state_dict())
    worker_count = min(max(cfg.rollout_workers, 1), max(cfg.episodes_per_collection, 1))
    pool = create_rollout_pool(cfg, worker_count)
    try:
        rollout, episode_rows, _ = collect_rollouts(
            cfg, base_trainer, args.episode_seed, pool, "first-ppo-diagnostic"
        )
    finally:
        close_rollout_pool(pool)

    tensors = rollout.tensors(cfg.gamma, cfg.gae_lambda, device)
    advantages = tensors["advantages"]
    tensors["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batches = fixed_minibatches(
        tensors["actions"].shape[0],
        cfg.ppo_update_epochs,
        cfg.ppo_minibatch_size,
        args.minibatch_seed,
    )

    initial_probe = pointer_probe(base_trainer.actor, tensors, args.probe_size)
    variant_results = [
        run_variant(
            variant,
            cfg,
            Trainer,
            initial_state,
            tensors,
            batches,
            args.probe_size,
        )
        for variant in args.variants
    ]

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "network_seed": args.network_seed,
        "episode_seed": args.episode_seed,
        "minibatch_seed": args.minibatch_seed,
        "rollout_size": len(rollout),
        "episode_reward_mean": float(np.mean([row["episode_reward"] for row in episode_rows])),
        "config": asdict(cfg),
        "advantage": advantage_diagnostics(tensors),
        "initial_probe": initial_probe,
        "variants": variant_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out_dir / "optimizer_steps.csv", [row for item in variant_results for row in item["steps"]])
    write_csv(
        out_dir / "pointer_probes.csv",
        [
            {"variant": item["variant"], **probe}
            for item in variant_results
            for probe in item["probes"]
        ],
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "rollout_size": len(rollout),
        "initial_probe": initial_probe,
        "advantage": result["advantage"],
        "final": {item["variant"]: item["final"] for item in variant_results},
    }, indent=2))


if __name__ == "__main__":
    main()
