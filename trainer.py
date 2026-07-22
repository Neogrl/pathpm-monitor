from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F

from config import Config
from global_node_graph import GlobalNodeGraph
from model import OptionActor
from ppo_buffer import PPORolloutBuffer


@dataclass
class TrainStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    loss: float
    reward_mean: float
    return_mean: float
    advantage_mean: float
    grad_norm: float
    ratio: float
    value_clipfrac: float
    mean_beta: float
    switch_loss: float
    learning_rate: float


class Trainer:
    def __init__(self, cfg: Config, device: Union[str, torch.device] = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = OptionActor(cfg).to(self.device)
        critic_params = []
        policy_params = []
        for name, param in self.actor.named_parameters():
            if name.startswith("critic_") or name.startswith("value_head"):
                critic_params.append(param)
            else:
                policy_params.append(param)
        self.optimizer = torch.optim.Adam(
            [
                {"params": policy_params, "lr": cfg.actor_lr},
                {"params": critic_params, "lr": cfg.critic_lr},
            ],
            eps=cfg.adam_eps,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.lr_decay_step,
            gamma=cfg.lr_decay_gamma,
        )
        self.update_count = 0
        graph = GlobalNodeGraph(cfg)
        self.global_edge_mask = torch.as_tensor(graph.edge_mask(), device=self.device).bool()
        self.global_node_padding_mask = torch.as_tensor(graph.node_padding_mask(), device=self.device).bool()

    def actor_args(self, batch: dict[str, torch.Tensor]) -> tuple:
        global_edge_mask = batch.get("global_edge_mask", self.global_edge_mask)
        global_node_padding_mask = batch.get("global_node_padding_mask", self.global_node_padding_mask)
        return (
            batch["global_node_inputs"].float(),
            batch["spatio_pos_encoding"].float(),
            global_edge_mask.bool(),
            global_node_padding_mask.bool(),
            batch["current_node_indices"].long(),
            batch["candidate_node_indices"].long(),
            batch["candidate_padding_mask"].bool(),
            batch["action_mask"].bool(),
            batch["uav_state"].float(),
            batch["prev_option"].long(),
        )

    def update(self, rollout: PPORolloutBuffer) -> TrainStats:
        tensors = rollout.tensors(self.cfg.gamma, self.cfg.gae_lambda, self.device)
        advantages = tensors["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        tensors["advantages"] = advantages

        stats: list[TrainStats] = []
        for _ in range(self.cfg.ppo_update_epochs):
            for batch in PPORolloutBuffer.minibatches(
                tensors,
                self.cfg.ppo_minibatch_size,
                self.cfg.ppo_num_minibatches,
            ):
                new_logp, entropy, values, beta = self.actor.evaluate_actions(
                    *self.actor_args(batch),
                    actions=batch["actions"].long(),
                    terminations=batch["terminations"].float(),
                )
                old_logp = batch["log_probs"].float()
                logratio = new_logp - old_logp.detach()
                ratio = torch.exp(logratio)
                adv = batch["advantages"].detach()
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.cfg.ppo_clip_coef, 1.0 + self.cfg.ppo_clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                returns = batch["returns"].float()
                old_values = batch["values"].float()
                value_pred_clipped = old_values + (values - old_values).clamp(
                    -self.cfg.ppo_clip_coef,
                    self.cfg.ppo_clip_coef,
                )
                if self.cfg.use_huber_loss:
                    value_loss_original = F.huber_loss(values, returns, delta=self.cfg.huber_delta, reduction="none")
                    value_loss_clipped = F.huber_loss(value_pred_clipped, returns, delta=self.cfg.huber_delta, reduction="none")
                else:
                    value_loss_original = F.mse_loss(values, returns, reduction="none")
                    value_loss_clipped = F.mse_loss(value_pred_clipped, returns, reduction="none")
                if self.cfg.use_clipped_value_loss:
                    value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
                else:
                    value_loss = value_loss_original.mean()
                entropy_loss = entropy.mean()
                switch_loss = beta.mean()
                loss = (
                    policy_loss
                    + self.cfg.ppo_value_coef * value_loss
                    - self.cfg.ppo_entropy_coef * entropy_loss
                    + self.cfg.ppo_switch_coef * switch_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.ppo_max_grad_norm)
                self.optimizer.step()
                self.update_count += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.cfg.ppo_clip_coef).float().mean()
                    value_clipfrac = ((values - old_values).abs() > self.cfg.ppo_clip_coef).float().mean()
                stats.append(
                    TrainStats(
                        policy_loss=float(policy_loss.detach().cpu()),
                        value_loss=float(value_loss.detach().cpu()),
                        entropy=float(entropy_loss.detach().cpu()),
                        approx_kl=float(approx_kl.detach().cpu()),
                        clipfrac=float(clipfrac.detach().cpu()),
                        loss=float(loss.detach().cpu()),
                        reward_mean=float(tensors["rewards"].mean().detach().cpu()),
                        return_mean=float(batch["returns"].mean().detach().cpu()),
                        advantage_mean=float(batch["advantages"].mean().detach().cpu()),
                        grad_norm=float(grad_norm.detach().cpu()),
                        ratio=float(ratio.mean().detach().cpu()),
                        value_clipfrac=float(value_clipfrac.detach().cpu()),
                        mean_beta=float(beta.mean().detach().cpu()),
                        switch_loss=float(switch_loss.detach().cpu()),
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    )
                )
        self.lr_scheduler.step()

        if not stats:
            return TrainStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.optimizer.param_groups[0]["lr"]))
        return TrainStats(
            policy_loss=float(sum(s.policy_loss for s in stats) / len(stats)),
            value_loss=float(sum(s.value_loss for s in stats) / len(stats)),
            entropy=float(sum(s.entropy for s in stats) / len(stats)),
            approx_kl=float(sum(s.approx_kl for s in stats) / len(stats)),
            clipfrac=float(sum(s.clipfrac for s in stats) / len(stats)),
            loss=float(sum(s.loss for s in stats) / len(stats)),
            reward_mean=float(sum(s.reward_mean for s in stats) / len(stats)),
            return_mean=float(sum(s.return_mean for s in stats) / len(stats)),
            advantage_mean=float(sum(s.advantage_mean for s in stats) / len(stats)),
            grad_norm=float(sum(s.grad_norm for s in stats) / len(stats)),
            ratio=float(sum(s.ratio for s in stats) / len(stats)),
            value_clipfrac=float(sum(s.value_clipfrac for s in stats) / len(stats)),
            mean_beta=float(sum(s.mean_beta for s in stats) / len(stats)),
            switch_loss=float(sum(s.switch_loss for s in stats) / len(stats)),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "update_count": self.update_count,
                "algorithm": "mappo_stage1",
            },
            path,
        )

    def load(self, path: Path) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        incompatible = self.actor.load_state_dict(ckpt["actor"], strict=False)
        allowed_missing = {"actor_agent_embedding.weight"}
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(f"Checkpoint model mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
        migrated_old_actor = bool(missing)
        if migrated_old_actor:
            # A zero embedding preserves the old shared-policy behavior for evaluation.
            torch.nn.init.zeros_(self.actor.actor_agent_embedding.weight)
        if "optimizer" in ckpt and not migrated_old_actor:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt and not migrated_old_actor:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.update_count = int(ckpt.get("update_count", 0))
