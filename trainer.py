from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F

from config import Config
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
    mean_beta: float
    switch_loss: float


class Trainer:
    def __init__(self, cfg: Config, device: Union[str, torch.device] = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = OptionActor(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.update_count = 0

    def actor_args(self, batch: dict[str, torch.Tensor]) -> tuple:
        return (
            batch["node_inputs"].float(),
            batch["node_padding_mask"].bool(),
            batch["action_mask"].bool(),
            batch["uav_state"].float(),
            batch["prev_option"].long(),
            batch["team_summary"].float(),
        )

    def update(self, rollout: PPORolloutBuffer) -> TrainStats:
        tensors = rollout.tensors(self.cfg.gamma, self.cfg.gae_lambda, self.device)
        advantages = tensors["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        tensors["advantages"] = advantages

        stats: list[TrainStats] = []
        for _ in range(self.cfg.ppo_update_epochs):
            for batch in PPORolloutBuffer.minibatches(tensors, self.cfg.ppo_minibatch_size):
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

                value_loss = F.mse_loss(values, batch["returns"].float())
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
                        mean_beta=float(beta.mean().detach().cpu()),
                        switch_loss=float(switch_loss.detach().cpu()),
                    )
                )

        if not stats:
            return TrainStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
            mean_beta=float(sum(s.mean_beta for s in stats) / len(stats)),
            switch_loss=float(sum(s.switch_loss for s in stats) / len(stats)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "algorithm": "ppo",
            },
            path,
        )

    def load(self, path: Path) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = int(ckpt.get("update_count", 0))
