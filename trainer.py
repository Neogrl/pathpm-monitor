from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model import OptionActor, TwinCritic
from replay_buffer import ReplayBuffer


@dataclass
class TrainStats:
    actor_loss: float
    q1_loss: float
    q2_loss: float
    termination_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    target_entropy: float
    q1_mean: float
    q2_mean: float
    reward_mean: float
    policy_grad_norm: float
    critic_grad_norm: float


class Trainer:
    def __init__(self, cfg: Config, device: Union[str, torch.device] = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = OptionActor(cfg).to(self.device)
        self.critic = TwinCritic(cfg).to(self.device)
        self.target_critic = TwinCritic(cfg).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor([-2.0], device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def critic_args(self, batch: dict[str, torch.Tensor], prefix: str = "") -> tuple:
        return (
            batch[prefix + "node_inputs"].float(),
            batch[prefix + "action_mask"].bool(),
            batch[prefix + "uav_state"].float(),
            batch[prefix + "prev_option"].long(),
            batch[prefix + "global_phd"].float(),
            batch[prefix + "global_search"].float(),
            batch[prefix + "true_target_states"].float(),
            batch[prefix + "discovered_memory"].float(),
        )

    def actor_args(self, batch: dict[str, torch.Tensor], prefix: str = "") -> tuple:
        return (
            batch[prefix + "node_inputs"].float(),
            batch[prefix + "node_padding_mask"].bool(),
            batch[prefix + "action_mask"].bool(),
            batch[prefix + "uav_state"].float(),
            batch[prefix + "prev_option"].long(),
            batch[prefix + "team_summary"].float(),
        )

    def update(self, replay: ReplayBuffer) -> TrainStats:
        batch = replay.sample(self.cfg.batch_size, self.device)
        reward = batch["reward"].float()
        done = batch["done"].float()
        if reward.ndim > 1:
            reward = reward.view(reward.shape[0])
        if done.ndim > 1:
            done = done.view(done.shape[0])
        actions = batch["actions"].long()

        with torch.no_grad():
            _, next_logits = self.actor(*self.actor_args(batch, "next_"))
            next_log_pi = F.log_softmax(next_logits, dim=-1)
            next_pi = torch.softmax(next_logits, dim=-1)
            tq1, tq2 = self.target_critic(*self.critic_args(batch, "next_"))
            next_q = torch.minimum(tq1, tq2)
            next_v_per_uav = torch.sum(next_pi * (next_q - self.alpha.detach() * next_log_pi), dim=-1)
            next_v = torch.mean(next_v_per_uav, dim=-1)
            target_q = reward + self.cfg.gamma * (1.0 - done) * next_v

        q1, q2 = self.critic(*self.critic_args(batch))
        chosen_q1 = torch.gather(q1, 2, actions.unsqueeze(-1)).squeeze(-1)
        chosen_q2 = torch.gather(q2, 2, actions.unsqueeze(-1)).squeeze(-1)
        q_target = target_q.unsqueeze(-1).expand_as(chosen_q1)
        q1_loss = F.mse_loss(chosen_q1, q_target)
        q2_loss = F.mse_loss(chosen_q2, q_target)
        critic_loss = q1_loss + q2_loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 20000)
        self.critic_opt.step()

        _, logits = self.actor(*self.actor_args(batch))
        log_pi = F.log_softmax(logits, dim=-1)
        pi = torch.softmax(logits, dim=-1)
        q1_pi, q2_pi = self.critic(*self.critic_args(batch))
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = torch.mean(torch.sum(pi * (self.alpha.detach() * log_pi - q_pi), dim=-1))

        term_loss = self.termination_loss(batch)
        total_actor = actor_loss + self.cfg.termination_coef * term_loss
        self.actor_opt.zero_grad()
        total_actor.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.actor_opt.step()

        valid_counts = torch.sum(~batch["action_mask"].bool(), dim=-1).float().clamp(min=1.0)
        target_entropy = 0.05 * torch.mean(torch.log(valid_counts))
        entropy = -torch.mean(torch.sum(pi * log_pi, dim=-1))
        alpha_loss = -(self.log_alpha * (entropy.detach() - target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.update_count += 1
        if self.update_count % self.cfg.target_update_interval == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        return TrainStats(
            actor_loss=float(actor_loss.detach().cpu()),
            q1_loss=float(q1_loss.detach().cpu()),
            q2_loss=float(q2_loss.detach().cpu()),
            termination_loss=float(term_loss.detach().cpu()),
            alpha_loss=float(alpha_loss.detach().cpu()),
            alpha=float(self.alpha.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            target_entropy=float(target_entropy.detach().cpu()),
            q1_mean=float(chosen_q1.detach().mean().cpu()),
            q2_mean=float(chosen_q2.detach().mean().cpu()),
            reward_mean=float(reward.detach().mean().cpu()),
            policy_grad_norm=float(policy_grad_norm.detach().cpu()),
            critic_grad_norm=float(critic_grad_norm.detach().cpu()),
        )

    def termination_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prev_option = batch["prev_option"].long()
        switched_option = 1 - prev_option
        actor_inputs = list(self.actor_args(batch))
        actor_inputs_keep = actor_inputs.copy()
        actor_inputs_switch = actor_inputs.copy()
        actor_inputs_keep[4] = prev_option
        actor_inputs_switch[4] = switched_option
        term_logits, logits_keep = self.actor(*actor_inputs_keep)
        _, logits_switch = self.actor(*actor_inputs_switch)
        pi_keep = torch.softmax(logits_keep, dim=-1)
        pi_switch = torch.softmax(logits_switch, dim=-1)
        critic_inputs_keep = list(self.critic_args(batch))
        critic_inputs_switch = list(self.critic_args(batch))
        critic_inputs_keep[3] = prev_option
        critic_inputs_switch[3] = switched_option
        q_keep = torch.minimum(*self.critic(*critic_inputs_keep)).detach()
        q_switch = torch.minimum(*self.critic(*critic_inputs_switch)).detach()
        v_keep = torch.sum(pi_keep * q_keep, dim=-1)
        v_switch = torch.sum(pi_switch * q_switch, dim=-1)
        adv_switch = (v_switch - v_keep).detach()
        beta = torch.sigmoid(term_logits)
        term = batch["terminations"].float()
        log_prob = term * torch.log(beta + 1e-8) + (1.0 - term) * torch.log(1.0 - beta + 1e-8)
        return -torch.mean(log_prob * adv_switch)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: Path) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "alpha_opt" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.update_count = int(ckpt.get("update_count", 0))
