import numpy as np
import torch
from typing import Optional, Union

from config import Config
from environment import CMUOMMTEnv
from metrics import final_metrics, reward_terms, weighted_reward
from model import OptionActor
from nodes import NodeBatch, NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief


def team_summary(cfg: Config, env: CMUOMMTEnv, target_belief: TargetBelief, search_belief: SearchBelief, selected: list[np.ndarray]) -> np.ndarray:
    uavs = env.uav_positions
    if len(selected):
        selected_arr = np.asarray(selected, dtype=np.float32)
        selected_mean = np.mean(selected_arr, axis=0) / cfg.map_size
    else:
        selected_mean = np.zeros(2, dtype=np.float32)
    if len(uavs) > 1:
        dists = []
        overlaps = []
        from utils import circle_overlap_area

        for i in range(len(uavs)):
            for j in range(i + 1, len(uavs)):
                d = np.linalg.norm(uavs[i] - uavs[j])
                dists.append(d / cfg.map_size)
                overlaps.append(circle_overlap_area(d, cfg.fov_radius) / cfg.fov_area)
    else:
        dists = [0.0]
        overlaps = [0.0]
    return np.asarray(
        [
            np.mean(uavs[:, 0]) / cfg.map_size,
            np.mean(uavs[:, 1]) / cfg.map_size,
            np.std(uavs[:, 0]) / cfg.map_size,
            np.std(uavs[:, 1]) / cfg.map_size,
            np.mean(dists),
            np.mean(overlaps),
            selected_mean[0],
            selected_mean[1],
            np.sum(target_belief.weights) / max(cfg.phd_prior_count, 1.0),
            np.mean(search_belief.search_belief),
            np.mean(search_belief.coverage_age) / cfg.search_age_scale,
            len(target_belief.peaks()) / max(cfg.max_target_candidates, 1) if not cfg.disable_phd_belief else 0.0,
        ],
        dtype=np.float32,
    )


def uav_state(cfg: Config, env: CMUOMMTEnv) -> np.ndarray:
    pos = env.uav_positions / cfg.map_size
    zeros = np.zeros_like(pos)
    return np.concatenate([pos, zeros], axis=1).astype(np.float32)


class RolloutWorker:
    def __init__(self, cfg: Config, actor: Optional[OptionActor] = None, device: Union[torch.device, str] = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = actor
        self.node_builder = NodeBuilder(cfg)

    def reset_stack(self, seed: int, n_targets: Optional[int] = None, eval_mode: bool = False):
        self.node_builder.reset()
        env = CMUOMMTEnv(self.cfg)
        env.reset(seed=seed, n_targets=n_targets)
        target = TargetBelief(self.cfg, eval_mode=eval_mode)
        target.reset(seed=seed + 101)
        search = SearchBelief(self.cfg)
        tracks = PseudoTrackMemory(self.cfg)
        prev_option = np.zeros(self.cfg.n_uavs, dtype=np.int64)
        return env, target, search, tracks, prev_option

    def sample_target_count(self, seed: int, eval_mode: bool = False) -> int:
        if eval_mode or not self.cfg.randomize_train_targets:
            return self.cfg.n_targets_true
        rng = np.random.default_rng(seed + 17)
        return int(rng.integers(self.cfg.n_targets_min, self.cfg.n_targets_max + 1))

    def _sequential_obs_and_actions(
        self,
        env: CMUOMMTEnv,
        target: TargetBelief,
        search: SearchBelief,
        tracks: PseudoTrackMemory,
        prev_option: np.ndarray,
        greedy: bool,
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, m = self.cfg.n_uavs, self.cfg.max_node_candidates
        node_inputs = np.zeros((n, m, 16), dtype=np.float32)
        node_padding_mask = np.ones((n, m), dtype=bool)
        action_mask = np.ones((n, m), dtype=bool)
        waypoints = np.zeros((n, m, 2), dtype=np.float32)
        actions = np.zeros(n, dtype=np.int64)
        options = prev_option.copy()
        terminations = np.zeros(n, dtype=bool)
        log_probs = np.zeros(n, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        betas = np.zeros(n, dtype=np.float32)
        selected: list[np.ndarray] = []
        for i in range(n):
            batch = self.node_builder.build(env.uav_positions, target, search, tracks, selected, step=env.step_count)
            node_inputs[i] = batch.node_inputs[i]
            node_padding_mask[i] = batch.node_padding_mask[i]
            action_mask[i] = batch.action_mask[i]
            waypoints[i] = batch.waypoints[i]
            if self.actor is None:
                valid = np.flatnonzero(~action_mask[i] & ~node_padding_mask[i])
                actions[i] = int(valid[0])
                options[i] = prev_option[i]
                terminations[i] = False
            else:
                obs = self._obs_dict_from_arrays(node_inputs, node_padding_mask, action_mask, env, target, search, prev_option)
                torch_obs = self._to_torch(obs, batch_dim=True)
                if hasattr(self.actor, "act_with_info"):
                    action_t, option_t, term_t, logp_t, value_t, beta_t = self.actor.act_with_info(**torch_obs, greedy=greedy)
                else:
                    action_t, option_t, term_t = self.actor.act(**torch_obs, greedy=greedy)
                    logp_t = torch.zeros_like(action_t, dtype=torch.float32)
                    value_t = torch.zeros_like(action_t, dtype=torch.float32)
                    beta_t = torch.zeros_like(action_t, dtype=torch.float32)
                actions[i] = int(action_t[0, i].cpu().item())
                options[i] = int(option_t[0, i].cpu().item())
                terminations[i] = bool(term_t[0, i].cpu().item())
                log_probs[i] = float(logp_t[0, i].cpu().item())
                values[i] = float(value_t[0, i].cpu().item())
                betas[i] = float(beta_t[0, i].cpu().item())
            selected.append(waypoints[i, actions[i]].copy())
        obs = self._obs_dict_from_arrays(node_inputs, node_padding_mask, action_mask, env, target, search, prev_option)
        return obs, actions, options, terminations, np.asarray(selected, dtype=np.float32), log_probs, values, betas

    def _obs_dict_from_arrays(
        self,
        node_inputs: np.ndarray,
        node_padding_mask: np.ndarray,
        action_mask: np.ndarray,
        env: CMUOMMTEnv,
        target: TargetBelief,
        search: SearchBelief,
        prev_option: np.ndarray,
    ) -> dict:
        return {
            "node_inputs": node_inputs.astype(np.float32),
            "node_padding_mask": node_padding_mask.astype(bool),
            "action_mask": action_mask.astype(bool),
            "uav_state": uav_state(self.cfg, env),
            "prev_option": prev_option.astype(np.int64),
            "team_summary": team_summary(self.cfg, env, target, search, []),
            "global_phd": target.summary(),
            "global_search": search.summary(),
        }

    def _to_torch(self, obs: dict, batch_dim: bool = False) -> dict[str, torch.Tensor]:
        keys = ["node_inputs", "node_padding_mask", "action_mask", "uav_state", "prev_option", "team_summary"]
        out = {}
        for key in keys:
            arr = obs[key]
            if batch_dim:
                arr = arr[None, ...]
            tensor = torch.as_tensor(arr, device=self.device)
            if key in ("node_padding_mask", "action_mask"):
                tensor = tensor.bool()
            elif key == "prev_option":
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            out[key] = tensor
        return out

    def _node_diagnostics(self, obs: dict, actions: np.ndarray) -> dict[str, float]:
        valid = ~obs["action_mask"] & ~obs["node_padding_mask"]
        features = obs["node_inputs"]
        selected_features = features[np.arange(self.cfg.n_uavs), actions]
        return {
            "valid_candidates_mean": float(np.mean(np.sum(valid, axis=1))),
            "target_node_count": float(np.sum(valid & (features[:, :, 12] > 0.5))),
            "search_node_count": float(np.sum(valid & (features[:, :, 13] > 0.5))),
            "maintenance_node_count": float(np.sum(valid & (features[:, :, 14] > 0.5))),
            "goal_node_count": float(np.sum(valid & (features[:, :, 15] > 0.5))),
            "selected_target_rate": float(np.mean(selected_features[:, 12] > 0.5)),
            "selected_search_rate": float(np.mean(selected_features[:, 13] > 0.5)),
            "selected_maintenance_rate": float(np.mean(selected_features[:, 14] > 0.5)),
            "selected_goal_rate": float(np.mean(selected_features[:, 15] > 0.5)),
        }

    def run_episode(
        self,
        seed: int,
        replay=None,
        greedy: bool = False,
        n_targets: Optional[int] = None,
        eval_mode: bool = False,
        randomize_targets: bool = False,
    ) -> dict:
        if n_targets is None and randomize_targets:
            n_targets = self.sample_target_count(seed, eval_mode=eval_mode)
        env, target, search, tracks, prev_option = self.reset_stack(seed, n_targets=n_targets, eval_mode=eval_mode)
        rewards: list[float] = []
        overlaps: list[float] = []
        diagnostics: dict[str, list[float]] = {
            "valid_candidates_mean": [],
            "target_node_count": [],
            "search_node_count": [],
            "maintenance_node_count": [],
            "goal_node_count": [],
            "selected_target_rate": [],
            "selected_search_rate": [],
            "selected_maintenance_rate": [],
            "selected_goal_rate": [],
            "switch_rate": [],
            "mean_beta": [],
            "option_0_ratio": [],
            "option_1_ratio": [],
            "search_belief_mean": [],
            "coverage_age_mean": [],
            "target_estimated_count": [],
            "track_count": [],
            "reward_observe": [],
            "reward_discover": [],
            "reward_continuity": [],
            "reward_search": [],
            "reward_miss": [],
            "reward_fairness_metric": [],
            "reward_overlap_metric": [],
            "reward_cost_metric": [],
            "reward_switch_metric": [],
        }
        for _ in range(self.cfg.episode_steps):
            target.predict()
            obs, actions, options, terminations, selected_waypoints, log_probs, values, betas = self._sequential_obs_and_actions(env, target, search, tracks, prev_option, greedy)
            node_diag = self._node_diagnostics(obs, actions)
            for key, value in node_diag.items():
                diagnostics[key].append(value)
            prev_search = float(np.mean(search.search_belief))
            info = env.step(selected_waypoints)
            target.update(info.measurements.points, env.uav_positions)
            peaks = [] if self.cfg.disable_phd_belief else target.peaks()
            tracks.update(env.step_count, info.measurements.points, peaks)
            search.update(env.uav_positions, info.measurements.points)
            diagnostics["switch_rate"].append(float(np.mean(terminations.astype(np.float32))))
            diagnostics["mean_beta"].append(float(np.mean(betas)))
            diagnostics["option_0_ratio"].append(float(np.mean(options == 0)))
            diagnostics["option_1_ratio"].append(float(np.mean(options == 1)))
            diagnostics["search_belief_mean"].append(float(np.mean(search.search_belief)))
            diagnostics["coverage_age_mean"].append(float(np.mean(search.coverage_age)))
            diagnostics["target_estimated_count"].append(float(np.sum(target.weights)))
            diagnostics["track_count"].append(float(len(tracks.tracks)))
            cur_search = float(np.mean(search.search_belief))
            terms = reward_terms(
                self.cfg,
                env.memory,
                len(info.detected_ids),
                info.newly_discovered,
                info.continuous_observed,
                prev_search,
                cur_search,
                env.uav_positions,
                info.step_distance,
                terminations.astype(np.float32),
            )
            reward = weighted_reward(terms, self.cfg)
            diagnostics["reward_observe"].append(terms["observe"])
            diagnostics["reward_discover"].append(terms["discover"])
            diagnostics["reward_continuity"].append(terms["continuity"])
            diagnostics["reward_search"].append(terms["search"])
            diagnostics["reward_miss"].append(terms["miss"])
            diagnostics["reward_fairness_metric"].append(terms["fairness"])
            diagnostics["reward_overlap_metric"].append(terms["overlap"])
            diagnostics["reward_cost_metric"].append(terms["cost"])
            diagnostics["reward_switch_metric"].append(terms["switch"])
            rewards.append(reward)
            overlaps.append(terms["overlap"])
            next_batch = self.node_builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
            next_obs = self._obs_dict_from_arrays(
                next_batch.node_inputs,
                next_batch.node_padding_mask,
                next_batch.action_mask,
                env,
                target,
                search,
                options,
            )
            if replay is not None:
                transition = {
                    "node_inputs": obs["node_inputs"],
                    "node_padding_mask": obs["node_padding_mask"],
                    "action_mask": obs["action_mask"],
                    "uav_state": obs["uav_state"],
                    "prev_option": obs["prev_option"],
                    "team_summary": obs["team_summary"],
                    "global_phd": obs["global_phd"],
                    "global_search": obs["global_search"],
                    "actions": actions,
                    "options": options,
                    "terminations": terminations.astype(np.float32),
                    "log_probs": log_probs.astype(np.float32),
                    "values": values.astype(np.float32),
                    "reward": np.asarray(reward, dtype=np.float32),
                    "done": np.asarray(float(env.done()), dtype=np.float32),
                    "next_node_inputs": next_obs["node_inputs"],
                    "next_node_padding_mask": next_obs["node_padding_mask"],
                    "next_action_mask": next_obs["action_mask"],
                    "next_uav_state": next_obs["uav_state"],
                    "next_prev_option": next_obs["prev_option"],
                    "next_team_summary": next_obs["team_summary"],
                    "next_global_phd": next_obs["global_phd"],
                    "next_global_search": next_obs["global_search"],
                }
                replay.add(transition)
            prev_option = options.copy()
            if env.done():
                break
        metrics = final_metrics(
            self.cfg,
            env.memory,
            rewards,
            overlaps,
            float(np.sum(target.weights)),
            env.target_states[:, 0:2],
            target.peaks(),
        )
        metrics["episode_reward"] = float(np.sum(rewards))
        metrics["steps"] = env.step_count
        metrics["seed"] = float(seed)
        metrics["n_targets"] = float(len(env.target_states))
        for key, values in diagnostics.items():
            metrics[key] = float(np.mean(values)) if values else 0.0
        return metrics
