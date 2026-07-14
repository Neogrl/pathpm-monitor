import numpy as np
import torch
from typing import Optional, Union

from config import Config
from environment import CMUOMMTEnv
from metrics import final_metrics, reward_terms, weighted_reward
from model import OptionActor
from nodes import NODE_INPUT_DIM, NODE_INPUT_INDEX, GlobalGraphBatch, NodeBatch, NodeBuilder
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief


def team_summary(cfg: Config, env: CMUOMMTEnv, target_belief: TargetBelief, search_belief: SearchBelief, selected: list[np.ndarray]) -> np.ndarray:
    uavs = env.uav_positions
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
            np.sum(target_belief.weights) / max(cfg.phd_prior_count, 1.0),
            np.mean(search_belief.search_belief),
            np.mean(search_belief.coverage_age) / cfg.search_age_scale,
            len(target_belief.peaks()) / max(cfg.max_target_candidates, 1) if not cfg.disable_phd_belief else 0.0,
        ],
        dtype=np.float32,
    )


def uav_state(cfg: Config, env: CMUOMMTEnv) -> np.ndarray:
    pos = env.uav_positions / cfg.map_size
    return pos.astype(np.float32)


class RolloutWorker:
    def __init__(self, cfg: Config, actor: Optional[OptionActor] = None, device: Union[torch.device, str] = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = actor
        self.node_builder = NodeBuilder(cfg)

    def reset_stack(self, seed: int, n_targets: Optional[int] = None, eval_mode: bool = False):
        self.node_builder.reset(seed=seed)
        start_rng = np.random.default_rng(seed + 909)
        uav_positions = self.node_builder.graph.sample_start_positions(self.cfg.n_uavs, start_rng)
        self.node_builder.reset(seed=seed, start_positions=uav_positions)
        env = CMUOMMTEnv(self.cfg)
        env.reset(seed=seed, n_targets=n_targets, uav_positions=uav_positions)
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

    def _joint_obs_and_actions(
        self,
        env: CMUOMMTEnv,
        target: TargetBelief,
        search: SearchBelief,
        tracks: PseudoTrackMemory,
        prev_option: np.ndarray,
        greedy: bool,
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, m = self.cfg.n_uavs, self.cfg.max_node_candidates
        node_inputs = np.zeros((n, m, NODE_INPUT_DIM), dtype=np.float32)
        node_padding_mask = np.ones((n, m), dtype=bool)
        action_mask = np.ones((n, m), dtype=bool)
        waypoints = np.zeros((n, m, 2), dtype=np.float32)
        candidate_node_indices = -np.ones((n, m), dtype=np.int64)
        actions = np.zeros(n, dtype=np.int64)
        options = prev_option.copy()
        terminations = np.zeros(n, dtype=bool)
        log_probs = np.zeros(n, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        betas = np.zeros(n, dtype=np.float32)
        batch = self.node_builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        node_inputs = batch.node_inputs.copy()
        node_padding_mask = batch.node_padding_mask.copy()
        action_mask = batch.action_mask.copy()
        waypoints = batch.waypoints.copy()
        candidate_node_indices = batch.candidate_node_indices.copy()
        global_batch = self.node_builder.global_batch_from_candidates(
            env.uav_positions,
            target,
            search,
            tracks,
            candidate_node_indices,
            node_padding_mask,
            action_mask,
            step=env.step_count,
        )
        obs = self._obs_dict_from_arrays(node_inputs, node_padding_mask, action_mask, env, target, search, prev_option, global_batch=global_batch)
        if self.actor is None:
            for i in range(n):
                valid = np.flatnonzero(~action_mask[i] & ~node_padding_mask[i])
                actions[i] = int(valid[0])
                options[i] = prev_option[i]
                terminations[i] = False
        else:
            torch_obs = self._to_torch(obs, batch_dim=True)
            if hasattr(self.actor, "act_with_info"):
                action_t, option_t, term_t, logp_t, value_t, beta_t = self.actor.act_with_info(**torch_obs, greedy=greedy)
            else:
                action_t, option_t, term_t = self.actor.act(**torch_obs, greedy=greedy)
                logp_t = torch.zeros_like(action_t, dtype=torch.float32)
                value_t = torch.zeros_like(action_t, dtype=torch.float32)
                beta_t = torch.zeros_like(action_t, dtype=torch.float32)
            actions = action_t[0].detach().cpu().numpy().astype(np.int64)
            options = option_t[0].detach().cpu().numpy().astype(np.int64)
            terminations = term_t[0].detach().cpu().numpy().astype(bool)
            log_probs = logp_t[0].detach().cpu().numpy().astype(np.float32)
            values = value_t[0].detach().cpu().numpy().astype(np.float32)
            betas = beta_t[0].detach().cpu().numpy().astype(np.float32)
        selected = waypoints[np.arange(n), actions].copy()
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
        global_batch: Optional[GlobalGraphBatch] = None,
    ) -> dict:
        obs = {
            "node_inputs": node_inputs.astype(np.float32),
            "node_padding_mask": node_padding_mask.astype(bool),
            "action_mask": action_mask.astype(bool),
            "uav_state": uav_state(self.cfg, env),
            "prev_option": prev_option.astype(np.int64),
            "global_phd": target.summary(),
            "global_search": search.summary(),
        }
        if global_batch is not None:
            obs.update(
                {
                    "global_node_inputs": global_batch.global_node_inputs.astype(np.float32),
                    "global_edge_mask": global_batch.global_edge_mask.astype(bool),
                    "global_node_padding_mask": global_batch.global_node_padding_mask.astype(bool),
                    "current_node_indices": global_batch.current_node_indices.astype(np.int64),
                    "candidate_node_indices": global_batch.candidate_node_indices.astype(np.int64),
                    "candidate_padding_mask": global_batch.candidate_padding_mask.astype(bool),
                    "global_action_mask": global_batch.action_mask.astype(bool),
                    "global_node_positions": global_batch.node_positions.astype(np.float32),
                }
            )
        return obs

    def _to_torch(self, obs: dict, batch_dim: bool = False) -> dict[str, torch.Tensor]:
        keys = [
            "global_node_inputs",
            "global_edge_mask",
            "global_node_padding_mask",
            "current_node_indices",
            "candidate_node_indices",
            "candidate_padding_mask",
            "action_mask",
            "uav_state",
            "prev_option",
        ]
        out = {}
        for key in keys:
            arr = obs[key]
            if batch_dim:
                arr = arr[None, ...]
            tensor = torch.as_tensor(arr, device=self.device)
            if key in ("global_edge_mask", "global_node_padding_mask", "candidate_padding_mask", "action_mask"):
                tensor = tensor.bool()
            elif key in ("prev_option", "current_node_indices", "candidate_node_indices"):
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            out[key] = tensor
        return out

    def _node_diagnostics(self, obs: dict, actions: np.ndarray) -> dict[str, float]:
        valid = ~obs["action_mask"] & ~obs["node_padding_mask"]
        features = obs["node_inputs"]
        selected_features = features[np.arange(self.cfg.n_uavs), actions]
        out = {
            "valid_candidates_mean": float(np.mean(np.sum(valid, axis=1))),
        }
        for name in (
            "candidate_distance_norm",
            "coverage_age_value",
            "overlap",
            "target_belief_value",
        ):
            idx = NODE_INPUT_INDEX[name]
            values = features[:, :, idx][valid]
            mean_key = f"{name}_mean" if name == "candidate_distance_norm" else f"candidate_{name}_mean"
            out[mean_key] = float(np.mean(values)) if len(values) else 0.0
            out[f"selected_{name}"] = float(np.mean(selected_features[:, idx])) if len(selected_features) else 0.0
        return out

    def _joint_action_diagnostics(self, obs: dict, actions: np.ndarray, selected_waypoints: np.ndarray) -> dict[str, float]:
        selected_indices = obs["candidate_node_indices"][np.arange(self.cfg.n_uavs), actions]
        pair_count = max(self.cfg.n_uavs * (self.cfg.n_uavs - 1) // 2, 1)
        same_node_pairs = 0
        near_overlap_pairs = 0
        for i in range(self.cfg.n_uavs):
            for j in range(i + 1, self.cfg.n_uavs):
                same_node_pairs += int(selected_indices[i] == selected_indices[j])
                dist = float(np.linalg.norm(selected_waypoints[i] - selected_waypoints[j]))
                near_overlap_pairs += int(dist < 2.0 * self.cfg.fov_radius)
        return {
            "same_node_conflict_rate": float(same_node_pairs / pair_count),
            "near_overlap_rate": float(near_overlap_pairs / pair_count),
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
        pending_transition: Optional[dict] = None
        diagnostics: dict[str, list[float]] = {
            "valid_candidates_mean": [],
            "candidate_distance_norm_mean": [],
            "candidate_coverage_age_value_mean": [],
            "candidate_overlap_mean": [],
            "selected_candidate_distance_norm": [],
            "selected_coverage_age_value": [],
            "selected_overlap": [],
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
            "reward_coverage": [],
            "reward_miss": [],
            "reward_fairness_metric": [],
            "reward_overlap_metric": [],
            "reward_cost_metric": [],
            "reward_switch_metric": [],
            "phd_position_error": [],
            "phd_number_error": [],
            "phd_total_weight": [],
            "phd_peak_count": [],
            "same_node_conflict_rate": [],
            "near_overlap_rate": [],
        }
        for _ in range(self.cfg.episode_steps):
            target.predict()
            obs, actions, options, terminations, selected_waypoints, log_probs, values, betas = self._joint_obs_and_actions(env, target, search, tracks, prev_option, greedy)
            if replay is not None and pending_transition is not None:
                # This policy call already produced V(s[t+1]), so reuse it instead of running a second full forward pass.
                pending_transition["next_values"] = values.astype(np.float32).copy()
                replay.add(pending_transition)
                pending_transition = None
            node_diag = self._node_diagnostics(obs, actions)
            for key, value in node_diag.items():
                diagnostics.setdefault(key, []).append(value)
            joint_diag = self._joint_action_diagnostics(obs, actions, selected_waypoints)
            for key, value in joint_diag.items():
                diagnostics.setdefault(key, []).append(value)
            previous_coverage_age = search.coverage_age.copy()
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
            terms = reward_terms(
                self.cfg,
                env.memory,
                len(info.detected_ids),
                info.newly_discovered,
                info.continuous_observed,
                peaks,
                env.target_states[:, 0:2],
                previous_coverage_age,
                env.uav_positions,
                info.step_distance,
                terminations.astype(np.float32),
            )
            reward = weighted_reward(terms, self.cfg)
            diagnostics["reward_observe"].append(terms["observe"])
            diagnostics["reward_discover"].append(terms["discover"])
            diagnostics["reward_continuity"].append(terms["continuity"])
            diagnostics["reward_search"].append(terms["search"])
            diagnostics["reward_coverage"].append(terms["coverage"])
            diagnostics["reward_miss"].append(terms["miss"])
            diagnostics["reward_fairness_metric"].append(terms["fairness"])
            diagnostics["reward_overlap_metric"].append(terms["overlap"])
            diagnostics["reward_cost_metric"].append(terms["cost"])
            diagnostics["reward_switch_metric"].append(terms["switch"])
            diagnostics["phd_position_error"].append(terms["phd_position_error"])
            diagnostics["phd_number_error"].append(terms["phd_number_error"])
            diagnostics["phd_total_weight"].append(float(np.sum(target.weights)))
            diagnostics["phd_peak_count"].append(float(len(peaks)))
            rewards.append(reward)
            overlaps.append(terms["overlap"])
            if replay is not None:
                pending_transition = {
                    "node_inputs": obs["node_inputs"],
                    "node_padding_mask": obs["node_padding_mask"],
                    "action_mask": obs["action_mask"],
                    "global_node_inputs": obs["global_node_inputs"],
                    "global_edge_mask": obs["global_edge_mask"],
                    "global_node_padding_mask": obs["global_node_padding_mask"],
                    "current_node_indices": obs["current_node_indices"],
                    "candidate_node_indices": obs["candidate_node_indices"],
                    "candidate_padding_mask": obs["candidate_padding_mask"],
                    "uav_state": obs["uav_state"],
                    "prev_option": obs["prev_option"],
                    "global_phd": obs["global_phd"],
                    "global_search": obs["global_search"],
                    "actions": actions,
                    "options": options,
                    "terminations": terminations.astype(np.float32),
                    "log_probs": log_probs.astype(np.float32),
                    "values": values.astype(np.float32),
                    "reward": np.asarray(reward, dtype=np.float32),
                    "done": np.asarray(float(env.done()), dtype=np.float32),
                }
            prev_option = options.copy()
            if env.done():
                break
        if replay is not None and pending_transition is not None:
            pending_transition["next_values"] = np.zeros_like(pending_transition["values"], dtype=np.float32)
            replay.add(pending_transition)
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
