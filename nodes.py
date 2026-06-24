from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import Config
from global_node_graph import GlobalNodeGraph
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief
from utils import circle_overlap_area


@dataclass
class NodeBatch:
    node_inputs: np.ndarray
    node_padding_mask: np.ndarray
    action_mask: np.ndarray
    waypoints: np.ndarray
    valid_counts: np.ndarray


class NodeBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graph = GlobalNodeGraph(cfg)
        self._last_update_step: Optional[int] = None

    def reset(self) -> None:
        self.graph.reset()
        self._last_update_step = None

    def build(
        self,
        uav_positions: np.ndarray,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        tracks: PseudoTrackMemory,
        selected_waypoints: Optional[list[np.ndarray]] = None,
        step: int = 0,
    ) -> NodeBatch:
        selected_waypoints = selected_waypoints or []
        n = self.cfg.n_uavs
        m = self.cfg.max_node_candidates
        node_inputs = np.zeros((n, m, 16), dtype=np.float32)
        padding_mask = np.ones((n, m), dtype=bool)
        action_mask = np.ones((n, m), dtype=bool)
        waypoints = np.zeros((n, m, 2), dtype=np.float32)
        valid_counts = np.zeros(n, dtype=np.int32)
        if self._last_update_step != step:
            self.graph.update_from_beliefs(step, uav_positions, target_belief, search_belief, tracks)
            self._last_update_step = step
        intents = self._global_intents(target_belief, search_belief, tracks)
        for i, pos in enumerate(uav_positions):
            node_indices = self.graph.action_node_indices(pos, selected_waypoints)
            count = min(len(node_indices), m)
            node_indices = node_indices[:count]
            intent_signals = self._project_intents_to_candidates(pos, node_indices, intents)
            for j in range(count):
                graph_idx = int(node_indices[j])
                wp = self.graph.positions[graph_idx]
                dist = float(np.linalg.norm(wp - pos))
                waypoints[i, j] = wp
                node_inputs[i, j] = self._features(
                    pos, wp, graph_idx, intent_signals[j], target_belief, search_belief, uav_positions, selected_waypoints
                )
                padding_mask[i, j] = False
                illegal = self._is_selected(wp, selected_waypoints) or dist > self.cfg.local_reachable_radius + 1e-6
                action_mask[i, j] = illegal
            if np.all(action_mask[i, :count]):
                graph_idx = self.graph.nearest_node_index(pos)
                wp = pos.astype(np.float32)
                waypoints[i, 0] = wp
                node_inputs[i, 0] = self._features(pos, wp, graph_idx, {}, target_belief, search_belief, uav_positions, selected_waypoints)
                padding_mask[i, 0] = False
                action_mask[i, 0] = False
                count = max(count, 1)
            valid_counts[i] = int(np.sum(~action_mask[i] & ~padding_mask[i]))
        return NodeBatch(node_inputs=node_inputs, node_padding_mask=padding_mask, action_mask=action_mask, waypoints=waypoints, valid_counts=valid_counts)

    def _global_intents(self, target_belief: TargetBelief, search_belief: SearchBelief, tracks: PseudoTrackMemory) -> list[tuple[np.ndarray, str, float]]:
        intents: list[tuple[np.ndarray, str, float]] = []
        if not self.cfg.disable_phd_belief:
            intents.extend((p.pos, "target", p.weight) for p in target_belief.peaks(self.cfg.max_target_candidates))
        intents.extend((pos, "maintenance", score) for pos, score in tracks.maintenance_intents())
        if not self.cfg.disable_search_belief:
            intents.extend((pos, "search", score) for pos, score in search_belief.peaks())
        intents.sort(key=lambda x: ({"maintenance": 0, "target": 1, "search": 2}[x[1]], -x[2]))
        merged: list[tuple[np.ndarray, str, float]] = []
        for pos, kind, score in intents:
            if all(np.linalg.norm(pos - p) >= self.cfg.merge_min_distance for p, _, _ in merged):
                merged.append((pos.astype(np.float32), kind, float(score)))
        return merged

    def _project_intents_to_candidates(
        self,
        uav_pos: np.ndarray,
        candidate_indices: np.ndarray,
        intents: list[tuple[np.ndarray, str, float]],
    ) -> list[dict[str, float]]:
        # Project each global intent onto the local action set through the traversable graph.
        signals: list[dict[str, float]] = [dict() for _ in range(len(candidate_indices))]
        if len(candidate_indices) == 0:
            return signals
        start_idx = self.graph.nearest_node_index(uav_pos)
        dist, prev = self.graph.shortest_tree_from(start_idx)
        for intent_pos, kind, score in intents:
            goal_idx = self.graph.nearest_node_index(intent_pos)
            local_j = self.graph.project_tree_path_to_candidates(start_idx, goal_idx, candidate_indices, dist, prev)
            if local_j < 0:
                continue
            signals[local_j][kind] = max(float(score), signals[local_j].get(kind, 0.0))
        return signals

    def _features(
        self,
        uav_pos: np.ndarray,
        waypoint: np.ndarray,
        graph_idx: int,
        intent_signal: dict[str, float],
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        uav_positions: np.ndarray,
        selected_waypoints: list[np.ndarray],
    ) -> np.ndarray:
        delta = waypoint - uav_pos
        dist = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0])) if dist > 1e-8 else 0.0
        if self.cfg.disable_phd_belief:
            expected = 0.0
            mean_v = np.zeros(2, dtype=np.float32)
        else:
            particles, weights = target_belief.particles_in_fov(waypoint, self.cfg.fov_radius)
            target_weight = float(np.sum(weights))
            expected = np.clip(target_weight / max(self.cfg.phd_prior_count, 1.0), 0.0, 1.0)
            if target_weight > 1e-8:
                mean_v = np.sum(particles[:, 2:4] * weights[:, None], axis=0) / target_weight / self.cfg.target_speed
            else:
                mean_v = np.zeros(2, dtype=np.float32)
        if self.cfg.disable_search_belief:
            search_value, age_value = 0.0, 0.0
        else:
            search_value, age_value = search_belief.stats_in_fov(waypoint, self.cfg.fov_radius)
        maintenance = float(self.graph.maintenance_value[graph_idx])
        target_flag = float("target" in intent_signal)
        search_flag = float("search" in intent_signal)
        maintenance_flag = float("maintenance" in intent_signal)
        goal_signal = float((target_flag > 0.5) or (search_flag > 0.5) or (maintenance_flag > 0.5))
        refs = selected_waypoints if selected_waypoints else [p for p in uav_positions if np.linalg.norm(p - uav_pos) > 1e-6]
        overlaps = [circle_overlap_area(np.linalg.norm(waypoint - ref), self.cfg.fov_radius) / self.cfg.fov_area for ref in refs]
        overlap = float(np.mean(overlaps)) if overlaps else 0.0
        return np.asarray(
            [
                delta[0] / self.cfg.map_size,
                delta[1] / self.cfg.map_size,
                dist / max(self.cfg.uav_speed, 1e-8),
                np.sin(angle),
                np.cos(angle),
                expected,
                mean_v[0],
                mean_v[1],
                max(search_value, float(self.graph.search_value[graph_idx])) if not self.cfg.disable_search_belief else 0.0,
                age_value,
                overlap,
                maintenance,
                target_flag,
                search_flag,
                maintenance_flag,
                goal_signal,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _is_selected(point: np.ndarray, selected: list[np.ndarray]) -> bool:
        return any(np.linalg.norm(point - s) < 1e-5 for s in selected)
