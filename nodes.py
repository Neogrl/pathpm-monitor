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
    candidate_node_indices: np.ndarray
    valid_counts: np.ndarray


@dataclass
class GlobalGraphBatch:
    global_node_inputs: np.ndarray
    global_edge_mask: np.ndarray
    global_node_padding_mask: np.ndarray
    current_node_indices: np.ndarray
    candidate_node_indices: np.ndarray
    candidate_padding_mask: np.ndarray
    action_mask: np.ndarray
    node_positions: np.ndarray


NODE_INPUT_FIELDS = [
    "delta_x_norm",
    "delta_y_norm",
    "candidate_distance_norm",
    "coverage_age_value",
    "overlap",
    "target_belief_value",
]
NODE_INPUT_INDEX = {name: idx for idx, name in enumerate(NODE_INPUT_FIELDS)}
NODE_INPUT_DIM = len(NODE_INPUT_FIELDS)


class NodeBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.node_input_dim != NODE_INPUT_DIM:
            self.cfg.node_input_dim = NODE_INPUT_DIM
        self.graph = GlobalNodeGraph(cfg)
        self._last_update_step: Optional[int] = None

    def reset(self, seed: Optional[int] = None, start_positions: Optional[np.ndarray] = None) -> None:
        self.graph.reset(seed=seed, start_positions=start_positions)
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
        node_inputs = np.zeros((n, m, NODE_INPUT_DIM), dtype=np.float32)
        padding_mask = np.ones((n, m), dtype=bool)
        action_mask = np.ones((n, m), dtype=bool)
        waypoints = np.zeros((n, m, 2), dtype=np.float32)
        candidate_node_indices = -np.ones((n, m), dtype=np.int64)
        valid_counts = np.zeros(n, dtype=np.int32)
        if self._last_update_step != step:
            self.graph.update_from_beliefs(step, uav_positions, target_belief, search_belief, tracks)
            self._last_update_step = step
        for i, pos in enumerate(uav_positions):
            node_indices = self.graph.action_node_indices(pos, selected_waypoints)
            count = min(len(node_indices), m)
            node_indices = node_indices[:count]
            for j in range(count):
                graph_idx = int(node_indices[j])
                wp = self.graph.positions[graph_idx]
                dist = float(np.linalg.norm(wp - pos))
                waypoints[i, j] = wp
                candidate_node_indices[i, j] = graph_idx
                node_inputs[i, j] = self._features(
                    pos, wp, graph_idx, search_belief, uav_positions, selected_waypoints
                )
                padding_mask[i, j] = False
                too_far = self.cfg.graph_type.lower() != "prm" and dist > self.cfg.local_reachable_radius + 1e-6
                illegal = self._is_selected(wp, selected_waypoints) or too_far
                action_mask[i, j] = illegal
            if np.all(action_mask[i, :count]):
                graph_idx = self.graph.nearest_node_index(pos)
                wp = self.graph.positions[graph_idx]
                waypoints[i, 0] = wp
                candidate_node_indices[i, 0] = graph_idx
                node_inputs[i, 0] = self._features(
                    pos, wp, graph_idx, search_belief, uav_positions, selected_waypoints
                )
                padding_mask[i, 0] = False
                action_mask[i, 0] = False
                count = max(count, 1)
            valid_counts[i] = int(np.sum(~action_mask[i] & ~padding_mask[i]))
        return NodeBatch(
            node_inputs=node_inputs,
            node_padding_mask=padding_mask,
            action_mask=action_mask,
            waypoints=waypoints,
            candidate_node_indices=candidate_node_indices,
            valid_counts=valid_counts,
        )

    def build_global(
        self,
        uav_positions: np.ndarray,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        tracks: PseudoTrackMemory,
        selected_waypoints: Optional[list[np.ndarray]] = None,
        step: int = 0,
    ) -> GlobalGraphBatch:
        selected_waypoints = selected_waypoints or []
        if self._last_update_step != step:
            self.graph.update_from_beliefs(step, uav_positions, target_belief, search_belief, tracks)
            self._last_update_step = step
        candidate_batch = self.build(uav_positions, target_belief, search_belief, tracks, selected_waypoints, step)
        return self.global_batch_from_candidates(
            uav_positions,
            target_belief,
            search_belief,
            tracks,
            candidate_batch.candidate_node_indices,
            candidate_batch.node_padding_mask,
            candidate_batch.action_mask,
            step,
        )

    def global_batch_from_candidates(
        self,
        uav_positions: np.ndarray,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        tracks: PseudoTrackMemory,
        candidate_node_indices: np.ndarray,
        candidate_padding_mask: np.ndarray,
        action_mask: np.ndarray,
        step: int = 0,
    ) -> GlobalGraphBatch:
        if self._last_update_step != step:
            self.graph.update_from_beliefs(step, uav_positions, target_belief, search_belief, tracks)
            self._last_update_step = step
        return GlobalGraphBatch(
            global_node_inputs=self._global_node_inputs(uav_positions, target_belief, search_belief),
            global_edge_mask=self.graph.edge_mask(),
            global_node_padding_mask=self.graph.node_padding_mask(),
            current_node_indices=np.asarray([self.graph.nearest_node_index(pos) for pos in uav_positions], dtype=np.int64),
            candidate_node_indices=candidate_node_indices.astype(np.int64).copy(),
            candidate_padding_mask=candidate_padding_mask.astype(bool).copy(),
            action_mask=action_mask.astype(bool).copy(),
            node_positions=self.graph.positions.copy(),
        )

    def _global_node_inputs(
        self,
        uav_positions: np.ndarray,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
    ) -> np.ndarray:
        n = self.cfg.n_uavs
        g = self.graph.n_nodes
        positions = self.graph.positions
        out = np.zeros((n, g, NODE_INPUT_DIM), dtype=np.float32)
        out[:, :, NODE_INPUT_INDEX["delta_x_norm"]] = (positions[None, :, 0] - uav_positions[:, None, 0]) / self.cfg.map_size
        out[:, :, NODE_INPUT_INDEX["delta_y_norm"]] = (positions[None, :, 1] - uav_positions[:, None, 1]) / self.cfg.map_size
        direct_dist = np.linalg.norm(positions[None, :, :] - uav_positions[:, None, :], axis=2)
        out[:, :, NODE_INPUT_INDEX["candidate_distance_norm"]] = np.clip(direct_dist / self.cfg.map_size, 0.0, 2.0).astype(np.float32)

        if not self.cfg.disable_search_belief:
            age = np.clip(search_belief.coverage_age / self.cfg.search_age_scale, 0.0, 1.0).reshape(-1)
            denom = np.maximum(np.sum(self.graph.node_cell_fov_mask, axis=1), 1)
            age_value = (self.graph.node_cell_fov_mask @ age / denom).astype(np.float32)
            out[:, :, NODE_INPUT_INDEX["coverage_age_value"]] = age_value[None, :]

        if not self.cfg.disable_phd_belief:
            out[:, :, NODE_INPUT_INDEX["target_belief_value"]] = self.graph.target_value[None, :]

        for i, pos in enumerate(uav_positions):
            refs = [p for j, p in enumerate(uav_positions) if j != i and np.linalg.norm(p - pos) > 1e-6]
            if not refs:
                continue
            overlaps = [
                self._circle_overlap_ratio(np.linalg.norm(positions - ref[None, :], axis=1))
                for ref in refs
            ]
            out[i, :, NODE_INPUT_INDEX["overlap"]] = np.mean(np.stack(overlaps, axis=0), axis=0).astype(np.float32)
        return out

    def _circle_overlap_ratio(self, distances: np.ndarray) -> np.ndarray:
        d = np.asarray(distances, dtype=np.float32)
        r = float(self.cfg.fov_radius)
        out = np.zeros_like(d, dtype=np.float32)
        inside = d < 2.0 * r
        same_center = d <= 1e-8
        out[same_center] = 1.0
        regular = inside & ~same_center
        if np.any(regular):
            dd = d[regular].astype(np.float64)
            area = 2 * r * r * np.arccos(dd / (2 * r)) - 0.5 * dd * np.sqrt(np.maximum(4 * r * r - dd * dd, 0.0))
            out[regular] = (area / self.cfg.fov_area).astype(np.float32)
        return out

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
        search_belief: SearchBelief,
        uav_positions: np.ndarray,
        selected_waypoints: list[np.ndarray],
    ) -> np.ndarray:
        delta = waypoint - uav_pos
        dist = float(np.linalg.norm(delta))
        distance_norm = float(np.clip(dist / self.cfg.map_size, 0.0, 2.0))
        if self.cfg.disable_search_belief:
            age_value = 0.0
        else:
            _, age_value = search_belief.stats_in_fov(waypoint, self.cfg.fov_radius)
        refs = selected_waypoints if selected_waypoints else [p for p in uav_positions if np.linalg.norm(p - uav_pos) > 1e-6]
        overlaps = [circle_overlap_area(np.linalg.norm(waypoint - ref), self.cfg.fov_radius) / self.cfg.fov_area for ref in refs]
        overlap = float(np.mean(overlaps)) if overlaps else 0.0
        return np.asarray(
            [
                delta[0] / self.cfg.map_size,
                delta[1] / self.cfg.map_size,
                distance_norm,
                age_value,
                overlap,
                0.0 if self.cfg.disable_phd_belief else float(self.graph.target_value[graph_idx]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _is_selected(point: np.ndarray, selected: list[np.ndarray]) -> bool:
        return any(np.linalg.norm(point - s) < 1e-5 for s in selected)
