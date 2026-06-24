from dataclasses import dataclass
import heapq

import numpy as np

from config import Config
from pseudo_tracks import PseudoTrackMemory
from search_belief import SearchBelief
from target_belief import TargetBelief


@dataclass
class GraphNodeState:
    positions: np.ndarray
    edge_indices: np.ndarray
    visited_count: np.ndarray
    last_covered_step: np.ndarray
    search_value: np.ndarray
    target_value: np.ndarray
    maintenance_value: np.ndarray
    target_flag: np.ndarray
    search_flag: np.ndarray
    maintenance_flag: np.ndarray


class GlobalNodeGraph:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        xs = np.arange(0.0, cfg.map_size + 1e-6, cfg.graph_node_spacing, dtype=np.float32)
        ys = np.arange(0.0, cfg.map_size + 1e-6, cfg.graph_node_spacing, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        self.positions = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
        self.n_nodes = len(self.positions)
        self.edge_indices = self._build_knn_edges()
        self.cell_centers = self._build_cell_centers()
        self.node_cell_fov_mask = (
            np.linalg.norm(self.positions[:, None, :] - self.cell_centers[None, :, :], axis=2) <= cfg.fov_radius
        )
        self.visited_count = np.zeros(self.n_nodes, dtype=np.float32)
        self.last_covered_step = -np.ones(self.n_nodes, dtype=np.float32)
        self.search_value = np.zeros(self.n_nodes, dtype=np.float32)
        self.target_value = np.zeros(self.n_nodes, dtype=np.float32)
        self.maintenance_value = np.zeros(self.n_nodes, dtype=np.float32)
        self.target_flag = np.zeros(self.n_nodes, dtype=bool)
        self.search_flag = np.zeros(self.n_nodes, dtype=bool)
        self.maintenance_flag = np.zeros(self.n_nodes, dtype=bool)
        self.node_passable = np.ones(self.n_nodes, dtype=bool)
        self._path_cache: dict[tuple[int, int], list[int]] = {}
        self._tree_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _build_cell_centers(self) -> np.ndarray:
        xs = (np.arange(self.cfg.search_bins) + 0.5) * self.cfg.cell_size
        ys = (np.arange(self.cfg.search_bins) + 0.5) * self.cfg.cell_size
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    def reset(self) -> None:
        self.visited_count[:] = 0.0
        self.last_covered_step[:] = -1.0
        self.search_value[:] = 0.0
        self.target_value[:] = 0.0
        self.maintenance_value[:] = 0.0
        self.target_flag[:] = False
        self.search_flag[:] = False
        self.maintenance_flag[:] = False

    def _build_knn_edges(self) -> np.ndarray:
        dist = np.linalg.norm(self.positions[:, None, :] - self.positions[None, :, :], axis=2)
        order = np.argsort(dist, axis=1)
        k = min(self.cfg.k_neighbors + 1, self.n_nodes)
        return order[:, :k].astype(np.int64)

    def set_node_passable(self, passable: np.ndarray) -> None:
        if passable.shape != (self.n_nodes,):
            raise ValueError(f"passable shape must be {(self.n_nodes,)}, got {passable.shape}")
        self.node_passable = passable.astype(bool).copy()
        self._path_cache.clear()
        self._tree_cache.clear()

    def _neighbors(self, node_idx: int) -> list[tuple[int, float]]:
        if not self.node_passable[node_idx]:
            return []
        src = self.positions[node_idx]
        out: list[tuple[int, float]] = []
        for nb in self.edge_indices[node_idx, 1:]:
            nb = int(nb)
            if nb == node_idx or not self.node_passable[nb]:
                continue
            cost = float(np.linalg.norm(src - self.positions[nb]))
            if cost > 0:
                out.append((nb, cost))
        return out

    def shortest_path(self, start_idx: int, goal_idx: int) -> list[int]:
        start_idx = int(start_idx)
        goal_idx = int(goal_idx)
        cache_key = (start_idx, goal_idx)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        if start_idx == goal_idx:
            self._path_cache[cache_key] = [start_idx]
            return self._path_cache[cache_key]
        if not self.node_passable[start_idx] or not self.node_passable[goal_idx]:
            self._path_cache[cache_key] = []
            return []
        dist = np.full(self.n_nodes, np.inf, dtype=np.float32)
        prev = -np.ones(self.n_nodes, dtype=np.int64)
        dist[start_idx] = 0.0
        heap: list[tuple[float, int]] = [(0.0, start_idx)]
        while heap:
            cur_dist, cur = heapq.heappop(heap)
            if cur == goal_idx:
                break
            if cur_dist > float(dist[cur]):
                continue
            for nb, cost in self._neighbors(cur):
                nxt = cur_dist + cost
                if nxt < float(dist[nb]):
                    dist[nb] = nxt
                    prev[nb] = cur
                    heapq.heappush(heap, (nxt, nb))
        if not np.isfinite(dist[goal_idx]):
            self._path_cache[cache_key] = []
            return []
        path = [goal_idx]
        cur = goal_idx
        while cur != start_idx:
            cur = int(prev[cur])
            if cur < 0:
                self._path_cache[cache_key] = []
                return []
            path.append(cur)
        path.reverse()
        self._path_cache[cache_key] = path
        return path

    def shortest_tree_from(self, start_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start_idx = int(start_idx)
        if start_idx in self._tree_cache:
            return self._tree_cache[start_idx]
        dist = np.full(self.n_nodes, np.inf, dtype=np.float32)
        prev = -np.ones(self.n_nodes, dtype=np.int64)
        if not self.node_passable[start_idx]:
            self._tree_cache[start_idx] = (dist, prev)
            return dist, prev
        dist[start_idx] = 0.0
        heap: list[tuple[float, int]] = [(0.0, start_idx)]
        while heap:
            cur_dist, cur = heapq.heappop(heap)
            if cur_dist > float(dist[cur]):
                continue
            for nb, cost in self._neighbors(cur):
                nxt = cur_dist + cost
                if nxt < float(dist[nb]):
                    dist[nb] = nxt
                    prev[nb] = cur
                    heapq.heappush(heap, (nxt, nb))
        self._tree_cache[start_idx] = (dist, prev)
        return dist, prev

    def project_path_to_candidates(self, start_idx: int, goal_idx: int, candidate_indices: np.ndarray) -> int:
        if len(candidate_indices) == 0:
            return -1
        lookup = {int(idx): j for j, idx in enumerate(candidate_indices)}
        path = self.shortest_path(start_idx, goal_idx)
        if path:
            for node_idx in path[1:]:
                if int(node_idx) in lookup:
                    return lookup[int(node_idx)]
            if len(path) == 1 and int(path[0]) in lookup:
                return lookup[int(path[0])]
        goal = self.positions[int(goal_idx)]
        cand = self.positions[candidate_indices.astype(np.int64)]
        return int(np.argmin(np.linalg.norm(cand - goal[None, :], axis=1)))

    def project_tree_path_to_candidates(
        self,
        start_idx: int,
        goal_idx: int,
        candidate_indices: np.ndarray,
        dist: np.ndarray,
        prev: np.ndarray,
    ) -> int:
        if len(candidate_indices) == 0:
            return -1
        lookup = {int(idx): j for j, idx in enumerate(candidate_indices)}
        if not np.isfinite(dist[int(goal_idx)]):
            goal = self.positions[int(goal_idx)]
            cand = self.positions[candidate_indices.astype(np.int64)]
            return int(np.argmin(np.linalg.norm(cand - goal[None, :], axis=1)))
        cur = int(goal_idx)
        path = []
        while cur >= 0:
            path.append(cur)
            if cur == int(start_idx):
                break
            cur = int(prev[cur])
        for node_idx in reversed(path[:-1]):
            if int(node_idx) in lookup:
                return lookup[int(node_idx)]
        if int(start_idx) in lookup:
            return lookup[int(start_idx)]
        goal = self.positions[int(goal_idx)]
        cand = self.positions[candidate_indices.astype(np.int64)]
        return int(np.argmin(np.linalg.norm(cand - goal[None, :], axis=1)))

    def update_from_beliefs(
        self,
        step: int,
        uav_positions: np.ndarray,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        tracks: PseudoTrackMemory,
    ) -> None:
        self._update_coverage(step, uav_positions)
        self._update_search(search_belief)
        self._update_target(target_belief)
        self._update_maintenance(tracks)

    def _update_coverage(self, step: int, uav_positions: np.ndarray) -> None:
        dist = np.linalg.norm(self.positions[:, None, :] - uav_positions[None, :, :], axis=2)
        covered = np.any(dist <= self.cfg.fov_radius, axis=1)
        self.visited_count[covered] += 1.0
        self.last_covered_step[covered] = float(step)

    def _update_search(self, search_belief: SearchBelief) -> None:
        score = search_belief.score().reshape(-1)
        denom = np.maximum(np.sum(self.node_cell_fov_mask, axis=1), 1)
        self.search_value = (self.node_cell_fov_mask @ score / denom).astype(np.float32)
        self.search_flag[:] = False
        for pos, _ in search_belief.peaks():
            idx = self.nearest_node_index(pos)
            self.search_flag[idx] = True

    def _update_target(self, target_belief: TargetBelief) -> None:
        dist = np.linalg.norm(self.positions[:, None, :] - target_belief.particles[None, :, 0:2], axis=2)
        in_fov = dist <= self.cfg.fov_radius
        self.target_value = np.clip((in_fov @ target_belief.weights) / max(self.cfg.phd_prior_count, 1.0), 0.0, 1.0).astype(np.float32)
        self.target_flag[:] = False
        for peak in target_belief.peaks(self.cfg.max_target_candidates):
            idx = self.nearest_node_index(peak.pos)
            self.target_flag[idx] = True

    def _update_maintenance(self, tracks: PseudoTrackMemory) -> None:
        self.maintenance_value[:] = 0.0
        self.maintenance_flag[:] = False
        for pos, score in tracks.maintenance_intents():
            dist2 = np.sum((self.positions - pos[None, :]) ** 2, axis=1)
            value = float(score) * np.exp(-dist2 / (2 * self.cfg.fov_radius ** 2))
            self.maintenance_value = np.maximum(self.maintenance_value, value.astype(np.float32))
            self.maintenance_flag[self.nearest_node_index(pos)] = True

    def nearest_node_index(self, point: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.positions - point[None, :]) ** 2, axis=1)))

    def action_node_indices(self, uav_pos: np.ndarray, selected_waypoints: list[np.ndarray]) -> np.ndarray:
        dist = np.linalg.norm(self.positions - uav_pos[None, :], axis=1)
        selected_mask = np.zeros(self.n_nodes, dtype=bool)
        for point in selected_waypoints:
            selected_mask |= np.linalg.norm(self.positions - point[None, :], axis=1) < self.cfg.merge_min_distance
        order = np.argsort(dist)
        reachable = order[
            (dist[order] <= self.cfg.local_reachable_radius + 1e-6)
            & (~selected_mask[order])
            & self.node_passable[order]
        ]
        k = min(self.cfg.action_k_neighbors, self.cfg.max_node_candidates)
        return reachable[:k].astype(np.int64)

    def snapshot(self) -> GraphNodeState:
        return GraphNodeState(
            positions=self.positions.copy(),
            edge_indices=self.edge_indices.copy(),
            visited_count=self.visited_count.copy(),
            last_covered_step=self.last_covered_step.copy(),
            search_value=self.search_value.copy(),
            target_value=self.target_value.copy(),
            maintenance_value=self.maintenance_value.copy(),
            target_flag=self.target_flag.copy(),
            search_flag=self.search_flag.copy(),
            maintenance_flag=self.maintenance_flag.copy(),
        )
