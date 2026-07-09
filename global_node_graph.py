from __future__ import annotations

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
    edge_mask: np.ndarray
    node_padding_mask: np.ndarray
    visited_count: np.ndarray
    last_covered_step: np.ndarray
    search_value: np.ndarray
    target_value: np.ndarray
    maintenance_value: np.ndarray
    target_flag: np.ndarray
    search_flag: np.ndarray
    maintenance_flag: np.ndarray


class GlobalNodeGraph:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.cell_centers = self._build_cell_centers()
        self._geometry_seed = seed
        self._path_cache: dict[tuple[int, int], list[int]] = {}
        self._tree_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._laplacian_pe_cache: dict[int, np.ndarray] = {}
        self._rebuild_geometry(seed=seed)

    def _build_grid_positions(self) -> np.ndarray:
        xs = np.arange(0.0, self.cfg.map_size + 1e-6, self.cfg.graph_node_spacing, dtype=np.float32)
        ys = np.arange(0.0, self.cfg.map_size + 1e-6, self.cfg.graph_node_spacing, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    def _build_boundary_positions(self) -> np.ndarray:
        count = max(int(self.cfg.prm_boundary_points_per_side), 2)
        values = np.linspace(0.0, self.cfg.map_size, count, dtype=np.float32)
        points = []
        for value in values:
            points.append((value, 0.0))
            points.append((value, self.cfg.map_size))
            points.append((0.0, value))
            points.append((self.cfg.map_size, value))
        return np.unique(np.asarray(points, dtype=np.float32), axis=0)

    def _build_prm_positions(self, seed: int | None, start_positions: np.ndarray | None = None) -> np.ndarray:
        rng = np.random.default_rng(0 if seed is None else seed)
        if self.cfg.prm_sampling.lower() == "stratified":
            positions = self._build_stratified_prm_positions(rng, include_boundary=False)
            return self._assemble_prm_positions(positions, start_positions)
        if self.cfg.prm_sampling.lower() != "uniform":
            raise ValueError(f"Unsupported prm_sampling={self.cfg.prm_sampling!r}; expected 'stratified' or 'uniform'")
        nodes: list[np.ndarray] = []
        min_dist = max(float(self.cfg.prm_min_node_distance), 0.0)
        max_attempts = max(int(self.cfg.prm_random_nodes) * 200, 1000)
        attempts = 0
        while len(nodes) < int(self.cfg.prm_random_nodes) and attempts < max_attempts:
            attempts += 1
            candidate = rng.uniform(0.0, self.cfg.map_size, size=2).astype(np.float32)
            if min_dist <= 0.0 or all(np.linalg.norm(candidate - node) >= min_dist for node in nodes):
                nodes.append(candidate)
        while len(nodes) < int(self.cfg.prm_random_nodes):
            nodes.append(rng.uniform(0.0, self.cfg.map_size, size=2).astype(np.float32))
        return self._assemble_prm_positions(np.stack(nodes, axis=0).astype(np.float32), start_positions)

    def _assemble_prm_positions(self, random_positions: np.ndarray, start_positions: np.ndarray | None) -> np.ndarray:
        parts = []
        if start_positions is not None:
            starts = np.asarray(start_positions, dtype=np.float32).reshape(-1, 2)
            parts.append(starts)
        parts.append(random_positions.astype(np.float32))
        if self.cfg.prm_include_boundary:
            parts.append(self._build_boundary_positions())
        return np.vstack(parts).astype(np.float32)

    def _build_stratified_prm_positions(self, rng: np.random.Generator, include_boundary: bool = True) -> np.ndarray:
        target = max(int(self.cfg.prm_random_nodes), 1)
        cols = int(np.ceil(np.sqrt(target)))
        rows = int(np.ceil(target / cols))
        cell_w = self.cfg.map_size / cols
        cell_h = self.cfg.map_size / rows
        jitter = float(np.clip(self.cfg.prm_jitter_ratio, 0.0, 0.45))
        nodes = []
        for r in range(rows):
            for c in range(cols):
                center = np.asarray([(c + 0.5) * cell_w, (r + 0.5) * cell_h], dtype=np.float32)
                offset = rng.uniform(-jitter, jitter, size=2).astype(np.float32) * np.asarray([cell_w, cell_h], dtype=np.float32)
                point = np.clip(center + offset, 0.0, self.cfg.map_size).astype(np.float32)
                nodes.append(point)
                if len(nodes) == target:
                    break
            if len(nodes) == target:
                break
        positions = np.stack(nodes, axis=0).astype(np.float32)
        if include_boundary and self.cfg.prm_include_boundary:
            positions = np.vstack([positions, self._build_boundary_positions()]).astype(np.float32)
        return positions

    def _build_positions(self, seed: int | None, start_positions: np.ndarray | None = None) -> np.ndarray:
        graph_type = self.cfg.graph_type.lower()
        if graph_type == "grid":
            return self._build_grid_positions()
        if graph_type == "prm":
            return self._build_prm_positions(seed, start_positions=start_positions)
        raise ValueError(f"Unsupported graph_type={self.cfg.graph_type!r}; expected 'grid' or 'prm'")

    def _rebuild_geometry(self, seed: int | None = None, start_positions: np.ndarray | None = None) -> None:
        self._geometry_seed = seed
        self.obstacles = self._build_obstacles(seed)
        self.positions = self._build_positions(seed, start_positions=start_positions)
        self.n_nodes = len(self.positions)
        self.node_passable = ~self._points_in_obstacles(self.positions)
        self.edge_indices = self._build_knn_edges()
        self.node_cell_fov_mask = (
            np.linalg.norm(self.positions[:, None, :] - self.cell_centers[None, :, :], axis=2) <= self.cfg.fov_radius
        )
        self.visited_count = np.zeros(self.n_nodes, dtype=np.float32)
        self.last_covered_step = -np.ones(self.n_nodes, dtype=np.float32)
        self.search_value = np.zeros(self.n_nodes, dtype=np.float32)
        self.target_value = np.zeros(self.n_nodes, dtype=np.float32)
        self.maintenance_value = np.zeros(self.n_nodes, dtype=np.float32)
        self.target_flag = np.zeros(self.n_nodes, dtype=bool)
        self.search_flag = np.zeros(self.n_nodes, dtype=bool)
        self.maintenance_flag = np.zeros(self.n_nodes, dtype=bool)
        self._path_cache.clear()
        self._tree_cache.clear()
        self._laplacian_pe_cache.clear()

    def _build_obstacles(self, seed: int | None) -> np.ndarray:
        if not self.cfg.obstacles_enabled or self.cfg.obstacle_count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        rng = np.random.default_rng((0 if seed is None else seed) + 10007)
        obstacles: list[tuple[float, float, float]] = []
        r_min = max(float(self.cfg.obstacle_radius_min), 0.0)
        r_max = max(float(self.cfg.obstacle_radius_max), r_min)
        margin = float(np.clip(self.cfg.obstacle_margin, r_max, self.cfg.map_size * 0.5))
        low = margin
        high = self.cfg.map_size - margin
        if high <= low:
            low, high = r_max, self.cfg.map_size - r_max
        attempts = 0
        max_attempts = max(500, int(self.cfg.obstacle_count) * 200)
        while len(obstacles) < int(self.cfg.obstacle_count) and attempts < max_attempts:
            attempts += 1
            radius = float(rng.uniform(r_min, r_max))
            center = rng.uniform(low, high, size=2)
            if all(np.linalg.norm(center - np.asarray([ox, oy])) >= radius + oradius + 3.0 for ox, oy, oradius in obstacles):
                obstacles.append((float(center[0]), float(center[1]), radius))
        while len(obstacles) < int(self.cfg.obstacle_count):
            radius = float(rng.uniform(r_min, r_max))
            center = rng.uniform(low, high, size=2)
            obstacles.append((float(center[0]), float(center[1]), radius))
        return np.asarray(obstacles, dtype=np.float32)

    def _points_in_obstacles(self, points: np.ndarray) -> np.ndarray:
        if len(self.obstacles) == 0:
            return np.zeros(len(points), dtype=bool)
        centers = self.obstacles[:, 0:2]
        radii = self.obstacles[:, 2] + max(float(self.cfg.obstacle_clearance), 0.0)
        dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        return np.any(dist <= radii[None, :], axis=1)

    def _edge_crosses_obstacle(self, a: np.ndarray, b: np.ndarray) -> bool:
        if len(self.obstacles) == 0:
            return False
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return bool(self._points_in_obstacles(a.reshape(1, 2))[0])
        for ox, oy, radius in self.obstacles:
            center = np.asarray([ox, oy], dtype=np.float32)
            t = float(np.clip(np.dot(center - a, ab) / denom, 0.0, 1.0))
            closest = a + t * ab
            if np.linalg.norm(closest - center) <= radius + max(float(self.cfg.obstacle_clearance), 0.0):
                return True
        return False

    def _build_cell_centers(self) -> np.ndarray:
        xs = (np.arange(self.cfg.search_bins) + 0.5) * self.cfg.cell_size
        ys = (np.arange(self.cfg.search_bins) + 0.5) * self.cfg.cell_size
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    def reset(self, seed: int | None = None, start_positions: np.ndarray | None = None) -> None:
        if self.cfg.graph_type.lower() == "prm":
            self._rebuild_geometry(seed=seed, start_positions=start_positions)
            return
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
        edge_indices = np.tile(np.arange(self.n_nodes, dtype=np.int64)[:, None], (1, k))
        max_radius = None
        if self.cfg.graph_type.lower() == "prm" and self.cfg.prm_edge_radius > 0:
            max_radius = self.cfg.prm_edge_radius
        for i in range(self.n_nodes):
            if not self.node_passable[i]:
                continue
            row = order[i]
            if max_radius is not None:
                keep = (row == i) | (dist[i, row] <= max_radius + 1e-6)
                row = row[keep]
            row = row[self.node_passable[row]]
            if len(self.obstacles) > 0:
                valid = []
                src = self.positions[i]
                for idx in row:
                    idx = int(idx)
                    if idx == i or not self._edge_crosses_obstacle(src, self.positions[idx]):
                        valid.append(idx)
                row = np.asarray(valid, dtype=np.int64)
            count = min(k, len(row))
            edge_indices[i, :count] = row[:count]
        return edge_indices.astype(np.int64)

    def edge_mask(self) -> np.ndarray:
        mask = np.ones((self.n_nodes, self.n_nodes), dtype=bool)
        for i in range(self.n_nodes):
            if self.node_passable[i]:
                valid = self.edge_indices[i][self.node_passable[self.edge_indices[i]]]
                mask[i, valid.astype(np.int64)] = False
            else:
                mask[i, i] = False
        return mask

    def node_padding_mask(self) -> np.ndarray:
        return (~self.node_passable).astype(bool)

    def set_node_passable(self, passable: np.ndarray) -> None:
        if passable.shape != (self.n_nodes,):
            raise ValueError(f"passable shape must be {(self.n_nodes,)}, got {passable.shape}")
        self.node_passable = passable.astype(bool).copy()
        self._path_cache.clear()
        self._tree_cache.clear()
        self._laplacian_pe_cache.clear()

    def laplacian_positional_encoding(self, dim: int) -> np.ndarray:
        dim = max(int(dim), 0)
        if dim == 0:
            return np.zeros((self.n_nodes, 0), dtype=np.float32)
        cached = self._laplacian_pe_cache.get(dim)
        if cached is not None:
            return cached.copy()
        adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for i in range(self.n_nodes):
            if not self.node_passable[i]:
                continue
            for nb in self.edge_indices[i, 1:]:
                nb = int(nb)
                if nb == i or not self.node_passable[nb]:
                    continue
                adjacency[i, nb] = 1.0
                adjacency[nb, i] = 1.0
        degree = adjacency.sum(axis=1)
        inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        valid = degree > 1e-8
        inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
        laplacian = np.eye(self.n_nodes, dtype=np.float32) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        try:
            _, eigenvectors = np.linalg.eigh(laplacian.astype(np.float64))
            pe = eigenvectors[:, 1 : dim + 1].astype(np.float32)
        except np.linalg.LinAlgError:
            pe = np.zeros((self.n_nodes, min(dim, max(self.n_nodes - 1, 0))), dtype=np.float32)
        if pe.shape[1] < dim:
            pe = np.pad(pe, ((0, 0), (0, dim - pe.shape[1])), mode="constant")
        pe[~self.node_passable] = 0.0
        self._laplacian_pe_cache[dim] = pe.astype(np.float32)
        return self._laplacian_pe_cache[dim].copy()

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

    def sample_start_positions(self, n_uavs: int, rng: np.random.Generator) -> np.ndarray:
        if self.cfg.graph_type.lower() == "prm":
            return self._sample_prm_start_positions(n_uavs, rng)
        margin = float(np.clip(self.cfg.uav_init_margin, 0.0, self.cfg.map_size * 0.49))
        candidates = self.positions[
            (self.positions[:, 0] >= margin)
            & (self.positions[:, 0] <= self.cfg.map_size - margin)
            & (self.positions[:, 1] >= margin)
            & (self.positions[:, 1] <= self.cfg.map_size - margin)
            & self.node_passable
        ]
        if len(candidates) == 0:
            candidates = self.positions[self.node_passable]
        if len(candidates) == 0:
            raise RuntimeError("No passable graph nodes available for UAV initialization.")
        min_sep = max(float(self.cfg.uav_init_min_separation), 0.0)
        positions: list[np.ndarray] = []
        max_attempts = max(200, int(n_uavs) * 200)
        for _ in range(max_attempts):
            candidate = candidates[int(rng.integers(0, len(candidates)))].astype(np.float32)
            if all(np.linalg.norm(candidate - pos) >= min_sep for pos in positions):
                positions.append(candidate)
                if len(positions) == n_uavs:
                    return np.stack(positions, axis=0).astype(np.float32)
        while len(positions) < n_uavs:
            positions.append(candidates[int(rng.integers(0, len(candidates)))].astype(np.float32))
        return np.stack(positions, axis=0).astype(np.float32)

    def _sample_prm_start_positions(self, n_uavs: int, rng: np.random.Generator) -> np.ndarray:
        margin = float(np.clip(self.cfg.uav_init_margin, 0.0, self.cfg.map_size * 0.49))
        low = margin
        high = self.cfg.map_size - margin
        if high <= low:
            low, high = 0.0, self.cfg.map_size
        min_sep = max(float(self.cfg.uav_init_min_separation), 0.0)
        positions: list[np.ndarray] = []
        max_attempts = max(500, int(n_uavs) * 300)
        for _ in range(max_attempts):
            candidate = rng.uniform(low, high, size=2).astype(np.float32)
            if self._points_in_obstacles(candidate.reshape(1, 2))[0]:
                continue
            if all(np.linalg.norm(candidate - pos) >= min_sep for pos in positions):
                positions.append(candidate)
                if len(positions) == n_uavs:
                    return np.stack(positions, axis=0).astype(np.float32)
        while len(positions) < n_uavs:
            candidate = rng.uniform(low, high, size=2).astype(np.float32)
            if not self._points_in_obstacles(candidate.reshape(1, 2))[0]:
                positions.append(candidate)
        return np.stack(positions, axis=0).astype(np.float32)

    def action_node_indices(self, uav_pos: np.ndarray, selected_waypoints: list[np.ndarray]) -> np.ndarray:
        current_idx = self.nearest_node_index(uav_pos)
        neighbor_indices = self.edge_indices[current_idx].astype(np.int64)
        dist = np.linalg.norm(self.positions[neighbor_indices] - uav_pos[None, :], axis=1)
        selected_mask = np.zeros(self.n_nodes, dtype=bool)
        for point in selected_waypoints:
            selected_mask |= np.linalg.norm(self.positions - point[None, :], axis=1) < self.cfg.merge_min_distance
        order = np.argsort(dist)
        ordered_neighbors = neighbor_indices[order]
        ordered_dist = dist[order]
        distance_ok = np.ones_like(ordered_dist, dtype=bool)
        if self.cfg.graph_type.lower() != "prm":
            distance_ok = ordered_dist <= self.cfg.local_reachable_radius + 1e-6
        reachable = ordered_neighbors[
            (ordered_neighbors != current_idx)
            & distance_ok
            & (~selected_mask[ordered_neighbors])
            & self.node_passable[ordered_neighbors]
        ]
        k = min(self.cfg.action_k_neighbors, self.cfg.max_node_candidates)
        return reachable[:k].astype(np.int64)

    def snapshot(self) -> GraphNodeState:
        return GraphNodeState(
            positions=self.positions.copy(),
            edge_indices=self.edge_indices.copy(),
            edge_mask=self.edge_mask(),
            node_padding_mask=self.node_padding_mask(),
            visited_count=self.visited_count.copy(),
            last_covered_step=self.last_covered_step.copy(),
            search_value=self.search_value.copy(),
            target_value=self.target_value.copy(),
            maintenance_value=self.maintenance_value.copy(),
            target_flag=self.target_flag.copy(),
            search_flag=self.search_flag.copy(),
            maintenance_flag=self.maintenance_flag.copy(),
        )
