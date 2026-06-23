from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import Config
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

    def build(
        self,
        uav_positions: np.ndarray,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        tracks: PseudoTrackMemory,
        selected_waypoints: Optional[list[np.ndarray]] = None,
    ) -> NodeBatch:
        selected_waypoints = selected_waypoints or []
        n = self.cfg.n_uavs
        m = self.cfg.max_node_candidates
        node_inputs = np.zeros((n, m, 16), dtype=np.float32)
        padding_mask = np.ones((n, m), dtype=bool)
        action_mask = np.ones((n, m), dtype=bool)
        waypoints = np.zeros((n, m, 2), dtype=np.float32)
        valid_counts = np.zeros(n, dtype=np.int32)
        intents = self._global_intents(target_belief, search_belief, tracks)
        for i, pos in enumerate(uav_positions):
            local_points, local_types = self._local_waypoints(pos, intents)
            count = min(len(local_points), m)
            if count == 0:
                local_points = [pos.copy()]
                local_types = [{"local": True}]
                count = 1
            for j in range(count):
                wp = local_points[j]
                waypoints[i, j] = wp
                node_inputs[i, j] = self._features(
                    pos, wp, local_types[j], target_belief, search_belief, tracks, uav_positions, selected_waypoints
                )
                padding_mask[i, j] = False
                illegal = self._is_selected(wp, selected_waypoints)
                action_mask[i, j] = illegal
            if np.all(action_mask[i, :count]):
                waypoints[i, 0] = pos
                node_inputs[i, 0] = self._features(pos, pos, {"local": True}, target_belief, search_belief, tracks, uav_positions, selected_waypoints)
                padding_mask[i, 0] = False
                action_mask[i, 0] = False
                count = max(count, 1)
            valid_counts[i] = int(np.sum(~action_mask[i] & ~padding_mask[i]))
        return NodeBatch(node_inputs=node_inputs, node_padding_mask=padding_mask, action_mask=action_mask, waypoints=waypoints, valid_counts=valid_counts)

    def _global_intents(self, target_belief: TargetBelief, search_belief: SearchBelief, tracks: PseudoTrackMemory) -> list[tuple[np.ndarray, str, float]]:
        intents: list[tuple[np.ndarray, str, float]] = []
        intents.extend((p.pos, "target", p.weight) for p in target_belief.peaks(self.cfg.max_target_candidates))
        intents.extend((pos, "maintenance", score) for pos, score in tracks.maintenance_intents())
        intents.extend((pos, "search", score) for pos, score in search_belief.peaks())
        intents.sort(key=lambda x: ({"maintenance": 0, "target": 1, "search": 2}[x[1]], -x[2]))
        merged: list[tuple[np.ndarray, str, float]] = []
        for pos, kind, score in intents:
            if all(np.linalg.norm(pos - p) >= self.cfg.merge_min_distance for p, _, _ in merged):
                merged.append((pos, kind, score))
        return merged

    def _local_waypoints(self, pos: np.ndarray, intents: list[tuple[np.ndarray, str, float]]) -> tuple[list[np.ndarray], list[dict]]:
        points: list[np.ndarray] = []
        types: list[dict] = []
        angles = np.linspace(0, 2 * np.pi, self.cfg.max_local_candidates_per_uav, endpoint=False)
        for angle in angles:
            point = pos + self.cfg.local_candidate_radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            points.append(np.clip(point, 0.0, self.cfg.map_size).astype(np.float32))
            types.append({"local": True})
        for intent_pos, kind, _ in intents:
            delta = intent_pos - pos
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-6:
                candidate = pos.astype(np.float32)
            elif dist <= self.cfg.local_reachable_radius:
                candidate = intent_pos.astype(np.float32)
            else:
                candidate = pos + self.cfg.local_candidate_radius * delta / dist
                candidate = np.clip(candidate, 0.0, self.cfg.map_size).astype(np.float32)
            merged = False
            for idx, point in enumerate(points):
                if np.linalg.norm(point - candidate) < self.cfg.merge_min_distance:
                    types[idx][kind] = True
                    merged = True
                    break
            if not merged:
                points.append(candidate)
                types.append({kind: True})
        return points, types

    def _features(
        self,
        uav_pos: np.ndarray,
        waypoint: np.ndarray,
        kind: dict,
        target_belief: TargetBelief,
        search_belief: SearchBelief,
        tracks: PseudoTrackMemory,
        uav_positions: np.ndarray,
        selected_waypoints: list[np.ndarray],
    ) -> np.ndarray:
        delta = waypoint - uav_pos
        dist = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0])) if dist > 1e-8 else 0.0
        particles, weights = target_belief.particles_in_fov(waypoint, self.cfg.fov_radius)
        target_weight = float(np.sum(weights))
        expected = np.clip(target_weight / max(self.cfg.phd_prior_count, 1.0), 0.0, 1.0)
        if target_weight > 1e-8:
            mean_v = np.sum(particles[:, 2:4] * weights[:, None], axis=0) / target_weight / self.cfg.target_speed
        else:
            mean_v = np.zeros(2, dtype=np.float32)
        search_value, age_value = search_belief.stats_in_fov(waypoint, self.cfg.fov_radius)
        maintenance = 0.0
        for track in tracks.tracks:
            if track.confidence < self.cfg.maintenance_track_min_confidence:
                continue
            gap_norm = min(track.current_gap / max(self.cfg.maintain_gap_threshold, 1), 1.0)
            pred = track.last_pos + min(track.current_gap, self.cfg.maintain_gap_threshold) * self.cfg.dt * track.last_velocity
            d2 = float(np.sum((pred - waypoint) ** 2))
            maintenance = max(maintenance, gap_norm * track.confidence * np.exp(-d2 / (2 * self.cfg.fov_radius ** 2)))
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
                search_value,
                age_value,
                overlap,
                maintenance,
                float(kind.get("target", False)),
                float(kind.get("search", False)),
                float(kind.get("maintenance", False)),
                float(kind.get("local", False)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _is_selected(point: np.ndarray, selected: list[np.ndarray]) -> bool:
        return any(np.linalg.norm(point - s) < 1e-5 for s in selected)
