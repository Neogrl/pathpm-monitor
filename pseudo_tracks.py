from dataclasses import dataclass

import numpy as np

from config import Config
from target_belief import Peak


@dataclass
class PseudoTrack:
    track_id: int
    last_pos: np.ndarray
    last_velocity: np.ndarray
    current_gap: int
    confidence: float
    last_update_step: int
    source_type: str


class PseudoTrackMemory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: list[PseudoTrack] = []
        self.next_id = 0

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 0

    def _merged_observations(self, measurements: np.ndarray, peaks: list[Peak]) -> list[tuple[np.ndarray, str]]:
        raw: list[tuple[np.ndarray, str]] = [(m.astype(np.float32), "measurement") for m in measurements]
        raw.extend((p.pos.astype(np.float32), "phd_peak") for p in peaks)
        merged: list[tuple[np.ndarray, str, int]] = []
        threshold = min(self.cfg.pseudo_track_assoc_gate, self.cfg.fov_radius * 0.4)
        for pos, source in raw:
            if not merged:
                merged.append((pos.copy(), source, 1))
                continue
            dists = [float(np.linalg.norm(pos - item[0])) for item in merged]
            idx = int(np.argmin(dists))
            if dists[idx] <= threshold:
                old_pos, old_source, count = merged[idx]
                new_count = count + 1
                new_pos = (old_pos * count + pos) / new_count
                if source not in old_source:
                    old_source = old_source + "+" + source
                merged[idx] = (new_pos.astype(np.float32), old_source, new_count)
            else:
                merged.append((pos.copy(), source, 1))
        return [(pos, source) for pos, source, _ in merged]

    def update(self, step: int, measurements: np.ndarray, peaks: list[Peak]) -> None:
        observations = self._merged_observations(measurements, peaks)
        matched_tracks: set[int] = set()
        for obs, source in observations:
            candidates = [(i, np.linalg.norm(obs - t.last_pos)) for i, t in enumerate(self.tracks) if i not in matched_tracks]
            if candidates:
                idx, dist = min(candidates, key=lambda x: x[1])
            else:
                idx, dist = -1, float("inf")
            if idx >= 0 and dist <= self.cfg.pseudo_track_assoc_gate:
                track = self.tracks[idx]
                velocity = (obs - track.last_pos) / self.cfg.dt
                track.last_velocity = 0.7 * track.last_velocity + 0.3 * velocity
                track.last_pos = obs
                track.current_gap = 0
                track.confidence = min(1.0, track.confidence + 0.2)
                track.last_update_step = step
                track.source_type = source
                matched_tracks.add(idx)
            else:
                self.tracks.append(
                    PseudoTrack(
                        track_id=self.next_id,
                        last_pos=obs,
                        last_velocity=np.zeros(2, dtype=np.float32),
                        current_gap=0,
                        confidence=0.5,
                        last_update_step=step,
                        source_type=source,
                    )
                )
                self.next_id += 1
                matched_tracks.add(len(self.tracks) - 1)
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.current_gap += 1
                track.confidence *= 0.98
        self.tracks = [
            t
            for t in self.tracks
            if t.current_gap <= self.cfg.pseudo_track_expire_steps and t.confidence >= 0.15
        ]

    def maintenance_intents(self) -> list[tuple[np.ndarray, float]]:
        intents: list[tuple[np.ndarray, float]] = []
        for track in self.tracks:
            if track.confidence < self.cfg.maintenance_track_min_confidence:
                continue
            gap = min(track.current_gap, self.cfg.maintain_gap_threshold)
            pred = track.last_pos + gap * self.cfg.dt * track.last_velocity
            pred = np.clip(pred, 0.0, self.cfg.map_size)
            score = min(track.current_gap / max(self.cfg.maintain_gap_threshold, 1), 1.0) * track.confidence
            intents.append((pred.astype(np.float32), float(score)))
        intents.sort(key=lambda x: -x[1])
        return intents[: self.cfg.max_maintenance_candidates]
