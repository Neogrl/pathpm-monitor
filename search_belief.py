import numpy as np

from config import Config
from utils import local_maxima_2d, non_max_suppression


class SearchBelief:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.search_belief = np.zeros((cfg.search_bins, cfg.search_bins), dtype=np.float32)
        self.coverage_age = np.zeros((cfg.search_bins, cfg.search_bins), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.search_belief = np.full((self.cfg.search_bins, self.cfg.search_bins), self.cfg.search_belief_init, dtype=np.float32)
        self.coverage_age = np.full((self.cfg.search_bins, self.cfg.search_bins), self.cfg.coverage_age_init, dtype=np.float32)

    def update(self, uav_positions: np.ndarray, measurement_points: np.ndarray) -> None:
        self.coverage_age += 1.0
        self.search_belief = np.minimum(1.0, self.search_belief + self.cfg.search_growth)
        centers = self.cell_centers()
        covered = np.zeros((self.cfg.search_bins, self.cfg.search_bins), dtype=bool)
        flat = centers.reshape(-1, 2)
        for pos in uav_positions:
            mask = np.linalg.norm(flat - pos[None, :], axis=1) <= self.cfg.fov_radius
            covered |= mask.reshape(self.cfg.search_bins, self.cfg.search_bins)
        self.coverage_age[covered] = 0.0
        self.search_belief[covered] = np.maximum(self.cfg.search_min, self.search_belief[covered] * self.cfg.search_decay_covered)
        for point in measurement_points:
            x, y = self.point_to_cell(point)
            self.coverage_age[y, x] = 0.0
            self.search_belief[y, x] = self.cfg.search_min

    def score(self) -> np.ndarray:
        age = np.clip(self.coverage_age / self.cfg.search_age_scale, 0.0, 1.0)
        return self.search_belief * (1.0 + age)

    def peaks(self) -> list[tuple[np.ndarray, float]]:
        score = self.score()
        coords, scores = local_maxima_2d(score, self.cfg.search_candidate_min_score)
        if len(coords) == 0:
            return []
        points = (coords + 0.5) * self.cfg.cell_size
        keep = non_max_suppression(points, scores, self.cfg.search_candidate_min_separation, self.cfg.max_search_candidates)
        return [(points[i].astype(np.float32), float(scores[i])) for i in keep]

    def point_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        cell = np.clip((point / self.cfg.cell_size).astype(int), 0, self.cfg.search_bins - 1)
        return int(cell[0]), int(cell[1])

    def cell_centers(self) -> np.ndarray:
        xs = (np.arange(self.cfg.search_bins) + 0.5) * self.cfg.cell_size
        ys = (np.arange(self.cfg.search_bins) + 0.5) * self.cfg.cell_size
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx, yy], axis=-1).astype(np.float32)

    def stats_in_fov(self, center: np.ndarray, radius: float) -> tuple[float, float]:
        centers = self.cell_centers().reshape(-1, 2)
        mask = np.linalg.norm(centers - center[None, :], axis=1) <= radius
        if not np.any(mask):
            return 0.0, 0.0
        score = self.score().reshape(-1)
        age = np.clip(self.coverage_age / self.cfg.search_age_scale, 0.0, 1.0).reshape(-1)
        return float(np.mean(score[mask])), float(np.mean(age[mask]))

    def summary(self) -> np.ndarray:
        age = np.clip(self.coverage_age / self.cfg.search_age_scale, 0.0, 1.0)
        return np.asarray(
            [np.mean(self.search_belief), np.max(self.search_belief), np.mean(age), np.max(age)],
            dtype=np.float32,
        )
