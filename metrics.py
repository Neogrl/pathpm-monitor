import numpy as np

from config import Config
from environment import DiscoveredTruthMemory
from target_belief import Peak
from utils import circle_overlap_area


def reward_terms(
    cfg: Config,
    memory: DiscoveredTruthMemory,
    detected_count: int,
    newly_discovered: int,
    continuous_observed: int,
    previous_search_mean: float,
    current_search_mean: float,
    uav_positions: np.ndarray,
    step_distance: np.ndarray,
    option_switched: np.ndarray,
) -> dict[str, float]:
    n_targets = len(memory.is_discovered)
    discovered_count = int(np.sum(memory.is_discovered))
    if discovered_count == 0:
        fairness = 0.0
        miss = 0.0
    else:
        counts = memory.observation_count[memory.is_discovered]
        fairness = float(np.clip(1.0 - np.std(counts) / (np.mean(counts) + 1e-8), 0.0, 1.0))
        miss = float(np.mean(memory.current_gap[memory.is_discovered] > cfg.maintain_gap_threshold))
    overlaps = []
    for i in range(len(uav_positions)):
        for j in range(i + 1, len(uav_positions)):
            overlaps.append(circle_overlap_area(np.linalg.norm(uav_positions[i] - uav_positions[j]), cfg.fov_radius) / cfg.fov_area)
    return {
        "observe": detected_count / max(n_targets, 1),
        "discover": newly_discovered / max(n_targets, 1),
        "fairness": fairness,
        "continuity": continuous_observed / max(discovered_count, 1),
        "search": max(0.0, previous_search_mean - current_search_mean),
        "overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "miss": miss,
        "cost": float(np.mean(step_distance / max(cfg.uav_speed, 1e-8))),
        "switch": float(np.mean(option_switched)),
    }


def weighted_reward(terms: dict[str, float]) -> float:
    return (
        1.0 * terms["observe"]
        + 0.8 * terms["discover"]
        + 0.55 * terms["fairness"]
        + 0.35 * terms["continuity"]
        + 0.30 * terms["search"]
        - 0.35 * terms["overlap"]
        - 0.50 * terms["miss"]
        - 0.05 * terms["cost"]
        - 0.02 * terms["switch"]
    )


def ospa_distance(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray,
    cutoff: float,
    order: int = 1,
) -> float:
    m = int(len(estimated_positions))
    n = int(len(true_positions))
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(cutoff)

    dist = np.linalg.norm(estimated_positions[:, None, :] - true_positions[None, :, :], axis=2)
    dist = np.minimum(dist, cutoff) ** order
    if m <= n:
        smaller, larger = m, n
        costs = dist
    else:
        smaller, larger = n, m
        costs = dist.T

    dp = {0: 0.0}
    for i in range(smaller):
        next_dp = {}
        for mask, cost in dp.items():
            for j in range(larger):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = cost + float(costs[i, j])
                if new_mask not in next_dp or new_cost < next_dp[new_mask]:
                    next_dp[new_mask] = new_cost
        dp = next_dp
    assign_cost = min(dp.values()) if dp else 0.0
    cardinality_cost = (larger - smaller) * (cutoff ** order)
    return float(((assign_cost + cardinality_cost) / max(m, n)) ** (1.0 / order))


def final_metrics(
    cfg: Config,
    memory: DiscoveredTruthMemory,
    rewards: list[float],
    overlap_values: list[float],
    estimated_count: float,
    true_positions: np.ndarray,
    estimated_peaks: list[Peak],
) -> dict[str, float]:
    n_targets = int(len(true_positions))
    discovered = memory.is_discovered
    first = memory.first_detection_step[discovered]
    gaps = memory.max_observation_gap[discovered] if np.any(discovered) else np.asarray([0])
    counts = memory.observation_count[discovered] if np.any(discovered) else np.asarray([0])
    fairness = 0.0 if len(counts) == 0 or np.mean(counts) == 0 else float(np.clip(1 - np.std(counts) / (np.mean(counts) + 1e-8), 0, 1))
    estimated_positions = np.asarray([peak.pos for peak in estimated_peaks], dtype=np.float32).reshape(-1, 2)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "discovery_rate": float(np.mean(discovered)) if len(discovered) else 0.0,
        "mean_first_detection_time": float(np.mean(first)) if len(first) else float("nan"),
        "observation_rate": float(np.sum(memory.observation_count) / max(len(rewards) * n_targets, 1)),
        "fairness": fairness,
        "max_observation_gap": float(np.max(gaps)),
        "mean_observation_gap": float(np.mean(memory.current_gap[discovered])) if np.any(discovered) else 0.0,
        "miss_violation_rate": float(np.mean(memory.current_gap[discovered] > cfg.maintain_gap_threshold)) if np.any(discovered) else 0.0,
        "overlap_penalty": float(np.mean(overlap_values)) if overlap_values else 0.0,
        "cardinality_error": float(abs(estimated_count - n_targets)),
        "estimated_count": float(estimated_count),
        "estimated_peak_count": float(len(estimated_peaks)),
        "true_target_count": float(n_targets),
        "OSPA": ospa_distance(estimated_positions, true_positions, cfg.ospa_cutoff, cfg.ospa_order),
    }
