from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_name: str = "CMUOMMT_planToGo_v1"
    map_size: float = 100.0
    dt: float = 1.0
    episode_steps: int = 60
    n_uavs: int = 3
    n_targets_true: int = 5
    n_targets_min: int = 3
    n_targets_max: int = 7
    max_true_targets: int = 8
    uav_speed: float = 5.5
    fov_radius: float = 12.0
    target_speed: float = 1.1
    target_velocity_noise_std: float = 0.06
    target_init_margin: float = 15.0
    p_detection: float = 0.92
    meas_std: float = 1.3
    clutter_mean: float = 0.7
    search_bins: int = 40
    search_belief_init: float = 0.65
    coverage_age_init: float = 8.0
    search_growth: float = 0.018
    search_decay_covered: float = 0.38
    search_min: float = 0.04
    search_age_scale: float = 25.0
    phd_prior_count: float = 4.0
    n_particles_train: int = 1200
    n_particles_eval: int = 2500
    transition_noise: float = 0.14
    death_probability: float = 0.004
    birth_rate: float = 0.45
    target_peak_min_weight: float = 0.15
    max_target_candidates: int = 8
    max_search_candidates: int = 16
    max_maintenance_candidates: int = 8
    max_local_candidates_per_uav: int = 8
    max_node_candidates: int = 32
    target_candidate_min_separation_ratio: float = 0.6
    search_candidate_min_separation_ratio: float = 0.75
    search_candidate_min_score: float = 0.35
    merge_min_distance_ratio: float = 0.45
    maintain_gap_threshold: int = 8
    pseudo_track_assoc_gate_ratio: float = 0.75
    pseudo_track_expire_steps: int = 16
    maintenance_track_min_confidence: float = 0.25
    hidden_dim: int = 128
    embed_dim: int = 128
    k_neighbors: int = 12
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 100000
    minimum_buffer_size: int = 5000
    updates_per_collection: int = 4
    actor_lr: float = 1e-5
    critic_lr: float = 1e-5
    alpha_lr: float = 1e-5
    termination_coef: float = 0.05
    target_update_interval: int = 128
    eval_interval: int = 10
    eval_episodes: int = 20
    log_interval: int = 10
    save_interval: int = 50
    validation_seed: int = 700
    test_seed: int = 500
    randomize_train_targets: bool = True
    ospa_cutoff: float = 25.0
    ospa_order: int = 1
    output_dir: str = "training_runs"
    device: str = "cpu"

    @property
    def cell_size(self) -> float:
        return self.map_size / self.search_bins

    @property
    def local_candidate_radius(self) -> float:
        return self.uav_speed

    @property
    def local_reachable_radius(self) -> float:
        return self.uav_speed

    @property
    def pseudo_track_assoc_gate(self) -> float:
        return self.pseudo_track_assoc_gate_ratio * self.fov_radius

    @property
    def target_candidate_min_separation(self) -> float:
        return self.target_candidate_min_separation_ratio * self.fov_radius

    @property
    def search_candidate_min_separation(self) -> float:
        return self.search_candidate_min_separation_ratio * self.fov_radius

    @property
    def merge_min_distance(self) -> float:
        return self.merge_min_distance_ratio * self.local_candidate_radius

    @property
    def fov_area(self) -> float:
        return 3.141592653589793 * self.fov_radius * self.fov_radius


def default_config() -> Config:
    return Config()


def ensure_output_dir(cfg: Config, run_name: str) -> Path:
    path = Path(cfg.output_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
