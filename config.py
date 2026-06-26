from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_name: str = "CMUOMMT_planToGo_v1"
    map_size: float = 100.0
    dt: float = 1.0
    episode_steps: int = 256
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
    filter_p_detection: float = 0.82
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
    max_search_candidates: int = 8
    max_maintenance_candidates: int = 8
    max_local_candidates_per_uav: int = 8
    max_node_candidates: int = 32
    graph_node_spacing: float = 5.0
    action_k_neighbors: int = 12
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
    gae_lambda: float = 0.95
    ppo_clip_coef: float = 0.20
    ppo_value_coef: float = 0.20
    ppo_entropy_coef: float = 0.00
    ppo_switch_coef: float = 0.01
    ppo_update_epochs: int = 4
    ppo_minibatch_size: int = 256
    ppo_max_grad_norm: float = 5.00
    episodes_per_collection: int = 16
    rollout_backend: str = "ray"
    rollout_workers: int = 8
    rollout_device: str = "cpu"
    rollout_cpus_per_worker: float = 1.0
    rollout_gpus_per_worker: float = 0.0
    worker_num_threads: int = 1
    actor_lr: float = 1e-4
    lr_decay_step: int = 250
    lr_decay_gamma: float = 0.96
    eval_interval: int = 0
    eval_episodes: int = 0
    log_interval: int = 10
    checkpoint_episode_interval: int = 100
    best_metric: str = "episode_reward"
    validation_seed: int = 700
    test_seed: int = 500
    randomize_train_targets: bool = True
    disable_search_belief: bool = False
    disable_phd_belief: bool = False
    disable_options: bool = False
    disable_termination: bool = False
    reward_observe_weight: float = 1.0
    reward_discover_weight: float = 1.0
    reward_continuity_weight: float = 0.5
    reward_search_weight: float = 0.3
    reward_miss_weight: float = -0.5
    ospa_cutoff: float = 25.0
    ospa_order: int = 1
    output_dir: str = "training_output"
    device: str = "cuda"

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
