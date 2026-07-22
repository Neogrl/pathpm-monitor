from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_name: str = "CMUOMMT_planToGo_v1"
    map_size: float = 100.0
    dt: float = 1.0
    episode_steps: int = 256
    n_uavs: int = 5
    n_targets_true: int = 7
    n_targets_min: int = 6
    n_targets_max: int = 8
    max_true_targets: int = 8
    uav_speed: float = 5.5
    fov_radius: float = 12.0
    target_speed: float = 0.55
    target_velocity_noise_std: float = 0.06
    target_init_margin: float = 15.0
    randomize_uav_start: bool = True
    uav_init_margin: float = 10.0
    uav_init_min_separation: float = 12.0
    p_detection: float = 0.92
    filter_p_detection: float = 0.92
    meas_std: float = 1.3
    clutter_mean: float = 0.7
    search_bins: int = 40
    search_belief_init: float = 0.65
    coverage_age_init: float = 8.0
    search_growth: float = 0.018
    search_decay_covered: float = 0.38
    search_min: float = 0.04
    search_age_scale: float = 25.0
    phd_prior_count: float = 7.0
    n_particles_train: int = 2500
    n_particles_eval: int = 2500
    transition_noise: float = 0.06
    death_probability: float = 0.0
    birth_rate: float = 0.0
    phd_birth_probability: float = 0.05
    phd_birth_scheme: str = "none"
    phd_measurement_proposal_enabled: bool = False
    phd_measurement_proposal_particles: int = 100
    phd_measurement_proposal_mass_fraction: float = 0.5
    phd_measurement_proposal_min_component_mass: float = 0.15
    phd_measurement_proposal_position_std: float = 1.3
    phd_resampling_mode: str = "component"
    phd_component_min_mass: float = 0.25
    phd_component_min_particles: int = 64
    phd_regularization_enabled: bool = True
    phd_regularization_min_scale: float = 0.05
    phd_regularization_max_scale: float = 0.50
    phd_initial_velocity_directions: int = 8
    max_target_candidates: int = 8
    max_search_candidates: int = 8
    max_maintenance_candidates: int = 8
    max_local_candidates_per_uav: int = 8
    max_node_candidates: int = 32
    graph_type: str = "prm"
    graph_node_spacing: float = 5.0
    prm_random_nodes: int = 240
    prm_sampling: str = "uniform"
    prm_jitter_ratio: float = 0.06
    prm_boundary_points_per_side: int = 11
    prm_include_boundary: bool = True
    prm_min_node_distance: float = 1.0
    prm_edge_radius: float = 0.0
    obstacles_enabled: bool = False
    obstacle_count: int = 3
    obstacle_radius_min: float = 7.0
    obstacle_radius_max: float = 11.0
    obstacle_margin: float = 16.0
    obstacle_clearance: float = 0.0
    action_k_neighbors: int = 12
    target_candidate_min_separation_ratio: float = 0.6
    search_candidate_min_separation_ratio: float = 0.75
    search_candidate_min_score: float = 0.35
    merge_min_distance_ratio: float = 0.45
    maintain_gap_threshold: int = 8
    maintenance_age_horizon: float = 8.0
    pseudo_track_assoc_gate_ratio: float = 0.75
    pseudo_track_expire_steps: int = 16
    maintenance_track_min_confidence: float = 0.25
    hidden_dim: int = 128
    embed_dim: int = 128
    graph_laplacian_pe_dim: int = 16
    graph_laplacian_pe_enabled: bool = True
    graph_encoder_layers: int = 1
    node_input_dim: int = 6
    uav_state_dim: int = 2
    k_neighbors: int = 12
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_coef: float = 0.20
    ppo_value_coef: float = 0.20
    ppo_entropy_coef: float = 0.005
    ppo_switch_coef: float = 0.01
    ppo_update_epochs: int = 5
    ppo_num_minibatches: int = 0
    ppo_minibatch_size: int = 64
    ppo_max_grad_norm: float = 5.00
    use_clipped_value_loss: bool = True
    use_huber_loss: bool = True
    huber_delta: float = 10.0
    episodes_per_collection: int = 16
    rollout_backend: str = "ray"
    rollout_workers: int = 8
    rollout_device: str = "cpu"
    rollout_cpus_per_worker: float = 1.0
    rollout_gpus_per_worker: float = 0.0
    worker_num_threads: int = 1
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    adam_eps: float = 1e-5
    lr_decay_step: int = 250
    lr_decay_gamma: float = 0.96
    eval_interval: int = 20
    eval_episodes: int = 5
    log_interval: int = 10
    checkpoint_episode_interval: int = 100
    best_metric: str = "val_observation_rate"
    validation_seed: int = 700
    test_seed: int = 500
    randomize_train_targets: bool = True
    disable_search_belief: bool = False
    disable_phd_belief: bool = False
    disable_options: bool = True
    disable_termination: bool = True
    reward_observe_weight: float = 0.0
    reward_discover_weight: float = 0.0
    reward_continuity_weight: float = 0.0
    reward_search_weight: float = 8.0
    reward_overlap_weight: float = 0.0
    reward_cost_weight: float = 0.0
    reward_miss_weight: float = 0.0
    reward_visibility_weight: float = 1.0
    reward_maintenance_age_weight: float = -0.5
    reward_duplicate_coverage_weight: float = -0.5
    reward_phd_position_weight: float = 0.0
    reward_phd_number_weight: float = 0.0
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
