# CMUOMMT planToGo V1

Clean implementation of the CMUOMMT V1 technical plan.

Current method:

```text
SMC-PHD-like target belief + Search Belief + pseudo tracks
-> local reachable graph waypoints
-> ORION-style single-graph option actor
-> PPO training
```

The actor uses ORION-style encoder/decoder/pointer modules and option termination, but does not use ORION's prior/current dual-graph fusion. PPO is the active training algorithm. Legal actions are strictly one-step reachable; candidate count may be smaller than `action_k_neighbors`, with padding used for the rest.

Smoke commands:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' run_smoke.py --episodes 2 --steps 10 --seed 1 --device cpu
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 2 --steps 5 --seed 2 --run-name smoke_ppo --episodes-per-collection 1 --rollout-workers 1 --ppo-minibatch-size 8 --ppo-update-epochs 2 --log-interval 1 --checkpoint-episode-interval 1 --device cpu
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 2 --steps 5 --seed 500 --checkpoint training_output/smoke_ppo/latest.pt --device cpu
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 1 --seed 600 --baseline heuristic --out-dir evaluation_runs/heuristic_smoke
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' pretrain_checks.py --episodes 20 --steps 60 --seed 900 --out-dir diagnostic_runs/pretrain_checks
```

Formal PPO training:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --run-name ppo_formal_5k
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 5000 --steps 256 --seed 10 --run-name ppo_formal_5k --episodes-per-collection 16 --rollout-backend ray --rollout-workers 8 --rollout-cpus-per-worker 1 --rollout-gpus-per-worker 0 --worker-num-threads 1 --rollout-device cpu --ppo-minibatch-size 256 --ppo-update-epochs 4 --log-interval 10 --checkpoint-episode-interval 100 --device cuda
```

The formal PPO defaults follow STAMP where it transfers cleanly: `episodes_per_collection=16`, `ppo_minibatch_size=256`, `actor_lr=1e-4`, `ppo_update_epochs=4`, `ppo_clip_coef=0.2`, `ppo_value_coef=0.2`, no entropy bonus, `ppo_max_grad_norm=5`, and StepLR decay with `lr_decay_step=250`, `lr_decay_gamma=0.96`. `gae_lambda` remains `0.95` because CMUOMMT has delayed discovery and maintenance rewards.

Training-time evaluation is disabled by default (`eval_interval=0`, `eval_episodes=0`) to match STAMP-style throughput. Use `evaluate.py` for fixed-seed validation and final reporting. Rollout collection uses STAMP-style persistent Ray actors by default (`rollout_backend=ray`, `rollout_workers=8`, `rollout_cpus_per_worker=1`, `rollout_gpus_per_worker=0`, `worker_num_threads=1`, `rollout_device=cpu`), while PPO updates run on `device`. The process-pool backend remains available as a fallback with `--rollout-backend process`.

Training logs:

```powershell
tensorboard --logdir training_output\ppo_formal_5k\tensorboard
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 5000 --steps 256 --seed 10 --run-name ppo_formal_5k_wandb --wandb
```

Console output is mirrored to `training_output/<run_name>/train.log` by default. Use `--log-file custom.log` to choose another filename. TensorBoard is enabled by default and writes to `training_output/<run_name>/tensorboard`. Use `--no-tensorboard` to disable it. W&B is disabled by default and enabled with `--wandb`.

A collection is one rollout batch: the trainer collects `episodes_per_collection` episodes with the current policy, merges them into one rollout buffer, and then runs one PPO update phase. `--log-interval` is counted in these outer collection iterations. It controls how often `training_metrics.csv` and `training_summary.json` are flushed; TensorBoard and `train.log` are still updated every collection.

Weights are saved in three ways: `latest.pt` is overwritten after every collection, `best.pt` is updated whenever `best_metric` improves, and numbered checkpoints are saved by cumulative episode count with `--checkpoint-episode-interval` such as every 100 episodes. `--save-interval` is kept only as a deprecated alias for `--checkpoint-episode-interval`.

The optimized reward uses only the main task terms:

```text
reward = observe + discover + 0.5 * continuity + 0.3 * search - 0.5 * miss
```

Fairness, overlap, movement cost, and option switch rate are recorded as diagnostics but do not affect the current PPO objective.

Baselines:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 256 --seed 500 --baseline random --out-dir evaluation_runs/random
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 256 --seed 500 --baseline coverage --out-dir evaluation_runs/coverage
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 256 --seed 500 --baseline search --out-dir evaluation_runs/search
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 256 --seed 500 --baseline phd --out-dir evaluation_runs/phd
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 256 --seed 500 --baseline heuristic --out-dir evaluation_runs/heuristic
```

Ablation entry points:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 256 --seed 10 --run-name ppo_no_search --ablation no_search
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 256 --seed 10 --run-name ppo_no_phd --ablation no_phd
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 256 --seed 10 --run-name ppo_no_option --ablation no_option
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 256 --seed 10 --run-name ppo_no_termination --ablation no_termination
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 256 --seed 10 --run-name ppo_no_discover_reward --ablation no_discover_reward
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 256 --seed 10 --run-name ppo_no_miss_penalty --ablation no_miss_penalty
```

Visualization tools:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' tools\visualize_setup.py --seed 900 --n-targets 5 --steps 60 --candidate-warmup-steps 20 --out-dir diagnostic_runs/setup_visualization
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' tools\visualize_phd.py --seed 900 --n-targets 5 --steps 60 --frames 0,5,10,20,40,60 --out-dir diagnostic_runs/phd_visualization
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' tools\visualize_planning_signals.py --seed 900 --n-targets 5 --frames 30 --out-dir diagnostic_runs/planning_signals_30f
```

`visualize_planning_signals.py` writes continuous rollout frames, a GIF, a signal count plot, a discrete graph overview, and a CSV trace of the RL-visible node signals.
