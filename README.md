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
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' run_smoke.py --episodes 2 --steps 10 --seed 1
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 2 --steps 5 --seed 2 --run-name smoke_ppo --updates-per-collection 1 --ppo-minibatch-size 8 --ppo-update-epochs 2 --eval-interval 1 --eval-episodes 1 --log-interval 1 --save-interval 1
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 2 --steps 5 --seed 500 --checkpoint training_runs/smoke_ppo/latest.pt
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 1 --seed 600 --baseline heuristic --out-dir evaluation_runs/heuristic_smoke
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' pretrain_checks.py --episodes 20 --steps 60 --seed 900 --out-dir diagnostic_runs/pretrain_checks
```

Default PPO training:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_pilot_1k --updates-per-collection 4 --ppo-minibatch-size 128 --ppo-update-epochs 4 --eval-interval 25 --eval-episodes 20 --log-interval 5 --save-interval 25
```

Training logs:

```powershell
tensorboard --logdir training_runs\ppo_pilot_1k\tensorboard
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_wandb_1k --wandb
```

TensorBoard is enabled by default and writes to `training_runs/<run_name>/tensorboard`. Use `--no-tensorboard` to disable it. W&B is disabled by default and enabled with `--wandb`.

The optimized reward uses only the main task terms:

```text
reward = observe + discover + 0.5 * continuity + 0.3 * search - 0.5 * miss
```

Fairness, overlap, movement cost, and option switch rate are recorded as diagnostics but do not affect the current PPO objective.

Baselines:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 60 --seed 500 --baseline random --out-dir evaluation_runs/random
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 60 --seed 500 --baseline coverage --out-dir evaluation_runs/coverage
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 60 --seed 500 --baseline search --out-dir evaluation_runs/search
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 60 --seed 500 --baseline phd --out-dir evaluation_runs/phd
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 20 --steps 60 --seed 500 --baseline heuristic --out-dir evaluation_runs/heuristic
```

Ablation entry points:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_no_search --ablation no_search
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_no_phd --ablation no_phd
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_no_option --ablation no_option
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_no_termination --ablation no_termination
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_no_discover_reward --ablation no_discover_reward
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name ppo_no_miss_penalty --ablation no_miss_penalty
```

Visualization tools:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' tools\visualize_setup.py --seed 900 --n-targets 5 --steps 60 --candidate-warmup-steps 20 --out-dir diagnostic_runs/setup_visualization
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' tools\visualize_phd.py --seed 900 --n-targets 5 --steps 60 --frames 0,5,10,20,40,60 --out-dir diagnostic_runs/phd_visualization
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' tools\visualize_planning_signals.py --seed 900 --n-targets 5 --frames 30 --out-dir diagnostic_runs/planning_signals_30f
```

`visualize_planning_signals.py` writes continuous rollout frames, a GIF, a signal count plot, a discrete graph overview, and a CSV trace of the RL-visible node signals.
