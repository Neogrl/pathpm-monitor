# CMUOMMT planToGo V1

Clean implementation of the CMUOMMT V1 technical plan.

Smoke commands:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' run_smoke.py --episodes 2 --steps 10 --seed 1
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 2 --steps 5 --seed 2 --run-name smoke_train --batch-size 8 --minimum-buffer-size 16 --updates-per-collection 1 --eval-interval 1 --eval-episodes 1 --log-interval 1 --save-interval 1
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 2 --seed 500 --checkpoint training_runs/smoke_train/latest.pt
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' evaluate.py --episodes 1 --seed 600 --baseline heuristic --out-dir evaluation_runs/heuristic_smoke
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' pretrain_checks.py --episodes 20 --steps 60 --seed 900 --out-dir diagnostic_runs/pretrain_checks
```

Default training uses a larger replay warmup:

```powershell
& 'C:\Users\15193\.conda\envs\pathpm\python.exe' train.py --updates 1000 --steps 60 --seed 10 --run-name pilot_1k --minimum-buffer-size 5000 --batch-size 64 --updates-per-collection 4 --eval-interval 25 --eval-episodes 20 --log-interval 5 --save-interval 25
```
