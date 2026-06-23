# CMUOMMT V1 技术实施计划（Claude 执行版）

> 本文档是给代码执行者使用的唯一实施依据。它不是论文草稿，也不是想法记录，而是用于指导后续代码重构、训练闭环实现、实验评估和结果汇报的技术计划文档。
>
> 执行者应当假设自己不了解本项目背景。开始修改代码前，必须先完整阅读本文档，再阅读当前仓库结构和已有代码。任何实现如果与本文档冲突，以本文档为准；如果当前代码与本文档不一致，应当按本文档进行重构。

---

## 0. 项目一句话说明

本项目要实现一个多无人机协同搜索与持续观测任务：

多架 UAV 在二维连续区域中，只能依靠有限视场内的带噪观测，逐步发现未知位置、未知数量的移动目标；发现目标后，系统既要持续重访和维护这些目标，又不能放弃对未覆盖区域的探索。最终希望通过 SMC-PHD 目标强度信念、Search Belief、伪轨迹记忆和 Option-Conditioned 多 UAV waypoint 强化学习策略，实现“探索未知目标”和“维护已发现目标”的动态平衡。

任务名称：

```text
Cooperative Multi-UAV Observation of Multiple Moving Targets under Unknown Target Cardinality
```

简称：

```text
CMUOMMT
```

中文描述：

```text
未知目标数量条件下的多无人机协同移动目标搜索与持续观测维护任务。
```

V1 方法建议名称：

```text
基于 SMC-PHD 目标强度信念与 Search Belief 的 Option-Conditioned 多 UAV Waypoint 强化学习方法
```

英文表述：

```text
An Option-Conditioned Multi-UAV Waypoint Policy with SMC-PHD Target Belief and Search Belief
```

---

## 1. 执行目标与交付物

### 1.1 最终目标

本轮重构不是简单补几个函数，而是把当前启发式 / BC / PPO 原型工程，重构为正式的 V1 方法闭环：

```text
environment
  -> noisy measurements
  -> SMC-PHD target belief
  -> search belief
  -> pseudo track memory
  -> global intent points
  -> local executable waypoints
  -> node features + action mask
  -> option-conditioned actor
  -> waypoint action
  -> centralized twin Q critic training
  -> evaluation and logging
```

正式训练方法固定为：

```text
ORION-style off-policy option actor-critic
```

正式主方法不使用：

```text
BC warm start
PPO
oracle maintenance candidates
ORION prior/current 双图融合
真实目标位置作为 actor 输入
真实目标数量作为 actor 输入
```

### 1.2 必须交付的代码能力

执行完成后，工程至少应支持以下能力：

1. 能运行 V1 环境。
2. 能生成 SMC-PHD target belief。
3. 能更新 search belief。
4. 能维护 discovered target memory。
5. 能维护不使用真值的 pseudo track memory。
6. 能生成 target / search / maintenance 三类 global intent points。
7. 能为每架 UAV 生成一步可达 local executable waypoints。
8. 能把远期 intent points 投影成 local waypoint 的 potential features。
9. 能生成 16 维节点特征。
10. 能生成严格合法的 action mask。
11. 能执行两个 option：Search / Maintain。
12. 能通过 termination head 控制 option 切换。
13. 能通过 option-conditioned actor 选择 waypoint。
14. 能通过 centralized twin Q critic 训练。
15. 能使用 replay buffer 和 SAC-style actor-critic loss。
16. 能记录训练日志、评估指标、reward 权重和关键配置。
17. 能运行 baseline、ablation 和 final test。
18. 能使用 validation seeds 保存 `best_eval.pt`。
19. 能输出最终主表指标。

### 1.3 建议交付文件

如果当前仓库已有对应文件，应优先在已有文件中修改，不要无意义重建。若仓库结构不清晰，可按下面功能拆分：

```text
configs/
  cmuommt_v1.yaml

envs/
  cmuommt_env.py

beliefs/
  smc_phd_belief.py
  search_belief.py
  pseudo_track_memory.py
  discovered_target_memory.py

planning/
  candidate_generator.py
  node_feature_builder.py
  action_mask.py

models/
  option_actor.py
  centralized_critic.py

rl/
  replay_buffer.py
  orion_style_trainer.py
  losses.py

eval/
  metrics.py
  baselines.py
  run_eval.py

scripts/
  train_v1.py
  eval_v1.py
  run_baselines.py
  run_ablations.py
```

如果当前工程文件命名不同，不要强行改名。核心要求是功能边界清晰、接口清晰、日志可复现。

---

## 2. 不可违反的硬约束

下面这些约束是 V1 的核心边界。任何实现、调参或简化都不能破坏这些约束。

### 2.1 Actor 不允许读取真值

正式 actor 输入不得包含：

```text
true target positions
true target velocities
true target ids
true target count
oracle discovered target positions
oracle maintenance candidates
```

真实目标状态只能用于：

```text
reward
critic privileged input
evaluation metrics
oracle diagnostic upper bound
```

### 2.2 PHD 初始目标数不能等于真实目标数

环境真实目标数为：

```text
n_targets_true
```

PHD 初始化先验目标数为：

```text
phd_prior_count = 4.0
```

要求：

```text
SMC-PHD 初始粒子权重总和 = phd_prior_count
phd_prior_count 不能自动设置成 n_targets_true
```

原因是目标数量未知，不能通过初始化把真实数量泄漏给 belief 或 actor。

### 2.3 正式动作只能是一步可达 waypoint

正式动作空间不是连续控制，也不是直接选择远处目标点，而是：

```text
每架 UAV 当前一步可到达的 local executable waypoints
```

远处的 PHD peak、search peak、maintenance pseudo target 只能作为 global intent points，通过 potential features 影响局部 waypoint 的价值，不能直接作为当前一步动作，除非该点本身也在一步可达半径内并被并入 local waypoint 集合。

### 2.4 正式 maintenance candidates 不能使用真实目标位置

正式方法中的 maintenance candidates 必须来自：

```text
measurement points
PHD target_peaks
pseudo track memory
```

不能来自：

```text
真实 target_id
真实 target position
真实 target memory 中的真值位置
```

允许额外实现 oracle maintenance candidates，但只能用于调试或性能上限诊断，不能进入主方法、主表或正式消融。

### 2.5 不采用 ORION 双图融合

V1 明确不采用：

```text
ORION prior map / current map 双图融合机制
```

可以借鉴 ORION 的机制包括：

```text
multi-agent rollout worker
padded node inputs
action mask
option-conditioned policy
option termination head
option-conditioned waypoint decoder
actor / critic separation
centralized critic
curriculum learning
replay buffer off-policy training
twin Q networks
target Q networks
entropy regularization
termination regularization
validation seeds 选择 best checkpoint
```

### 2.6 正式训练不使用 PPO 或 BC

当前工程中如果已有 BC 和 PPO 原型，只能保留为调试工具或历史对照，不属于正式 V1 方法。

正式训练固定为：

```text
ORION-style off-policy option actor-critic
```

---

## 3. 场景与环境设定

### 3.1 地图与仿真参数

| 参数 | V1 取值 | 说明 |
|---|---:|---|
| `map_type` | `continuous_2d_square` | 二维连续正方形区域 |
| `map_size` | `100.0` | 地图边长 |
| `boundary` | `reflective` | UAV 和目标触边反弹或裁剪回边界 |
| `obstacle_enabled` | `False` | V1 不加入障碍物 |
| `dt` | `1.0` | 仿真步长 |
| `episode_steps_train` | `60` | 训练 episode 长度 |
| `episode_steps_eval` | `60` | 测试 episode 长度 |

V1 不加入障碍物。后续如果扩展障碍物，应作为 V2 方向，不要混入 V1。

### 3.2 UAV 参数

| 参数 | V1 取值 |
|---|---:|
| `n_uavs` | `3` |
| `uav_state` | `[x, y]` |
| `uav_speed` | `5.5` |
| `fov_radius` | `12.0` |
| `local_reachable_radius` | `uav_speed` |
| `max_local_candidates_per_uav` | `8` |

初始位置：

```text
uav_start_positions =
[
  [8.0, 20.0],
  [8.0, 50.0],
  [8.0, 80.0]
]
```

重要修改：

```text
旧 demo 使用 fov_radius = 18.0，偏大。
正式 V1 使用 fov_radius = 12.0。
```

### 3.3 目标参数

主实验：

```text
n_targets_true = 5
```

训练扩展：

```text
n_targets_train_range = [3, 7]
```

测试扩展：

```text
n_targets_eval_set = [3, 5, 7]
```

目标状态：

```text
target_state = [x, vx, y, vy]
```

目标运动模型：

```text
constant velocity + Gaussian velocity noise + boundary reflection
```

默认参数：

| 参数 | V1 取值 |
|---|---:|
| `target_speed` | `1.1` |
| `target_velocity_noise_std` | `0.06` |
| `target_init_margin` | `15.0` |

目标真实状态只属于环境。actor 不可见真实目标位置、真实目标速度、真实目标编号和真实目标数量。

### 3.4 观测模型

| 参数 | V1 取值 | 说明 |
|---|---:|---|
| `p_detection` | `0.92` | 环境真实检测概率 |
| `filter_p_detection` | `0.82` | PHD 滤波器内部检测概率 |
| `meas_std` | `1.3` | 测量噪声标准差 |
| `clutter_mean` | `0.7` | 每步杂波均值 |
| `clutter_distribution` | `Poisson` | 杂波分布 |
| `measurement` | `[x, y] + Gaussian noise` | 观测形式 |

观测规则：

```text
若目标进入任意 UAV FOV，则以 p_detection 产生观测。
若目标不在任意 UAV FOV 内，不视为漏检。
杂波点在 UAV FOV 内生成。
```

这点很重要：未覆盖目标不是漏检，只有进入 FOV 后未检测到才是观测失败。

### 3.5 Search belief 栅格参数

| 参数 | V1 取值 |
|---|---:|
| `search_bins` | `40` |
| `search_grid_cell_size` | `2.5` |
| `search_belief_init` | `0.65` |
| `coverage_age_init` | `8.0` |
| `search_growth` | `0.018` |
| `search_decay_covered` | `0.38` |
| `search_min` | `0.04` |
| `search_age_scale` | `25.0` |

### 3.6 SMC-PHD 参数

| 参数 | V1 取值 |
|---|---:|
| `n_particles_train` | `1200` |
| `n_particles_eval` | `2500` |
| `transition_model` | `constant_velocity` |
| `transition_noise` | `0.14` |
| `death_probability` | `0.004` |
| `birth_probability` | `0.18` |
| `birth_rate` | `0.45` |
| `resampler` | `systematic_resampler` |
| `constraint` | `particles outside map are invalid` |
| `phd_prior_count` | `4.0` |

Birth 语义：

```text
birth intensity 表示环境中原本存在但尚未发现目标的先验强度；
不表示物理世界中新目标随机出生。
```

Birth 来源包括：

1. 全局均匀 birth。
2. 当前 measurement-adaptive birth。
3. previous peak memory birth。
4. search-region birth。

---

## 4. 系统状态与 belief 设计

V1 使用三类主要 belief / memory，并额外维护伪轨迹：

```text
target belief
search belief
discovered target memory
pseudo track memory
```

其中 target belief 和 search belief 可以进入 actor；discovered target memory 的真值部分不能进入 actor；pseudo track memory 是正式 actor 维护候选的主要来源。

### 4.1 Target Belief

实现：

```text
Stone Soup SMC-PHD
```

输入：

```text
current measurements
previous particles
previous weights
motion model
birth model
survival / death model
detection model
clutter model
```

输出：

```text
particles: [x, vx, y, vy]
weights: particle weights
estimated_count = sum(weights)
target_peaks = extracted PHD peaks or weighted particle clusters
```

PHD 强度含义：

```text
某区域内 PHD 强度积分 = 该区域内期望目标数量
```

PHD 用于：

1. 生成 target waypoint candidates。
2. 计算 local waypoint 的 `expected_targets_in_fov`。
3. 计算 local waypoint 的 `mean_target_vx` 和 `mean_target_vy`。
4. 计算 PHD 估计指标，例如 cardinality error、OSPA、peak localization error。
5. 为 pseudo track memory 提供 PHD peak 输入。

PHD 不负责：

1. 目标身份维护。
2. 公平性历史统计。
3. 未覆盖区域搜索价值。
4. 真实 discovered target memory。

### 4.2 Search Belief

Search belief 表示某区域仍可能存在未发现目标、因此值得搜索的程度。

状态：

```text
search_belief[y, x] in [0, 1]
coverage_age[y, x] >= 0
```

初始化：

```text
search_belief = 0.65
coverage_age = 8.0
```

自然增长：

```text
coverage_age += 1
search_belief = min(1.0, search_belief + search_growth)
```

若 cell 被任意 UAV FOV 覆盖：

```text
coverage_age = 0
search_belief = max(search_min, search_belief * search_decay_covered)
```

若 cell 中产生目标观测：

```text
coverage_age = 0
search_belief = search_min
```

Search score：

```text
search_score = search_belief * (1 + clip(coverage_age / search_age_scale, 0, 1))
```

Search belief 用于：

1. 生成 search candidates。
2. 计算 local waypoint 的 `search_belief` 特征。
3. 计算 local waypoint 的 `coverage_age` 特征。
4. 计算搜索奖励 `R_search`。

### 4.3 Discovered Target Memory

Discovered target memory 用于 reward、critic privileged input 和 evaluation。

目的：

```text
PHD 不维护目标 ID，但公平性、发现率、重访间隔必须有目标级历史。
```

V1 中允许使用环境真实 target_id 维护 discovered memory，但必须严格限制用途：

```text
target_id 不输入 actor；
target_id 只用于 reward、critic privileged input 和 evaluation。
```

每个目标保存：

```text
is_discovered
first_detection_step
last_observed_step
observation_count
current_gap
max_observation_gap
is_observed_now
```

更新规则：

```text
若目标当前被成功检测：
    is_discovered = True
    若首次检测，first_detection_step = step
    observation_count += 1
    current_gap = 0
    last_observed_step = step
    is_observed_now = True

否则：
    is_observed_now = False
    若 is_discovered:
        current_gap += 1
        max_observation_gap = max(max_observation_gap, current_gap)
```

### 4.4 Pseudo Track Memory

Pseudo track memory 用于给正式 actor 提供非真值 maintenance candidates。

输入：

```text
当前步 measurement points
当前步 PHD target_peaks
上一时刻 pseudo tracks
```

每条 pseudo track 保存：

```text
track_id
last_pos
last_velocity
last_update_step
current_gap
confidence
source_type in {measurement, phd_peak}
```

更新规则：

```text
1. 对 measurement points 和 PHD target_peaks 合并去重，得到 pseudo_observations。
2. 对每个 pseudo_observation z，找距离最近的未匹配 track。
3. 若 distance(z, track.last_pos) <= pseudo_track_assoc_gate，则匹配该 track：
       new_velocity = (z - track.last_pos) / dt
       track.last_velocity = 0.7 * track.last_velocity + 0.3 * new_velocity
       track.last_pos = z
       track.last_update_step = step
       track.current_gap = 0
       track.confidence = min(1.0, track.confidence + 0.2)
   否则新建 track：
       last_pos = z
       last_velocity = [0, 0]
       current_gap = 0
       confidence = 0.5
4. 未匹配 track：
       current_gap += 1
       confidence = 0.98 * confidence
5. 删除满足任一条件的 track：
       current_gap > pseudo_track_expire_steps
       confidence < 0.15
```

维护候选只能使用 pseudo track 的以下字段：

```text
last_pos
last_velocity
current_gap
confidence
```

不得使用真实目标位置或真实 target_id。

---

## 5. Candidate Waypoint 与动作空间设计

V1 的动作空间是一步可达 waypoint 选择。候选体系分成两层：

```text
global intent points
local executable waypoints
```

### 5.1 两层候选的语义

Global intent points 表示远期任务意图，由 belief 和 memory 生成，包括：

```text
target candidates
maintenance candidates
search candidates
```

它们可以离 UAV 很远，不直接作为当前动作。

Local executable waypoints 是每架 UAV 当前一步真正可以执行的动作，由 UAV 当前位置周围固定角度采样生成，距离当前 UAV 不超过：

```text
local_reachable_radius = uav_speed
```

策略网络的 logits 只在 local executable waypoints 上计算。

### 5.2 数据流

每一步执行如下流程：

```text
SMC-PHD target belief
  -> target_candidates

pseudo track memory
  -> maintenance_candidates

search belief
  -> search_candidates

target_candidates + maintenance_candidates + search_candidates
  -> global_intent_points

for each UAV:
  current UAV position
    -> fixed-angle local motion samples
    -> local executable waypoints

global_intent_points + local executable waypoints
  -> intent-to-local potential features
  -> node_features
  -> action_mask
  -> actor chooses local waypoint
```

### 5.3 Target Candidates

来源：

```text
SMC-PHD target_peaks
```

生成方式：

1. 将 SMC-PHD 粒子投影到 `search_bins x search_bins` 网格，得到：

```text
phd_grid[y, x] = sum particle_weight
```

2. 对 `phd_grid` 使用 `3 x 3` 局部极大值检测。
3. 只保留：

```text
phd_grid[y, x] >= target_peak_min_weight
```

4. 峰值坐标取该 cell 内粒子的加权平均位置；若 cell 内粒子数为 0，则取 cell center。
5. 按峰值强度从高到低排序。
6. 使用 `target_candidate_min_separation` 做非极大值抑制。
7. 保留前 `max_target_candidates` 个点。

参数：

| 参数 | V1 取值 |
|---|---:|
| `max_target_candidates` | `8` |
| `target_candidate_min_separation` | `fov_radius * 0.6` |
| `target_peak_min_weight` | `0.15` |

### 5.4 Maintenance Candidates

来源：

```text
pseudo tracks
```

不允许直接使用真实目标位置。

生成方式：

1. 对 pseudo tracks 读取 `current_gap`。
2. 选择 `current_gap` 较大的 pseudo tracks。
3. 预测位置：

```text
track_pred_pos = last_pos + clamp(current_gap, 0, maintain_gap_threshold) * dt * last_velocity
```

4. 若 track 没有可靠速度：

```text
track_pred_pos = last_pos
```

5. 将 `track_pred_pos` 裁剪到地图边界内。
6. 删除：

```text
confidence < maintenance_track_min_confidence
```

7. 按以下值从大到小排序：

```text
current_gap * confidence
```

8. 保留前 `max_maintenance_candidates` 个点。
9. 若没有可用 pseudo track，则 maintenance candidates 为空；PHD peak 仍由 target candidates 处理，不伪装成 maintenance candidate。

参数：

| 参数 | V1 取值 |
|---|---:|
| `max_maintenance_candidates` | `8` |
| `maintain_gap_threshold` | `8` |
| `pseudo_track_assoc_gate` | `fov_radius * 0.75` |
| `pseudo_track_expire_steps` | `2 * maintain_gap_threshold` |
| `maintenance_track_min_confidence` | `0.25` |

### 5.5 Search Candidates

来源：

```text
search_score = search_belief * (1 + clip(coverage_age / search_age_scale, 0, 1))
```

生成方式：

1. 计算 `search_score`。
2. 对 `search_score` 使用 `3 x 3` 局部极大值检测。
3. 只保留：

```text
search_score[y, x] >= search_candidate_min_score
```

4. 峰值坐标取 cell center。
5. 按 `search_score` 从高到低排序。
6. 使用 `search_candidate_min_separation` 做非极大值抑制。
7. 保留前 `max_search_candidates` 个点。

参数：

| 参数 | V1 取值 |
|---|---:|
| `max_search_candidates` | `16` |
| `search_candidate_min_separation` | `fov_radius * 0.75` |
| `search_candidate_min_score` | `0.35` |

### 5.6 Local Motion Candidates

每架 UAV 当前位置周围固定角度采样：

```text
angles = linspace(0, 2*pi, max_local_candidates_per_uav)
point = uav_position + local_candidate_radius * [cos(angle), sin(angle)]
point = clip(point, [0, 0], [map_size, map_size])
```

参数：

| 参数 | V1 取值 |
|---|---:|
| `max_local_candidates_per_uav` | `8` |
| `local_candidate_radius` | `uav_speed` |
| `local_reachable_radius` | `uav_speed` |

### 5.7 候选合并

Global intent points 合并优先级：

```text
maintenance > target > search
```

合并规则：

1. 按优先级加入候选。
2. 若新候选与已有候选距离小于 `merge_min_distance`，则丢弃或合并 intent type 标记。
3. 所有候选点裁剪到地图范围内。

参数：

```text
merge_min_distance = local_candidate_radius * 0.45
```

Local executable waypoints 生成规则：

```text
每架 UAV 独立生成 max_local_candidates_per_uav 个局部候选点；
每个局部候选点必须满足 distance <= local_reachable_radius；
不足 max_node_candidates 时使用 padding；
若某个 global intent point 恰好在 local_reachable_radius 内，将其并入局部候选点；
若并入点与已有 local waypoint 距离小于 merge_min_distance，则保留 intent type 标记并合并到已有 local waypoint。
```

### 5.8 Intent-to-local potential 投影

远期意图点不直接作为一步动作。每个 local waypoint `v` 都根据“执行 `v` 后的 FOV”计算 potential features。

定义：

```text
FOV(v) = { position p | ||p - v|| <= fov_radius }
```

Target potential：

```text
target_potential(v)
  = sum_j particle_weight_j * I(particle_pos_j in FOV(v))

expected_targets_in_fov(v)
  = clip(target_potential(v) / max(phd_prior_count, 1.0), 0, 1)
```

目标速度特征：

```text
mean_target_vx(v)
  = weighted_mean(particle_vx_j for particles in FOV(v))
  = 0 if target_potential(v) < eps

mean_target_vy(v)
  = weighted_mean(particle_vy_j for particles in FOV(v))
  = 0 if target_potential(v) < eps
```

搜索价值：

```text
search_belief(v)
  = mean(search_score[y, x] for grid cells whose centers are in FOV(v))

coverage_age(v)
  = mean(min(coverage_age[y, x] / search_age_scale, 1.0) for grid cells in FOV(v))
```

维护压力：

```text
maintenance_pressure(v)
  = max_k gap_norm_k * exp(-||track_pred_pos_k - v||^2 / (2 * fov_radius^2))

gap_norm_k = min(current_gap_k / maintain_gap_threshold, 1.0)
```

视场重叠：

```text
expected_overlap(v)
  = mean_m circle_intersection_area(v, selected_or_current_uav_pos_m, fov_radius) / fov_area
```

---

## 6. Action Mask 设计

Action mask 用于把 local executable waypoints 进一步筛成当前 UAV 允许选择的合法动作。

### 6.1 Mask 规则

以下节点必须 mask：

```text
padding local node
距离当前 UAV 超过 local_reachable_radius 的 waypoint
已经被前序 UAV 选择的 waypoint
越界 waypoint
```

执行约束：

```text
UAV 每一步最多移动 uav_speed。
policy logits 只在 action_mask == 0 的节点上有效。
```

### 6.2 多 UAV 顺序决策

每步按照固定 UAV 顺序依次选择动作。第 i 架 UAV 选择动作后，后续 UAV 的 action mask 需要屏蔽前序 UAV 已选择的 local waypoint，以减少重复占位和无效重叠。

顺序决策过程：

```text
selected_waypoints = []

for uav_i in range(n_uavs):
    build action_mask_i using selected_waypoints
    actor chooses action_i
    selected_waypoints.append(action_i)
```

### 6.3 合法动作兜底

若某个 UAV 的合法动作数量为 0：

```text
1. 添加当前位置作为 stay waypoint。
2. stay waypoint 仅在无其他合法动作时 unmasked。
3. 执行 stay waypoint 时 step_distance = 0。
4. 训练日志记录 zero_valid_action_count。
```

必须记录：

```text
valid_action_count
zero_valid_action_count
masked_padding_count
masked_unreachable_count
masked_selected_count
```

这些日志用于排查训练失败。

---

## 7. 节点特征设计

V1 节点特征固定为 16 维。

对 UAV `a` 和候选 waypoint `v`：

```text
node_feature(a, v) =
[
  dx,
  dy,
  distance,
  bearing_sin,
  bearing_cos,
  expected_targets_in_fov,
  mean_target_vx,
  mean_target_vy,
  search_belief,
  coverage_age,
  expected_overlap,
  maintenance_pressure,
  is_target_candidate,
  is_maintenance_candidate,
  is_search_candidate,
  is_local_candidate
]
```

### 7.1 归一化规则

```text
dx = (waypoint_x - uav_x) / map_size
dy = (waypoint_y - uav_y) / map_size
distance = ||waypoint - uav|| / map_size
bearing_sin = sin(angle from UAV to waypoint)
bearing_cos = cos(angle from UAV to waypoint)
mean_target_vx = weighted_mean_vx / target_speed
mean_target_vy = weighted_mean_vy / target_speed
coverage_age = min(raw_coverage_age / search_age_scale, 1.0)
maintenance_pressure = min(raw_maintenance_pressure, 1.0)
```

### 7.2 字段定义

`expected_targets_in_fov`：

```text
clip(target_potential(v) / max(phd_prior_count, 1.0), 0, 1)
```

`mean_target_vx / mean_target_vy`：

```text
对 FOV(v) 内 PHD 粒子的速度做权重均值，再除以 target_speed；
若 FOV(v) 内 PHD 权重和小于 eps，则置 0。
```

`search_belief`：

```text
mean(search_score cells in FOV(v))
```

`coverage_age`：

```text
mean(min(raw_coverage_age / search_age_scale, 1.0) cells in FOV(v))
```

`expected_overlap`：

```text
candidate FOV 与其他 UAV 已选或当前位置 FOV 的平均重叠比例
```

`maintenance_pressure`：

```text
由 pseudo track 的预测位置和 current_gap 计算；
不得由真实目标位置直接计算。
```

`is_target_candidate / is_maintenance_candidate / is_search_candidate`：

```text
仅当该局部 waypoint 由一步可达的对应 global intent point 合并而来时置 1。
```

`is_local_candidate`：

```text
由固定角度采样生成的 local waypoint 置 1；
由一步可达 global intent point 合并进来的节点置 0，除非它也与某个 local sample 合并。
```

### 7.3 删除旧固定角色特征

V1 中 UAV 不再固定 tracker / explorer 角色。

必须删除：

```text
role_tracker
role_explorer
```

探索 / 维护倾向由 option state 动态决定，多 UAV 协同补位由顺序决策、action mask、expected_overlap 和 reward 共同形成。

---

## 8. Option 机制

V1 使用两个 option，与 ORION 的二值 option 机制保持一致。

```text
Option 0: Search / Discover
Option 1: Maintain / Observe
```

### 8.1 Option 语义

Option 0：Search / Discover

```text
优先搜索高 search_belief、高 coverage_age 的区域；
目标是发现尚未确认的移动目标。
```

Option 1：Maintain / Observe

```text
优先维护已发现目标；
目标是提高持续观测率、公平性，并降低 miss violation。
```

不单独设置 Assist option。协同补位通过以下机制实现：

```text
多 UAV 顺序决策
action mask 屏蔽前序 UAV 已选 waypoint
expected_overlap 输入 actor
R_overlap 惩罚重复 FOV
```

### 8.2 Option 状态

每架 UAV 都有自己的当前 option：

```text
option_state[i] in {Search / Discover, Maintain / Observe}
```

UAV 不绑定固定角色。

### 8.3 二值 termination 规则

ORION 中的 option 切换是二值翻转。V1 也采用该规则：

```text
if terminate:
    current_option = 1 - previous_option
else:
    current_option = previous_option
```

训练时：

```text
terminate ~ Bernoulli(beta_i)
```

评估时：

```text
terminate = beta_i > 0.5
```

其中：

```text
beta_i = P(terminate current option | current belief, current option)
```

### 8.4 Termination regularization

训练中需要加入：

```text
termination regularization
```

目的：

```text
避免 option 每步频繁抖动；
鼓励 UAV 在一个短时间窗口内保持行为一致性。
```

默认参数：

```text
termination_entropy_coef = 0.01
termination_switch_cost = 0.02
```

必须记录：

```text
option_switch_rate
mean_beta
option_0_ratio
option_1_ratio
```

---

## 9. Actor 与 Critic 接口

### 9.1 Actor 输入

Actor 执行时只能读取非真值 observation。

```text
node_inputs:       [n_uav, max_node_candidates, 16]
node_padding_mask: [n_uav, max_node_candidates]
action_mask:       [n_uav, max_node_candidates]
uav_state:         [n_uav, state_dim]
current_option:    [n_uav]
```

其中：

```text
node_padding_mask = 1 表示 padding node
action_mask = 1 表示非法动作
```

### 9.2 Actor 输出

```text
termination_logits: [n_uav, 1]
waypoint_logits:    [n_uav, max_node_candidates]
```

动作执行流程：

```text
1. 每架 UAV 先根据 termination head 决定是否切换 option。
2. 根据 current option 和 node embeddings 计算 waypoint logits。
3. UAV 按顺序选择 waypoint。
4. 后续 UAV 的 action mask 屏蔽前面 UAV 已选择的候选点。
```

### 9.3 Centralized Critic 输入

V1 采用 centralized twin Q critic。critic 只在训练时使用，允许使用 privileged information。

```text
critic_node_inputs:      [batch, n_uav, max_node_candidates, 16]
critic_action_mask:      [batch, n_uav, max_node_candidates]
all_uav_positions:       [batch, n_uav, 2]
all_selected_actions:    [batch, n_uav]
current_options:         [batch, n_uav]
global_phd_features:     [batch, phd_feature_dim]
global_search_features:  [batch, search_feature_dim]
true_target_states:      [batch, max_true_targets, 5]
discovered_memory_state: [batch, max_true_targets, memory_dim]
```

`true_target_states` 的最后一维：

```text
[x / map_size, y / map_size, vx / target_speed, vy / target_speed, valid_target_mask]
```

`global_phd_features`：

```text
[
  estimated_count / max(phd_prior_count, 1.0),
  cardinality_variance_proxy,
  max_peak_weight,
  mean_peak_weight,
  num_peaks / max_target_candidates
]
```

`global_search_features`：

```text
[
  mean(search_belief),
  max(search_belief),
  mean(coverage_age / search_age_scale),
  max(coverage_age / search_age_scale)
]
```

Critic 输出：

```text
Q1_values: [batch, n_uav, max_node_candidates]
Q2_values: [batch, n_uav, max_node_candidates]
```

Q 语义：

```text
Q_i(s, a_i | a_<i, team_state, option_i)
```

其中 `a_<i` 为顺序决策中前序 UAV 已选择的动作。

### 9.4 CTDE 原则

V1 满足 centralized training, decentralized execution：

```text
训练时 critic 可以读取真实目标状态和 discovered memory。
执行时 actor 不能读取真实目标状态。
```

必须在代码中保证 actor observation 和 critic state 分开构造，不能共用一个包含真值的大字典后再临时过滤，以免后续误用。

---

## 10. 网络架构

V1 网络使用单候选图 attention，不使用 ORION 的 prior/current 双图融合。

### 10.1 Node Encoder

```text
Linear(16, hidden_dim)
ReLU
Linear(hidden_dim, embed_dim)
ReLU
```

默认参数：

```text
hidden_dim = 128
embed_dim = 128
```

### 10.2 Candidate Attention Encoder

对 local action nodes 构造 kNN adjacency：

```text
k_neighbors = 12
```

使用 masked self-attention：

```text
valid local node 只关注自身和 k 个空间近邻；
padding local node 不参与 attention；
k = min(k_neighbors, valid_node_count - 1)。
```

输出：

```text
candidate_embeddings: [n_uav, max_node_candidates, embed_dim]
```

### 10.3 UAV Context Encoder

对每架 UAV 构造 context：

```text
uav_position
previous_action
current_option_embedding
team_summary_embedding
```

team summary 原始特征：

```text
team_summary_raw =
[
  mean_uav_x / map_size,
  mean_uav_y / map_size,
  std_uav_x / map_size,
  std_uav_y / map_size,
  mean_pairwise_uav_distance / map_size,
  mean_pairwise_fov_overlap,
  mean_selected_waypoint_x / map_size,
  mean_selected_waypoint_y / map_size,
  estimated_count / max(phd_prior_count, 1.0),
  mean_search_belief,
  mean_coverage_age / search_age_scale,
  discovered_count / max_expected_targets
]
```

```text
team_summary_embedding = MLP(team_summary_raw)
```

### 10.4 Termination Head

```text
termination_logit = MLP(uav_context + current_option_embedding)
beta = sigmoid(termination_logit)
```

### 10.5 Option-Conditioned Waypoint Decoder

```text
query = MLP(uav_context + current_option_embedding)
key = candidate_embeddings
waypoint_logits = attention(query, key)
```

Mask 规则：

```text
padding node -> -inf
already selected local waypoint -> -inf
action_mask == 1 -> -inf
```

### 10.6 Centralized Twin Q Critic

Critic 应包含两个独立 Q 网络：

```text
Q1
Q2
```

两个 Q 网络输入相同，参数独立。输出 shape：

```text
[batch, n_uav, max_node_candidates]
```

对 padding 或 masked action 的 Q 值不参与 loss 和 policy expectation。

---

## 11. Reward 设计

V1 reward：

```text
R =
  w_obs      * R_observe
+ w_fair     * R_fairness
+ w_cont     * R_continuity
+ w_search   * R_search
+ w_discover * R_discover
- w_overlap  * R_overlap
- w_miss     * R_miss
- w_cost     * R_cost
- w_switch   * R_switch
```

### 11.1 Reward 分量

`R_observe`：

```text
R_observe = 当前步成功检测到的真实目标数 / n_targets_true
```

`R_discover`：

```text
R_discover = 当前步首次发现目标数 / n_targets_true
```

`R_fairness`：

```text
若 discovered_count = 0:
    R_fairness = 0
否则:
    cv = std(observation_count) / (mean(observation_count) + eps)
    R_fairness = clip(1 - cv, 0, 1)
```

`R_continuity`：

```text
R_continuity =
当前步被观测且上一时刻也被观测的已发现目标数 / max(1, discovered_count)
```

`R_search`：

```text
R_search = max(0, previous_search_belief_mean - current_search_belief_mean)
```

`R_overlap`：

```text
R_overlap = mean(pairwise_overlap_area / fov_area)
```

两圆半径均为 `fov_radius = r`，圆心距离为 `d`：

```text
if d >= 2r:
    pairwise_overlap_area = 0
elif d <= eps:
    pairwise_overlap_area = pi * r^2
else:
    pairwise_overlap_area =
        2 * r^2 * arccos(d / (2r))
        - 0.5 * d * sqrt(4r^2 - d^2)
```

`R_miss`：

```text
R_miss = 已发现目标中 current_gap > maintain_gap_threshold 的比例
```

`R_cost`：

```text
R_cost = mean(step_distance / uav_speed)
```

`R_switch`：

```text
R_switch = 当前步 option 发生切换的 UAV 比例
```

### 11.2 默认权重

| 权重 | 取值 |
|---|---:|
| `w_obs` | `1.0` |
| `w_discover` | `0.8` |
| `w_fair` | `0.55` |
| `w_cont` | `0.35` |
| `w_search` | `0.30` |
| `w_overlap` | `0.35` |
| `w_miss` | `0.50` |
| `w_cost` | `0.05` |
| `w_switch` | `0.02` |

所有 reward 权重必须写入训练日志。

### 11.3 Reward 日志

每个 episode 至少记录：

```text
episode_reward
R_observe_mean
R_discover_mean
R_fairness_mean
R_continuity_mean
R_search_mean
R_overlap_mean
R_miss_mean
R_cost_mean
R_switch_mean
```

---

## 12. ORION-style Off-policy Option Actor-Critic 训练

正式 V1 不采用 PPO，也不采用 BC 作为主训练路线。

ORION 原项目的训练方式是离策略 actor-critic：使用 replay buffer 收集 rollout，再用 SAC-style 损失更新 policy、twin Q networks、entropy temperature 和 option termination head。

因此，本任务正式采用：

```text
ORION-style off-policy option actor-critic
```

### 12.1 Transition 保存内容

每一步保存 transition：

```text
observation_t
action_t
option_t
termination_t
reward_t
done_t
observation_t+1
critic_state_t
critic_state_t+1
```

actor observation 不包含真实目标状态。

critic_state_t 保存字段：

```text
critic_node_inputs_t
critic_action_mask_t
all_uav_positions_t
all_selected_actions_t
current_options_t
global_phd_features_t
global_search_features_t
true_target_states_t
discovered_memory_state_t
```

必须保存：

```text
action_mask_t
action_mask_t+1
node_padding_mask_t
node_padding_mask_t+1
```

原因：

```text
Q 网络和 actor 更新只能在合法候选动作上计算。
padding local node、不可达 local waypoint、已被其他 UAV 选择的 local waypoint 不能参与 action probability 或 max / expectation 计算。
```

### 12.2 Replay Buffer 参数

```text
replay_size = 100000
minimum_buffer_size = 5000
batch_size = 64
updates_per_collection = 4
```

### 12.3 Actor 更新

actor 输出当前合法候选 waypoint 上的概率分布：

```text
masked_logits = waypoint_logits.masked_fill(action_mask == 1, -inf)
pi(a | observation, current_option) = softmax(masked_logits)
log_pi = log_softmax(masked_logits)
```

actor loss 使用 SAC-style 形式：

```text
Q_min = min(Q1_values, Q2_values)

L_actor =
mean over batch,uav [
    sum_a pi(a|o) * (alpha * log_pi(a|o) - Q_min(s, a, option))
]
```

Mask 规则：

```text
sum_a 只对 action_mask == 0 的动作计算；
action_mask == 1 的动作 pi = 0；
若 valid_action_count = 1，则该 UAV 的 entropy 项为 0。
```

### 12.4 Twin Q Critic 更新

使用两个 Q 网络：

```text
Q1, Q2
```

target value 使用较小 Q 值：

```text
Q_min = min(Q1, Q2)
```

TD target：

```text
next_masked_logits = next_waypoint_logits.masked_fill(next_action_mask == 1, -inf)
next_pi = softmax(next_masked_logits)
next_log_pi = log_softmax(next_masked_logits)
next_Q_min = min(target_Q1_values_next, target_Q2_values_next)

target_Q =
reward + gamma * (1 - done) *
mean over uav [
    sum_a' next_pi(a') * (next_Q_min(s', a') - alpha * next_log_pi(a'))
]
```

critic loss：

```text
chosen_Q1 = gather(Q1_values, action_t)
chosen_Q2 = gather(Q2_values, action_t)

L_Q1 = MSE(chosen_Q1, target_Q)
L_Q2 = MSE(chosen_Q2, target_Q)
```

### 12.5 Entropy Temperature

V1 使用可学习 entropy temperature：

```text
alpha
```

目标熵：

```text
target_entropy = 0.05 * (-log(1 / valid_action_count_mean))
```

作用：

```text
候选动作较多时保持足够探索；
候选动作较少时避免无意义随机。
```

### 12.6 Option Termination 更新

termination head 输出：

```text
beta = P(terminate previous option | observation, previous_option)
```

二值 option 切换：

```text
若 terminate:
    current_option = 1 - previous_option
否则:
    current_option = previous_option
```

termination loss：

```text
beta = sigmoid(termination_logit)
option_keep = previous_option
option_switch = 1 - previous_option

Q_keep_values = Q(s, a, option_keep)
Q_switch_values = Q(s, a, option_switch)

V_keep = sum_a pi_keep(a|o) * Q_keep_values(a)
V_switch = sum_a pi_switch(a|o) * Q_switch_values(a)

adv_switch = stop_gradient(V_switch - V_keep)

termination_log_prob =
    termination_t * log(beta + eps)
    + (1 - termination_t) * log(1 - beta + eps)

L_termination = - mean(termination_log_prob * adv_switch)
```

总损失：

```text
L_total =
    L_actor
    + L_Q1
    + L_Q2
    + termination_coef * L_termination
    + termination_switch_cost * mean(termination_t)
```

默认训练参数：

| 参数 | 取值 |
|---|---:|
| `gamma` | `0.99` |
| `actor_lr` | `1e-5` |
| `critic_lr` | `1e-5` |
| `alpha_lr` | `1e-5` |
| `termination_coef` | `0.05` |
| `termination_switch_cost` | `0.02` |
| `target_update_interval` | `128` |
| `max_actor_grad_norm` | `100` |
| `max_critic_grad_norm` | `20000` |

### 12.7 BC 的定位

Behavior Cloning，简称 BC，不属于正式 V1 方法。

BC 只允许作为工程调试工具，用于检查：

```text
node_inputs 是否正确
action_mask 是否正确
candidate waypoint label 是否可学习
网络 forward / backward 是否正常
```

正式实验表格、主方法描述和消融实验不使用 BC warm start。

### 12.8 模型选择

不能只按训练 rollout reward 保存 best。

必须使用固定 validation seeds 保存：

```text
best_eval.pt
```

Validation：

```text
eval_interval = 10
eval_episodes = 20
eval_seed = 700
```

Test：

```text
test_seed = 500
test_episodes = 50
```

### 12.9 Curriculum Learning

V1 采用课程学习。

阶段 A：

```text
n_targets = 3
fov_radius = 15
clutter_mean = 0.3
episode_steps = 40
target_speed = 0.8
```

阶段 B：

```text
n_targets = 5
fov_radius = 12
clutter_mean = 0.7
episode_steps = 60
target_speed = 1.1
```

阶段 C：

```text
n_targets sampled from [3, 7]
fov_radius = 12
clutter_mean sampled from [0.5, 1.2]
p_detection sampled from [0.85, 0.95]
episode_steps = 60
```

最终主实验在阶段 B 设置上报告。

---

## 13. 评价指标

### 13.1 搜索指标

```text
discovery_rate
mean_first_detection_time
all_targets_discovered_step
```

### 13.2 维护指标

```text
observation_rate
average_observation_count
minimum_observation_count
fairness
mean_observation_gap
max_observation_gap
miss_violation_rate
continuity
```

### 13.3 协同指标

```text
overlap_penalty
total_travel_distance
mean_pairwise_uav_distance
option_switch_rate
```

### 13.4 PHD 估计指标

```text
estimated_count
cardinality_error = abs(estimated_count - n_targets_true)
OSPA
peak_precision
peak_recall
peak_F1
mean_peak_localization_error
```

### 13.5 主表指标

主表固定使用：

```text
mean_reward
discovery_rate
mean_first_detection_time
observation_rate
fairness
max_observation_gap
miss_violation_rate
overlap_penalty
cardinality_error
OSPA
```

所有指标需要输出：

```text
mean
std
num_episodes
seed_range
```

---

## 14. Baseline 与消融实验

### 14.1 Baselines

必须实现或保留以下 baseline：

1. Random waypoint。
2. Coverage / lawnmower。
3. Search greedy。
4. Target / PHD greedy。
5. Heuristic score policy。
6. Full proposed ORION-style option actor-critic。

### 14.2 Ablations

必须实现以下消融：

1. No search belief。
2. No PHD target belief。
3. No option。
4. No termination head。
5. No centralized critic。
6. No discover reward。
7. No miss penalty。
8. No overlap penalty。

### 14.3 消融解释要求

每个消融不仅要能跑，还要能解释它删除了什么模块：

```text
No search belief:
    search_belief 和 coverage_age 不进入候选生成和节点特征，R_search 可同时关闭。

No PHD target belief:
    target candidates、expected_targets_in_fov、mean_target_vx/vy 不使用 PHD。

No option:
    固定单一 policy，不输入 current_option，不使用 termination head。

No termination head:
    option 不动态切换，可使用固定 option 或按简单规则切换，但不能使用可学习 beta。

No centralized critic:
    critic 不使用 true_target_states 和 discovered_memory_state。

No discover reward:
    w_discover = 0。

No miss penalty:
    w_miss = 0。

No overlap penalty:
    w_overlap = 0。
```

---

## 15. 当前实现状态与缺口

### 15.1 当前已完成

当前工程已经具备以下基础能力：

1. SMC-PHD target belief。
2. search belief。
3. target / search / local waypoint candidates。
4. 16 维旧节点特征。
5. 启发式闭环。
6. 多 seed 评估。
7. 可视化输出。

当前工程中已有 BC 和 PPO 原型代码；它们只作为早期调试和对照记录，不属于正式 V1 方法。

### 15.2 必须修改

下面是必须完成的 V1 缺口：

1. `fov_radius` 从 `18` 改为 `12`。
2. 加入 `phd_prior_count`，并与真实目标数解耦。
3. 加入 discovered target memory。
4. 将正式动作空间改为一步可达 local executable waypoints。
5. 增加非 oracle maintenance candidates，正式方法不得使用真实目标位置生成维护候选。
6. 增加 global intent points 到局部 waypoint 的 potential 特征投影。
7. 删除固定 UAV role 特征，改为动态 option。
8. 将 option 改为 Search / Maintain 两种。
9. 实现 option termination head。
10. 实现 option-conditioned waypoint decoder。
11. 实现 centralized twin Q critic，训练时允许使用真实目标状态。
12. reward 补齐 discover / miss / cost / switch。
13. 评估补齐 discovery / gap / cardinality / OSPA。
14. 增加 baselines 和 ablations。
15. 使用 validation seeds 选择 `best_eval.pt`。
16. 将训练从 BC / PPO 原型替换为 ORION-style off-policy option actor-critic。

---

## 16. 推荐实施顺序

Claude 执行时应按阶段推进，不要一次性重写全部代码。每一阶段完成后先运行最小测试，再进入下一阶段。

### 阶段 1：配置与环境约束

任务：

1. 统一配置文件。
2. 修改 `fov_radius = 12.0`。
3. 加入 `phd_prior_count = 4.0`。
4. 确保 `phd_prior_count` 不随 `n_targets_true` 改变。
5. 明确 train / eval / test episode steps。
6. 记录所有 reward 权重和关键环境参数。

验收：

```text
打印一次 reset 后的配置；
n_targets_true = 5 时，phd_prior_count 仍为 4.0；
fov_radius 日志显示为 12.0；
actor observation 中没有 true target states。
```

### 阶段 2：环境 memory

任务：

1. 实现 discovered target memory。
2. 实现 pseudo track memory。
3. 将 discovered memory 限制为 reward / critic / eval 使用。
4. 将 pseudo track memory 接入 maintenance candidates。

验收：

```text
目标被检测后 discovered 状态变为 True；
current_gap 随未观测步数增加；
pseudo track 可根据 measurement / PHD peak 新建、更新、过期；
正式 maintenance candidate 不读取真实目标位置。
```

### 阶段 3：候选生成与动作空间

任务：

1. 拆分 global intent points 和 local executable waypoints。
2. 实现 target candidates。
3. 实现 search candidates。
4. 实现 maintenance candidates。
5. 实现 local motion candidates。
6. 实现候选合并和 intent type 标记。
7. 实现 max node padding。

验收：

```text
global intent points 可以远离 UAV；
actor logits 只对应 local executable waypoints；
每个合法 local waypoint 距离当前 UAV 不超过 uav_speed；
padding node 被 mask。
```

### 阶段 4：节点特征与 mask

任务：

1. 实现 16 维节点特征。
2. 删除旧 role_tracker / role_explorer。
3. 实现 target/search/maintenance potential 投影。
4. 实现 expected_overlap。
5. 实现 action mask。
6. 实现无合法动作时的 stay waypoint 兜底。

验收：

```text
node_feature shape = [n_uav, max_node_candidates, 16]；
role 特征不存在；
action_mask 正确屏蔽 padding / unreachable / selected / out-of-bound；
valid_action_count 不应长期为 0；
zero_valid_action_count 被记录。
```

### 阶段 5：Actor 与 option

任务：

1. 实现二值 option state。
2. 实现 termination head。
3. 实现 option-conditioned waypoint decoder。
4. 实现训练采样和评估 greedy 两种 termination 逻辑。
5. 记录 option 相关日志。

验收：

```text
current_option shape = [n_uav]；
termination_logits shape = [n_uav, 1]；
waypoint_logits shape = [n_uav, max_node_candidates]；
eval 时 terminate = beta > 0.5；
option_switch_rate 可被记录。
```

### 阶段 6：Centralized twin Q critic

任务：

1. 实现 critic privileged input。
2. 实现 Q1 / Q2。
3. 确保 actor observation 和 critic state 分离。
4. 实现 masked Q 输出处理。

验收：

```text
critic 输出 shape = [batch, n_uav, max_node_candidates]；
critic 可以读取 true_target_states；
actor forward 路径无法读取 true_target_states；
masked actions 不参与 Q loss 和 actor expectation。
```

### 阶段 7：Reward 与 metrics

任务：

1. 补齐 V1 reward 分量。
2. 记录每个 reward 分量。
3. 实现 discovery / gap / fairness / continuity / cardinality / OSPA 等指标。
4. 输出主表指标。

验收：

```text
每个 episode 输出 reward 总值和分量；
discovery_rate、observation_rate、max_observation_gap、miss_violation_rate 可计算；
PHD estimated_count 和 cardinality_error 可计算；
OSPA 可计算或明确在缺少依赖时给出可替代实现。
```

### 阶段 8：Replay buffer 与 off-policy 训练

任务：

1. 实现 replay buffer。
2. 保存 actor observation、critic state、action、option、termination、reward、done、next observation、next critic state。
3. 实现 SAC-style actor loss。
4. 实现 twin Q critic loss。
5. 实现 learnable alpha。
6. 实现 termination loss。
7. 实现 target Q update。
8. 实现 gradient clipping。

验收：

```text
minimum_buffer_size 前不更新；
batch 采样 shape 正确；
actor loss、Q loss、alpha loss、termination loss 均为有限值；
训练不会因为 masked logits 出现 NaN；
best_eval.pt 由 validation seeds 选择。
```

### 阶段 9：Baseline、ablation 与最终测试

任务：

1. 跑 baselines。
2. 跑 ablations。
3. 跑 curriculum。
4. 使用 `test_seed = 500` 和 `test_episodes = 50` 输出最终主表。
5. 保存所有配置、日志、checkpoint 和结果表。

验收：

```text
每个 baseline 可独立运行；
每个 ablation 可独立运行；
最终主表包含固定 10 个指标；
结果文件包含 mean / std / num_episodes / seed_range。
```

---

## 17. 最小测试清单

Claude 每完成一个阶段，都应优先运行最小测试，而不是直接长时间训练。

### 17.1 Reset 测试

检查：

```text
env.reset()
actor_obs keys
critic_state keys
n_targets_true
phd_prior_count
uav positions
target states not in actor_obs
```

### 17.2 单步 rollout 测试

检查：

```text
env.step(actions)
measurements
PHD update
search belief update
discovered memory update
pseudo track update
reward components
done flag
```

### 17.3 候选生成测试

检查：

```text
target_candidates shape
maintenance_candidates shape
search_candidates shape
local_waypoints shape
node_features shape
action_mask shape
valid_action_count
```

### 17.4 Actor forward 测试

检查：

```text
termination_logits
waypoint_logits
masked softmax
sampled actions
no NaN
```

### 17.5 Critic forward 测试

检查：

```text
Q1_values
Q2_values
chosen_Q
masked Q ignored
no NaN
```

### 17.6 Replay buffer 测试

检查：

```text
push transition
sample batch
batch tensor shapes
next action mask exists
critic state exists
```

### 17.7 训练 smoke test

运行极小训练：

```text
episodes = 3
episode_steps = 10
batch_size = 4
minimum_buffer_size = 8
```

验收：

```text
能完成 forward、backward、optimizer step；
loss 为有限值；
checkpoint 可保存；
eval 可运行。
```

---

## 18. 训练与结果保存规范

### 18.1 日志必须包含

```text
config snapshot
random seed
episode reward
reward components
actor loss
critic loss
alpha loss
termination loss
alpha value
valid_action_count
zero_valid_action_count
option_switch_rate
option_0_ratio
option_1_ratio
estimated_count
cardinality_error
discovery_rate
observation_rate
fairness
max_observation_gap
miss_violation_rate
overlap_penalty
OSPA
```

### 18.2 文件保存建议

```text
outputs/
  cmuommt_v1/
    config.yaml
    train_log.csv
    eval_log.csv
    best_eval.pt
    last.pt
    metrics_main_table.csv
    metrics_baselines.csv
    metrics_ablations.csv
    visualizations/
```

### 18.3 Checkpoint 选择

保存规则：

```text
last.pt: 最新训练状态。
best_eval.pt: 固定 validation seeds 上主评价指标最优的模型。
```

不允许只按训练 reward 保存 best。

---

## 19. 常见误解与禁止实现

### 19.1 不要把 PHD 当成轨迹管理器

PHD 输出的是强度和期望数量，不维护目标身份。目标级公平性、gap、first detection 需要 discovered target memory。正式 actor 的维护候选来自 pseudo track，不来自真实 target memory。

### 19.2 不要把远处候选点直接作为动作

远处的 target peak / search peak / pseudo target 是 intent，不是一步动作。actor 只能选择一步可达 local waypoint。

### 19.3 不要让 actor 偷看真值

这包括显式真值，也包括通过 oracle candidate、true target gap、真实 target_id 编码后的隐式泄漏。

### 19.4 不要把 BC / PPO 当成正式方法

BC 可以做 smoke test。PPO 可以作为历史对照。但 V1 正式方法是 off-policy option actor-critic。

### 19.5 不要加入障碍物

V1 不考虑障碍物。障碍物会引入路径规划和可达性问题，属于下一阶段扩展。

### 19.6 不要新增第三个 Assist option

V1 只有两个 option。协同补位由顺序动作、mask、overlap feature 和 reward 完成。

---

## 20. Claude 执行提示词

可以把下面这段直接交给 Claude，要求它基于当前仓库执行：

```text
请完整阅读 PROJECT_PLAN_V1_RESTRUCTURED_FOR_CLAUDE.md，并基于当前仓库实现 CMUOMMT V1。

你需要把当前工程从启发式 / BC / PPO 原型，重构为正式的 ORION-style off-policy option actor-critic 方法。不要一次性大改全部文件，请先扫描仓库结构，列出当前已有模块与本文档要求之间的映射关系，然后按文档第 16 节的阶段顺序逐步修改。

实现时必须遵守以下硬约束：
1. actor 不能读取真实目标位置、真实目标速度、真实 target_id 或真实目标数量。
2. phd_prior_count 必须与 n_targets_true 解耦，默认 phd_prior_count = 4.0。
3. 正式动作只能是一步可达 local executable waypoints。
4. 远处 PHD peak / search peak / maintenance pseudo target 只能作为 global intent points，通过 potential features 影响 local waypoint。
5. 正式 maintenance candidates 必须来自 measurements、PHD peaks 或 pseudo track memory，不能使用真实目标位置。
6. V1 不使用 ORION prior/current 双图融合。
7. 正式训练不使用 PPO 或 BC warm start，而是使用 replay buffer + SAC-style actor loss + twin Q critic + entropy temperature + option termination loss。
8. V1 只有两个 option：Search / Discover 和 Maintain / Observe。
9. 必须使用 validation seeds 保存 best_eval.pt。
10. 每个阶段完成后先做 smoke test，再进入下一阶段。

请先输出：
- 当前仓库结构理解；
- 已有代码与 V1 要求的差距；
- 你准备修改的文件列表；
- 第一阶段的具体修改计划。

然后开始实现第一阶段。
```

---

## 21. 最终验收标准

当以下条件全部满足时，可以认为 V1 工程实现完成：

1. 配置、环境、belief、candidate、actor、critic、trainer、eval 的边界清晰。
2. actor observation 中没有任何真实目标信息。
3. critic privileged input 与 actor observation 明确分离。
4. `fov_radius = 12.0` 生效。
5. `phd_prior_count = 4.0` 且不等于真实目标数。
6. 正式动作全部是一步可达 local waypoints。
7. maintenance candidates 不使用真值。
8. 16 维节点特征与本文档一致。
9. option termination head 可训练、可评估。
10. replay buffer 保存完整 transition。
11. SAC-style actor loss、twin Q loss、alpha loss、termination loss 可正常反向传播。
12. validation seeds 能保存 `best_eval.pt`。
13. baselines 和 ablations 能独立运行。
14. final test 能输出主表固定 10 个指标。
15. 所有结果能通过配置和 seed 复现。
