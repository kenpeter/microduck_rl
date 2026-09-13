# Microduck Training Abstraction

## How Training Works

```
MJCF (Onshape) ──> microduck_constants.py:MICRODUCK_WALK_ROBOT_CFG
                        │
                        v
make_microduck_velocity_env_cfg(play, rough)  [microduck_velocity_env_cfg.py:193]
  ├─ Robot: 14 XL330 BAM actuator (friction_dr_bam.py)  ctrl==joint idx
  ├─ Sensors: feet_ground_contact + self_collision + foot_height_scan
  ├─ Commands: twist(3) + head_pose(4) + body_pose(6) ──> 13D command block
  ├─ Obs: 61D actor = 48 proprio + 13 command  [actor 61D, critic 76D]
  │       proprio: base_ang_vel(3)+projected_gravity(3)+joint_pos(14)+joint_vel(14)+actions(14)
  │       + twist(3)+head(4)+body(6) = 61. Critic adds base_lin_vel(3)+foot obs.
  ├─ Domain Randomization (reset/startup events)
  └─ Curriculum (step = iter*24)
                        │
                        v
              MuJoCo Warp (mjlab)  1024-4096 envs @50Hz, GPU batched
                        │  rollout 24 steps/env/iter
                        v
              PPO (rsl_rl, PpoWithSymmetryCfg)  [microduck_velocity_env_cfg.py:912]
                actor 512-256-128 ELU, critic 512-256-128, gamma 0.99 lam 0.95
                        │
                        v
              Checkpoint logs/rsl_rl/velocity/model_*.pt ──> ONNX export (scripts/export.py)
                        │
                        v
              Deployment: pollen-robotics/microduck runtime hot-swaps ONNX @50Hz
```

## Reward Flow (per env step)

```
State (qpos/qvel, contacts, sensors, commands)
  │
  ├─► track_linear_velocity  w=+2.0 std=0.316  Gaussian on |vel_xy - cmd_xy|  ─┐
  ├─► track_angular_velocity w=+2.0 std=0.707  Gaussian on |yaw - cmd_yaw|      │
  ├─► upright                w=+2.0 std=0.224  Gaussian on trunk tilt            │  TASK
  ├─► pose                   w=+1.0            Gaussian on leg joints vs HOME  │  REWARDS
  ├─► air_time               w=+3.0            window 0.125-0.30s if |cmd|>0.01 │  (positive)
  ├─► head_pose_tracking     w=+2.0 std=0.5    mean Gaussian over 4 head joints │
  │                                                              ────────────────┤
  ├─► body_ang_vel           w=-0.05  L2 on trunk ang vel                       │
  ├─► angular_momentum       w=-0.02  L1                                        │
  ├─► action_rate_l2         w=-0.1→-1.0  L2 on Δaction (curriculum)            │  REGULARIZERS
  ├─► dof_pos_limits         w<0              last 7.5% of joint range         │  (negative)
  ├─► foot_clearance/slip/swing  w=-0.1..     foot height/slip penalties       │
  ├─► self_collisions        w=-1.0           trunk self-contact               │
  ├─► head_pose_bias         w=0→3.0  L1 on 1s EMA of head error (ramped)     ─┘
  │                                                              │
  └──────────────── Sum (weighted) ──► Episode_Reward/* (wandb) ──► PPO advantage
                                     total Mean reward 2→15 when walking

Sign convention: mjlab-base costs ≥0 with negative weight; self-negating penalties ≤0 with positive weight.
Check: every Episode_Reward/<penalty> ≤0 or sign is flipped (mdp.py).
```

## All Rewards (Velocity-Flat) — from microduck_velocity_env_cfg.py

> **How to read:** `+` = reward (bigger is better), `-` = penalty (negative weight, keep near 0).  
> Gaussian `exp(-err²/std²)` = 1.0 when perfect, decays smoothly. Check `Episode_Reward/*` in log.

### 🟢 Task Rewards (make it walk)

- **track_linear_velocity — `+2.0` · `std 0.316`**
  - `exp(-|vel_xy - cmd_xy|² / std²)` · MAIN walking drive
  - Iter 81: `0.04` → walking: `>1.5`

- **track_angular_velocity — `+2.0` · `std 0.707`**
  - `exp(-|yaw - cmd_yaw|² / std²)` · turning

- **upright — `+2.0` · `std 0.224` on `trunk_base`**
  - Gaussian on trunk tilt

- **pose — `+1.0`**
  - Gaussian on leg joints vs HOME · tight when standing (`0.05-0.15`), loose when walking (`0.05-0.4`)
  - Joints: `^(?!passive_|.*neck.*|.*head.*).*` (legs only)

- **air_time — `+3.0`**
  - Rewards foot swing `0.125–0.30s` when `|cmd|>0.01` · zero if fallen

- **head_pose_tracking — `+2.0` · `std 0.5`**
  - Mean Gaussian over 4 head joints (`neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`) · partial credit

- **body_pose_tracking — `0.0` (disabled)**
  - `xy 0.05 / z 0.02 / angle 15°` · kept alive for obs shape, not trained in vel env

### 🔴 Regularizers (penalties, keep near 0)

- **action_rate_l2 — `-0.1 → -1.0` (curriculum)**
  - `||action - prev_action||²` · smoothness · ramps `-0.1 / -0.2 / -0.4 / -0.6 / -0.8 / -1.0` at iters `0 / 500 / 750 / 1000 / 1250 / 1500`

- **body_ang_vel — `-0.05`** on `trunk_base` · trunk wobble

- **angular_momentum — `-0.02`** · kept low — dynamic motion needs it

- **head_pose_bias — `0 → 3.0` (ramped after iter 600)**
  - `L1` on 1-second EMA of head error · fixes DC head droop, not oscillation

- **dof_pos_limits — `-`** · fires only in last ~7.5% of joint range

- **foot_clearance — `-`** · `target 0.02m` if `cmd>0.01` · anti-drag

- **foot_swing_height — `-`** · `target 0.02m` · force lift

- **foot_slip — `-0.1`** on `left/right_foot` · deliberately weak (pivot turning)

- **self_collisions — `-1.0`** · `sensor=self_collision` · leg-trunk hit

- **foot_contact_forces — critic only** · NaN-safe privileged obs

### 📦 Other families (not active in Velocity-Flat, see mdp.py:546-933)

`body_upright_linear` / `gaussian`, `upright_progress` (Δcos tilt), `height_progress` (Δz), `fallen_state_penalty`, `recovery_success`, `com_upward_velocity`, `standing_composite_score` — used in VelStand / StandUp / Roulade.

Extra utils in mdp.py: `joint_accelerations_l2`, `leg/neck_action_rate_l2`, `wheel_glide_reward`, `descent_speed_reward`, etc.

## Full Training Pipeline (text diagram)

```
Commands ─┬─ twist: lin(-0.4..0.4, -0.3..0.3) ang(-1.0..1.0) + 15% turn-in-place bucket
          ├─ head_pose: 4D curriculum 0.05→1.10/1.40/0.31 rad over 0-2000 iters
          └─ body_pose: 6D small (±0.005m, ±0.05rad) weight 0

Observations: actor 61D = base_ang_vel(3)+gravity(3)+joint_pos(14)+joint_vel(14)+actions(14)+commands(13)
              critic 76D = actor + base_lin_vel(3)+foot_height(2)+foot_air_time(2)+foot_contact(2)+foot_contact_forces(6)
              noise: ang_vel ±0.03, gravity ±0.01, joint_pos ±0.001, joint_vel ±0.25
              delays: IMU 0-1 step, joint_vel 1 step; encoder bias ±0.015 rad; IMU misalign 6°
DR: com ±3→15mm, head_com ±3→10mm, joint_friction 0.9-1.1 (BAM scale), armature 0.9-1.1, mass 0.95-1.05, foot_friction 0.7-1.3, pushes ±0.3 m/s every 3-6s
Curricula: action_rate -0.1→-1.0, standing 2%→25%, head_pose widening, com ramps, head_bias 0→3.0 after iter 600
Terminations: fell_over, nan_state (with sensor force check), fallen_too_long, time_out
PPO: value_loss 1.0 clip 0.2 entropy 0.01 5 epochs 4 minibatches lr 1e-3 adaptive gamma 0.99 lam 0.95
```

## Text Flow Diagram (data)

```
Command Sampler ──► UniformVelocityCommand (twist) ─┐
                  UniformPoseCommand (head/body) ────┤
                                                     v
Robot Sim ──► Sensors ──► Observations ──► Obs Normalizer (baked into ONNX)
                                    │
                                    └─► Actor (61D) ──► Action (14) ──► BAM Actuator ──► Sim
                                         Critic (76D) ──► Value ──► PPO update

DR Events (per reset): randomize_com, head_com, joint_friction(BAM friction_scale), armature, mass(inertia), encoder_bias, IMU misalign, velocity pushes
Curricula (per step): action_rate_weight, standing_envs, head_pose_range, com_range
Terminations: fell_over, time_out, nan_state, out_of_terrain_bounds
```

## How to Train (this repo, 4070 Ti 12GB)

```bash
uv sync
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 1024 --agent.logger tensorboard --agent.max_iterations 24000
# monitor
tail -f /tmp/train.log ; tensorboard --logdir logs/rsl_rl/velocity
# walking when: track_linear_velocity >1.0, fell_over <5%, episode_length 300-500, Mean reward 12-20
# eval
uv run play Mjlab-Velocity-Flat-MicroDuck --agent.load-checkpoint logs/rsl_rl/velocity/<run>/model_*.pt
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --agent.load-checkpoint logs/.../model_*.pt --output out.onnx
uv run scripts/infer_policy.py --walking out.onnx
# keep only final
ls -t logs/rsl_rl/velocity/*/model_*.pt | tail -n +2 | xargs rm
```

## Invariants

- 61D obs shared across all policies for hot-swap. Zero-pad unused command slots, never delete.
- BAM actuator: friction via friction_scale, dof_frictionloss is zeroed.
- passive_* prefix for unactuated joints; selectors use `^(?!passive_).*`
- DR non-accumulating (restore-then-apply). Obs normalization baked into ONNX.
- Policies unfiltered (no action EMA) unless runtime flag matches.
