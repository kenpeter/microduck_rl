---
tags:
- reinforcement-learning
- locomotion
- bipedal
- microduck
- pollen-robotics
---

# Microduck RL — 2.0 m/s running push

Fork of [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) training
Pollin Robotics **Microduck** (25 cm open-source biped, XL330 servos) to run at a
**measured mean 2.0 m/s** forward speed via PPO (rsl_rl + MuJoCo Warp / BAM).

## Honest status (2026-09-13)

- Recipe: exact DuckEMW running config, **only deviation = speed cap raised to 2.0**
  (`MICRODUCK_RUNNING_SPEED_CAP=2.0`, `MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.0`).
  No flight reward, no Su-style rewards, no achievement gate — reverted to the proven base.
- Measured mean forward speed (world-displacement eval, `scripts/eval_sprint_speed.py`,
  checkpoint `logs/rsl_rl/running/2026-09-12_16-16-33_running-max-speed/model_76500.pt`):
  - cmd 0.4 → 0.11 m/s (16 resets)
  - cmd 0.8 → 1.27 m/s
  - cmd 1.2 → 1.55 m/s
  - cmd 1.6 → 1.61 m/s
  - **cmd 2.0 → 1.66 m/s** (p10 1.49, p90 1.85, max 1.93, 1 reset)
- Goal (mean 2.0 @ cmd 2.0) **NOT yet reached** — training still climbing.

## How speed is measured (no lies)

`scripts/eval_sprint_speed.py` measures true world-frame displacement
(`root_link_pos_w` delta / time) across 64 envs, reporting mean / p10 / p90 / max / resets
per command. Reward/curriculum metrics are NEVER reported as speed.

## Run it

```bash
export MICRODUCK_RUNNING_SPEED_CAP=2.0
export MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.0
.venv/bin/train Mjlab-Running-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 80000
```

## Files

- `src/mjlab_microduck/tasks/microduck_running_env_cfg.py` — running task cfg + knobs
- `src/mjlab_microduck/tasks/mdp.py` — rewards + speed curriculum
- `scripts/eval_sprint_speed.py` — lie-proof world-displacement eval
- `model_76500.pt` — latest checkpoint (measured 1.66 m/s @ cmd 2.0)
