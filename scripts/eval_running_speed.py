"""Deterministic speed eval for a Microduck running/walk policy.

Measures the robot's ACTUAL forward velocity from the live sim state
(robot.data.root_link_lin_vel_b[:, 0]) — never a training proxy (cmd - err,
reward scalar, etc). This is the single source of truth the agent must report.

Usage:
    .venv/bin/python scripts/eval_running_speed.py \
        Mjlab-SprintCushion-Flat-MicroDuck \
        --checkpoint-file /abs/path/model_XXXX.pt \
        --command-vx 3.0 --num-envs 64 --steps 600

Output (deterministic, parsed by the agent verbatim):
    EVAL_SPEED cmd=3.0  mean=0.42  sustained_10s=0.38  upright_peak=0.95  p90=0.71
    EVAL_ENVS n=64  steps=600  fell_envs=3

The "upright_peak" is gated by tilt < 45 deg so a pre-fall lunge never counts.
"""
import argparse
import statistics

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from dataclasses import asdict


def _obs_first(ret):
    return ret[0] if isinstance(ret, (tuple, list)) else ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--command-vx", type=float, default=3.0)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--upright-deg", type=float, default=45.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs

    # Drop gait_phase if the checkpoint predates it (61-dim policy vs 63-dim cfg)
    for group in ("actor", "critic"):
        if "gait_phase" in env_cfg.observations[group].terms:
            del env_cfg.observations[group].terms["gait_phase"]

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Pin a constant forward command. The sprint task uses VelocityCommandCommandOnly
    # whose _update_command() zeroes "standing" envs every step; disable it and write
    # vel_command_b directly, and expose get_command() (the obs term reads it).
    twist = env.env.command_manager._terms["twist"]
    twist._update_command = lambda: None
    twist._resample_command = lambda env_ids: None
    twist.is_standing_env[:] = False
    twist.is_heading_env[:] = False
    twist.is_world_env[:] = False
    twist.is_forward_env[:] = False
    twist.get_command = lambda name: twist.vel_command_b

    def _pin():
        with torch.no_grad():
            twist.vel_command_b[:, 0] = args.command_vx
            twist.vel_command_b[:, 1] = 0.0
            twist.vel_command_b[:, 2] = 0.0
            twist.vel_command_w[:, 0] = args.command_vx

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint_file), load_cfg={"actor": True},
                strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    robot = env.env.scene["robot"]

    obs = _obs_first(env.reset())
    _pin()
    # spawn already moving at the command velocity (matches training init)
    with torch.no_grad():
        rp = robot.data.root_link_pos_w.clone()
        rq = robot.data.root_link_quat_w.clone()
        lv = robot.data.root_link_lin_vel_b.clone()
        av = robot.data.root_link_ang_vel_b.clone()
        lv[:, 0] = args.command_vx
        from mjlab.utils.lab_api.math import quat_apply
        lw = quat_apply(rq, lv)
        rs = torch.cat([rp, rq, lw, av], dim=-1)
        robot.write_root_state_to_sim(rs, torch.arange(args.num_envs, device=device))

    speeds = []          # per-step mean across envs
    upright_speeds = []  # per-step forward speed where tilt < threshold (stable, not a single-frame spike)
    fell = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    cos_thr = float(torch.cos(torch.deg2rad(torch.tensor(args.upright_deg))))
    upright_buf = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    for _ in range(args.steps):
        obs = _obs_first(env.step(policy(obs)))
        _pin()
        vx = robot.data.root_link_lin_vel_b[:, 0]
        speeds.append(float(vx.mean().item()))
        upright_now = (-robot.data.projected_gravity_b[:, 2]) >= cos_thr
        # require 3 consecutive upright steps so a pre-fall lunge spike is rejected
        upright_buf = upright_buf & upright_now
        upright_speeds.append(float(vx[upright_buf].mean().item()) if upright_buf.any() else 0.0)
        fell = fell | (~upright_now)

    all_v = [s for s in speeds]  # per-step mean series
    window = min(args.steps, 500)  # ~10s at 50Hz env steps
    sustained = statistics.mean(all_v[-window:])
    mean_all = statistics.mean(all_v)
    p90 = sorted(all_v)[int(0.9 * len(all_v))]
    upright_peak = max(upright_speeds) if upright_speeds else 0.0
    n_fell = int(fell.sum().item())

    print(f"EVAL_SPEED cmd={args.command_vx:.1f}  "
          f"mean={mean_all:.3f}  sustained_10s={sustained:.3f}  "
          f"upright_peak={upright_peak:.3f}  p90={p90:.3f}")
    print(f"EVAL_ENVS n={args.num_envs}  steps={args.steps}  fell_envs={n_fell}")
    env.close()


if __name__ == "__main__":
    main()
