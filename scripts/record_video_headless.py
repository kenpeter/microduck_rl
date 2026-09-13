"""Headless video recorder for a trained MicroDuck policy.

Reuses mjlab's exact play setup (env cfg + agent cfg + checkpoint load) but
replaces the interactive viewer with a fixed-step loop and a final env.close(),
which is what finalizes the Gym VideoRecorder mp4. Designed for batch use:

    uv run python scripts/record_video_headless.py <TASK_ID> \
        --checkpoint-file /abs/path/model_2499.pt --video-length 1000

Video lands in <run_dir>/videos/play/rl-video-step-0.mp4 (same place `play` would
write it). Run with MUJOCO_GL=egl for headless rendering.
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


def _obs_first(ret):
    """rsl_rl reset() may return obs or (obs, infos)."""
    return ret[0] if isinstance(ret, (tuple, list)) else ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--video-length", type=int, default=1000)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-terminations", action="store_true")
    args = ap.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    if args.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    resume_path = Path(args.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(resume_path)
    log_dir = resume_path.parent
    print(f"[INFO]: Loading checkpoint: {resume_path.name}  (log_dir={log_dir})")

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    render_mode = "rgb_array"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    env = VideoRecorder(
        env,
        video_folder=str(log_dir / "videos" / "play"),
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    out_dir = log_dir / "videos" / "play"
    try:
        obs = _obs_first(env.reset())
        n = args.video_length
        for i in range(n):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
        print(f"[INFO]: Stepped {n} steps")
    finally:
        env.close()
        print(f"[INFO]: Done. Video(s) in {out_dir}")


if __name__ == "__main__":
    main()
