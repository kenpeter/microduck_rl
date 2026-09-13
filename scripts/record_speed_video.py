"""Record a headless video of the MicroDuck sprint policy with a live
measured-speed overlay.

Pins the `twist` command to a constant forward velocity (default 3.0 m/s, the
value that yielded the ~2.3 m/s baseline at model_26996.pt) and draws the
*measured* body-frame forward speed onto every frame from
`robot.data.root_link_lin_vel_b[:, 0]`.

    MUJOCO_GL=egl python scripts/record_speed_video.py \
        Mjlab-SprintCushion-Flat-MicroDuck \
        --checkpoint-file /abs/path/model_26996.pt \
        --command-vx 3.0 --video-length 600 --num-envs 1

Outputs <ckpt_dir>/videos/sprint_vXpXX.mp4
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def _obs_first(ret):
    return ret[0] if isinstance(ret, (tuple, list)) else ret


def _font(size):
    # try a few common monospace fonts, fall back to PIL default
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--command-vx", type=float, default=3.0)
    ap.add_argument("--video-length", type=int, default=600)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup", type=int, default=60,
                    help="steps to let the duck settle before recording")
    ap.add_argument("--no-terminations", action="store_true")
    args = ap.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs

    # model_26996.pt (the ~2.3 m/s baseline) was trained with 61-dim obs
    # (pre-gait_phase). Drop the gait_phase term so the actor + obs_normalizer
    # dims match the checkpoint - the policy never saw gait_phase anyway.
    for group in ("actor", "critic"):
        g = env_cfg.observations[group]
        if "gait_phase" in g.terms:
            del g.terms["gait_phase"]
            print(f"[INFO]: removed gait_phase from {group} obs (restoring 61-dim)")

    resume_path = Path(args.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(resume_path)
    log_dir = resume_path.parent

    # render on gpu via egl; keep device for policy = same
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- pin the twist command to a constant forward velocity ---
    # The command manager's _update_command() runs every step and zeroes
    # "standing" envs, wiping any pin. Disable it and write vel_command_b directly.
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
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True,
                map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Mimic training's init_velocity_prob: spawn the robot already moving at the
    # command velocity so it doesn't fall from rest before the policy engages.
    robot = env.env.scene["robot"]
    with torch.no_grad():
        root_pos = robot.data.root_link_pos_w.clone()
        root_quat = robot.data.root_link_quat_w.clone()
        lin_vel_b = robot.data.root_link_lin_vel_b.clone()
        ang_vel_b = robot.data.root_link_ang_vel_b.clone()
        lin_vel_b[:, 0] = args.command_vx
        from mjlab.utils.lab_api.math import quat_apply  # type: ignore
        lin_vel_w = quat_apply(root_quat, lin_vel_b)
        root_state = torch.cat([root_pos, root_quat, lin_vel_w, ang_vel_b], dim=-1)
        robot.write_root_state_to_sim(root_state, torch.arange(args.num_envs, device=device))

    font = _font(28)
    out_dir = log_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sprint_v{args.command_vx:+.2f}.mp4"

    frames = []
    obs = _obs_first(env.reset())
    _pin()  # re-pin after reset (reset resamples)
    # spawn already moving (mimics training init_velocity_prob) so the duck
    # doesn't fall from rest before the policy engages
    with torch.no_grad():
        rp = robot.data.root_link_pos_w.clone()
        rq = robot.data.root_link_quat_w.clone()
        lv = robot.data.root_link_lin_vel_b.clone()
        av = robot.data.root_link_ang_vel_b.clone()
        lv[:, 0] = args.command_vx
        lin_vel_w = quat_apply(rq, lv)
        rs = torch.cat([rp, rq, lin_vel_w, av], dim=-1)
        robot.write_root_state_to_sim(rs, torch.arange(args.num_envs, device=device))

    # warmup: settle into gait, no recording
    for _ in range(args.warmup):
        obs = _obs_first(env.step(policy(obs)))
        _pin()

    robot = env.env.scene["robot"]
    last_speed = 0.0
    for i in range(args.video_length):
        obs = _obs_first(env.step(policy(obs)))
        _pin()
        # measured forward speed (body frame, m/s)
        bvel = robot.data.root_link_lin_vel_b  # (B, 3)
        speed = float(bvel[:, 0].mean().item())
        last_speed = speed
        # render frame
        frame = env.env.render()  # (H, W, 3) uint8
        if frame is None:
            continue
        img = Image.fromarray(np.asarray(frame)).convert("RGB")
        draw = ImageDraw.Draw(img)
        txt = f"v = {speed:.2f} m/s   (cmd {args.command_vx:+.1f})"
        draw.rectangle([8, 8, 8 + 22 * len(txt) + 16, 52], fill=(0, 0, 0, 160))
        draw.text((18, 16), txt, font=font, fill=(0, 255, 120))
        frames.append(np.asarray(img))

    if frames:
        import imageio.v2 as imageio
        # env step = decimation * physics_dt = 4 * 0.005 = 0.02 s -> 50 fps real-time
        fps = 50.0
        imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
        print(f"[INFO] wrote {out_path}  ({len(frames)} frames, ~{last_speed:.2f} m/s final)")
    else:
        print("[WARN] no frames captured")


if __name__ == "__main__":
    main()
