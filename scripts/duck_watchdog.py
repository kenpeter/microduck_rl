#!/home/kenpeter/work/microduck_rl/.venv/bin/python3
"""Microduck RL watchdog — prints honest measured speed, no LLM needed.

Evaluates the latest checkpoint (by mtime) of the active cap-2.5 run and
reports the world-displacement mean at each command. Delivered verbatim by
the cron engine, so it never fails on a model-provider outage.
"""
import os
import glob
import subprocess

ROOT = "/home/kenpeter/work/microduck_rl"
LOG_DIR = os.path.join(ROOT, "logs", "rsl_rl", "running")
CKPT_GLOB = os.path.join(LOG_DIR, "*running-max-speed", "model_*.pt")
EVAL = os.path.join(ROOT, "scripts", "eval_sprint_speed.py")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python3")

# 1. Is training alive?
ps = subprocess.run(
    ["ps", "-eo", "args"], capture_output=True, text=True
).stdout
alive = "Mjlab-Running-Flat-MicroDuck" in ps
print("=== MICRODUCK RL WATCHDOG ===")
print(f"Process: {'ALIVE' if alive else 'DOWN (not auto-relaunched)'}")

# 2. Latest checkpoint by mtime
ckpts = sorted(glob.glob(CKPT_GLOB), key=os.path.getmtime, reverse=True)
if not ckpts:
    print("No checkpoints found.")
    raise SystemExit(0)
latest = ckpts[0]
print(f"Latest ckpt: {os.path.basename(latest)}")

# 3. Honest eval (full sweep)
print("--- HONEST EVAL (world displacement, 64 envs) ---")
try:
    out = subprocess.run(
        [VENV_PY, EVAL, "--task", "Mjlab-Running-Flat-MicroDuck",
         "--checkpoint", latest, "--num-envs", "64"],
        cwd=ROOT, capture_output=True, text=True, timeout=580,
    )
    for line in out.stdout.splitlines():
        if "cmd=" in line or "Error" in line or "Traceback" in line:
            print(line)
except subprocess.TimeoutExpired:
    print("eval timed out (>580s)")
except Exception as e:  # noqa
    print(f"eval error: {e}")

print("--- VERDICT ---")
print("Goal: measured mean @ cmd 2.0 >= 1.98 m/s. Report only eval numbers above.")
