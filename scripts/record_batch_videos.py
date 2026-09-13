"""Batch headless video recorder for all finished (DONE) auto5000 tasks.

For each task in the queue's DONE file, finds its current-run final checkpoint
(model_2499/4999.pt), records a play video via scripts/record_video_headless.py,
and writes it next to the checkpoint. Skips tasks that already have a video.
Runs sequentially (one at a time) to stay training-safe on the shared GPU.

    uv run python scripts/record_batch_videos.py
"""
import os, glob, re, time, subprocess, sys
from collections import Counter

REPO = "/home/kenpeter/work/microduck_rl"
LOG_ROOT = os.path.join(REPO, "logs", "rsl_rl")
DONE = os.path.expanduser("~/.hermes/profiles/agent-2/duck_queue_done.txt")
RECORDER = os.path.join(REPO, "scripts", "record_video_headless.py")
PYTHON = os.path.join(REPO, ".venv", "bin", "python")
VIDEO_LEN = 1000  # steps @ 50 Hz = 20 s clip

done = [l.strip() for l in open(DONE) if l.strip()]
CUTOFF = time.mktime(time.strptime("2026-09-04 00:00:00", "%Y-%m-%d %H:%M:%S"))

final = []
for d in glob.glob(os.path.join(LOG_ROOT, "*", "*auto5000")):
    if not os.path.isdir(d) or os.path.getmtime(d) < CUTOFF:
        continue
    pts = glob.glob(os.path.join(d, "model_*.pt"))
    nums = [int(re.search(r"model_(\d+)\.pt", p).group(1)) for p in pts]
    if nums and max(nums) >= 2499:
        final.append((os.path.getmtime(d), d, max(nums)))
final.sort()
n = min(len(done), len(final))
if n < len(done):
    print(f"WARN: DONE has {len(done)} tasks but only {len(final)} matching checkpoints; processing first {n}", flush=True)

results = []
for i, (mt, d, it) in enumerate(final[:n], 1):
    task = done[i - 1]
    ckpt = os.path.join(d, f"model_{it}.pt")
    vdir = os.path.join(d, "videos", "play")
    existing = glob.glob(os.path.join(vdir, "*.mp4"))
    if existing:
        print(f"[{i}/{len(done)}] SKIP {task} (has {os.path.basename(existing[0])})", flush=True)
        results.append((task, "skip"))
        continue
    print(f"[{i}/{len(done)}] RECORD {task}  iter={it}", flush=True)
    env = dict(os.environ, MUJOCO_GL="egl")
    cmd = [PYTHON, RECORDER, task, "--checkpoint-file", ckpt,
           "--video-length", str(VIDEO_LEN), "--num-envs", "1"]
    try:
        r = subprocess.run(cmd, cwd=REPO, env=env, timeout=600,
                            capture_output=True, text=True)
        if r.returncode == 0 and glob.glob(os.path.join(vdir, "*.mp4")):
            sz = os.path.getsize(glob.glob(os.path.join(vdir, "*.mp4"))[0])
            print(f"    OK ({sz} bytes) -> {vdir}", flush=True)
            results.append((task, "ok"))
        else:
            print(f"    FAIL rc={r.returncode}\n--- stdout ---\n{r.stdout[-1200:]}\n--- stderr ---\n{r.stderr[-1200:]}", flush=True)
            results.append((task, "fail"))
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", flush=True)
        results.append((task, "timeout"))

print("\n=== SUMMARY ===", flush=True)
print(dict(Counter(s for _, s in results)), flush=True)
for t, s in results:
    if s not in ("ok", "skip"):
        print("  PROBLEM:", t, s, flush=True)
