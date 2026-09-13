#!/bin/bash
# Remainder of the VelStand retrain: tasks 3-4 (Rough variants) that OOM'd at
# 2400 envs. Re-run at 2048 envs (the main batch's safe count for rough terrain),
# warm-started from the latest auto5000 model_2499.pt. Tasks 1-2 already done.
set -e
cd /home/kenpeter/work/microduck_rl
export WANDB_MODE=offline

LOG_ROOT=logs/rsl_rl/velstand
CKPTS=(
  "Mjlab-VelStand-Rough-Backlash-MicroDuck:$LOG_ROOT/2026-09-05_04-30-44_auto5000/model_2499.pt"
  "Mjlab-VelStand-Rough-MicroDuck:$LOG_ROOT/2026-09-05_06-30-48_auto5000/model_2499.pt"
)
ITERS=4000
ENVS=2048

for entry in "${CKPTS[@]}"; do
  TASK="${entry%%:*}"
  CKPT="${entry##*:}"
  if [ ! -f "$CKPT" ]; then
    echo "[retrain] SKIP $TASK (checkpoint missing: $CKPT)" >&2
    continue
  fi
  echo "[retrain] $TASK  warm-start from $CKPT  iters=$ITERS envs=$ENVS" >&2
  uv run train "$TASK" \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$ITERS" \
    --agent.save-interval 500 \
    --agent.run-name velstand_retrain \
    --agent.resume False \
    --agent.load-checkpoint "$CKPT"
  echo "[retrain] $TASK DONE" >&2
done

touch /home/kenpeter/.hermes/profiles/agent-2/velstand_retrain.done
echo "[retrain] remainder (tasks 3-4) DONE -> velstand_retrain.done" >&2
