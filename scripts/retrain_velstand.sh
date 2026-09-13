#!/bin/bash
# Retrain the 4 VelStand tasks, WARM-STARTED from the latest (auto5000) VelStand
# checkpoints, training MORE iterations because floor recovery is weak.
# Each task resumes from its model_2499.pt (the latest batch) and trains ITERS more.
set -e
cd /home/kenpeter/work/microduck_rl
export WANDB_MODE=offline

LOG_ROOT=logs/rsl_rl/velstand
# task -> latest auto5000 checkpoint (from the just-finished batch, Sep 5)
CKPTS=(
  "Mjlab-VelStand-Flat-Backlash-MicroDuck:$LOG_ROOT/2026-09-05_02-53-21_auto5000/model_2499.pt"
  "Mjlab-VelStand-Flat-MicroDuck:$LOG_ROOT/2026-09-05_03-48-59_auto5000/model_2499.pt"
  "Mjlab-VelStand-Rough-Backlash-MicroDuck:$LOG_ROOT/2026-09-05_04-30-44_auto5000/model_2499.pt"
  "Mjlab-VelStand-Rough-MicroDuck:$LOG_ROOT/2026-09-05_06-30-48_auto5000/model_2499.pt"
)
ITERS=4000          # additional iterations on top of the loaded latest model
ENVS=2400

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
echo "[retrain] ALL VelStand retrain done -> velstand_retrain.done" >&2
