#!/usr/bin/env bash
# 3-skill (0,1,2) adaptation experiment, one round of HG-DAgger:
#   1. DART  : clean per-skill base demos (no injected failures)
#   2. COLLECT: model-less HG-DAgger adaptation data (takeover 0-10, supervision 50-125)
#   3. TRAIN : transformer on DART + DAgger (expert-gated)
set -euo pipefail
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
RUN=runs/smerl_lowalpha_ckpt
BASE=src/smerl/$RUN

echo "=== [1/3] DART base demos (skills 0,1,2, no failures) — ~400 eps ==="
$PY -m src.smerl.collect_trajectories \
  --run $RUN --ckpt-file ckpt_step010000.pt \
  --skills 0,1,2 --value-norm-skills 0,1,2 --n-per-skill 134 \
  --p 0.0 --action-noise-frac 0.05 --radius 0.1 --max-steps 75 \
  --out $BASE/trajectories_dart012 --device cpu

echo "=== [2/3] DAgger collection (iter 0, model-less, decoupled, INTERVENED-ONLY) — 400 eps ==="
$PY -m src.smerl.collect_demos \
  --run $RUN --ckpt-file ckpt_step010000.pt \
  --iteration 0 --skills 0,1,2 --value-norm-skills 0,1,2 \
  --takeover-range 0,10 --supervision-range 50,125 --decouple-delay 3 \
  --n-episodes 400 --keep-intervened-only --radius 0.1 --max-steps 100 \
  --out $BASE/trajectories_dagger012_iter0 --device cpu

echo "=== [3/3] Transformer training (decoupled + switch_weight + logic; DART + DAgger) ==="
$PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_adapt_3skill.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_3skill --run-name bc_adapt_3skill_balanced

echo "=== DONE -> $BASE/bc_adapt_3skill.pt ==="
