#!/bin/bash
# Companion to phase1_r2.sh: waits for the round-2 meta_r2 collection to finish, then
# trains+evals the switch_share=0.4 variant on GPU 3 (in parallel with the share=0.2 run).
# Tests whether sparser switch demos need a higher share to keep within-episode switching.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "[s40] waiting for round-2 collection (meta_r2 manifest)..."
while [ ! -f $B/trajectories_dagger12_meta_r2/manifest.json ]; do sleep 20; done
echo "[s40] collection done -> train share0.4"

CUDA_VISIBLE_DEVICES=3 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase1_12_r2_s40.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase1 --run-name bc_phase1_12_r2_s40 \
  --out runs/smerl_lowalpha_ckpt/bc_phase1_12_r2_s40.pt 2>&1 | tail -8

echo "=== [P1 r2 s40] eval cross-episode latching ==="
CUDA_VISIBLE_DEVICES=3 $PY -m src.smerl.eval_adapt_multiep \
  --model bc_phase1_12_r2_s40.pt --skills 1,2 --n-meta 80 --n-sub 4 --max-steps 100 \
  --device cuda --out adapt_multiep_phase1_r2_s40.png 2>&1 | tee $B/eval_phase1_r2_s40_multiep.log

echo "=== [P1 r2 s40] within-episode sanity ==="
CUDA_VISIBLE_DEVICES=3 $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
  --models bc_phase1_12_r2_s40.pt --skills 1,2 --value-norm-skills 1,2 --reset-on-switch \
  --theta-bad 0.95 --n-episodes 200 --max-steps 150 --device cuda 2>&1 \
  | tee $B/eval_phase1_r2_s40_within.log

echo "=== PHASE 1 R2 S40 DONE ==="
