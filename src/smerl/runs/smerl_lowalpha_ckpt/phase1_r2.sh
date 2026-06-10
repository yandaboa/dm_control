#!/bin/bash
# Phase 1 round 2 — fix cross-episode latching (skills 1,2).
# On-policy collection from the round-1 model (bc_phase1_12) with LONG supervision so
# the expert demonstrates sustained commitment across 3-4 chained sub-episodes (round-1
# demos were ~1 sub-ep => OOD at eval sub-eps 1-3). Retrain on accumulated stores, eval.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P1 r2] collect ON-POLICY (driver=bc_phase1_12), LONG supervision (multi-sub-ep) ==="
CUDA_VISIBLE_DEVICES=2 $PY -m src.smerl.collect_demos \
  --model bc_phase1_12.pt --skills 1,2 --value-norm-skills 1,2 \
  --keep-intervened-only --n-episodes 400 --max-steps 100 \
  --takeover-range 5,15 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --device cuda --seed 1 \
  --out $B/trajectories_dagger12_meta_r2 2>&1 | tail -15

echo "=== [P1 r2] train bc_phase1_12_r2 (rope@0.2, dart12 + meta + meta_r2) ==="
CUDA_VISIBLE_DEVICES=2 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase1_12_r2.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase1 --run-name bc_phase1_12_r2 \
  --out runs/smerl_lowalpha_ckpt/bc_phase1_12_r2.pt 2>&1 | tail -8

echo "=== [P1 r2] eval cross-episode latching (full policy) ==="
CUDA_VISIBLE_DEVICES=2 $PY -m src.smerl.eval_adapt_multiep \
  --model bc_phase1_12_r2.pt --skills 1,2 --n-meta 80 --n-sub 4 --max-steps 100 \
  --device cuda --out adapt_multiep_phase1_r2.png 2>&1 | tee $B/eval_phase1_r2_multiep.log

echo "=== [P1 r2] eval cross-episode latching (TEACHER-ACTION: isolates skill-latching) ==="
CUDA_VISIBLE_DEVICES=2 $PY -m src.smerl.eval_adapt_multiep \
  --model bc_phase1_12_r2.pt --skills 1,2 --n-meta 80 --n-sub 4 --max-steps 100 \
  --teacher-action --device cuda --out adapt_multiep_phase1_r2_teacher.png 2>&1 \
  | tee $B/eval_phase1_r2_multiep_teacher.log

echo "=== [P1 r2] within-episode sanity ==="
CUDA_VISIBLE_DEVICES=2 $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
  --models bc_phase1_12_r2.pt --skills 1,2 --value-norm-skills 1,2 --reset-on-switch \
  --theta-bad 0.95 --n-episodes 200 --max-steps 150 --device cuda 2>&1 \
  | tee $B/eval_phase1_r2_within.log

echo "=== PHASE 1 R2 DONE ==="
