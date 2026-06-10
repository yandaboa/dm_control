#!/bin/bash
# Phase 2 round 3 — does a THIRD on-policy round keep improving 3-skill latching?
# (r1 partial, r2 clearly better; testing the "rounds scale with #skills" thesis.)
# Driver = bc_phase2_r2; collect meta_r3 (long-sup), train on dart012 + meta_r1..r3, eval.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt
GPU=${1:-3}
run(){ CUDA_VISIBLE_DEVICES=$GPU $PY "$@"; }

echo "=== [P2 r3] collect meta-episodic on-policy (skills 0,1,2) LONG-sup, driver=bc_phase2_r2 ==="
run -m src.smerl.collect_demos --model bc_phase2_r2.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --keep-intervened-only --n-episodes 500 --max-steps 100 \
  --takeover-range 5,15 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --device cuda --seed 3 \
  --out $B/trajectories_dagger012_meta_r3 2>&1 | tail -12

echo "=== [P2 r3] train bc_phase2_r3 (rope@0.4, dart012 + meta_r1..r3) ==="
run -m src.smerl.train_seq --config src/smerl/configs/bc_phase2_r3.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase2 --run-name bc_phase2_r3 \
  --out runs/smerl_lowalpha_ckpt/bc_phase2_r3.pt 2>&1 | tail -8

echo "=== [P2 r3] eval cross-episode latching ==="
run -m src.smerl.eval_adapt_multiep --model bc_phase2_r3.pt --skills 0,1,2 \
  --n-meta 90 --n-sub 4 --max-steps 100 --device cuda \
  --out adapt_multiep_phase2_r3.png 2>&1 | tee $B/eval_phase2_r3_multiep.log

echo "=== PHASE 2 R3 DONE ==="
