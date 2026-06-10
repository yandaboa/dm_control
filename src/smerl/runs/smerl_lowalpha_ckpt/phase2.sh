#!/bin/bash
# Phase 2 — scale to 3 behavior modes (skills 0,1,2), multi-round on-policy.
# Incorporates the Phase-1 lessons: LONG supervision (--supervision-range 150,300,
# --max-learner-steps 120) so demos span the full multi-sub-ep eval horizon and teach
# "a sub-episode-boundary value-reset is NOT a stall"; switch_share 0.4 (sparse switch
# mix); RoPE+bs8. r0 = bootstrap within-episode base; r1,r2 = on-policy meta-episodic.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt
GPU=${1:-0}
run(){ CUDA_VISIBLE_DEVICES=$GPU $PY "$@"; }

echo "=== [P2 r0] train bootstrap 3-skill base (within-episode) ==="
run -m src.smerl.train_seq --config src/smerl/configs/bc_phase2_r0.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase2 --run-name bc_phase2_r0 \
  --out runs/smerl_lowalpha_ckpt/bc_phase2_r0.pt 2>&1 | tail -8

for R in 1 2; do
  PREV=$((R-1)); DRIVER=bc_phase2_r${PREV}.pt
  echo "=== [P2 r$R] collect meta-episodic on-policy (skills 0,1,2) LONG-sup, driver=$DRIVER ==="
  run -m src.smerl.collect_demos --model $DRIVER --skills 0,1,2 --value-norm-skills 0,1,2 \
    --keep-intervened-only --n-episodes 500 --max-steps 100 \
    --takeover-range 5,15 --supervision-range 150,300 --delay-range 5,10 \
    --max-learner-steps 120 --device cuda --seed $R \
    --out $B/trajectories_dagger012_meta_r${R} 2>&1 | tail -12

  echo "=== [P2 r$R] train bc_phase2_r$R (rope@0.4) ==="
  run -m src.smerl.train_seq --config src/smerl/configs/bc_phase2_r${R}.yaml --device cuda \
    --wandb-project 2d --wandb-group adapt_phase2 --run-name bc_phase2_r${R} \
    --out runs/smerl_lowalpha_ckpt/bc_phase2_r${R}.pt 2>&1 | tail -8

  echo "=== [P2 r$R] eval cross-episode latching (matched dynamics = headline) ==="
  run -m src.smerl.eval_adapt_multiep --model bc_phase2_r${R}.pt --skills 0,1,2 \
    --n-meta 90 --n-sub 4 --max-steps 100 --device cuda \
    --out adapt_multiep_phase2_r${R}.png 2>&1 | tee $B/eval_phase2_r${R}_multiep.log
done

echo "=== PHASE 2 DONE ==="
