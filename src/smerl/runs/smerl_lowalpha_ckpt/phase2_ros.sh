#!/bin/bash
# Phase 2 RETRY — 3 skills (0,1,2), env terminate-on-switch, 2 on-policy rounds.
# Changes vs phase2.sh: (1) --reset-on-switch in collection + matched eval;
# (2) 1000 meta-eps/round, ALL intervened AND >=1 post-intervention success
# (--require-success); (3) round 2 takeover-range 70-85 so the learner self-generates
# 1-2 sub-episodes of history before the expert may correct (curriculum over context
# depth); (4) trains on PURE ros stores (dart012 + ros rounds only).
# r1 driver = bc_phase2_r0.pt (existing within-ep bootstrap; driver only, not data).
# Main pipeline on GPU 0; round evals run on GPU 1 via phase2_ros_eval.sh.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P2 ros r1] collect 1000 (driver=bc_phase2_r0, takeover 5-15, ros, require-success) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.collect_demos \
  --model bc_phase2_r0.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --keep-intervened-only --require-success --n-episodes 1000 --max-steps 100 \
  --takeover-range 5,15 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --reset-on-switch --device cuda --seed 1 \
  --out $B/trajectories_dagger012_meta_ros_r1 2>&1 | tail -4

echo "=== [P2 ros r1] train bc_phase2_ros_r1 (rope@0.4, dart012 + ros_r1) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase2_ros_r1.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase2_ros --run-name bc_phase2_ros_r1 \
  --out runs/smerl_lowalpha_ckpt/bc_phase2_ros_r1.pt 2>&1 | tail -4

echo "=== [P2 ros r1] evals on GPU 1 (background); r2 collection starts on GPU 0 ==="
bash $B/phase2_ros_eval.sh r1 1 &
EVAL1=$!

echo "=== [P2 ros r2] collect 1000 (driver=bc_phase2_ros_r1, TAKEOVER 70-85, ros, require-success) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.collect_demos \
  --model bc_phase2_ros_r1.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --keep-intervened-only --require-success --n-episodes 1000 --max-steps 100 \
  --takeover-range 70,85 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --reset-on-switch --device cuda --seed 2 \
  --out $B/trajectories_dagger012_meta_ros_r2 2>&1 | tail -4

echo "=== [P2 ros r2] train bc_phase2_ros_r2 (rope@0.4, dart012 + ros_r1 + ros_r2) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase2_ros_r2.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase2_ros --run-name bc_phase2_ros_r2 \
  --out runs/smerl_lowalpha_ckpt/bc_phase2_ros_r2.pt 2>&1 | tail -4

wait $EVAL1
echo "=== [P2 ros r2] evals ==="
bash $B/phase2_ros_eval.sh r2 1

echo "=== PHASE 2 ROS DONE ==="
