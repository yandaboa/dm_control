#!/bin/bash
# Phase 2 ros v2 — overfit fix: 500 meta-eps/round (was 1000; 50 epochs over the
# doubled set doubled gradient steps and overfit r1). r1 retrains on the first-500
# subset of the existing ros_r1 store (no re-collection); r2 collects 500 fresh with
# TAKEOVER 70-85 driven by the new r1. KV-cached inference (seq_model/adaptive_
# transformer) speeds up collection + eval. Evals: phase2_ros_eval.sh (matched
# dynamics, train+legacy theta).
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P2 ros v2 r1] retrain on 500-ep subset (rope@0.4, dart012 + ros_r1_500) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase2_ros_r1.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase2_ros --run-name bc_phase2_ros_r1_n500 \
  --out runs/smerl_lowalpha_ckpt/bc_phase2_ros_r1.pt 2>&1 | tail -4

echo "=== [P2 ros v2 r1] evals on GPU 1 (background); r2 collection starts on GPU 0 ==="
bash $B/phase2_ros_eval.sh r1 1 &
EVAL1=$!

echo "=== [P2 ros v2 r2] collect 500 (driver=bc_phase2_ros_r1, TAKEOVER 70-85) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.collect_demos \
  --model bc_phase2_ros_r1.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --keep-intervened-only --require-success --n-episodes 500 --max-steps 100 \
  --takeover-range 70,85 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --reset-on-switch --device cuda --seed 2 \
  --out $B/trajectories_dagger012_meta_ros_r2 2>&1 | tail -4

echo "=== [P2 ros v2 r2] train bc_phase2_ros_r2 (rope@0.4, dart012 + ros_r1_500 + ros_r2) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase2_ros_r2.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase2_ros --run-name bc_phase2_ros_r2 \
  --out runs/smerl_lowalpha_ckpt/bc_phase2_ros_r2.pt 2>&1 | tail -4

wait $EVAL1 || echo "(r1 eval exited nonzero)"
echo "=== [P2 ros v2 r2] evals ==="
bash $B/phase2_ros_eval.sh r2 1

echo "=== PHASE 2 ROS V2 DONE ==="
