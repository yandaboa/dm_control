#!/bin/bash
# Eval phase of phase1_ros.sh, run standalone (training was ended early at epoch ~40/50;
# best-val checkpoints already saved). Matched reset-on-switch dynamics throughout.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P1 ros] eval: cross-episode latching, MATCHED reset-on-switch dynamics (GPUs 0-3) ==="
pids=()
g=0
for s in 20 30 40 50; do
  ( CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.eval_adapt_multiep \
      --model bc_phase1_ros_s${s}.pt --skills 1,2 --n-meta 80 --n-sub 4 --max-steps 100 \
      --reset-on-switch --device cuda --out adapt_multiep_phase1_ros_s${s}.png \
      > $B/eval_phase1_ros_s${s}_multiep.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
      --models bc_phase1_ros_s${s}.pt --skills 1,2 --value-norm-skills 1,2 \
      --reset-on-switch --theta-bad 0.95 --n-episodes 200 --max-steps 150 --device cuda \
      > $B/eval_phase1_ros_s${s}_within.log 2>&1 ) &
  pids+=($!)
  g=$((g+1))
done
for i in 0 1 2 3; do wait ${pids[$i]}; done
for s in 20 30 40 50; do
  echo "--- s${s} multiep (reset-on-switch) ---"
  grep -A5 "\[adapt\]" $B/eval_phase1_ros_s${s}_multiep.log
  echo "--- s${s} within-ep ---"
  tail -3 $B/eval_phase1_ros_s${s}_within.log
done
echo "=== PHASE 1 ROS EVAL DONE ==="
