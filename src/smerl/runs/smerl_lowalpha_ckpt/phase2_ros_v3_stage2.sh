#!/bin/bash
# phase2_ros_v3 stage 2 only (collection already done): the two trainings + evals.
# Sources one() from phase2_ros_v3.sh by re-defining it here would drift — instead
# just call the fixed script's tail: train concat (GPU 3) and weighted (GPU 7).
set -e
cd /mnt/storage/lti/dm_control
B=src/smerl/runs/smerl_lowalpha_ckpt
source <(sed -n '/^one()/,/^}/p' $B/phase2_ros_v3.sh)
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python

one r2w 3 & P1=$!
one r2w_wt 7 & P2=$!
wait $P1 $P2

for v in r2w r2w_wt; do
  m=bc_phase2_ros_${v}
  echo "===== ${v} train-theta multiep ====="
  grep -A7 "\[adapt\]" $B/eval_${m}_multiep_traintheta.log
  echo "===== ${v} legacy multiep ====="
  grep -A6 "\[adapt\]" $B/eval_${m}_multiep.log
  echo "===== ${v} within-ep train-theta ====="
  tail -2 $B/eval_${m}_within_traintheta.log
done
echo "=== PHASE 2 ROS V3 DONE ==="
