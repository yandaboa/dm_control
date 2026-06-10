#!/bin/bash
# Eval one phase2_ros round: $1 = r1|r2, $2 = GPU. Matched terminate-on-switch
# dynamics throughout. Runs the multiep eval in BOTH theta modes:
#   train  (headline) — latent sampled exactly like collection (sample_theta:
#           one forced-bad + one forced-good, third play skill ~Beta -> can be
#           2-bad-1-good), matching the training distribution;
#   legacy — 1 bad + all-others-good, comparable to the old phase-2 numbers.
# Invoked fresh by phase2_ros.sh per round (safe to edit between invocations).
set -e
R=$1; G=$2
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

CUDA_VISIBLE_DEVICES=$G $PY -m src.smerl.eval_adapt_multiep \
  --model bc_phase2_ros_${R}.pt --skills 0,1,2 --n-meta 90 --n-sub 4 --max-steps 100 \
  --reset-on-switch --theta-mode train --device cuda \
  --out adapt_multiep_phase2_ros_${R}_traintheta.png \
  > $B/eval_phase2_ros_${R}_multiep_traintheta.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY -m src.smerl.eval_adapt_multiep \
  --model bc_phase2_ros_${R}.pt --skills 0,1,2 --n-meta 90 --n-sub 4 --max-steps 100 \
  --reset-on-switch --theta-mode legacy --device cuda \
  --out adapt_multiep_phase2_ros_${R}.png \
  > $B/eval_phase2_ros_${R}_multiep.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
  --models bc_phase2_ros_${R}.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --reset-on-switch --theta-bad 0.95 --n-episodes 200 --max-steps 150 --device cuda \
  > $B/eval_phase2_ros_${R}_within.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
  --models bc_phase2_ros_${R}.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --reset-on-switch --theta-mode train --theta-bad 0.95 --n-episodes 200 \
  --max-steps 150 --device cuda \
  > $B/eval_phase2_ros_${R}_within_traintheta.log 2>&1
echo "=== [P2 ros $R] evals done ==="
echo "--- train-theta (matched latent) ---"
grep -A7 "\[adapt\]" $B/eval_phase2_ros_${R}_multiep_traintheta.log
echo "--- legacy theta (old-numbers comparable) ---"
grep -A5 "\[adapt\]" $B/eval_phase2_ros_${R}_multiep.log
echo "--- within-ep probe (legacy / train-theta) ---"
tail -2 $B/eval_phase2_ros_${R}_within.log
tail -2 $B/eval_phase2_ros_${R}_within_traintheta.log
