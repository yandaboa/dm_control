#!/bin/bash
# Phase 2 ros v3 round 3: $1 = driver model (better round-2 variant, e.g.
# bc_phase2_ros_r2w.pt or bc_phase2_ros_r2w_wt.pt), $2 = collect GPU.
# Takeover upper bound raised again (5-120, min fixed at 5); max-learner-steps 140
# keeps a >=20-step intervention window. Then concat (GPU 3) and recency-weighted
# [0.1,0.2,0.3,0.4] (GPU 7) trainings + matched evals, as in phase2_ros_v3.sh.
set -e
DRIVER=$1; CG=$2
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P2 ros r3] collect 500 r3w (driver=${DRIVER}, TAKEOVER 5-120) ==="
CUDA_VISIBLE_DEVICES=$CG $PY -m src.smerl.collect_demos \
  --model ${DRIVER} --skills 0,1,2 --value-norm-skills 0,1,2 \
  --keep-intervened-only --require-success --n-episodes 500 --max-steps 100 \
  --takeover-range 5,120 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 140 --reset-on-switch --device cuda --seed 4 \
  --out $B/trajectories_dagger012_meta_ros_r3w 2>&1 | tail -4

one() {  # one(variant, gpu): train + the three matched evals
  local v=$1 g=$2
  local m=bc_phase2_ros_${v}   # separate local: ${v} expands BEFORE local assigns
  echo "=== [P2 ros r3] train ${m} (GPU $g) ==="
  CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.train_seq \
    --config src/smerl/configs/${m}.yaml --device cuda \
    --wandb-project 2d --wandb-group adapt_phase2_ros --run-name ${m} \
    --out runs/smerl_lowalpha_ckpt/${m}.pt > $B/train_${m}.log 2>&1
  echo "[${v}] train done: $(grep '\[done\]' $B/train_${m}.log | tail -1)"
  CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.eval_adapt_multiep \
    --model ${m}.pt --skills 0,1,2 --n-meta 90 --n-sub 4 --max-steps 100 \
    --reset-on-switch --theta-mode train --device cuda \
    --out adapt_multiep_${m}_traintheta.png > $B/eval_${m}_multiep_traintheta.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.eval_adapt_multiep \
    --model ${m}.pt --skills 0,1,2 --n-meta 90 --n-sub 4 --max-steps 100 \
    --reset-on-switch --theta-mode legacy --device cuda \
    --out adapt_multiep_${m}.png > $B/eval_${m}_multiep.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
    --models ${m}.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
    --reset-on-switch --theta-mode train --theta-bad 0.95 --n-episodes 200 \
    --max-steps 150 --device cuda > $B/eval_${m}_within_traintheta.log 2>&1
  echo "=== [${v}] evals done ==="
}

one r3w 3 & P1=$!
one r3w_wt 7 & P2=$!
wait $P1 $P2

for v in r3w r3w_wt; do
  m=bc_phase2_ros_${v}
  echo "===== ${v} train-theta multiep ====="
  grep -A7 "\[adapt\]" $B/eval_${m}_multiep_traintheta.log
  echo "===== ${v} legacy multiep ====="
  grep -A6 "\[adapt\]" $B/eval_${m}_multiep.log
  echo "===== ${v} within-ep train-theta ====="
  tail -2 $B/eval_${m}_within_traintheta.log
done
echo "=== PHASE 2 ROS R3 DONE ==="
