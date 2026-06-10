#!/bin/bash
# Phase 2 ros v3 — widened takeover schedule + DAgger recency weighting.
# r2w: collect 500 metas driven by bc_phase2_ros_r1 with takeover 5-100 (min FIXED
# at r1's 5, only the upper bound grows -> upper tail = learner-solo histories
# with self-recovery before the expert may correct). Then train two variants in
# parallel on the same data: plain concat (r2w) vs per-store gradient shares
# [0.3, 0.3, 0.4] (r2w_wt), and run the matched evals for both.
# GPUs: collect=2, concat train+eval=3, weighted train+eval=7
# (GPU 0/1 = phase2_ros_v2 70-85 arm, GPUs 4-6 reserved for UWLab).
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P2 ros v3] collect 500 r2w (driver=bc_phase2_ros_r1, TAKEOVER 5-100) ==="
CUDA_VISIBLE_DEVICES=2 $PY -m src.smerl.collect_demos \
  --model bc_phase2_ros_r1.pt --skills 0,1,2 --value-norm-skills 0,1,2 \
  --keep-intervened-only --require-success --n-episodes 500 --max-steps 100 \
  --takeover-range 5,100 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --reset-on-switch --device cuda --seed 3 \
  --out $B/trajectories_dagger012_meta_ros_r2w 2>&1 | tail -4

one() {  # one(variant, gpu): train + the three matched evals
  local v=$1 g=$2
  local m=bc_phase2_ros_${v}   # separate local: ${v} expands BEFORE local assigns
  echo "=== [P2 ros v3] train ${m} (GPU $g) ==="
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
