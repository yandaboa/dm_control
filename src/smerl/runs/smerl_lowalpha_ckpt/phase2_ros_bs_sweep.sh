#!/bin/bash
# Batch-size sweep at the phase2-ros r1 setting (long ~830-token sequences):
# bs16/32/64 on GPUs 2/3/7, identical data (dart012 + ros_r1_500), epochs, LR.
# The concurrently-training bc_phase2_ros_r1 (bs8, GPU 0) is the baseline.
# Each model gets the same matched-dynamics evals as phase2_ros_eval.sh.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

one() {  # one(bs, gpu): train + eval on a dedicated GPU
  local bs=$1 g=$2 m=bc_phase2_ros_r1_bs$1
  CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.train_seq \
    --config src/smerl/configs/${m}.yaml --device cuda \
    --wandb-project 2d --wandb-group adapt_phase2_ros --run-name ${m}_n500 \
    --out runs/smerl_lowalpha_ckpt/${m}.pt > $B/train_${m}.log 2>&1
  echo "[bs${bs}] train done: $(grep '\[done\]' $B/train_${m}.log | tail -1)"
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
    --reset-on-switch --theta-bad 0.95 --n-episodes 200 --max-steps 150 --device cuda \
    > $B/eval_${m}_within.log 2>&1
  echo "[bs${bs}] evals done"
}

one 16 2 & P1=$!
one 32 3 & P2=$!
one 64 7 & P3=$!
wait $P1 $P2 $P3

for bs in 16 32 64; do
  m=bc_phase2_ros_r1_bs${bs}
  echo "===== bs${bs} train-theta multiep ====="
  grep -A7 "\[adapt\]" $B/eval_${m}_multiep_traintheta.log
  echo "===== bs${bs} legacy multiep ====="
  grep -A5 "\[adapt\]" $B/eval_${m}_multiep.log
  echo "===== bs${bs} within-ep ====="
  tail -2 $B/eval_${m}_within.log
done
echo "=== BS SWEEP DONE ==="
