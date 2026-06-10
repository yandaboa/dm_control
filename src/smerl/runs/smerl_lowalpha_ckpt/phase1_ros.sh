#!/bin/bash
# Phase 1 RETRY — env-side terminate-on-switch (reset_on_switch=True), skills 1,2.
# Mirrors phase1_r2.sh except: (a) collection runs with --reset-on-switch (a switch
# teleports to a fresh start = give up & retry, keeping the step clock), 800 eps to
# match r2's total switch-demo volume; (b) training uses dart12 + the PURE ros store
# (old non-teleport meta stores excluded); (c) switch_share swept 0.2/0.3/0.4/0.5 in
# parallel on GPUs 0-3; (d) ALL evals use matched --reset-on-switch dynamics.
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P1 ros] collect ON-POLICY (driver=bc_phase1_12), LONG supervision, reset-on-switch ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.collect_demos \
  --model bc_phase1_12.pt --skills 1,2 --value-norm-skills 1,2 \
  --keep-intervened-only --n-episodes 800 --max-steps 100 \
  --takeover-range 5,15 --supervision-range 150,300 --delay-range 5,10 \
  --max-learner-steps 120 --reset-on-switch --device cuda --seed 1 \
  --out $B/trajectories_dagger12_meta_ros 2>&1 | tail -5

echo "=== [P1 ros] train switch_share sweep 0.2/0.3/0.4/0.5 in parallel (GPUs 0-3) ==="
pids=()
g=0
for s in 20 30 40 50; do
  CUDA_VISIBLE_DEVICES=$g $PY -m src.smerl.train_seq \
    --config src/smerl/configs/bc_phase1_ros_s${s}.yaml --device cuda \
    --wandb-project 2d --wandb-group adapt_phase1_ros --run-name bc_phase1_ros_s${s} \
    --out runs/smerl_lowalpha_ckpt/bc_phase1_ros_s${s}.pt \
    > $B/train_phase1_ros_s${s}.log 2>&1 &
  pids+=($!)
  g=$((g+1))
done
for i in 0 1 2 3; do
  wait ${pids[$i]}
  s=$((20 + 10*i))
  echo "  s${s} train done: $(grep '\[done\]' $B/train_phase1_ros_s${s}.log | tail -1)"
done

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

echo "=== PHASE 1 ROS DONE ==="
