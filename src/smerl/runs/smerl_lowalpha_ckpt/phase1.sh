#!/bin/bash
# Phase 1 — cross-episode latching, skills 1,2.
#  1. ONE meta-episodic on-policy collection (driver = bc_bs8_rope_share20, the adopted
#     Phase-0 winner) -> trajectories_dagger12_meta (chained sub-eps, shared theta).
#  2. Train TWO models on that store: adopted RoPE+share0.2 (bc_phase1_12) and a
#     wpe+share0.4 baseline (bc_phase1_12_wpe), to test if RoPE's Phase-0 edge carries
#     to cross-episode latching.
#  3. Eval both: cross-episode (eval_adapt_multiep) + within-episode (eval_bamdp_adapt).
set -e
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt

echo "=== [P1] collect meta-episodic on-policy (skills 1,2), driver=bc_bs8_rope_share20 ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.collect_demos \
  --model bc_bs8_rope_share20.pt --skills 1,2 --value-norm-skills 1,2 \
  --keep-intervened-only --n-episodes 400 --max-steps 100 \
  --takeover-range 5,15 --supervision-range 20,40 --delay-range 5,10 \
  --device cuda --seed 0 \
  --out $B/trajectories_dagger12_meta 2>&1 | tail -15

echo "=== [P1] train BOTH variants in parallel (rope@0.2 on GPU0, wpe@0.4 on GPU1) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase1_12.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase1 --run-name bc_phase1_12_rope \
  --out runs/smerl_lowalpha_ckpt/bc_phase1_12.pt > $B/train_phase1_rope.log 2>&1 &
P_ROPE=$!
CUDA_VISIBLE_DEVICES=1 $PY -m src.smerl.train_seq \
  --config src/smerl/configs/bc_phase1_12_wpe.yaml --device cuda \
  --wandb-project 2d --wandb-group adapt_phase1 --run-name bc_phase1_12_wpe \
  --out runs/smerl_lowalpha_ckpt/bc_phase1_12_wpe.pt > $B/train_phase1_wpe.log 2>&1 &
P_WPE=$!
wait $P_ROPE; echo "  rope train done: $(grep '\[done\]' $B/train_phase1_rope.log | tail -1)"
wait $P_WPE;  echo "  wpe  train done: $(grep '\[done\]' $B/train_phase1_wpe.log  | tail -1)"

for tag in rope:bc_phase1_12.pt wpe:bc_phase1_12_wpe.pt; do
  name=${tag%%:*}; mdl=${tag##*:}
  echo "=== [P1:$name] eval cross-episode latching (eval_adapt_multiep) ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.eval_adapt_multiep \
    --model $mdl --skills 1,2 --n-meta 80 --n-sub 4 --max-steps 100 \
    --device cuda --out adapt_multiep_phase1_${name}.png 2>&1 | tee $B/eval_phase1_${name}_multiep.log
  echo "=== [P1:$name] within-episode sanity (eval_bamdp_adapt) ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
    --models $mdl --skills 1,2 --value-norm-skills 1,2 --reset-on-switch \
    --theta-bad 0.95 --n-episodes 200 --max-steps 150 --device cuda 2>&1 \
    | tee $B/eval_phase1_${name}_within.log
done

echo "=== PHASE 1 DONE ==="
