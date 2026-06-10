#!/bin/bash
# 16 runs: {wpe,rope} x switch_share{0.2,0.3,0.4,0.5} x seed{1,2}, batch 8, constant LR.
# All 8 GPUs free -> fixed round-robin, waves of 8 (one job per GPU per wave). wandb on.
cd /mnt/storage/lti/dm_control
PY=/home/ubuntu/miniforge3/envs/SMERL/bin/python
B=src/smerl/runs/smerl_lowalpha_ckpt
GROUP=adapt_p0_bs8_seedsweep

JOBS=()
for v in wpe rope; do for sh in 20 30 40 50; do for sd in 1 2; do JOBS+=("$v $sh $sd"); done; done; done

run_wave(){  # $1=mode (train|eval)
  local i=0 PIDS=()
  for job in "${JOBS[@]}"; do
    set -- $job; v=$1; sh=$2; sd=$3
    local g=$((i % 8)) name="bc_bs8_${v}_share${sh}_seed${sd}"
    if [ "$MODE" = train ]; then
      local share rope=""; share=$(echo "scale=2;$sh/100"|bc); [ "$v" = "rope" ] && rope="--rope"
      CUDA_VISIBLE_DEVICES=$g nohup $PY -m src.smerl.train_seq --config src/smerl/configs/bc_adapt12.yaml \
        --device cuda --batch 8 $rope --switch-share $share --seed $sd \
        --wandb-project 2d --wandb-group $GROUP --run-name $name \
        --out runs/smerl_lowalpha_ckpt/${name}.pt > $B/train_${name}.log 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$g nohup $PY -m src.smerl.eval_bamdp_adapt --run runs/smerl_lowalpha_ckpt \
        --models ${name}.pt --skills 1,2 --value-norm-skills 1,2 --reset-on-switch --theta-bad 0.95 \
        --n-episodes 200 --max-steps 150 --device cuda > $B/eval_${name}.log 2>&1 &
    fi
    PIDS+=($!); i=$((i+1))
    if [ $((i % 8)) -eq 0 ]; then for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); fi
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
}

MODE=train run_wave; echo "=== ALL 16 TRAININGS DONE ==="
MODE=eval  run_wave; echo "=== ALL 16 EVALS DONE ==="

printf "%-34s %8s %12s %8s %9s\n" model success "succ|failed" "%failed" switches
for v in wpe rope; do for sh in 20 30 40 50; do for sd in 1 2; do
  name="bc_bs8_${v}_share${sh}_seed${sd}"; grep -E "^ +${name}\.pt" $B/eval_${name}.log | tail -1
done; done; done
