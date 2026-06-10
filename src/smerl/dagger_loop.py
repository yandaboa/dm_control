"""HG-DAgger driver: iterate collect_demos -> train_seq with growing history.

Iteration 0 bootstraps without a model (the learner commits to a random skill each
sub-episode). Each later iteration samples on-policy skills from the model trained
on all data so far, with the takeover/supervision ranges grown by a fixed step — so
the model first learns to adapt over short stall histories, then longer ones.
Training always merges every iteration's store collected so far.

    python -m src.smerl.dagger_loop --iters 4 --n-episodes 800 --skills 1,2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


def _range(s):
    lo, hi = (int(x) for x in s.split(","))
    return lo, hi


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=800)
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="1,2")
    ap.add_argument("--base-config", type=str, default="configs/bc_adapt.yaml")
    ap.add_argument("--name", type=str, default="bc_adapt",
                    help="checkpoint name stem; iteration K -> <name>_iterK.pt")
    ap.add_argument("--takeover0", type=_range, default=(5, 15))
    ap.add_argument("--supervision0", type=_range, default=(60, 120))
    ap.add_argument("--takeover-growth", type=_range, default=(10, 20),
                    help="added to the takeover range each iteration")
    ap.add_argument("--supervision-growth", type=_range, default=(40, 80))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--wandb", action="store_true", help="enable wandb in training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = os.path.join("src/smerl", args.run)
    base_cfg_path = os.path.join("src/smerl", args.base_config)
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    cfg_dir = os.path.join(run_dir, "_dagger_configs")
    os.makedirs(cfg_dir, exist_ok=True)

    stores = []                         # cumulative store list (relative to src/smerl)
    py = [sys.executable, "-m"]
    for k in range(args.iters):
        tk = (args.takeover0[0] + k * args.takeover_growth[0],
              args.takeover0[1] + k * args.takeover_growth[1])
        sup = (args.supervision0[0] + k * args.supervision_growth[0],
               args.supervision0[1] + k * args.supervision_growth[1])
        store_rel = f"{args.run}/trajectories_dagger_iter{k}"
        stores.append(store_rel)
        prev_model = f"{args.name}_iter{k-1}.pt" if k > 0 else None

        # ---- collect ----
        collect = py + ["src.smerl.collect_demos",
                        "--run", args.run, "--ckpt-file", args.ckpt_file,
                        "--value-net", args.value_net, "--value-norm-skills", args.skills,
                        "--iteration", str(k), "--skills", args.skills,
                        "--n-episodes", str(args.n_episodes),
                        "--takeover-range", f"{tk[0]},{tk[1]}",
                        "--supervision-range", f"{sup[0]},{sup[1]}",
                        "--device", args.device, "--seed", str(args.seed + k)]
        if prev_model is not None:
            collect += ["--model", prev_model]
        run(collect)

        # ---- train on all data so far ----
        cfg = dict(base_cfg)
        cfg["data"] = {"store": list(stores)}
        cfg["out"] = f"{args.run}/{args.name}_iter{k}.pt"
        cfg["device"] = args.device
        cfg["seed"] = args.seed
        cfg_path = os.path.join(cfg_dir, f"iter{k}.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        train = py + ["src.smerl.train_seq", "--config", cfg_path, "--device", args.device]
        if not args.wandb:
            train += ["--no-wandb"]
        else:
            train += ["--run-name", f"{args.name}_iter{k}", "--wandb-group", args.name]
        run(train)
        print(f"[dagger] iteration {k} done: takeover={tk} supervision={sup} "
              f"-> {cfg['out']}  (trained on {len(stores)} stores)")

    print(f"\n[dagger] {args.iters} iterations complete; final model "
          f"{args.run}/{args.name}_iter{args.iters-1}.pt")


if __name__ == "__main__":
    main()
