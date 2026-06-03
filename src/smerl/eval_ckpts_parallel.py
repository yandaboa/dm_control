"""Evaluate a directory of BC checkpoints by forced-skill closed-loop success,
parallelized across all free GPUs.

Each checkpoint is scored by the per-skill forced-z success rate (clamp the skill
token to z for the whole episode, measure goal-reaching) over a fixed set of
starts — the same metric the single-process selector uses, but fanned out so the
slow iterative loop becomes ~Nx faster (N = free GPUs).

Design:
  * one PERSISTENT worker process per free GPU; each loads the teacher env + value
    net + start set ONCE, then pulls checkpoints off a shared queue. So a GPU that
    finishes a checkpoint immediately grabs the next — natural load balancing /
    "queue the next job when a GPU opens up".
  * free GPUs are auto-detected from nvidia-smi memory use (override with --gpus).
  * tqdm over completed checkpoints gives a live ETA.

The core scorer ``forced_skill_success`` is importable, so an in-training eval can
reuse it on the live model later.

    python -m src.smerl.eval_ckpts_parallel \
        --ckpt-dir src/smerl/runs/smerl_lowalpha_ckpt/bc_multimodal_150_ckpts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import multiprocessing as mp


def forced_skill_success(model, cfg, value_net, starts, device, n_skills):
    """Per-skill forced-z closed-loop success [n_skills] for one BC model.
    Importable for reuse (e.g. in-training eval on the live model)."""
    import numpy as np
    from src.smerl.skill_decode import build_env
    from src.smerl.eval_bc_multimodal import rollout_multimodal
    succ = np.zeros(n_skills)
    for z in range(n_skills):
        ok = [rollout_multimodal(model, value_net, build_env(cfg, start=tuple(s)),
                                 device, force_z=z)[0] for s in starts]
        succ[z] = float(np.mean(ok))
    return succ


def _free_gpus(thresh_mib):
    try:
        txt = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"]).decode()
        return [int(l.split(",")[0]) for l in txt.strip().splitlines()
                if int(l.split(",")[1].strip()) < thresh_mib]
    except Exception:
        return [0]


def _worker(gpu_id, in_q, out_q, dispatched, C):
    """Persistent per-GPU worker: load once, then drain the checkpoint queue,
    exiting when the queue is empty (so it works whether spawned at the start or
    mid-run when a GPU frees up). Reports __ready__ on success, __failed__ if its
    GPU won't initialize (e.g. grabbed by another job)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from queue import Empty
    import numpy as np
    import torch
    from src.smerl.skill_decode import load_agent, sample_in_disk
    from src.smerl.train_task_value import load_value_net
    from src.smerl.eval_bc_multimodal import load_bc
    try:
        device = torch.device("cuda:0")
        torch.zeros(1, device=device)              # claim the GPU early
        _, cfg, nominal_start, _ = load_agent(C["agent_ckpt"], device)
        cfg["max_episode_steps"] = C["max_steps"]
        nominal_start = np.asarray(nominal_start, np.float32)
        n_skills = cfg["n_skills"]
        value_net, _ = load_value_net(C["value_net"], device)
        starts = [sample_in_disk(nominal_start, C["radius"],
                                 np.random.default_rng(C["seed"] + i))
                  for i in range(C["n_episodes"])]
    except Exception:
        import traceback
        out_q.put(("__failed__", gpu_id, traceback.format_exc()))
        return
    out_q.put(("__ready__", gpu_id, None))
    while True:
        try:
            path = in_q.get(timeout=1.0)
        except Empty:
            if dispatched.is_set():                # all tasks queued, none left
                break
            continue
        try:
            model = load_bc(path, device)
            succ = forced_skill_success(model, cfg, value_net, starts, device,
                                        n_skills)
            out_q.put((path, succ.tolist(), None))
        except Exception:
            import traceback
            out_q.put((path, None, traceback.format_exc()))


def _teacher_success(C):
    """Teacher per-skill success on the same start set (CPU, parent process)."""
    import numpy as np
    import torch
    from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
    from src.smerl.eval_bc_multimodal import rollout_teacher
    agent, cfg, nominal_start, _ = load_agent(C["agent_ckpt"], torch.device("cpu"))
    cfg["max_episode_steps"] = C["max_steps"]
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    starts = [sample_in_disk(nominal_start, C["radius"],
                             np.random.default_rng(C["seed"] + i))
              for i in range(C["n_episodes"])]
    return [float(np.mean([rollout_teacher(agent, build_env(cfg, start=tuple(s)), z)[0]
                           for s in starts])) for z in range(n_skills)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, required=True,
                    help="directory of BC checkpoints to evaluate")
    ap.add_argument("--glob", type=str, default="*.pt",
                    help="checkpoint glob within --ckpt-dir")
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n-episodes", type=int, default=40, help="starts per skill")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gpus", type=str, default="auto",
                    help="'auto' (free GPUs) or comma list e.g. 0,1,2")
    ap.add_argument("--mem-thresh", type=int, default=1000,
                    help="a GPU is 'free' if memory.used < this many MiB")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--poll-interval", type=float, default=60.0,
                    help="when --gpus auto, re-check for newly-free GPUs every N "
                         "seconds and add workers (other jobs may end mid-eval)")
    ap.add_argument("--out", type=str, default=None,
                    help="results json (default <ckpt-dir>/forced_eval.json); "
                         "also the cache of already-evaluated checkpoints")
    ap.add_argument("--select-out", type=str, default=None,
                    help="copy the best checkpoint to this path")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate every checkpoint, ignoring the cache")
    args = ap.parse_args()

    from tqdm import tqdm

    ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(
        "src/smerl", args.ckpt_dir) if not os.path.exists(args.ckpt_dir) else args.ckpt_dir
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, args.glob)))
    if not ckpts:
        raise SystemExit(f"no checkpoints matched {os.path.join(ckpt_dir, args.glob)}")

    run_dir = os.path.join("src/smerl", args.run)
    C = {"agent_ckpt": os.path.join(run_dir, args.ckpt_file),
         "value_net": os.path.join(run_dir, args.value_net),
         "n_episodes": args.n_episodes, "radius": args.radius,
         "max_steps": args.max_steps, "seed": args.seed}

    auto = args.gpus == "auto"

    def free_gpus():
        return (_free_gpus(args.mem_thresh) if auto
                else [int(g) for g in args.gpus.split(",") if g.strip() != ""])

    # cache: reuse already-evaluated checkpoints (and the teacher) from --out json
    out = args.out or os.path.join(ckpt_dir, "forced_eval.json")
    cache = {}
    if os.path.exists(out) and not args.force:
        with open(out) as f:
            cache = json.load(f)
    cached_results = {} if args.force else {
        bn: rec["success"] for bn, rec in cache.get("results", {}).items()}
    results = {c: cached_results[os.path.basename(c)] for c in ckpts
               if os.path.basename(c) in cached_results}
    to_eval = [c for c in ckpts if os.path.basename(c) not in cached_results]
    print(f"[eval] {len(ckpts)} checkpoints  |  {len(results)} cached, "
          f"{len(to_eval)} to eval  |  gpus={'auto' if auto else free_gpus()}  "
          f"poll={args.poll_interval}s  |  {args.n_episodes} starts/skill")
    if not to_eval:
        print("[eval] nothing to do — all checkpoints already evaluated "
              "(use --force to re-run)")

    if not args.force and cache.get("teacher") is not None \
            and cache.get("n_episodes") == args.n_episodes:
        tsucc = cache["teacher"]
        print(f"[teacher] reusing cached reference={[round(t,2) for t in tsucc]}")
    else:
        print("[teacher] computing reference (CPU) ...")
        tsucc = _teacher_success(C)
    feas = [t > 0.5 for t in tsucc]
    print(f"[teacher] feasible skills={[i for i,f in enumerate(feas) if f]}")

    errors = {}
    if to_eval:
        import time
        from queue import Empty
        in_q, out_q = mp.Queue(), mp.Queue()
        for c in to_eval:
            in_q.put(c)
        dispatched = mp.Event()
        dispatched.set()                 # all tasks are already queued
        claimed, procs = set(), []

        def claim_free_gpus():
            """Spawn a worker on each free GPU we haven't claimed yet."""
            for g in sorted(set(free_gpus()) - claimed):
                for _ in range(args.workers_per_gpu):
                    p = mp.Process(target=_worker,
                                   args=(g, in_q, out_q, dispatched, C), daemon=True)
                    p.start()
                    procs.append((g, p))
                claimed.add(g)

        claim_free_gpus()
        if not claimed:
            raise SystemExit("no free GPUs to start; raise --mem-thresh or "
                             "pass --gpus")
        print(f"[eval] claimed GPUs {sorted(claimed)}; evaluating ...")

        done, last_poll = 0, time.time()
        pbar = tqdm(total=len(to_eval), desc="eval ckpts", unit="ckpt")
        while done < len(to_eval):
            try:
                tag, b, c = out_q.get(timeout=2.0)
                if tag == "__ready__":
                    tqdm.write(f"[eval] worker ready on GPU {b}")
                elif tag == "__failed__":
                    tqdm.write(f"[warn] GPU {b} worker failed to init; skipping")
                else:
                    if c is not None:
                        errors[tag] = c
                    else:
                        results[tag] = b
                    done += 1
                    pbar.update(1)
            except Empty:
                pass
            # periodically grab GPUs freed by other jobs; recover lost tasks
            if auto and time.time() - last_poll > args.poll_interval:
                last_poll = time.time()
                if not any(p.is_alive() for _, p in procs):
                    missing = [c for c in to_eval if c not in results
                               and c not in errors]
                    if missing and in_q.empty():     # workers died with tasks in flight
                        for c in missing:
                            in_q.put(c)
                        claimed.clear()              # allow re-claiming GPUs
                claim_free_gpus()
                n_alive = sum(p.is_alive() for _, p in procs)
                pbar.set_postfix_str(f"{n_alive} workers, {len(claimed)} gpus")
        pbar.close()
        for _, p in procs:
            p.join(timeout=5)

    def feas_mean(s):
        vals = [s[i] for i in range(len(s)) if feas[i]]
        return sum(vals) / max(len(vals), 1)

    rows = sorted(results.items(), key=lambda kv: feas_mean(kv[1]), reverse=True)
    print(f"\n{'checkpoint':>20} {'per-skill success':>28} {'feas-mean':>10}")
    for path, succ in rows:
        print(f"{os.path.basename(path):>20} "
              f"{str([round(x,2) for x in succ]):>28} {feas_mean(succ):>10.3f}")
    if errors:
        print(f"\n[errors] {len(errors)} checkpoints failed:")
        for path, e in errors.items():
            print(f"  {os.path.basename(path)}: {e.strip().splitlines()[-1]}")

    payload = {"teacher": tsucc, "feasible": feas, "n_episodes": args.n_episodes,
               "radius": args.radius, "max_steps": args.max_steps,
               "results": {os.path.basename(p): {"success": s,
                           "feas_mean": feas_mean(s)} for p, s in results.items()}}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[json] saved {out}")

    if rows and args.select_out:
        import shutil
        best_path, best_succ = rows[0]
        dst = args.select_out if os.path.isabs(args.select_out) else os.path.join(
            run_dir, args.select_out)
        shutil.copyfile(best_path, dst)
        print(f"[select] best={os.path.basename(best_path)} "
              f"(feas-mean={feas_mean(best_succ):.3f}) -> {dst}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
