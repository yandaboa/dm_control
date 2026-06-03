"""Plot transformer forced-skill rollouts against the teacher skill they clone.

Two jobs:
  1. (optional) checkpoint SELECTION — closed-loop success is unstable across
     epochs at flat val loss, so picking by val loss is unreliable. Given a set of
     candidate checkpoints (e.g. the periodic saves from train_seq --save-every),
     evaluate each by forced-skill closed-loop success and keep the best.
  2. PLOT — for each skill z, overlay the SMERL teacher's deterministic
     trajectories (grey) and the transformer's forced-z rollouts (colored) from an
     identical set of starts, with per-skill success annotated. This is a direct
     visual check that the transformer reproduces each skill's behavior, and shows
     where (and which) skills drift off the teacher path.

    python -m src.smerl.plot_skill_replication --run runs/smerl_lowalpha_ckpt \
        --select --models "bc_multimodal_ckpts/ep*.pt"
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
from src.smerl.train_task_value import load_value_net
from src.smerl.eval_bc_multimodal import load_bc, rollout_multimodal, rollout_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_multimodal.pt",
                    help="model to plot (if not selecting)")
    ap.add_argument("--models", type=str, default=None,
                    help="glob (within run dir) of candidate checkpoints to select from")
    ap.add_argument("--select", action="store_true",
                    help="pick the best candidate by closed-loop success, save it")
    ap.add_argument("--select-out", type=str, default="bc_multimodal.pt",
                    help="filename (in run dir) to copy the selected best model to")
    ap.add_argument("--out-plot", type=str, default="bc_skill_replication.png",
                    help="filename (in run dir) for the replication plot")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n-select", type=int, default=40, help="starts for selection")
    ap.add_argument("--n-plot", type=int, default=15, help="trajectories/skill to draw")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    goal = build_env(cfg).goal
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)

    def starts(n):
        return [sample_in_disk(nominal_start, args.radius,
                               np.random.default_rng(args.seed + i)) for i in range(n)]

    def forced_success(model, n):
        """per-skill closed-loop success of the BC model (forced z)."""
        sts = starts(n)
        succ = np.zeros(n_skills)
        for z in range(n_skills):
            ok = [rollout_multimodal(model, value_net,
                                     build_env(cfg, start=tuple(s)), device,
                                     force_z=z)[0] for s in sts]
            succ[z] = np.mean(ok)
        return succ

    # teacher per-skill success (reference, also tells us which skills are feasible)
    tsucc = np.zeros(n_skills)
    for z in range(n_skills):
        tsucc[z] = np.mean([rollout_teacher(agent, build_env(cfg, start=tuple(s)), z)[0]
                            for s in starts(args.n_select)])

    # ---- selection ----
    model_path = os.path.join(run_dir, args.model)
    if args.select and args.models:
        cands = sorted(glob.glob(os.path.join(run_dir, args.models)))
        print(f"[select] {len(cands)} candidates; teacher per-skill succ="
              f"{np.round(tsucc,2).tolist()}")
        feas = tsucc > 0.5     # score only on skills the teacher can actually do
        best_score, best = -1.0, None
        for p in cands:
            m = load_bc(p, device)
            s = forced_success(m, args.n_select)
            score = float(s[feas].mean()) if feas.any() else float(s.mean())
            star = ""
            if score > best_score:
                best_score, best = score, p
                star = "  *"
            print(f"  {os.path.basename(p):>16}  succ={np.round(s,2).tolist()}  "
                  f"feas-mean={score:.3f}{star}")
        model_path = os.path.join(run_dir, args.select_out)
        shutil.copyfile(best, model_path)
        print(f"[select] best={os.path.basename(best)} (feas-mean={best_score:.3f}) "
              f"-> {model_path}")

    model = load_bc(model_path, device)

    # ---- per-skill trajectory plot (teacher vs forced-z BC, same starts) ----
    sts = starts(args.n_plot)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, n_skills, figsize=(3.3 * n_skills, 3.7),
                             squeeze=False)
    for z in range(n_skills):
        ax = axes[0, z]
        bc_ok = []
        for s in sts:
            ts, _, _, _, tstates = rollout_teacher(
                agent, build_env(cfg, start=tuple(s)), z)
            ax.plot(tstates[:, 0], tstates[:, 1], color="0.55", lw=1.0,
                    alpha=0.7, zorder=2)
            ok, _, _, _, bstates, _ = rollout_multimodal(
                model, value_net, build_env(cfg, start=tuple(s)), device, force_z=z)
            ax.plot(bstates[:, 0], bstates[:, 1], color=cmap(z), lw=1.0,
                    alpha=0.8, zorder=3)
            bc_ok.append(ok)
        ax.scatter(*nominal_start, marker="*", c="k", s=160, zorder=5)
        ax.scatter(goal[0], goal[1], marker="X", c="k", s=130, zorder=5)
        ax.add_patch(plt.Circle(goal, cfg.get("success_radius", 0.05),
                                color="k", fill=False, ls="--", alpha=0.5))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"z={z}\nteacher {tsucc[z]:.2f}  BC {np.mean(bc_ok):.2f}",
                     fontsize=10)
    # legend proxies
    axes[0, 0].plot([], [], color="0.55", lw=2, label="teacher (det)")
    axes[0, 0].plot([], [], color=cmap(0), lw=2, label="transformer (forced z)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Transformer forced-skill rollouts vs teacher  "
                 f"(grey=teacher, color=BC; {args.n_plot} starts, "
                 f"r={args.radius}, T<={args.max_steps})", fontsize=12)
    plt.tight_layout()
    out = os.path.join(run_dir, args.out_plot)
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
