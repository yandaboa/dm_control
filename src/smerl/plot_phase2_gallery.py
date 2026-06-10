"""Gallery of individual multi-episode runs from the converged Phase-2 model.

Rolls out N meta-episodes (shared theta per meta, persistent transformer context,
no reset_on_switch -- matched collect_demos dynamics) with the adopted 3-skill
model bc_phase2_r3.pt, and saves ONE png per meta-episode into a new directory.
Each panel shows the BAMDP value over time, colored by the skill the model
committed to, with sub-episode boundaries marked and a per-sub success tick.

    python -m src.smerl.plot_phase2_gallery --n-meta 24 --n-sub 4
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import make_base_env_fn, wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.eval_adapt_multiep import rollout_meta


def plot_meta(ax, bad, subs, play, cmap):
    x0 = 0
    for i, sub in enumerate(subs):
        xs = np.arange(x0, x0 + len(sub["value"]))
        ax.plot(xs, sub["value"], color="0.7", lw=1, zorder=1)
        ax.scatter(xs, sub["value"], c=[cmap(z) for z in sub["skills"]], s=14, zorder=2)
        ax.axvline(x0 - 0.5, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.text(x0 + 1, 1.04, f"#{i}{'✓' if sub['success'] else '✗'}", fontsize=8)
        x0 += len(sub["value"])
    ax.set_ylim(-0.02, 1.15)
    ax.set_ylabel("value")
    ax.set_xlabel("timestep (sub-episodes concatenated; : = reset)")
    first_str = ",".join(str(s["first"]) for s in subs)
    ax.set_title(f"bad skill = {bad}  |  first-step skill per sub-ep: [{first_str}]  "
                 f"(color = committed skill)", fontsize=10)
    for z in play:
        ax.plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
    ax.legend(fontsize=8, ncol=len(play), loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_phase2_r3.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="0,1,2")
    ap.add_argument("--n-meta", type=int, default=24)
    ap.add_argument("--n-sub", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--teacher-action", action="store_true")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--theta-bad", type=float, default=0.95)
    ap.add_argument("--theta-good", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default="phase2_multiep_gallery")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    outdir = os.path.join(run_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    agent, cfg, nominal_start, _ = load_agent(os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    play = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    value_net, _, _ = wrap_value_norm(value_net, play, agent, cfg, nominal_start,
                                      args.radius, device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius, args.max_steps)

    cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=np.full(n_skills, float(args.max_steps)),
                        n_sub_episodes=10**9, schedule="budget", continue_on_failure=True,
                        expose_value=True, expose_failing=True, plateau_prob=0.5,
                        certain_skill=True, reset_on_switch=False)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b,
                                  np.random.default_rng(args.seed), device, value_net=value_net)

    cmap = plt.get_cmap("tab10")
    rng = np.random.default_rng(args.seed)
    succ_by_idx = np.zeros(args.n_sub); cnt_by_idx = np.zeros(args.n_sub)
    badfirst_by_idx = np.zeros(args.n_sub)
    for m in range(args.n_meta):
        bad = int(rng.choice(play))
        theta = np.full(n_skills, args.theta_bad)
        for z in play:
            if z != bad:
                theta[z] = args.theta_good
        bamdp.rng = np.random.default_rng(args.seed + 1 + m)
        subs = rollout_meta(model, agent, bamdp, device, theta, args.n_sub,
                            args.max_steps, teacher_action=args.teacher_action)
        if not subs:
            continue
        for i, sub in enumerate(subs):
            cnt_by_idx[i] += 1
            succ_by_idx[i] += int(sub["success"])
            badfirst_by_idx[i] += int(sub["first"] == bad)
        fig, ax = plt.subplots(figsize=(11, 3.0))
        plot_meta(ax, bad, subs, play, cmap)
        fig.tight_layout()
        fn = os.path.join(outdir, f"meta_{m:03d}_bad{bad}.png")
        fig.savefig(fn, dpi=130)
        plt.close(fig)
        print(f"[viz] {fn}  ({len(subs)} sub-eps, "
              f"success={[int(s['success']) for s in subs]})")

    print(f"\n[summary] {int(cnt_by_idx[0])} meta-eps plotted into {outdir}")
    print("  success by sub-ep:  " + "  ".join(
        f"#{i}:{succ_by_idx[i]/max(cnt_by_idx[i],1):.2f}" for i in range(args.n_sub)))
    print("  P(first==bad):      " + "  ".join(
        f"#{i}:{badfirst_by_idx[i]/max(cnt_by_idx[i],1):.2f}" for i in range(args.n_sub)))


if __name__ == "__main__":
    main()
