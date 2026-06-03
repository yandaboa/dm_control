"""Visualize what each SMERL skill actually does in state space.

For each skill z we roll out the (stochastic) policy many times and overlay the
trajectories, coloured by skill. Two columns: rollouts from the fixed nominal
start (intra-skill variability only) and from perturbed starts in a disk (the
distribution the discriminator must separate). A per-skill panel row makes the
modes individually legible. This is the qualitative companion to the
decodability probe (skill_decode.py) and the BAMDP leakage numbers.

    python -m src.smerl.visualize_skills --run runs/smerl_point2d_c
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, sample_in_disk, build_env


def rollouts_for_skill(agent, cfg, nominal_start, z, n, radius, rng,
                       deterministic=False):
    out = []
    for _ in range(n):
        start = (nominal_start if radius == 0.0
                 else sample_in_disk(nominal_start, radius, rng))
        env = build_env(cfg, start=tuple(start))
        obs, _ = env.reset()
        states = [obs[:2].copy()]
        terminated = truncated = False
        success = False
        while not (terminated or truncated):
            a = agent.act(obs, z=z, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(a)
            states.append(obs[:2].copy())
            success = bool(info.get("is_success", False))
        out.append((np.asarray(states), success, len(states)))
    return out, env.goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_point2d_c")
    ap.add_argument("--n", type=int, default=40, help="rollouts per skill")
    ap.add_argument("--radius", type=float, default=0.25,
                    help="perturbed-start disk radius (right column)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="single panel: one deterministic arc per skill")
    ap.add_argument("--ckpt-file", type=str, default="agent.pt",
                    help="checkpoint filename within the run dir "
                         "(e.g. ckpt_step020000.pt for a mid-training policy)")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join("src/smerl", args.run, args.ckpt_file), device)
    n_skills = cfg["n_skills"]
    nominal_start = np.asarray(nominal_start, dtype=np.float32)
    cmap = plt.get_cmap("tab10")
    # tag outputs with the checkpoint step so mid-training plots don't clobber
    tag = "" if args.ckpt_file == "agent.pt" else "_" + os.path.splitext(
        os.path.basename(args.ckpt_file))[0].replace("ckpt_", "")

    if args.clean:
        fig, ax = plt.subplots(figsize=(7, 7))
        goal = None
        for z in range(n_skills):
            # faint stochastic spread for context
            spread, goal = rollouts_for_skill(agent, cfg, nominal_start, z,
                                              args.n, 0.0, rng, deterministic=False)
            for traj, _, _ in spread:
                ax.plot(traj[:, 0], traj[:, 1], color=cmap(z), alpha=0.10, lw=1)
            # one clean deterministic arc
            det, goal = rollouts_for_skill(agent, cfg, nominal_start, z, 1, 0.0,
                                           rng, deterministic=True)
            traj, ok, T = det[0]
            ax.plot(traj[:, 0], traj[:, 1], color=cmap(z), lw=2.5,
                    label=f"z={z}  ({'goal' if ok else 'miss'}, T={T})")
        ax.scatter(*nominal_start, marker="*", c="k", s=220, zorder=5,
                   label="start")
        ax.scatter(*goal, marker="X", c="k", s=220, zorder=5, label="goal")
        ax.add_patch(plt.Circle(goal, cfg.get("success_radius", 0.05),
                                color="k", fill=False, ls="--", alpha=0.5))
        ax.set_title(f"{cfg.get('run', args.run)} — skills from nominal start")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
        out = os.path.join("src/smerl", args.run, f"skills_clean{tag}.png")
        plt.tight_layout(); plt.savefig(out, dpi=130)
        print(f"[plot] saved {out}")
        return

    # ---- top: overlay of all skills, nominal vs perturbed start ----
    fig = plt.figure(figsize=(3.0 * n_skills, 5.6 + 3.4))
    gs = fig.add_gridspec(2, 2 * n_skills, height_ratios=[5.6, 3.4],
                          hspace=0.35, wspace=0.6)
    mid = n_skills  # split the 2*n_skills columns in half
    overlay_axes = [fig.add_subplot(gs[0, :mid]), fig.add_subplot(gs[0, mid:])]
    goal = None
    for col, (rad, title) in enumerate([(0.0, "nominal start"),
                                        (args.radius, f"perturbed r={args.radius}")]):
        ax = overlay_axes[col]
        for z in range(n_skills):
            rs, goal = rollouts_for_skill(agent, cfg, nominal_start, z, args.n,
                                          rad, rng)
            succ = np.mean([r[1] for r in rs])
            for traj, _, _ in rs:
                ax.plot(traj[:, 0], traj[:, 1], color=cmap(z), alpha=0.18, lw=1)
            ax.plot([], [], color=cmap(z), lw=2,
                    label=f"z={z}  succ={succ:.0%}")
        ax.scatter(*nominal_start, marker="*", c="k", s=160, zorder=5)
        ax.scatter(*goal, marker="X", c="k", s=160, zorder=5)
        circ = plt.Circle(goal, cfg.get("success_radius", 0.05), color="k",
                          fill=False, ls="--", alpha=0.5)
        ax.add_patch(circ)
        ax.set_title(f"{cfg.get('run', args.run)} — all skills ({title})")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")

    # ---- bottom: one panel per skill (perturbed starts) ----
    for z in range(n_skills):
        ax = fig.add_subplot(gs[1, 2 * z:2 * z + 2])
        rs, goal = rollouts_for_skill(agent, cfg, nominal_start, z, args.n,
                                      args.radius, rng)
        succ = np.mean([r[1] for r in rs])
        meanT = np.mean([r[2] for r in rs])
        for traj, ok, _ in rs:
            ax.plot(traj[:, 0], traj[:, 1], color=cmap(z),
                    alpha=0.35 if ok else 0.5,
                    lw=1, ls="-" if ok else ":")
        ax.scatter(*nominal_start, marker="*", c="k", s=120, zorder=5)
        ax.scatter(*goal, marker="X", c="k", s=120, zorder=5)
        ax.set_title(f"z={z}  succ={succ:.0%}  T={meanT:.0f}", fontsize=10)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    out = os.path.join("src/smerl", args.run, f"skill_trajectories{tag}.png")
    plt.savefig(out, dpi=120)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
