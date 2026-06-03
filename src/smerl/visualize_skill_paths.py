"""Per-skill paths + skill-conditioned value profiles, to choose 2 skills with
distinct paths AND identifiable (clearly-climbing) values for the switch task.

    python -m src.smerl.visualize_skill_paths
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    K = cfg["n_skills"]
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, K, figsize=(3.2 * K, 6.0), squeeze=False)
    print(f"{'skill':>5} {'reached':>8} {'len':>5} {'Vstart':>7} {'Vend':>6} "
          f"{'climb/step':>11}")
    for z in range(K):
        env = build_env(cfg, start=tuple(nominal_start))
        obs, _ = env.reset()
        states = [obs.copy()]
        term = trunc = False
        ok = False
        while not (term or trunc):
            a = agent.act(obs, z=z, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            states.append(obs.copy())
            ok = bool(info.get("is_success", False))
        S = np.asarray(states, np.float32)
        V = value_net(torch.as_tensor(S, device=device),
                      torch.full((len(S),), z, dtype=torch.long, device=device)).cpu().numpy()
        climb = (V[-1] - V[0]) / max(len(V) - 1, 1)
        print(f"{z:>5} {('yes' if ok else 'NO'):>8} {len(S):>5} {V[0]:>7.3f} "
              f"{V[-1]:>6.3f} {climb:>11.4f}")
        ax = axes[0, z]
        ax.plot(S[:, 0], S[:, 1], color=cmap(z), lw=2)
        ax.scatter(*nominal_start, marker="*", c="k", s=180, zorder=5)
        ax.scatter(goal[0], goal[1], marker="X", c="k", s=140, zorder=5)
        ax.add_patch(plt.Circle(goal, cfg.get("success_radius", 0.25), color="k",
                                fill=False, ls="--", alpha=0.4))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"skill {z}  ({'goal' if ok else 'miss'}, T={len(S)})", fontsize=10)
        axb = axes[1, z]
        axb.plot(V, color=cmap(z), lw=2)
        axb.set_ylim(-0.02, 1.02); axb.grid(alpha=.3); axb.set_xlabel("timestep")
        if z == 0:
            axb.set_ylabel("V(s, z)  (skill-conditioned)")
        axb.set_title(f"climb={climb:.3f}/step", fontsize=9)
    fig.suptitle("Skills: paths (top) and skill-conditioned value V(s,z) (bottom) "
                 "— pick 2 with distinct paths + clearly-climbing value", fontsize=12)
    plt.tight_layout()
    out = os.path.join(run_dir, "skill_paths_values.png")
    plt.savefig(out, dpi=130)
    print(f"[viz] saved {out}")


if __name__ == "__main__":
    main()
