"""Visualize the SAC soft-value of each skill along its rollout.

SAC has an implicit state-value function — the *soft value*

    V(s, z) = E_{a~pi(.|s,z)} [ min(Q1,Q2)(s,z,a) - alpha * log pi(a|s,z) ],

the bootstrap target the critic already regresses to (smerl_sac.py). We estimate
it by sampling actions from the policy. alpha (the entropy coef) is not stored in
the checkpoint, so we read it from the training history by step.

For each skill we roll out the deterministic policy and plot, in a column:
  top   — the xy trajectory, points coloured by V(s,z)
  bottom— V(s,z) over time (per visited state), same colormap
so a point on the path matches a value on the line below it.

    python -m src.smerl.visualize_critic_values --run runs/smerl_lowalpha_ckpt
    python -m src.smerl.visualize_critic_values --run runs/smerl_lowalpha_ckpt \
        --ckpt-file ckpt_step020000.pt
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env


def load_critic(agent, ckpt_path, device):
    """skill_decode.load_agent restores only actor+disc; the value function needs
    the trained Q-net, so load it here."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.critic.load_state_dict(ckpt["critic"])
    return agent


def ent_coef_for(run, ckpt_file):
    """Recover alpha (entropy coef) from the training history by checkpoint step.
    agent.pt -> last logged value; ckpt_stepNNNNNN.pt -> nearest logged step."""
    summ = os.path.join("src/smerl", run, "summary.json")
    try:
        hist = json.load(open(summ)).get("history", [])
        logged = [(h["step"], h.get("ent_coef")) for h in hist
                  if h.get("ent_coef") is not None]
    except (OSError, json.JSONDecodeError):
        logged = []
    if not logged:
        print("[warn] no ent_coef in history; using alpha=0 (pure E[min Q])")
        return 0.0
    if ckpt_file == "agent.pt":
        return float(logged[-1][1])
    step = int("".join(c for c in ckpt_file if c.isdigit()))
    s, a = min(logged, key=lambda kv: abs(kv[0] - step))
    print(f"[info] alpha={a:.4g} (history step {s}, requested {step})")
    return float(a)


@torch.no_grad()
def soft_value(agent, obs, z, alpha, device, n_samples=64):
    """V(s,z) = E_a[min Q(s,z,a) - alpha*log pi(a|s,z)] for a batch [N, obs_dim]."""
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
    if obs_t.ndim == 1:
        obs_t = obs_t[None, :]
    n = obs_t.shape[0]
    z_oh = agent.z_onehot(np.full(n, int(z), dtype=np.int64))
    acc = torch.zeros(n, device=device)
    for _ in range(n_samples):
        a, logp, _ = agent.actor.sample(obs_t, z_oh)        # stochastic sample
        q1, q2 = agent.critic(obs_t, z_oh, a)
        v = torch.min(q1, q2).squeeze(-1) - alpha * logp.squeeze(-1)
        acc += v
    return (acc / n_samples).cpu().numpy()


def rollout_full(agent, cfg, start, z):
    """Deterministic rollout; returns (full_obs[T, obs_dim], goal, success)."""
    env = build_env(cfg, start=tuple(start))
    obs, _ = env.reset()
    states = [obs.copy()]
    terminated = truncated = False
    success = False
    while not (terminated or truncated):
        a = agent.act(obs, z=z, deterministic=True)
        obs, _, terminated, truncated, info = env.step(a)
        states.append(obs.copy())
        success = bool(info.get("is_success", False))
    return np.asarray(states), env.goal, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="agent.pt")
    ap.add_argument("--n-samples", type=int, default=64,
                    help="action samples for the soft-value expectation")
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt_path = os.path.join("src/smerl", args.run, args.ckpt_file)
    agent, cfg, nominal_start, _ = load_agent(ckpt_path, device)
    load_critic(agent, ckpt_path, device)
    alpha = ent_coef_for(args.run, args.ckpt_file)
    n_skills = cfg["n_skills"]
    nominal_start = np.asarray(nominal_start, dtype=np.float32)
    cmap = plt.get_cmap(args.cmap)
    tag = "" if args.ckpt_file == "agent.pt" else "_" + os.path.splitext(
        os.path.basename(args.ckpt_file))[0].replace("ckpt_", "")

    # --- compute everything first so we can share one color scale ---
    panels = []
    all_vals = []
    for z in range(n_skills):
        traj, goal, ok = rollout_full(agent, cfg, nominal_start, z)
        vals = soft_value(agent, traj, z, alpha, device, n_samples=args.n_samples)
        panels.append((z, traj, vals, ok, goal))
        all_vals.append(vals)
    cat = np.concatenate(all_vals)
    vmin, vmax = np.percentile(cat, 1), np.percentile(cat, 99)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- plot: per skill, trajectory (top) over value-vs-time (bottom) ---
    fig, axes = plt.subplots(2, n_skills, figsize=(3.4 * n_skills, 6.2),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [3.0, 2.0]})
    sc = None
    for col, (z, traj, vals, ok, goal) in enumerate(panels):
        axt, axb = axes[0, col], axes[1, col]

        # top: trajectory coloured by value
        axt.plot(traj[:, 0], traj[:, 1], color="0.6", lw=0.8, zorder=2)
        sc = axt.scatter(traj[:, 0], traj[:, 1], c=vals, cmap=cmap, norm=norm,
                         s=22, edgecolors="k", linewidths=0.3, zorder=3)
        axt.scatter(*nominal_start, marker="*", c="k", s=240, zorder=5)
        axt.scatter(*goal, marker="X", c="k", s=180, zorder=5)
        axt.add_patch(plt.Circle(goal, cfg.get("success_radius", 0.05),
                                 color="k", fill=False, ls="--", alpha=0.5))
        axt.set_title(f"z={z}  ({'goal' if ok else 'miss'}, T={len(traj)})",
                      fontsize=11)
        axt.set_xlim(-1, 1); axt.set_ylim(-1, 1); axt.set_aspect("equal")
        axt.grid(alpha=0.3)

        # bottom: value over time, same colormap
        t = np.arange(len(vals))
        axb.plot(t, vals, color="0.6", lw=0.8, zorder=2)
        axb.scatter(t, vals, c=vals, cmap=cmap, norm=norm, s=18,
                    edgecolors="k", linewidths=0.2, zorder=3)
        axb.set_xlabel("timestep")
        if col == 0:
            axb.set_ylabel("V(s, z)  (soft value)")
        axb.set_ylim(vmin, vmax)
        axb.grid(alpha=0.3)

    fig.suptitle(f"{cfg.get('run', args.run)} — SAC soft-value V(s,z) along each "
                 f"skill's rollout  (alpha={alpha:.3g}, {args.n_samples} samples)",
                 fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.91, top=0.9, bottom=0.08,
                        wspace=0.25, hspace=0.28)
    cax = fig.add_axes([0.93, 0.12, 0.013, 0.74])
    fig.colorbar(sc, cax=cax, label="V(s, z)")

    out = os.path.join("src/smerl", args.run, f"critic_values{tag}.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
