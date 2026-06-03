"""Ablation: task-oriented V(s) vs skill-conditioned V(s,z), at matched horizon.

We collect ONE set of rollouts from the (preferred) policy and fit two value nets
on identical data and targets:

  * task  — V(s):    skill-agnostic, the BAMDP-exposed value (single number/state)
  * skill — V(s,z):  z conditioned via one-hot (the pre-ablation design)

Then we roll out each skill deterministically and overlay both values over time,
reporting a roughness metric per skill so "how smooth" is quantified:

    roughness = mean | second difference of V along the rollout |   (lower=smoother)

We also report monotonicity violation (mean negative step-to-step change) and the
training MSE of each fit.

    python -m src.smerl.ablate_value --run runs/smerl_lowalpha_ckpt \
        --ckpt-file ckpt_step010000.pt --max-steps 75
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import (TaskValueNet, SkillValueNet, collect,
                                        train_value, rollout_det)


def train_skill_value(X, Z, G, obs_dim, n_skills, device, hidden=(64, 64),
                      epochs=200, batch=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = SkillValueNet(obs_dim, n_skills, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(X, device=device)
    Zt = torch.as_tensor(Z, device=device)
    Gt = torch.as_tensor(G, device=device)
    n = Xt.shape[0]
    final = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(net(Xt[idx], Zt[idx]), Gt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        final = tot / n
    return net, final


def mse_of(net, X, G, device, Z=None):
    Xt = torch.as_tensor(X, device=device)
    Gt = torch.as_tensor(G, device=device)
    with torch.no_grad():
        pred = net(Xt) if Z is None else net(Xt, torch.as_tensor(Z, device=device))
        return float(F.mse_loss(pred, Gt))


def roughness(v):
    """Mean magnitude of the discrete second difference (curvature)."""
    return float(np.mean(np.abs(np.diff(v, 2)))) if len(v) >= 3 else 0.0


def mono_violation(v):
    """Mean downward step (0 = perfectly non-decreasing)."""
    d = np.diff(v)
    return float(-d[d < 0].sum() / max(len(d), 1))


@torch.no_grad()
def task_value_of(net, states, device):
    return net(torch.as_tensor(np.asarray(states, np.float32), device=device)).cpu().numpy()


@torch.no_grad()
def skill_value_of(net, states, z, device):
    Xt = torch.as_tensor(np.asarray(states, np.float32), device=device)
    Zt = torch.full((Xt.shape[0],), int(z), dtype=torch.long, device=device)
    return net(Xt, Zt).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-per-skill", type=int, default=150)
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, obs_dim = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    n_skills = cfg["n_skills"]
    nominal_start = np.asarray(nominal_start, np.float32)

    print(f"[collect] {args.n_per_skill} rollouts/skill (T<={args.max_steps}) ...")
    X, Z, G, succ = collect(agent, cfg, nominal_start, n_skills,
                            args.n_per_skill, args.radius, args.gamma, rng)
    print(f"[collect] {X.shape[0]} states; per-skill success: "
          f"{np.round(succ, 2).tolist()}")

    print("[train] task-oriented V(s) ...")
    task_net = train_value(X, G, obs_dim, device, epochs=args.epochs,
                           seed=args.seed)
    print("[train] skill-conditioned V(s,z) ...")
    skill_net, _ = train_skill_value(X, Z, G, obs_dim, n_skills, device,
                                     epochs=args.epochs, seed=args.seed)
    mse_task = mse_of(task_net, X, G, device)
    mse_skill = mse_of(skill_net, X, G, device, Z=Z)
    print(f"[fit] train MSE  task={mse_task:.4f}  skill={mse_skill:.4f}")

    # persist the skill net (task net already saved by train_task_value)
    torch.save({"state_dict": skill_net.state_dict(), "obs_dim": obs_dim,
                "hidden": [64, 64], "n_skills": n_skills, "mode": "skill",
                "gamma": args.gamma, "run": cfg.get("run", args.run),
                "ckpt_file": args.ckpt_file, "per_skill_success": succ.tolist()},
               os.path.join(run_dir, "value_net_skill.pt"))

    # --- evaluate along each skill's deterministic rollout ---
    fig, axes = plt.subplots(1, n_skills, figsize=(3.4 * n_skills, 4.2),
                             squeeze=False)
    r_task, r_skill = [], []
    for col in range(n_skills):
        traj, goal, ok = rollout_det(agent, cfg, nominal_start, col)
        vt = task_value_of(task_net, traj, device)
        vs = skill_value_of(skill_net, traj, col, device)
        r_task.append(roughness(vt)); r_skill.append(roughness(vs))
        ax = axes[0, col]
        t = np.arange(len(traj))
        ax.plot(t, vt, "-o", ms=3, color="#1f77b4", lw=1.4,
                label=f"task V(s)  rgh={roughness(vt):.4f}")
        ax.plot(t, vs, "-o", ms=3, color="#d62728", lw=1.4,
                label=f"skill V(s,z)  rgh={roughness(vs):.4f}")
        ax.set_title(f"z={col} ({'goal' if ok else 'miss'}, T={len(traj)}, "
                     f"succ={succ[col]:.0%})", fontsize=10)
        ax.set_xlabel("timestep"); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("V along rollout")
        ax.legend(fontsize=7, loc="upper left")

    rt, rs = float(np.mean(r_task)), float(np.mean(r_skill))
    print(f"[roughness] mean curvature  task={rt:.4f}  skill={rs:.4f}  "
          f"(skill is {rt/max(rs,1e-9):.1f}x smoother)")
    fig.suptitle(f"{cfg.get('run', args.run)} [{args.ckpt_file}] — value ablation "
                 f"(T<={args.max_steps})   task MSE={mse_task:.3f} rgh={rt:.4f}  |  "
                 f"skill MSE={mse_skill:.3f} rgh={rs:.4f}", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.13, wspace=0.22)
    out = os.path.join(run_dir, "value_ablation.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
