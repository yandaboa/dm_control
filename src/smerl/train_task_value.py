"""Train a task-oriented, success-only value function V(s) and persist it.

The critic baked into agent.pt values the *gated* SMERL reward
(r_env + alpha*indicator*r_tilde), so its value field is contaminated by the
diversity bonus, and it is skill-conditioned Q(s,z,a). Here we fit a clean,
**skill-agnostic, task-oriented** value function from scratch on a
**success-only** reward:

    r_t = 1 at the transition that reaches the goal, 0 otherwise.

The Monte-Carlo return-to-go is then G_t = gamma^(t_success - t) for a successful
episode and 0 for a miss, so the regressed value

    V(s) = E[ gamma^(steps-to-goal) | s ]  in [0, 1]

is exactly "discounted probability/proximity of task success" — 1 at the goal,
0 on trajectories that never get there. It is a pure function of the state s
(NOT the skill z): we pool rollouts across all skills and fit one MLP V(s) by
MC regression on stochastic rollouts from perturbed starts (for state coverage).

The trained net is saved to ``<run>/value_net.pt`` so the BAMDP wrapper can load
it and expose V(s) as an observation. We also plot V along each skill's
deterministic rollout: xy trajectory on top, V-over-time directly below.

    python -m src.smerl.train_task_value --run runs/smerl_point2d_c \
        --ckpt-file agent.pt
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk


# ----------------------------- value network -----------------------------


class TaskValueNet(nn.Module):
    """Task-oriented state-value V(s) -> [0,1] via sigmoid. Skill-agnostic."""

    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        sizes = [obs_dim, *hidden, 1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim

    def forward(self, obs):
        return torch.sigmoid(self.net(obs)).squeeze(-1)


class SkillValueNet(nn.Module):
    """Skill-conditioned value V(s, z) -> [0,1]; z passed as one-hot, concatenated
    to obs. The pre-ablation design — one value per skill in a single net."""

    def __init__(self, obs_dim, n_skills, hidden=(64, 64)):
        super().__init__()
        sizes = [obs_dim + n_skills, *hidden, 1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.n_skills = n_skills

    def forward(self, obs, z_idx):
        z_oh = F.one_hot(z_idx, num_classes=self.n_skills).float()
        return torch.sigmoid(self.net(torch.cat([obs, z_oh], dim=-1))).squeeze(-1)


def load_value_net(path, device="cpu"):
    """Load a value net saved by this script. Returns (net, meta). Handles both
    the task-oriented V(s) and the skill-conditioned V(s,z) variants."""
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("mode") == "skill":
        net = SkillValueNet(blob["obs_dim"], blob["n_skills"],
                            tuple(blob["hidden"])).to(device)
    else:
        net = TaskValueNet(blob["obs_dim"], tuple(blob["hidden"])).to(device)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob


class NormalizedSkillValueNet(nn.Module):
    """Wrap a skill-conditioned value net and rescale V(s,z) per skill to [0,1]:
    clip((V - vmin[z]) / (vmax[z] - vmin[z]), 0, 1). Steepens each skill's value
    climb so a failure/stall is far more legible. Unlisted skills (vmin=0,vmax=1)
    pass through unchanged. Keeps .n_skills so the BAMDP treats it as V(s,z)."""

    def __init__(self, base, vmin, vmax):
        super().__init__()
        self.base = base
        self.n_skills = base.n_skills
        self.register_buffer("vmin", torch.as_tensor(np.asarray(vmin, np.float32)))
        self.register_buffer("vmax", torch.as_tensor(np.asarray(vmax, np.float32)))

    def forward(self, obs, z_idx):
        v = self.base(obs, z_idx)
        lo, hi = self.vmin[z_idx], self.vmax[z_idx]
        return ((v - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


@torch.no_grad()
def compute_skill_value_norm(agent, cfg, value_net, skills, nominal_start, radius,
                             n_roll, rng, device, n_skills, lo_pct=1.0, hi_pct=99.0):
    """Per-skill (vmin, vmax) from deterministic rollouts off perturbed starts —
    the [lo_pct, hi_pct] percentiles of V(s,z) over the visited states."""
    vmin = np.zeros(n_skills, np.float32)
    vmax = np.ones(n_skills, np.float32)
    nominal_start = np.asarray(nominal_start, np.float32)
    for z in skills:
        vals = []
        for _ in range(n_roll):
            start = sample_in_disk(nominal_start, radius, rng)
            env = build_env(cfg, start=tuple(start))
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                v = float(value_net(torch.as_tensor(obs[None], dtype=torch.float32,
                          device=device), torch.tensor([z], device=device)).item())
                vals.append(v)
                obs, _, terminated, truncated, _ = env.step(
                    agent.act(obs, z=z, deterministic=True))
        vals = np.asarray(vals)
        vmin[z] = float(np.percentile(vals, lo_pct))
        vmax[z] = float(np.percentile(vals, hi_pct))
    return vmin.tolist(), vmax.tolist()


# ----------------------------- data collection ---------------------------


def collect(agent, cfg, nominal_start, n_skills, n_per_skill, radius, gamma,
            rng, max_steps_cap=None):
    """Stochastic rollouts from perturbed starts; MC success-return per state.

    Returns (obs[N,obs_dim], z[N], G[N]) and a per-skill success rate. z is kept
    only for per-skill diagnostics/plots; the value fit itself ignores it."""
    obs_all, z_all, g_all = [], [], []
    succ_count = np.zeros(n_skills)
    for z in range(n_skills):
        for _ in range(n_per_skill):
            start = sample_in_disk(nominal_start, radius, rng) if radius > 0 \
                else nominal_start
            env = build_env(cfg, start=tuple(start))
            obs, _ = env.reset()
            states = [obs.copy()]
            terminated = truncated = False
            success = False
            while not (terminated or truncated):
                a = agent.act(obs, z=z, deterministic=False)
                obs, _, terminated, truncated, info = env.step(a)
                states.append(obs.copy())
                success = bool(info.get("is_success", False))
            T = len(states)
            # success-only return-to-go: gamma^(t_succ - t) if success else 0
            if success:
                t_succ = T - 1
                g = gamma ** (t_succ - np.arange(T)).astype(np.float64)
                succ_count[z] += 1
            else:
                g = np.zeros(T)
            obs_all.append(np.asarray(states, dtype=np.float32))
            z_all.append(np.full(T, z, dtype=np.int64))
            g_all.append(g.astype(np.float32))
    X = np.concatenate(obs_all)
    Z = np.concatenate(z_all)
    G = np.concatenate(g_all)
    return X, Z, G, succ_count / n_per_skill


def train_value(X, G, obs_dim, device, hidden=(64, 64), epochs=200, batch=512,
                lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = TaskValueNet(obs_dim, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(X, device=device)
    Gt = torch.as_tensor(G, device=device)
    n = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = net(Xt[idx])
            loss = F.mse_loss(pred, Gt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"   [value] epoch {ep:3d}  mse={tot / n:.4f}")
    return net


# ----------------------------- eval + plot -------------------------------


def rollout_det(agent, cfg, start, z):
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


@torch.no_grad()
def value_of(net, states, device):
    Xt = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
    return net(Xt).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-per-skill", type=int, default=150,
                    help="training rollouts per skill")
    ap.add_argument("--radius", type=float, default=0.25,
                    help="perturbed-start disk radius for state coverage")
    ap.add_argument("--max-steps", type=int, default=75,
                    help="episode horizon (overrides the run's stored value)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    ckpt_path = os.path.join("src/smerl", args.run, args.ckpt_file)
    agent, cfg, nominal_start, obs_dim = load_agent(ckpt_path, device)
    cfg["max_episode_steps"] = args.max_steps   # fit V(s) under the eval horizon
    n_skills = cfg["n_skills"]
    nominal_start = np.asarray(nominal_start, dtype=np.float32)
    cmap = plt.get_cmap(args.cmap)
    tag = "" if args.ckpt_file == "agent.pt" else "_" + os.path.splitext(
        os.path.basename(args.ckpt_file))[0].replace("ckpt_", "")

    print(f"[collect] {args.n_per_skill} rollouts/skill, radius={args.radius} ...")
    X, Z, G, succ = collect(agent, cfg, nominal_start, n_skills,
                            args.n_per_skill, args.radius, args.gamma, rng)
    print(f"[collect] {X.shape[0]} states; per-skill success rate: "
          f"{np.round(succ, 2).tolist()}")
    print(f"[train] fitting task-oriented V(s) on success-only return ...")
    hidden = (64, 64)
    net = train_value(X, G, obs_dim, device, hidden=hidden, epochs=args.epochs,
                      seed=args.seed)

    # --- persist the value net for the BAMDP wrapper ---
    out_dir = os.path.join("src/smerl", args.run)
    net_path = os.path.join(out_dir, "value_net.pt")
    torch.save({"state_dict": net.state_dict(), "obs_dim": obs_dim,
                "hidden": list(hidden), "gamma": args.gamma,
                "run": cfg.get("run", args.run), "ckpt_file": args.ckpt_file,
                "per_skill_success": succ.tolist()}, net_path)
    print(f"[save] wrote {net_path}")

    # --- eval along each skill's deterministic rollout ---
    panels = []
    for z in range(n_skills):
        traj, goal, ok = rollout_det(agent, cfg, nominal_start, z)
        vals = value_of(net, traj, device)
        panels.append((z, traj, vals, ok, goal))

    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    fig, axes = plt.subplots(2, n_skills, figsize=(3.4 * n_skills, 6.2),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [3.0, 2.0]})
    sc = None
    for col, (z, traj, vals, ok, goal) in enumerate(panels):
        axt, axb = axes[0, col], axes[1, col]
        axt.plot(traj[:, 0], traj[:, 1], color="0.6", lw=0.8, zorder=2)
        sc = axt.scatter(traj[:, 0], traj[:, 1], c=vals, cmap=cmap, norm=norm,
                         s=22, edgecolors="k", linewidths=0.3, zorder=3)
        axt.scatter(*nominal_start, marker="*", c="k", s=240, zorder=5)
        axt.scatter(*goal, marker="X", c="k", s=180, zorder=5)
        axt.add_patch(plt.Circle(goal, cfg.get("success_radius", 0.05),
                                 color="k", fill=False, ls="--", alpha=0.5))
        axt.set_title(f"z={z}  ({'goal' if ok else 'miss'}, T={len(traj)}, "
                      f"train_succ={succ[z]:.0%})", fontsize=10)
        axt.set_xlim(-1, 1); axt.set_ylim(-1, 1); axt.set_aspect("equal")
        axt.grid(alpha=0.3)

        t = np.arange(len(vals))
        axb.plot(t, vals, color="0.6", lw=0.8, zorder=2)
        axb.scatter(t, vals, c=vals, cmap=cmap, norm=norm, s=18,
                    edgecolors="k", linewidths=0.2, zorder=3)
        axb.set_xlabel("timestep")
        if col == 0:
            axb.set_ylabel("V(s) = E[$\\gamma^{\\,steps\\,to\\,goal}$]")
        axb.set_ylim(-0.02, 1.02)
        axb.grid(alpha=0.3)

    fig.suptitle(f"{cfg.get('run', args.run)} [{args.ckpt_file}] — task-oriented "
                 f"value V(s) along each skill  (gamma={args.gamma})",
                 fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.9, bottom=0.08,
                        wspace=0.25, hspace=0.28)
    cax = fig.add_axes([0.93, 0.12, 0.013, 0.74])
    fig.colorbar(sc, cax=cax, label="V(s)  (discounted success)")

    out = os.path.join(out_dir, f"task_value{tag}.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
