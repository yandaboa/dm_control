"""Skill decodability probe.

Question: if we spawn the SMERL policy at random points around its nominal
training start (uniform in a 0.1-radius disk) with a randomly drawn skill z, and
roll it out, can a freshly-trained classifier recover z from the visited states?

This is a post-hoc measure of how distinguishable the learned modes actually are.
It mirrors the DIAYN/SMERL discriminator q_phi(z|s), but the classifier here is
trained from scratch on held-out collected data (the in-training discriminator
that ships in agent.pt is reported alongside as a baseline).

Run from the repo root (SMERL conda env):

    python -m src.smerl.skill_decode
    python -m src.smerl.skill_decode --runs runs/smerl_point2d --n-rollouts 400
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
import torch.nn.functional as F

from src.smerl.point2d_env import Point2DGoalEnv
from src.smerl.smerl_sac import Discriminator, SMERLAgent, SMERLConfig


# ----------------------------- env / agent reconstruction ----------------


def build_env(cfg: dict, start=None, success_radius=None) -> Point2DGoalEnv:
    """Reconstruct the run's env. ``start`` overrides the nominal start."""
    sr = success_radius if success_radius is not None else cfg.get("success_radius", 0.05)
    mes = cfg.get("max_episode_steps", 200)
    if "start" in cfg and "goal" in cfg:
        s = tuple(cfg["start"]) if start is None else tuple(start)
        return Point2DGoalEnv(start=s, goal=tuple(cfg["goal"]), success_radius=sr,
                              max_episode_steps=mes)
    # Run A: start/goal were derived from env_seed, not stored explicitly.
    if start is None:
        return Point2DGoalEnv(seed=cfg["env_seed"], success_radius=sr,
                              max_episode_steps=mes)
    g = Point2DGoalEnv(seed=cfg["env_seed"]).goal
    return Point2DGoalEnv(start=tuple(start), goal=(float(g[0]), float(g[1])),
                          success_radius=sr, max_episode_steps=mes)


def load_agent(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    probe = build_env(cfg)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    scfg = SMERLConfig(n_skills=cfg["n_skills"], alpha_div=cfg["alpha_div"],
                       eps_frac=cfg["eps_frac"], R_SAC=cfg["R_SAC"])
    agent = SMERLAgent(obs_dim, act_dim, scfg, device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.disc.load_state_dict(ckpt["disc"])  # in-training discriminator (baseline)
    nominal_start = probe.start
    return agent, cfg, nominal_start, obs_dim


# ----------------------------- rollout collection ------------------------


def sample_in_disk(center: np.ndarray, radius: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Uniform sample in a disk of the given radius around ``center``."""
    r = radius * np.sqrt(rng.random())
    theta = 2.0 * np.pi * rng.random()
    out = center + r * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    return np.clip(out, -0.95, 0.95).astype(np.float32)


def collect_rollouts(agent: SMERLAgent, cfg: dict, nominal_start: np.ndarray,
                     n_rollouts: int, radius: float, rng: np.random.Generator,
                     deterministic: bool):
    """Returns a list of (obs_array[T,obs_dim], z) for each rollout."""
    n_skills = cfg["n_skills"]
    rollouts = []
    for _ in range(n_rollouts):
        z = int(rng.integers(n_skills))
        start = sample_in_disk(nominal_start, radius, rng)
        env = build_env(cfg, start=start)
        obs, _ = env.reset()
        states = [obs.copy()]
        terminated = truncated = False
        while not (terminated or truncated):
            a = agent.act(obs, z=z, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(a)
            states.append(obs.copy())
        rollouts.append((np.asarray(states, dtype=np.float32), z))
    return rollouts


# ----------------------------- discriminator probe -----------------------


def train_probe(train_states, train_z, obs_dim, n_skills, device,
                epochs=300, batch_size=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    disc = Discriminator(obs_dim, n_skills).to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=lr)
    X = torch.as_tensor(train_states, device=device)
    y = torch.as_tensor(train_z, device=device, dtype=torch.long)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(disc(X[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return disc


@torch.no_grad()
def eval_probe(disc, rollouts_test, n_skills, device):
    """Per-state and per-trajectory accuracy + confusion matrix (rows=true)."""
    conf = np.zeros((n_skills, n_skills), dtype=np.int64)
    state_correct = state_total = 0
    traj_correct = 0
    for states, z in rollouts_test:
        X = torch.as_tensor(states, device=device)
        probs = F.softmax(disc(X), dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()
        state_correct += int((preds == z).sum())
        state_total += len(preds)
        for p in preds:
            conf[z, p] += 1
        # trajectory verdict: mean softmax over the visited states
        traj_pred = int(probs.mean(dim=0).argmax().item())
        traj_correct += int(traj_pred == z)
    return {
        "state_acc": state_correct / max(state_total, 1),
        "traj_acc": traj_correct / max(len(rollouts_test), 1),
        "confusion": conf.tolist(),
    }


def split_rollouts(rollouts, test_frac, rng):
    idx = np.arange(len(rollouts))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(rollouts)))
    test_idx, train_idx = set(idx[:n_test].tolist()), idx[n_test:]
    train = [rollouts[i] for i in train_idx]
    test = [rollouts[i] for i in sorted(test_idx)]
    return train, test


def stack_states(rollouts):
    states = np.concatenate([s for s, _ in rollouts], axis=0)
    z = np.concatenate([np.full(len(s), zz, dtype=np.int64)
                        for s, zz in rollouts], axis=0)
    return states, z


# ----------------------------- per-run experiment ------------------------


def run_experiment(run_dir, n_rollouts, radius, test_frac, device,
                   deterministic, seed):
    rng = np.random.default_rng(seed)
    ckpt_path = os.path.join("src/smerl", run_dir, "agent.pt")
    agent, cfg, nominal_start, obs_dim = load_agent(ckpt_path, device)
    n_skills = cfg["n_skills"]
    chance = 1.0 / n_skills

    rollouts = collect_rollouts(agent, cfg, nominal_start, n_rollouts, radius,
                                rng, deterministic)
    z_counts = np.bincount([z for _, z in rollouts], minlength=n_skills)
    train, test = split_rollouts(rollouts, test_frac, rng)
    tr_states, tr_z = stack_states(train)

    probe = train_probe(tr_states, tr_z, obs_dim, n_skills, device, seed=seed)
    fresh = eval_probe(probe, test, n_skills, device)
    pretrained = eval_probe(agent.disc, test, n_skills, device)

    name = os.path.basename(run_dir.rstrip("/"))
    print(f"\n[{name}] {n_skills} skills | {n_rollouts} rollouts "
          f"(start∈disk r={radius} @ {np.round(nominal_start,3).tolist()}) | "
          f"{'det' if deterministic else 'stochastic'} | chance={chance:.2f}")
    print(f"   rollouts/skill: {z_counts.tolist()}  "
          f"(train states={len(tr_z)}, test rollouts={len(test)})")
    print(f"   fresh probe     : state_acc={fresh['state_acc']:.3f}  "
          f"traj_acc={fresh['traj_acc']:.3f}")
    print(f"   in-train disc   : state_acc={pretrained['state_acc']:.3f}  "
          f"traj_acc={pretrained['traj_acc']:.3f}")

    return {
        "run": name,
        "n_skills": n_skills,
        "chance": chance,
        "n_rollouts": n_rollouts,
        "radius": radius,
        "nominal_start": np.round(nominal_start, 4).tolist(),
        "rollouts_per_skill": z_counts.tolist(),
        "deterministic": deterministic,
        "fresh_probe": fresh,
        "pretrained_disc": pretrained,
    }


def plot_confusions(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n), squeeze=False)
    for row, res in enumerate(results):
        for col, (key, title) in enumerate(
                [("fresh_probe", "fresh probe"),
                 ("pretrained_disc", "in-training disc")]):
            ax = axes[row, col]
            conf = np.array(res[key]["confusion"], dtype=float)
            norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(norm, vmin=0, vmax=1, cmap="viridis")
            ax.set_xlabel("predicted z")
            ax.set_ylabel("true z")
            ax.set_xticks(range(res["n_skills"]))
            ax.set_yticks(range(res["n_skills"]))
            ax.set_title(f"{res['run']} — {title}\n"
                         f"state_acc={res[key]['state_acc']:.2f}  "
                         f"traj_acc={res[key]['traj_acc']:.2f}")
            for i in range(res["n_skills"]):
                for j in range(res["n_skills"]):
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            color="w" if norm[i, j] < 0.6 else "k", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs/smerl_point2d", "runs/smerl_point2d_b",
                             "runs/smerl_point2d_c"])
    ap.add_argument("--n-rollouts", type=int, default=400)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--deterministic", action="store_true",
                    help="use the deterministic policy mean (default: stochastic)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="src/smerl/runs")
    args = ap.parse_args()

    device = torch.device(args.device)
    results = [run_experiment(r, args.n_rollouts, args.radius, args.test_frac,
                              device, args.deterministic, args.seed)
               for r in args.runs]

    os.makedirs(args.out_dir, exist_ok=True)
    plot_confusions(results, os.path.join(args.out_dir, "skill_decode.png"))
    with open(os.path.join(args.out_dir, "skill_decode.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] saved {os.path.join(args.out_dir, 'skill_decode.json')}")


if __name__ == "__main__":
    main()
