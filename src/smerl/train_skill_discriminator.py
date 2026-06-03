"""Train and persist a reusable skill discriminator q(z | s) for a SMERL run.

With a uniform skill prior, a well-fit discriminator gives

    q(z | s) = rho_z(s) / sum_z' rho_z'(s)

i.e. the normalized state-visitation ratio across skills at state s. That makes
the saved softmax directly usable downstream (e.g. as a per-skill occupancy /
belief signal in a BAMDP).

Defaults to run C (smerl_point2d_c), the run whose skills are stable and
separable. Data is collected the same way as the decodability probe: random
skill, start uniform in a 0.1-radius disk around the policy's nominal start.

    python -m src.smerl.train_skill_discriminator
    # later, in your own code:
    from src.smerl.train_skill_discriminator import load_discriminator, skill_posterior
    disc, meta = load_discriminator("src/smerl/runs/smerl_point2d_c/skill_discriminator.pt")
    q = skill_posterior(disc, meta, x=0.1, y=0.2)          # -> array over skills
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

from src.smerl.skill_decode import (
    build_env, collect_rollouts, eval_probe, load_agent, split_rollouts,
    stack_states, train_probe,
)
from src.smerl.smerl_sac import Discriminator


# ----------------------------- load / query helpers ----------------------


def load_discriminator(path: str, device: str | torch.device = "cpu"):
    """Reconstruct the saved discriminator and its metadata."""
    device = torch.device(device)
    art = torch.load(path, map_location=device, weights_only=False)
    disc = Discriminator(art["obs_dim"], art["n_skills"],
                         tuple(art["hidden"])).to(device)
    disc.load_state_dict(art["state_dict"])
    disc.eval()
    return disc, art


@torch.no_grad()
def skill_posterior(disc: Discriminator, meta: dict, x: float, y: float,
                    vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
    """q(z | s) at a single state. Goal is filled from the run's metadata,
    so the caller only supplies the agent's (position, velocity)."""
    gx, gy = meta["goal"]
    obs = torch.tensor([[x, y, vx, vy, gx, gy]], dtype=torch.float32,
                       device=next(disc.parameters()).device)
    return F.softmax(disc(obs), dim=-1).cpu().numpy()[0]


# ----------------------------- visitation map ----------------------------


def plot_visitation_map(disc, meta, out_path, n=200):
    device = next(disc.parameters()).device
    gx, gy = meta["goal"]
    sx, sy = meta["nominal_start"]
    grid = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(grid, grid)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.zeros(X.size),
                    np.full(X.size, gx), np.full(X.size, gy)], axis=1)
    with torch.no_grad():
        probs = F.softmax(disc(torch.as_tensor(pts, dtype=torch.float32,
                                               device=device)), dim=-1).cpu().numpy()
    argmax = probs.argmax(axis=1).reshape(n, n)
    conf = probs.max(axis=1).reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    k = meta["n_skills"]
    im0 = axes[0].imshow(argmax, origin="lower", extent=[-1, 1, -1, 1],
                         cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title(f"{meta['run']} — argmax_z q(z|s)  (vel=0)")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, ticks=range(k))
    cb0.set_label("most-likely skill z")

    im1 = axes[1].imshow(conf, origin="lower", extent=[-1, 1, -1, 1],
                         cmap="magma", vmin=1.0 / k, vmax=1.0)
    axes[1].set_title("max_z q(z|s)  (posterior confidence)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.plot(sx, sy, "w*", ms=14, mec="k", label="nominal start")
        ax.plot(gx, gy, "wX", ms=12, mec="k", label="goal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved {out_path}")


# ----------------------------- main --------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_point2d_c")
    ap.add_argument("--n-rollouts", type=int, default=400)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    ckpt = os.path.join("src/smerl", args.run, "agent.pt")
    agent, cfg, nominal_start, obs_dim = load_agent(ckpt, device)
    n_skills = cfg["n_skills"]
    goal = build_env(cfg).goal

    print(f"[collect] {args.n_rollouts} rollouts, start∈disk(r={args.radius}) "
          f"@ {np.round(nominal_start, 3).tolist()}  goal={np.round(goal,3).tolist()}")
    rollouts = collect_rollouts(agent, cfg, nominal_start, args.n_rollouts,
                                args.radius, rng, deterministic=False)

    # Honest held-out estimate first ...
    train, test = split_rollouts(rollouts, args.test_frac, rng)
    tr_states, tr_z = stack_states(train)
    probe = train_probe(tr_states, tr_z, obs_dim, n_skills, device,
                        epochs=args.epochs, seed=args.seed)
    metrics = eval_probe(probe, test, n_skills, device)
    print(f"[held-out] state_acc={metrics['state_acc']:.3f}  "
          f"traj_acc={metrics['traj_acc']:.3f}  (chance={1/n_skills:.2f})")

    # ... then retrain on ALL collected data for the deployable artifact.
    all_states, all_z = stack_states(rollouts)
    disc = train_probe(all_states, all_z, obs_dim, n_skills, device,
                       epochs=args.epochs, seed=args.seed)

    out_dir = os.path.join("src/smerl", args.run)
    art_path = os.path.join(out_dir, "skill_discriminator.pt")
    torch.save({
        "state_dict": disc.state_dict(),
        "obs_dim": obs_dim,
        "n_skills": n_skills,
        "hidden": [32, 32],
        "goal": [float(goal[0]), float(goal[1])],
        "nominal_start": [float(nominal_start[0]), float(nominal_start[1])],
        "radius": args.radius,
        "run": os.path.basename(args.run.rstrip("/")),
        "trained_on": "all_rollouts",
        "held_out_state_acc": metrics["state_acc"],
        "held_out_traj_acc": metrics["traj_acc"],
        "note": "softmax(disc(obs)) = q(z|s) = normalized per-skill visitation ratio",
    }, art_path)
    print(f"[save] discriminator -> {art_path}")

    plot_visitation_map(disc, {
        "goal": [float(goal[0]), float(goal[1])],
        "nominal_start": [float(nominal_start[0]), float(nominal_start[1])],
        "n_skills": n_skills,
        "run": os.path.basename(args.run.rstrip("/")),
    }, os.path.join(out_dir, "skill_visitation.png"))


if __name__ == "__main__":
    main()
