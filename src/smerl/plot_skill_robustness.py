"""Plot SMERL skill trajectories under start-position perturbation.

For each task (A/B/C), draw a row of subplots, one per noise sigma. Within each
subplot, sample a perturbed start and roll each of the K skills deterministically
for one episode, then overlay the trajectories.

Usage:
    python -m src.smerl.plot_skill_robustness
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.eval_robustness import load_smerl, perturbed_start
from src.smerl.point2d_env import Point2DGoalEnv


@dataclass
class TaskSpec:
    name: str
    smerl_dir: str
    nominal_start: tuple[float, float]
    goal: tuple[float, float]
    success_radius: float


def rollout_skill(agent, z: int, env: Point2DGoalEnv) -> tuple[np.ndarray, bool, int]:
    obs, _ = env.reset()
    traj = [env._pos.copy()]
    done = False
    info: dict = {}
    while not done:
        a = agent.act(obs, z=z, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        traj.append(env._pos.copy())
        done = term or trunc
    return np.stack(traj), bool(info.get("is_success", False)), len(traj) - 1


def plot_task(task: TaskSpec, noise_sigmas: list[float], device: torch.device,
              rng: np.random.Generator, out_path: str) -> None:
    probe = Point2DGoalEnv(start=task.nominal_start, goal=task.goal,
                           success_radius=task.success_radius)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]

    agent, cfg = load_smerl(f"src/smerl/{task.smerl_dir}/agent.pt",
                            obs_dim, act_dim, device)
    K = int(cfg["n_skills"])
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    n_cols = len(noise_sigmas)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 3.8),
                             squeeze=False)
    axes = axes[0]

    goal = np.asarray(task.goal, dtype=np.float32)
    nominal_start = np.asarray(task.nominal_start, dtype=np.float32)

    for col, sigma in enumerate(noise_sigmas):
        ax = axes[col]
        start = perturbed_start(task.nominal_start, sigma, rng)
        succ_count = 0
        for z in range(K):
            env = Point2DGoalEnv(start=start, goal=task.goal,
                                 success_radius=task.success_radius)
            traj, succ, length = rollout_skill(agent, z, env)
            ls = "-" if succ else "--"
            ax.plot(traj[:, 0], traj[:, 1], ls, color=colors[z], lw=1.6,
                    alpha=0.9, label=f"z={z} ({'✓' if succ else '✗'}, T={length})")
            ax.plot(traj[-1, 0], traj[-1, 1], "s", color=colors[z], ms=6)
            succ_count += int(succ)

        ax.plot(*nominal_start, "*", color="gray", ms=10, alpha=0.6,
                label="nominal start")
        ax.plot(*start, "*", color="black", ms=14, label="perturbed start")
        ax.plot(*goal, "X", color="black", ms=12, label="goal")
        theta = np.linspace(0, 2 * np.pi, 96)
        ax.plot(goal[0] + task.success_radius * np.cos(theta),
                goal[1] + task.success_radius * np.sin(theta),
                "k--", alpha=0.5)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"σ={sigma}   succ={succ_count}/{K}", fontsize=10)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

    fig.suptitle(
        f"{task.name}  —  one deterministic episode per skill, "
        f"starts perturbed by N(0, σ²I)  (solid = success, dashed = failure)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main():
    rng = np.random.default_rng(7)
    device = torch.device("cpu")

    tasks = [
        TaskSpec(
            name="task A (seed=0, r=0.05)",
            smerl_dir="runs/smerl_point2d",
            nominal_start=(0.27392337, -0.46042657),
            goal=(-0.91805297, -0.96694475),
            success_radius=0.05,
        ),
        TaskSpec(
            name="task B (manual, r=0.05)",
            smerl_dir="runs/smerl_point2d_b",
            nominal_start=(-0.25, -0.5),
            goal=(0.66, 0.66),
            success_radius=0.05,
        ),
        TaskSpec(
            name="task C (manual, r=0.25)",
            smerl_dir="runs/smerl_point2d_c",
            nominal_start=(-0.25, -0.5),
            goal=(0.66, 0.66),
            success_radius=0.25,
        ),
    ]

    noise_sigmas = [0.0, 0.10, 0.20, 0.30, 0.50]
    out_dir = "src/smerl/runs"

    for task in tasks:
        out_path = os.path.join(
            out_dir, f"skill_robustness_{task.smerl_dir.split('/')[-1]}.png"
        )
        plot_task(task, noise_sigmas, device, rng, out_path)


if __name__ == "__main__":
    main()
