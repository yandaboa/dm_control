"""Robustness eval — perturb the initial position with Gaussian noise and
measure (a) plain SAC, (b) SMERL averaged over skills, (c) SMERL best-of-K
(Algorithm 2 from the paper: roll each skill once, keep the best return).

Usage:
    python -m src.smerl.eval_robustness
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from src.smerl.point2d_env import Point2DGoalEnv
from src.smerl.smerl_sac import SMERLAgent, SMERLConfig


# ----------------------------- loading -----------------------------------


def load_smerl(ckpt_path: str, obs_dim: int, act_dim: int,
               device: torch.device) -> tuple[SMERLAgent, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = SMERLConfig(
        n_skills=cfg_dict["n_skills"],
        alpha_div=cfg_dict["alpha_div"],
        eps_frac=cfg_dict["eps_frac"],
        R_SAC=cfg_dict["R_SAC"],
    )
    agent = SMERLAgent(obs_dim, act_dim, cfg, device)
    agent.actor.load_state_dict(ckpt["actor"])
    return agent, cfg_dict


# ----------------------------- eval helpers ------------------------------


def perturbed_start(nominal_start: tuple[float, float], sigma: float,
                    rng: np.random.Generator) -> tuple[float, float]:
    if sigma <= 0:
        return tuple(nominal_start)
    noise = rng.normal(0.0, sigma, size=2)
    out = np.asarray(nominal_start, dtype=np.float32) + noise
    out = np.clip(out, -0.95, 0.95)  # keep away from the wall
    return float(out[0]), float(out[1])


def rollout_sac(model: SAC, env: Point2DGoalEnv) -> tuple[float, bool]:
    obs, _ = env.reset()
    ret = 0.0
    done = False
    info: dict = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        done = term or trunc
    return ret, bool(info.get("is_success", False))


def rollout_smerl(agent: SMERLAgent, z: int, env: Point2DGoalEnv) -> tuple[float, bool]:
    obs, _ = env.reset()
    ret = 0.0
    done = False
    info: dict = {}
    while not done:
        a = agent.act(obs, z=z, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        done = term or trunc
    return ret, bool(info.get("is_success", False))


@dataclass
class TaskSpec:
    name: str
    smerl_dir: str
    sac_dir: str
    nominal_start: tuple[float, float]
    goal: tuple[float, float]
    success_radius: float


def evaluate_task(task: TaskSpec, noise_sigmas: list[float], n_trials: int,
                  device: torch.device, rng: np.random.Generator) -> dict:
    probe = Point2DGoalEnv(start=task.nominal_start, goal=task.goal,
                           success_radius=task.success_radius)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]

    sac_path = f"src/smerl/{task.sac_dir}/final_model.zip"
    smerl_path = f"src/smerl/{task.smerl_dir}/agent.pt"
    sac_model = SAC.load(sac_path, device=device)
    smerl_agent, smerl_cfg = load_smerl(smerl_path, obs_dim, act_dim, device)
    n_skills = int(smerl_cfg["n_skills"])
    print(f"\n[{task.name}] loaded SAC + SMERL ({n_skills} skills)  "
          f"nominal start={task.nominal_start}  goal={task.goal}  "
          f"r={task.success_radius}")

    sigmas = []
    sac_R_mean, sac_R_sem = [], []
    sac_S_mean, sac_S_sem = [], []
    smerl_mean_R_mean, smerl_mean_R_sem = [], []
    smerl_mean_S_mean, smerl_mean_S_sem = [], []
    smerl_best_R_mean, smerl_best_R_sem = [], []
    smerl_best_S_mean, smerl_best_S_sem = [], []
    per_skill_S_curves = [[] for _ in range(n_skills)]

    for sigma in noise_sigmas:
        sac_R = np.zeros(n_trials)
        sac_S = np.zeros(n_trials, dtype=bool)
        smerl_R = np.zeros((n_trials, n_skills))
        smerl_S = np.zeros((n_trials, n_skills), dtype=bool)

        for t in range(n_trials):
            start = perturbed_start(task.nominal_start, sigma, rng)

            env_sac = Point2DGoalEnv(start=start, goal=task.goal,
                                     success_radius=task.success_radius)
            r, s = rollout_sac(sac_model, env_sac)
            sac_R[t] = r
            sac_S[t] = s

            for z in range(n_skills):
                env_z = Point2DGoalEnv(start=start, goal=task.goal,
                                       success_radius=task.success_radius)
                r, s = rollout_smerl(smerl_agent, z, env_z)
                smerl_R[t, z] = r
                smerl_S[t, z] = s

        best_idx = smerl_R.argmax(axis=1)
        best_R = smerl_R[np.arange(n_trials), best_idx]
        best_S = smerl_S[np.arange(n_trials), best_idx]

        sigmas.append(sigma)
        sac_R_mean.append(sac_R.mean());        sac_R_sem.append(sac_R.std() / np.sqrt(n_trials))
        sac_S_mean.append(sac_S.mean());        sac_S_sem.append(sac_S.std() / np.sqrt(n_trials))
        smerl_mean_R_mean.append(smerl_R.mean()); smerl_mean_R_sem.append(smerl_R.std() / np.sqrt(n_trials * n_skills))
        smerl_mean_S_mean.append(smerl_S.mean()); smerl_mean_S_sem.append(smerl_S.std() / np.sqrt(n_trials * n_skills))
        smerl_best_R_mean.append(best_R.mean()); smerl_best_R_sem.append(best_R.std() / np.sqrt(n_trials))
        smerl_best_S_mean.append(best_S.mean()); smerl_best_S_sem.append(best_S.std() / np.sqrt(n_trials))
        for z in range(n_skills):
            per_skill_S_curves[z].append(smerl_S[:, z].mean())

        print(f"  sigma={sigma:.2f}  "
              f"SAC[ succ={sac_S.mean():.2f} R={sac_R.mean():6.2f} ]  "
              f"SMERL mean[ succ={smerl_S.mean():.2f} R={smerl_R.mean():6.2f} ]  "
              f"SMERL best-of-{n_skills}[ succ={best_S.mean():.2f} R={best_R.mean():6.2f} ]")

    return {
        "task_name": task.name,
        "noise_sigmas": list(map(float, sigmas)),
        "sac_R_mean": sac_R_mean, "sac_R_sem": sac_R_sem,
        "sac_S_mean": sac_S_mean, "sac_S_sem": sac_S_sem,
        "smerl_mean_R_mean": smerl_mean_R_mean, "smerl_mean_R_sem": smerl_mean_R_sem,
        "smerl_mean_S_mean": smerl_mean_S_mean, "smerl_mean_S_sem": smerl_mean_S_sem,
        "smerl_best_R_mean": smerl_best_R_mean, "smerl_best_R_sem": smerl_best_R_sem,
        "smerl_best_S_mean": smerl_best_S_mean, "smerl_best_S_sem": smerl_best_S_sem,
        "per_skill_S_curves": per_skill_S_curves,
        "n_trials": n_trials,
        "n_skills": n_skills,
    }


# ----------------------------- plotting ----------------------------------


def plot_results(results: list[dict], out_path: str) -> None:
    n_tasks = len(results)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4.5 * n_tasks),
                             squeeze=False)
    for row, res in enumerate(results):
        sigmas = res["noise_sigmas"]
        ax_S = axes[row, 0]
        ax_R = axes[row, 1]

        ax_S.errorbar(sigmas, res["sac_S_mean"], yerr=res["sac_S_sem"],
                      fmt="o-", color="C0", label="SAC baseline (1 policy)")
        ax_S.errorbar(sigmas, res["smerl_mean_S_mean"],
                      yerr=res["smerl_mean_S_sem"],
                      fmt="s-", color="C1", label="SMERL — mean over skills")
        ax_S.errorbar(sigmas, res["smerl_best_S_mean"],
                      yerr=res["smerl_best_S_sem"],
                      fmt="^-", color="C2",
                      label=f"SMERL — best of {res['n_skills']} (Alg 2)")
        # Optional faint per-skill curves
        for z, curve in enumerate(res["per_skill_S_curves"]):
            ax_S.plot(sigmas, curve, "--", alpha=0.25, color="C1")
        ax_S.set_xlabel("start-noise σ")
        ax_S.set_ylabel("success rate")
        ax_S.set_title(f"{res['task_name']} — success rate vs noise")
        ax_S.set_ylim(-0.05, 1.05)
        ax_S.grid(alpha=0.3)
        ax_S.legend(loc="lower left", fontsize=8)

        ax_R.errorbar(sigmas, res["sac_R_mean"], yerr=res["sac_R_sem"],
                      fmt="o-", color="C0", label="SAC baseline (1 policy)")
        ax_R.errorbar(sigmas, res["smerl_mean_R_mean"],
                      yerr=res["smerl_mean_R_sem"],
                      fmt="s-", color="C1", label="SMERL — mean over skills")
        ax_R.errorbar(sigmas, res["smerl_best_R_mean"],
                      yerr=res["smerl_best_R_sem"],
                      fmt="^-", color="C2",
                      label=f"SMERL — best of {res['n_skills']} (Alg 2)")
        ax_R.set_xlabel("start-noise σ")
        ax_R.set_ylabel("mean return")
        ax_R.set_title(f"{res['task_name']} — return vs noise")
        ax_R.grid(alpha=0.3)
        ax_R.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n[plot] saved {out_path}")


# ----------------------------- main --------------------------------------


def main():
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    tasks = [
        TaskSpec(
            name="task A (seed=0, r=0.05)",
            smerl_dir="runs/smerl_point2d",
            sac_dir="runs/sac_point2d",
            nominal_start=(0.27392337, -0.46042657),
            goal=(-0.91805297, -0.96694475),
            success_radius=0.05,
        ),
        TaskSpec(
            name="task B (manual, r=0.05)",
            smerl_dir="runs/smerl_point2d_b",
            sac_dir="runs/sac_point2d_b",
            nominal_start=(-0.25, -0.5),
            goal=(0.66, 0.66),
            success_radius=0.05,
        ),
        TaskSpec(
            name="task C (manual, r=0.25)",
            smerl_dir="runs/smerl_point2d_c",
            sac_dir="runs/sac_point2d_c",
            nominal_start=(-0.25, -0.5),
            goal=(0.66, 0.66),
            success_radius=0.25,
        ),
    ]

    noise_sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    n_trials = 30

    results = []
    for task in tasks:
        results.append(evaluate_task(task, noise_sigmas, n_trials, device, rng))

    out_dir = "src/smerl/runs"
    plot_results(results, os.path.join(out_dir, "robustness.png"))
    with open(os.path.join(out_dir, "robustness.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] saved {os.path.join(out_dir, 'robustness.json')}")


if __name__ == "__main__":
    main()
