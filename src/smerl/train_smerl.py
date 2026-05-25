"""SMERL training loop (Algorithm 1 in Kumar et al. 2020).

Episode rollout: sample z ~ Uniform(Z) at episode start, keep it fixed.
At episode end, compute task return; if R >= R_SAC - eps, retroactively
add alpha * r_tilde to every per-step reward before pushing to the buffer.

Run from the repo root:

    python -m src.smerl.train_smerl --total-timesteps 80000 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from src.smerl.point2d_env import Point2DGoalEnv
from src.smerl.smerl_sac import SMERLAgent, SMERLConfig


def make_env(seed: int, start: tuple[float, float] | None = None,
             goal: tuple[float, float] | None = None) -> Point2DGoalEnv:
    return Point2DGoalEnv(seed=seed, start=start, goal=goal)


def evaluate_skills(agent: SMERLAgent, env_seed: int, n_skills: int,
                    n_episodes_per_skill: int = 3,
                    start: tuple[float, float] | None = None,
                    goal: tuple[float, float] | None = None) -> dict:
    """Roll out each latent skill deterministically and report per-skill stats."""
    per_skill = []
    for z in range(n_skills):
        env = make_env(env_seed, start=start, goal=goal)
        returns, successes, lens, end_dists = [], [], [], []
        for _ in range(n_episodes_per_skill):
            obs, _ = env.reset()
            ret, length = 0.0, 0
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                a = agent.act(obs, z=z, deterministic=True)
                obs, r, terminated, truncated, info = env.step(a)
                ret += float(r)
                length += 1
            returns.append(ret)
            successes.append(bool(info.get("is_success", False)))
            lens.append(length)
            end_dists.append(float(info.get("distance_to_goal", np.nan)))
        per_skill.append({
            "z": z,
            "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lens)),
            "mean_final_dist": float(np.mean(end_dists)),
        })
    best = max(per_skill, key=lambda r: r["mean_return"])
    return {
        "per_skill": per_skill,
        "best_z": best["z"],
        "best_return": best["mean_return"],
        "best_success_rate": best["success_rate"],
        "mean_return_across_skills": float(np.mean([s["mean_return"] for s in per_skill])),
        "mean_success_across_skills": float(np.mean([s["success_rate"] for s in per_skill])),
    }


def collect_skill_trajectories(agent: SMERLAgent, env_seed: int,
                               n_skills: int,
                               start: tuple[float, float] | None = None,
                               goal: tuple[float, float] | None = None) -> dict:
    """Single deterministic episode per skill — used to inspect diversity."""
    out = {}
    for z in range(n_skills):
        env = make_env(env_seed, start=start, goal=goal)
        obs, _ = env.reset()
        traj = [env._pos.copy()]
        terminated = truncated = False
        while not (terminated or truncated):
            a = agent.act(obs, z=z, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(a)
            traj.append(env._pos.copy())
        out[int(z)] = np.stack(traj).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--algo-seed", type=int, default=42)
    ap.add_argument("--total-timesteps", type=int, default=80_000)
    ap.add_argument("--log-dir", type=str,
                    default="src/smerl/runs/smerl_point2d")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-skills", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--eps-frac", type=float, default=0.05)
    ap.add_argument("--R-SAC", type=float, default=-8.72,
                    help="baseline SAC return; eps = eps_frac * |R_SAC|")
    ap.add_argument("--learning-starts", type=int, default=1_000)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--start", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="explicit start position (overrides --env-seed)")
    ap.add_argument("--goal", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="explicit goal position (overrides --env-seed)")
    args = ap.parse_args()
    start = tuple(args.start) if args.start is not None else None
    goal = tuple(args.goal) if args.goal is not None else None

    torch.manual_seed(args.algo_seed)
    np.random.seed(args.algo_seed)

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(args.device)

    env = make_env(args.env_seed, start=start, goal=goal)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"[env]   start={env.start.tolist()}  goal={env.goal.tolist()}  "
          f"d={float(np.linalg.norm(env.goal - env.start)):.3f}")

    cfg = SMERLConfig(
        n_skills=args.n_skills,
        alpha_div=args.alpha,
        eps_frac=args.eps_frac,
        R_SAC=args.R_SAC,
        learning_starts=args.learning_starts,
    )
    eps = cfg.eps_frac * abs(cfg.R_SAC)
    threshold = cfg.R_SAC - eps
    print(f"[smerl] |Z|={cfg.n_skills} alpha={cfg.alpha_div} "
          f"eps={eps:.4f} (R_SAC={cfg.R_SAC}, threshold={threshold:.4f})")

    agent = SMERLAgent(obs_dim, act_dim, cfg, device)

    # --- rollout state ---
    obs, _ = env.reset()
    z = int(np.random.randint(cfg.n_skills))
    episode_buffer: list[tuple] = []
    episode_task_return = 0.0
    episode_steps = 0

    history = []
    n_episodes = 0
    n_gated_episodes = 0
    last_log_t = time.time()

    for t in range(1, args.total_timesteps + 1):
        # action
        if t <= cfg.learning_starts:
            a = env.action_space.sample().astype(np.float32)
        else:
            a = agent.act(obs, z=z, deterministic=False)

        next_obs, r_env, terminated, truncated, info = env.step(a)
        r_tilde = agent.diversity_reward(next_obs, z=z)
        episode_buffer.append(
            (obs.copy(), a.astype(np.float32), float(r_env), float(r_tilde),
             next_obs.copy(), bool(terminated), int(z))
        )
        episode_task_return += float(r_env)
        episode_steps += 1
        obs = next_obs

        if terminated or truncated:
            indicator = 1.0 if episode_task_return >= threshold else 0.0
            for (s, ac, re, rt, ns, d, zz) in episode_buffer:
                r_smerl = re + cfg.alpha_div * indicator * rt
                agent.replay.add(s, ac, r_smerl, ns, d, zz)

            n_episodes += 1
            if indicator > 0:
                n_gated_episodes += 1

            episode_buffer = []
            obs, _ = env.reset()
            z = int(np.random.randint(cfg.n_skills))
            episode_task_return = 0.0
            episode_steps = 0

        # gradient step(s)
        if t > cfg.learning_starts:
            for _ in range(cfg.grad_steps_per_env_step):
                agent.update()

        if t % args.eval_every == 0:
            stats = evaluate_skills(agent, args.env_seed, cfg.n_skills,
                                    n_episodes_per_skill=2,
                                    start=start, goal=goal)
            elapsed = time.time() - last_log_t
            last_log_t = time.time()
            log = {
                "step": t,
                "episodes": n_episodes,
                "gated_frac": (n_gated_episodes / max(n_episodes, 1)),
                "buffer_size": agent.replay.size,
                "elapsed_s": round(elapsed, 1),
                **stats,
            }
            history.append(log)
            top = stats["per_skill"]
            print(f"[t={t:>6d}] eps={n_episodes:4d}  gated={log['gated_frac']:.2f}  "
                  f"mean_R={stats['mean_return_across_skills']:.2f}  "
                  f"best_R(z={stats['best_z']})={stats['best_return']:.2f}  "
                  f"succ={stats['mean_success_across_skills']:.2f}  "
                  f"dt={elapsed:.1f}s")
            for skill in top:
                print(f"           z={skill['z']}  R={skill['mean_return']:7.2f}  "
                      f"succ={skill['success_rate']:.2f}  "
                      f"len={skill['mean_length']:.1f}  "
                      f"end_d={skill['mean_final_dist']:.3f}")

    # ---- save artifacts ----
    final_stats = evaluate_skills(agent, args.env_seed, cfg.n_skills,
                                  n_episodes_per_skill=5,
                                  start=start, goal=goal)
    skill_trajs = collect_skill_trajectories(agent, args.env_seed, cfg.n_skills,
                                             start=start, goal=goal)

    out = {
        "config": {
            "n_skills": cfg.n_skills,
            "alpha_div": cfg.alpha_div,
            "eps_frac": cfg.eps_frac,
            "eps": eps,
            "R_SAC": cfg.R_SAC,
            "threshold": threshold,
            "total_timesteps": args.total_timesteps,
            "env_seed": args.env_seed,
            "algo_seed": args.algo_seed,
            "start": list(env.start.tolist()),
            "goal": list(env.goal.tolist()),
        },
        "history": history,
        "final": final_stats,
        "trajectories": skill_trajs,
    }
    with open(os.path.join(args.log_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved summary to {args.log_dir}/summary.json")
    print(f"[final] mean_R={final_stats['mean_return_across_skills']:.2f}  "
          f"best_z={final_stats['best_z']} (R={final_stats['best_return']:.2f})  "
          f"succ={final_stats['mean_success_across_skills']:.2f}")

    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "disc": agent.disc.state_dict(),
            "config": out["config"],
        },
        os.path.join(args.log_dir, "agent.pt"),
    )


if __name__ == "__main__":
    main()
