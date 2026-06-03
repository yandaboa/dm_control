"""Collect SMERL-skill trajectories through the failure BAMDP and store them.

Drives the (preferred) latent-conditioned policy through the
SyntheticFailureBAMDP with continuing (non-terminating) forced failures, and
writes each rollout to an on-disk trajectory store (``trajectory_store.py``) as
aligned (state, action) sequences with the exposed task value, the failing flag,
and per-episode labels (skill z, latent theta, outcome, failure step/mode).

Each episode draws the driven skill's failure rate p_z ~ Beta(beta_a, beta_b)
(default uniform), so the dataset spans clean successes through plateau/decline
failures — the supervision a downstream transformer needs.

    python -m src.smerl.collect_trajectories --run runs/smerl_lowalpha_ckpt \
        --ckpt-file ckpt_step010000.pt --n-per-skill 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
from src.smerl.point2d_env import Point2DGoalEnv
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.trajectory_store import TrajectoryWriter


def make_base_env_fn(cfg, nominal_start, goal, radius, max_steps):
    sr = cfg.get("success_radius", 0.05)

    def base_env_fn(rng):
        start = sample_in_disk(np.asarray(nominal_start, dtype=np.float32),
                               radius, rng)
        return Point2DGoalEnv(start=(float(start[0]), float(start[1])),
                              goal=(float(goal[0]), float(goal[1])),
                              success_radius=sr, max_episode_steps=max_steps)
    return base_env_fn


def wrap_value_norm(value_net, norm_skills, agent, cfg, nominal_start, radius,
                    device, n_skills, n_roll=30):
    """Per-skill renormalize V(s,z)->[0,1] for ``norm_skills`` (fixed seed so DART
    and DAgger collections share the same normalization). Returns (net, vmin, vmax)."""
    from src.smerl.train_task_value import (compute_skill_value_norm,
                                            NormalizedSkillValueNet)
    rng = np.random.default_rng(12345)
    vmin, vmax = compute_skill_value_norm(agent, cfg, value_net, norm_skills,
                                          nominal_start, radius, n_roll, rng,
                                          device, n_skills)
    net = NormalizedSkillValueNet(value_net, vmin, vmax).to(device).eval()
    print(f"[value-norm] skills={norm_skills}  "
          f"vmin={[round(vmin[z],3) for z in norm_skills]}  "
          f"vmax={[round(vmax[z],3) for z in norm_skills]}")
    return net, vmin, vmax


def collect_episode(bamdp, agent, z, deterministic, action_noise=0.0, rng=None):
    """Drive skill z for one sub-episode; return aligned arrays + labels.

    DART: if ``action_noise`` > 0, the env is stepped with the teacher's clean
    (deterministic) action plus uniform noise U(-action_noise, action_noise) per
    dim, but the RECORDED target is the clean teacher action. The noise perturbs
    the visited-state distribution so the clone sees (and learns to recover from)
    states near where it will drift at test time, while the labels stay clean."""
    obs = bamdp.reset()
    states = [obs["state"].astype(np.float32)]
    values = [float(obs["value"][0])]
    failing = [bool(obs["failing"][0])]
    actions, rewards = [], []
    fail_step, info = None, {}
    terminated = truncated = False
    dart = action_noise > 0
    while not (terminated or truncated):
        a = np.asarray(agent.act(obs["state"], z=z,
                                 deterministic=(deterministic or dart)),
                       dtype=np.float32)
        if dart:
            xi = rng.uniform(-action_noise, action_noise,
                             size=a.shape).astype(np.float32)
            a_apply = np.clip(a + xi, -1.0, 1.0)
        else:
            a_apply = a
        obs, r, terminated, truncated, info = bamdp.step(a_apply)
        actions.append(a)               # record the clean teacher action (target)
        rewards.append(float(r))
        states.append(obs["state"].astype(np.float32))
        values.append(float(obs["value"][0]))
        failing.append(bool(obs["failing"][0]))
        if fail_step is None and info["forced_failure"]:
            fail_step = len(actions)        # index into states/values
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "failing": np.asarray(failing, dtype=bool),
        "z": int(z), "theta": info["theta"].copy(),
        "success": bool(info.get("is_success", False)),
        "fail_step": fail_step, "fail_mode": info.get("fail_mode"),
    }


def estimate_action_magnitude(base_env_fn, agent, skills, n_per_skill, rng):
    """Mean absolute action component of the clean (deterministic) teacher, over a
    few rollouts per skill. Used to scale the DART noise."""
    mags = []
    for z in skills:
        for _ in range(n_per_skill):
            env = base_env_fn(rng)
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                a = np.asarray(agent.act(obs, z=z, deterministic=True),
                               dtype=np.float32)
                mags.append(np.abs(a))
                obs, _, terminated, truncated, _ = env.step(a)
    return float(np.mean(np.concatenate([m.reshape(-1) for m in mags])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--n-per-skill", type=int, default=200)
    ap.add_argument("--skills", type=str, default=None,
                    help="comma list of skills to collect (default: all)")
    ap.add_argument("--p", type=float, default=None,
                    help="fixed per-episode failure rate; default samples Beta")
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--radius", type=float, default=0.1,
                    help="perturbed-start disk radius")
    ap.add_argument("--beta-a", type=float, default=1.0)
    ap.add_argument("--beta-b", type=float, default=1.0)
    ap.add_argument("--schedule", type=str, default="budget")
    ap.add_argument("--decline-decay", type=float, default=0.95)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--action-noise-frac", type=float, default=0.0,
                    help="DART: per-dim uniform action-noise half-width as a "
                         "fraction of the avg |action| (e.g. 0.05 = ±5%%). The "
                         "noise is applied to the env; the clean action is the "
                         "recorded target. 0 disables (clean demos).")
    ap.add_argument("--calib-episodes", type=int, default=5,
                    help="clean rollouts per skill to estimate avg |action|")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt",
                    help="value net file in <run> (skill-conditioned by default)")
    ap.add_argument("--value-norm-skills", type=str, default=None,
                    help="comma list of skills to per-skill renormalize V(s,z)->[0,1]")
    ap.add_argument("--out", type=str, default=None,
                    help="store dir (default: <run>/trajectories)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    run_dir = os.path.join("src/smerl", args.run)

    agent, cfg, nominal_start, obs_dim = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    act_dim = build_env(cfg).action_space.shape[0]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    n_skills = cfg["n_skills"]
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius,
                                   args.max_steps)
    value_norm = None
    if args.value_norm_skills:
        ns = [int(s) for s in args.value_norm_skills.split(",")]
        value_net, vmin, vmax = wrap_value_norm(value_net, ns, agent, cfg,
                                                nominal_start, args.radius, device,
                                                n_skills)
        value_norm = {"skills": ns, "vmin": vmin, "vmax": vmax}

    cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=np.full(n_skills,
                        float(args.max_steps)), n_sub_episodes=10**9,
                        schedule=args.schedule, beta_a=args.beta_a,
                        beta_b=args.beta_b, continue_on_failure=True,
                        expose_value=True, expose_failing=True,
                        plateau_prob=0.5, decline_decay=args.decline_decay)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b, rng, device,
                                  value_net=value_net)

    skills = ([int(s) for s in args.skills.split(",")] if args.skills
              else list(range(n_skills)))

    # DART: scale the action noise to the teacher's typical action magnitude
    action_noise = 0.0
    avg_mag = None
    if args.action_noise_frac > 0:
        avg_mag = estimate_action_magnitude(base_env_fn, agent, skills,
                                            args.calib_episodes, rng)
        action_noise = args.action_noise_frac * avg_mag
        print(f"[dart] avg|action|={avg_mag:.4f}  noise_frac={args.action_noise_frac}"
              f"  -> per-dim noise U(±{action_noise:.4f})")

    out_dir = args.out or os.path.join(run_dir, "trajectories")
    meta = {"run": cfg.get("run", args.run), "ckpt_file": args.ckpt_file,
            "obs_dim": obs_dim, "act_dim": act_dim, "n_skills": n_skills,
            "max_steps": args.max_steps, "goal": np.asarray(goal).tolist(),
            "radius": args.radius, "schedule": args.schedule,
            "beta": [args.beta_a, args.beta_b],
            "decline_decay": args.decline_decay,
            "deterministic": args.deterministic,
            "action_noise_frac": args.action_noise_frac,
            "action_noise": action_noise, "avg_action_mag": avg_mag,
            "value_norm": value_norm,
            "obs_keys": ["state", "value", "failing"]}
    writer = TrajectoryWriter(out_dir, meta)
    from tqdm import tqdm
    n_succ = 0
    n_fail = {"plateau": 0, "decline": 0}
    total = len(skills) * args.n_per_skill
    pbar = tqdm(total=total, desc="DART collect", unit="ep")
    for z in skills:
        for _ in range(args.n_per_skill):
            # per-episode latent: fixed --p, else draw the failure rate from prior
            theta = np.zeros(n_skills)
            theta[z] = (float(args.p) if args.p is not None
                        else float(rng.beta(args.beta_a, args.beta_b)))
            bamdp.active_skill = z          # condition the exposed V(s,z) on z
            bamdp.reset_meta(theta=theta)
            ep = collect_episode(bamdp, agent, z, args.deterministic,
                                 action_noise=action_noise, rng=rng)
            writer.add(ep)
            if ep["success"]:
                n_succ += 1
            elif ep["fail_mode"] in n_fail:
                n_fail[ep["fail_mode"]] += 1
            pbar.update(1)
    pbar.close()
    path = writer.close()

    print(f"[collect] {total} episodes -> {out_dir}")
    print(f"  success={n_succ}  failed: plateau={n_fail['plateau']} "
          f"decline={n_fail['decline']}")
    print(f"[manifest] {path}")


if __name__ == "__main__":
    main()
