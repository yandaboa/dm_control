"""Pure-strategy calibration for the synthetic-failure BAMDP (BAMDP.md, "Validation").

Run each expert in isolation and confirm the realized failure rate matches the
target p_i. This is the operational test that the hazard injection enforces the
intended per-strategy rates. We compare the two schedules ("fixed" vs "budget").

    python -m src.smerl.bamdp_calibrate
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

from src.smerl.point2d_env import Point2DGoalEnv
from src.smerl.skill_decode import load_agent, sample_in_disk
from src.smerl.train_skill_discriminator import load_discriminator
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP


def make_base_env_fn(nominal_start, goal, success_radius, radius):
    def base_env_fn(rng):
        start = sample_in_disk(np.asarray(nominal_start, dtype=np.float32),
                               radius, rng)
        return Point2DGoalEnv(start=(float(start[0]), float(start[1])),
                              goal=(float(goal[0]), float(goal[1])),
                              success_radius=success_radius)
    return base_env_fn


def run_episodes(bamdp, agent, z, n_episodes, deterministic):
    """Drive the BAMDP with the SMERL policy at fixed skill z; return per-episode
    (forced_failure, is_success, length, mean_w_z, T_z) where T_z = sum_t w_z."""
    out = []
    for _ in range(n_episodes):
        obs = bamdp.reset()
        terminated = truncated = False
        info = {}
        w_z_sum, steps = 0.0, 0
        while not (terminated or truncated):
            a = agent.act(obs, z=z, deterministic=deterministic)
            obs, _, terminated, truncated, info = bamdp.step(a)
            w_z_sum += float(info["w"][z]); steps += 1
        out.append((info["forced_failure"], info["is_success"], steps,
                    w_z_sum / max(steps, 1), w_z_sum))
    return out


def measure_constants(agent, disc, base_env_fn, n_skills, device, rng, n=80,
                      deterministic=False):
    """Per-skill calibration constants from injection-free isolation rollouts.

    n_i is defined as the expected time-integrated discriminator self-mass
    E[T_i] = E[sum_t w_i^(t)], NOT raw episode length: this is the unit in which
    the hazard guarantee T_i/n_i = 1 holds for a pure strategy even when the
    discriminator is not perfectly confident (e.g. near the shared start)."""
    zero = BAMDPConfig(n_skills=n_skills, skill_lengths=np.ones(n_skills),
                       n_sub_episodes=n, schedule="fixed")
    bamdp = SyntheticFailureBAMDP(base_env_fn, disc, zero, rng, device,
                                  theta_override=np.zeros(n_skills))
    n_i = np.zeros(n_skills)       # E[T_i] (mass units)
    raw_len = np.zeros(n_skills)   # E[episode length] (steps), for reference
    success = np.zeros(n_skills)
    for z in range(n_skills):
        bamdp.reset_meta(theta=np.zeros(n_skills))
        res = run_episodes(bamdp, agent, z, n, deterministic)
        n_i[z] = float(np.mean([r[4] for r in res]))
        raw_len[z] = float(np.mean([r[2] for r in res]))
        success[z] = float(np.mean([r[1] for r in res]))
    return n_i, raw_len, success


def calibrate(run="runs/smerl_point2d_c", n_episodes=200, device="cpu", seed=0,
              p_grid=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85), success_thresh=0.8,
              deterministic=False):
    device = torch.device(device)
    rng = np.random.default_rng(seed)

    agent, cfg, nominal_start, obs_dim = load_agent(
        os.path.join("src/smerl", run, "agent.pt"), device)
    disc, meta = load_discriminator(
        os.path.join("src/smerl", run, "skill_discriminator.pt"), device)
    n_skills = cfg["n_skills"]
    base_env_fn = make_base_env_fn(nominal_start, meta["goal"],
                                   cfg.get("success_radius", 0.05),
                                   meta["radius"])

    n_i, raw_len, nat_success = measure_constants(
        agent, disc, base_env_fn, n_skills, device, rng,
        deterministic=deterministic)
    strategies = [z for z in range(n_skills) if nat_success[z] >= success_thresh]
    print(f"[constants] n_i=E[T_i]={np.round(n_i,1).tolist()}  "
          f"raw_len={np.round(raw_len,1).tolist()}  "
          f"natural_success={np.round(nat_success,2).tolist()}")
    print(f"[strategies] calibrating z in {strategies} "
          f"(natural success >= {success_thresh})\n")

    results = {"run": meta["run"], "n_i": n_i.tolist(),
               "raw_len": raw_len.tolist(),
               "natural_success": nat_success.tolist(),
               "strategies": strategies, "p_grid": list(p_grid),
               "n_episodes": n_episodes, "schedules": {}}

    for schedule in ("fixed", "budget"):
        cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=n_i,
                            n_sub_episodes=n_episodes, schedule=schedule)
        bamdp = SyntheticFailureBAMDP(base_env_fn, disc, cfg_b, rng, device)
        sched_res = {}
        print(f"=== schedule: {schedule} ===")
        for z in strategies:
            realized, w_means = [], []
            for p in p_grid:
                theta = np.zeros(n_skills)
                theta[z] = p
                bamdp.reset_meta(theta=theta)
                res = run_episodes(bamdp, agent, z, n_episodes, deterministic)
                fail_rate = float(np.mean([r[0] or (not r[1]) for r in res]))
                forced_rate = float(np.mean([r[0] for r in res]))
                realized.append(forced_rate)
                w_means.append(float(np.mean([r[3] for r in res])))
                print(f"  z={z}  target p={p:.2f}  realized_forced={forced_rate:.3f}  "
                      f"total_fail={fail_rate:.3f}  mean_w_z={w_means[-1]:.2f}")
            sched_res[str(z)] = {"realized": realized, "mean_w_z": w_means}
        results["schedules"][schedule] = sched_res
        print()

    return results


def run_episodes_joint(bamdp, agent, z, n_episodes, deterministic):
    """Drive skill z under whatever theta the meta-episode currently holds.
    Returns per-episode (forced_failure, is_success, T_vec) where T_vec is the
    full per-strategy integrated discriminator mass at episode end."""
    out = []
    for _ in range(n_episodes):
        obs = bamdp.reset()
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            a = agent.act(obs, z=z, deterministic=deterministic)
            obs, _, terminated, truncated, info = bamdp.step(a)
        out.append((info["forced_failure"], info["is_success"],
                    info["T_i"].copy()))
    return out


def calibrate_joint(run="runs/smerl_point2d_c", n_draws=60, n_per_draw=50,
                    device="cpu", seed=0, beta_a=1.0, beta_b=1.0,
                    success_thresh=0.8, deterministic=False):
    """Joint-prior calibration: instead of a one-hot theta, sample the FULL
    theta = (p_1,...,p_K) ~ Beta(beta_a, beta_b) per meta-episode (as the agent
    actually experiences it) and check that the realized per-skill forced-failure
    rate still tracks that skill's own p_z. Cross-talk from discriminator-mass
    leakage onto the other (now nonzero-hazard) strategies should push the
    realized rate ABOVE p_z; we report the leakage fraction to attribute it."""
    device = torch.device(device)
    rng = np.random.default_rng(seed)

    agent, cfg, nominal_start, obs_dim = load_agent(
        os.path.join("src/smerl", run, "agent.pt"), device)
    disc, meta = load_discriminator(
        os.path.join("src/smerl", run, "skill_discriminator.pt"), device)
    n_skills = cfg["n_skills"]
    base_env_fn = make_base_env_fn(nominal_start, meta["goal"],
                                   cfg.get("success_radius", 0.05),
                                   meta["radius"])

    n_i, raw_len, nat_success = measure_constants(
        agent, disc, base_env_fn, n_skills, device, rng,
        deterministic=deterministic)
    strategies = [z for z in range(n_skills) if nat_success[z] >= success_thresh]
    print(f"[constants] n_i=E[T_i]={np.round(n_i,1).tolist()}  "
          f"natural_success={np.round(nat_success,2).tolist()}")
    print(f"[strategies] calibrating z in {strategies} "
          f"(natural success >= {success_thresh})")
    print(f"[joint] {n_draws} theta draws ~ Beta({beta_a},{beta_b}), "
          f"{n_per_draw} episodes/draw\n")

    results = {"run": meta["run"], "n_i": n_i.tolist(),
               "natural_success": nat_success.tolist(),
               "strategies": strategies, "n_draws": n_draws,
               "n_per_draw": n_per_draw, "beta_a": beta_a, "beta_b": beta_b,
               "schedules": {}}

    for schedule in ("fixed", "budget"):
        cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=n_i,
                            n_sub_episodes=n_per_draw, schedule=schedule,
                            beta_a=beta_a, beta_b=beta_b)
        bamdp = SyntheticFailureBAMDP(base_env_fn, disc, cfg_b, rng, device)
        sched_res = {str(z): {"target": [], "realized": [], "leak_frac": []}
                     for z in strategies}
        print(f"=== schedule: {schedule} ===")
        excess = {z: [] for z in strategies}
        for d in range(n_draws):
            theta = rng.beta(beta_a, beta_b, size=n_skills)
            for z in strategies:
                bamdp.reset_meta(theta=theta)
                res = run_episodes_joint(bamdp, agent, z, n_per_draw,
                                         deterministic)
                forced = float(np.mean([r[0] for r in res]))
                T = np.array([r[2] for r in res]).mean(0)   # mean per-skill mass
                leak = T.sum() - T[z]
                leak_frac = float(leak / max(T.sum(), 1e-9))
                sched_res[str(z)]["target"].append(float(theta[z]))
                sched_res[str(z)]["realized"].append(forced)
                sched_res[str(z)]["leak_frac"].append(leak_frac)
                excess[z].append(forced - float(theta[z]))
        for z in strategies:
            lf = np.mean(sched_res[str(z)]["leak_frac"])
            print(f"  z={z}  mean(realized - target)={np.mean(excess[z]):+.3f}  "
                  f"mean_leak_frac={lf:.2f}")
        results["schedules"][schedule] = sched_res
        print()

    return results


def plot_joint(results, out_path):
    strategies = results["strategies"]
    schedules = results["schedules"]
    fig, axes = plt.subplots(1, len(schedules),
                             figsize=(6 * len(schedules), 5.2), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for col, (schedule, sched_res) in enumerate(schedules.items()):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal (y=x)")
        for i, z in enumerate(strategies):
            r = sched_res[str(z)]
            t = np.asarray(r["target"]); y = np.asarray(r["realized"])
            ax.scatter(t, y, s=18, alpha=0.6, color=cmap(i),
                       label=f"z={z} (Δ={np.mean(y - t):+.2f}, "
                             f"leak={np.mean(r['leak_frac']):.2f})")
        ax.set_xlabel("target failure rate $p_z$ (from full $\\theta$ draw)")
        ax.set_ylabel("realized forced-failure rate")
        ax.set_title(f"{results['run']} — {schedule} schedule (joint prior)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved {out_path}")


def plot_calibration(results, out_path):
    p_grid = results["p_grid"]
    strategies = results["strategies"]
    fig, axes = plt.subplots(1, len(results["schedules"]),
                             figsize=(6 * len(results["schedules"]), 5.2),
                             squeeze=False)
    for col, (schedule, sched_res) in enumerate(results["schedules"].items()):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal (y=x)")
        for z in strategies:
            r = sched_res[str(z)]
            ax.plot(p_grid, r["realized"], "o-",
                    label=f"z={z} (mean w_z={np.mean(r['mean_w_z']):.2f})")
        ax.set_xlabel("target failure rate $p_i$")
        ax.set_ylabel("realized forced-failure rate")
        ax.set_title(f"{results['run']} — {schedule} schedule")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_point2d_c")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--joint", action="store_true",
                    help="sample full theta ~ Beta and check per-skill rates")
    ap.add_argument("--n-draws", type=int, default=60)
    ap.add_argument("--n-per-draw", type=int, default=50)
    args = ap.parse_args()

    out_dir = os.path.join("src/smerl", args.run)
    if args.joint:
        results = calibrate_joint(run=args.run, n_draws=args.n_draws,
                                  n_per_draw=args.n_per_draw, device=args.device,
                                  seed=args.seed, deterministic=args.deterministic)
        plot_joint(results, os.path.join(out_dir, "bamdp_calibration_joint.png"))
        out_json = os.path.join(out_dir, "bamdp_calibration_joint.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[json] saved {out_json}")
        return

    results = calibrate(run=args.run, n_episodes=args.n_episodes,
                        device=args.device, seed=args.seed,
                        deterministic=args.deterministic)
    plot_calibration(results, os.path.join(out_dir, "bamdp_calibration.png"))
    with open(os.path.join(out_dir, "bamdp_calibration.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] saved {os.path.join(out_dir, 'bamdp_calibration.json')}")


if __name__ == "__main__":
    main()
