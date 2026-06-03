"""Visualize the BAMDP's exposed task-value observation under continuing failures.

Drives the SMERL skill policy through the SyntheticFailureBAMDP with
``continue_on_failure=True`` and ``expose_value=True`` (dict observation). For
each strategy we inject a per-step failure rate and roll out several episodes:

  top    — one representative *failed* trajectory, points coloured by the EXPOSED
           value (obs["value"]); the failure instant is ringed. After failure the
           exposed value decouples from position and plateaus/declines.
  bottom — exposed value over time for every episode, coloured by outcome
           (success climbs to ~1; failure freezes then plateaus or declines),
           with the failure step marked.

The wrapper uses the policy's own in-checkpoint discriminator as the belief
encoder, so no external discriminator file is needed.

    python -m src.smerl.bamdp_value_obs --run runs/smerl_lowalpha_ckpt \
        --ckpt-file ckpt_step010000.pt
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP


def make_base_env_fn(cfg, nominal_start, goal, radius, max_steps):
    sr = cfg.get("success_radius", 0.05)

    def base_env_fn(rng):
        start = sample_in_disk(np.asarray(nominal_start, dtype=np.float32),
                               radius, rng)
        from src.smerl.point2d_env import Point2DGoalEnv
        return Point2DGoalEnv(start=(float(start[0]), float(start[1])),
                              goal=(float(goal[0]), float(goal[1])),
                              success_radius=sr, max_episode_steps=max_steps)
    return base_env_fn


def run_episode(bamdp, agent, z, deterministic):
    """One sub-episode driving skill z. Returns the exposed-value series, failing
    series, xy path and outcome. Skill policy is fed obs["state"]."""
    obs = bamdp.reset()
    vals = [float(obs["value"][0])]
    failing = [bool(obs["failing"][0])]
    xs, ys = [float(obs["state"][0])], [float(obs["state"][1])]
    fail_step = None
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        a = agent.act(obs["state"], z=z, deterministic=deterministic)
        obs, _, terminated, truncated, info = bamdp.step(a)
        vals.append(float(obs["value"][0]))
        failing.append(bool(obs["failing"][0]))
        xs.append(float(obs["state"][0])); ys.append(float(obs["state"][1]))
        if fail_step is None and info["forced_failure"]:
            fail_step = len(vals) - 1
    return {"vals": np.asarray(vals), "failing": np.asarray(failing),
            "xy": np.column_stack([xs, ys]), "fail_step": fail_step,
            "fail_mode": info.get("fail_mode"),
            "success": bool(info.get("is_success", False)),
            "goal": bamdp.env.goal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--p", type=float, default=0.5,
                    help="injected per-strategy failure rate")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--schedule", type=str, default="budget")
    ap.add_argument("--decline-decay", type=float, default=0.95)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    run_dir = os.path.join("src/smerl", args.run)

    ckpt_path = os.path.join(run_dir, args.ckpt_file)
    agent, cfg, nominal_start, obs_dim = load_agent(ckpt_path, device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    value_net, vmeta = load_value_net(
        os.path.join(run_dir, "value_net_skill.pt"), device)
    n_skills = cfg["n_skills"]
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius,
                                   args.max_steps)

    strategies = list(range(n_skills))
    n_i = np.full(n_skills, float(args.max_steps))   # mass-unit length proxy
    print(f"[strategies] {strategies}  p={args.p}  schedule={args.schedule}  "
          f"max_steps={args.max_steps}")

    cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=n_i,
                        n_sub_episodes=args.n_episodes, schedule=args.schedule,
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, plateau_prob=0.5,
                        decline_decay=args.decline_decay)
    # belief encoder = the policy's own in-checkpoint discriminator
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b, rng, device,
                                  value_net=value_net)

    per_z = {}
    for z in strategies:
        theta = np.zeros(n_skills); theta[z] = args.p
        bamdp.active_skill = z          # expose V(s,z) for the driven skill
        eps = []
        for _ in range(args.n_episodes):
            bamdp.reset_meta(theta=theta)
            eps.append(run_episode(bamdp, agent, z, args.deterministic))
        n_fail = sum(not e["success"] for e in eps)
        npl = sum(e["fail_mode"] == "plateau" for e in eps if not e["success"])
        ndc = sum(e["fail_mode"] == "decline" for e in eps if not e["success"])
        print(f"  z={z}: {n_fail}/{len(eps)} failed ({npl} plateau / {ndc} decline)")
        per_z[z] = eps

    # --- plot ---
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    nz = len(strategies)
    fig, axes = plt.subplots(2, nz, figsize=(3.6 * nz, 7.4), squeeze=False,
                             gridspec_kw={"height_ratios": [3.0, 2.2]})
    colors = {"success": "#2ca02c", "plateau": "#ff7f0e", "decline": "#d62728"}
    for col, z in enumerate(strategies):
        eps = per_z[z]
        axt, axb = axes[0, col], axes[1, col]

        failed = [e for e in eps if not e["success"]]
        rep = failed[0] if failed else eps[0]
        xy, vals = rep["xy"], rep["vals"]
        axt.plot(xy[:, 0], xy[:, 1], color="0.6", lw=0.8, zorder=2)
        axt.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, norm=norm, s=22,
                    edgecolors="k", linewidths=0.3, zorder=3)
        axt.scatter(*xy[0], marker="*", c="k", s=220, zorder=5)
        axt.scatter(*rep["goal"], marker="X", c="k", s=150, zorder=5)
        axt.add_patch(plt.Circle(rep["goal"], cfg.get("success_radius", 0.05),
                                 color="k", fill=False, ls="--", alpha=0.5))
        if rep["fail_step"] is not None:
            axt.scatter(*xy[rep["fail_step"]], s=240, facecolors="none",
                        edgecolors="r", linewidths=2.0, zorder=6,
                        label="failure event")
            axt.legend(fontsize=8, loc="upper left")
        axt.set_title(f"z={z} — "
                      f"{'failed ('+str(rep['fail_mode'])+')' if failed else 'success'}",
                      fontsize=10)
        axt.set_xlim(-1, 1); axt.set_ylim(-1, 1); axt.set_aspect("equal")
        axt.grid(alpha=0.3)

        for e in eps:
            c = colors["success"] if e["success"] else colors.get(
                e["fail_mode"], "#d62728")
            axb.plot(e["vals"], color=c, lw=1.2, alpha=0.7)
            if e["fail_step"] is not None:
                axb.scatter(e["fail_step"], e["vals"][e["fail_step"]], color=c,
                            s=28, zorder=5, edgecolors="k", linewidths=0.4)
        axb.set_xlabel("timestep")
        if col == 0:
            axb.set_ylabel("exposed value  obs[\"value\"]")
        axb.set_ylim(-0.02, 1.05)
        axb.grid(alpha=0.3)
        handles = [plt.Line2D([], [], color=colors[k], lw=2, label=k)
                   for k in ("success", "plateau", "decline")]
        axb.legend(handles=handles, fontsize=7, loc="lower right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.suptitle(f"{vmeta.get('run', args.run)} [{args.ckpt_file}] — BAMDP exposed "
                 f"task-value under continuing failures "
                 f"(p={args.p}, T<={args.max_steps}, decay={args.decline_decay})",
                 fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.91, bottom=0.08,
                        wspace=0.22, hspace=0.26)
    cax = fig.add_axes([0.925, 0.55, 0.011, 0.34])
    fig.colorbar(sm, cax=cax, label="exposed value (top)")

    out = os.path.join(run_dir, "bamdp_value_obs.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
