"""Visualize the multimodal transformer adapting in the synthetic-failure BAMDP.

Drives the model in the failure BAMDP (argmax skill each step; an agent-initiated
skill change calls bamdp.switch_skill). Picks a few episodes where a failure was
injected AND the model switched AND it still reached the goal, then draws, per
episode (one column): the xy path colored by the committed skill (so the switch
shows as a colour change) and, below it, the exposed value over time with the
failure-onset and switch instants marked (stagnation -> switch -> recovery).

    python -m src.smerl.visualize_bamdp_switch --model bc_dagger_2layer.pt --n 4
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import make_base_env_fn
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.adaptive_transformer import AdaptiveTransformer


@torch.no_grad()
def rollout_record(model, bamdp, device, force_first=None):
    at = AdaptiveTransformer(model, device)
    obs = bamdp.reset()
    xy, skill, value, failing, teleports, stuck = [], [], [], [], [], []
    active, switches, fail_onset = None, [], None
    terminated = truncated = False
    info = {}
    t = 0
    while not (terminated or truncated):
        s = obs["state"].astype(np.float32)
        at.update(s)
        z = at.sample_skill(force=force_first if active is None else None)
        tele = False
        if active is None:
            active = z; bamdp.set_active_skill(z)
        elif z != active:
            bamdp.switch_skill(z); switches.append(t); active = z
            if bamdp.cfg.reset_on_switch:        # teleport: act from the fresh start
                obs = bamdp._observe()
                s = obs["state"].astype(np.float32)
                at.revise_state(s)
                tele = True
        v = float(bamdp._v_obs)
        a = at.sample_action()
        at.push_value(v)
        xy.append(s[:2]); skill.append(active); value.append(v); teleports.append(tele)
        obs, r, terminated, truncated, info = bamdp.step(a)
        failing.append(bool(info["failing"]))
        stuck.append(bool(info.get("stuck", False)))   # wall envs: contact detection
        if fail_onset is None and info["failing"]:
            fail_onset = t
        t += 1
    return {"xy": np.asarray(xy), "skill": np.asarray(skill),
            "value": np.asarray(value), "failing": np.asarray(failing),
            "switches": switches, "fail_onset": fail_onset,
            "teleports": np.asarray(teleports), "stuck": np.asarray(stuck),
            "success": bool(info.get("is_success", False)),
            "theta": bamdp.theta.copy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_logic_dec.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n", type=int, default=4, help="# episodes to draw")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--beta-a", type=float, default=0.5)
    ap.add_argument("--beta-b", type=float, default=0.5)
    ap.add_argument("--skills", type=str, default="1,2")
    ap.add_argument("--reset-on-switch", action="store_true", default=True)
    ap.add_argument("--theta-bad", type=float, default=0.95)
    ap.add_argument("--value-norm-skills", type=str, default="1,2")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    model = load_bc(os.path.join(run_dir, args.model), device)
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius, args.max_steps)
    if args.value_norm_skills:
        from src.smerl.collect_trajectories import wrap_value_norm
        ns = [int(s) for s in args.value_norm_skills.split(",")]
        value_net, _, _ = wrap_value_norm(value_net, ns, agent, cfg, nominal_start,
                                          args.radius, device, n_skills)
    cfg_b = BAMDPConfig(n_skills=n_skills,
                        skill_lengths=np.full(n_skills, float(args.max_steps)),
                        n_sub_episodes=10**9, schedule="budget",
                        beta_a=args.beta_a, beta_b=args.beta_b,
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, certain_skill=True,
                        reset_on_switch=args.reset_on_switch)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b,
                                  np.random.default_rng(0), device, value_net=value_net)
    allowed = [int(s) for s in args.skills.split(",")]

    # gather switching episodes; record switch-timing stats
    picks, i, sw_t, sw_dfail, sw_val = [], 0, [], [], []
    while i < 600 and len(picks) < args.n * 6:
        bamdp.rng = np.random.default_rng(args.seed + i)
        bad = int(bamdp.rng.choice(allowed))
        theta = np.full(n_skills, 0.02); theta[bad] = args.theta_bad
        bamdp.reset_meta(theta=theta)
        ep = rollout_record(model, bamdp, device)
        i += 1
        if ep["switches"] and ep["fail_onset"] is not None:
            s0 = ep["switches"][0]
            sw_t.append(s0); sw_dfail.append(s0 - ep["fail_onset"])
            sw_val.append(float(ep["value"][s0 - 1]) if s0 > 0 else float(ep["value"][0]))
            if ep["success"]:
                picks.append(ep)
    picks = picks[:args.n]
    print(f"[viz] {len(picks)} drawn; switch timing over {len(sw_t)} switched eps:")
    if sw_t:
        print(f"  switch timestep: mean={np.mean(sw_t):.1f} (std {np.std(sw_t):.1f})  "
              f"steps fail->switch: mean={np.mean(sw_dfail):.1f} (std {np.std(sw_dfail):.1f})  "
              f"value just before switch: mean={np.mean(sw_val):.2f}")
    if not picks:
        return

    cmap = plt.get_cmap("tab10")
    N = len(picks)
    fig, axes = plt.subplots(2, N, figsize=(3.6 * N, 6.4), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 2]})
    for c, ep in enumerate(picks):
        xy, sk, val, tel = ep["xy"], ep["skill"], ep["value"], ep["teleports"]
        sw = ep["switches"][0]
        z0, zf = int(sk[0]), int(sk[-1])
        ax = axes[0, c]
        for t in range(len(xy) - 1):
            if tel[t + 1]:                  # teleport: don't draw the jump segment
                continue
            ax.plot(xy[t:t + 2, 0], xy[t:t + 2, 1], color=cmap(sk[t]), lw=2, zorder=2)
        ax.scatter(*xy[0], marker="*", c="k", s=220, zorder=5)              # start
        ax.scatter(goal[0], goal[1], marker="X", c="k", s=160, zorder=5)
        ax.add_patch(plt.Circle(goal, cfg.get("success_radius", 0.25), color="k",
                                fill=False, ls="--", alpha=0.5))
        ax.scatter(*xy[sw - 1], marker="s", facecolors="none", edgecolors="r",
                   s=120, linewidths=2, zorder=6)      # where it decided to switch
        ax.scatter(*xy[sw], marker="*", c="lime", s=200, edgecolors="k",
                   zorder=6)                            # retry fresh start (post-teleport)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"skill {z0} → {zf}   (switch @ t={sw})", fontsize=11)

        axb = axes[1, c]
        axb.plot(val, color="0.5", lw=1, zorder=1)
        axb.scatter(range(len(val)), val, c=[cmap(z) for z in sk], s=16, zorder=2)
        if ep["fail_onset"] is not None:
            axb.axvline(ep["fail_onset"], color="r", ls=":", lw=1.5, label="failure")
        axb.axvline(sw, color="k", ls="-", lw=1.5, label="switch (teleport)")
        axb.set_ylim(-0.02, 1.02); axb.set_xlabel("timestep")
        if c == 0:
            axb.set_ylabel("exposed value (normalized)")
            axb.legend(fontsize=8, loc="lower right")
        axb.grid(alpha=.3)

    for z in range(n_skills):
        axes[0, 0].plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
    axes[0, 0].legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle(f"Transformer adapting in the failure BAMDP — {args.model}\n"
                 f"path colored by committed skill; circle = switch; value stalls "
                 f"then recovers after switching", fontsize=12)
    plt.tight_layout()
    out = os.path.join(run_dir, "bamdp_switch_episodes.png")
    plt.savefig(out, dpi=130)
    print(f"[viz] saved {out}  ({N} episodes)")


if __name__ == "__main__":
    main()
