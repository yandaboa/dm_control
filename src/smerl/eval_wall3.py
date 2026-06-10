"""3-skill physical-obstacle (OOD) eval: TWO walls, find the third skill.

Per trial, two of the three play skills are blocked — each gets a circular
freeze-on-contact wall on its own path (placed to spare the escape skill's path)
— and the policy must discover the single unobstructed skill. NO failures are
injected (theta = 0): the only failure signal is physical — the state freezes
against a wall and the exposed value V(s, z) stagnates. This is the 3-skill
version of eval_wall.py, and is doubly OOD for a synthetic-failure-trained
policy: real obstacle dynamics AND two simultaneously "bad" skills.

Walls are validated per-skill (blocks its target, geometrically spares the
escape path) and then jointly (with BOTH walls present, both blocked skills
fail to reach the goal and the escape skill still succeeds).

    python -m src.smerl.eval_wall3 --model bc_phase2_ros_r2.pt --n-trials 60
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
from src.smerl.collect_trajectories import wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.wall_env import WallPoint2DGoalEnv, rollout_path, unique_walls
from src.smerl.visualize_bamdp_switch import rollout_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_phase2_ros_r2.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="0,1,2")
    ap.add_argument("--n-trials", type=int, default=60,
                    help="total trials, split evenly over the 3 escape skills")
    ap.add_argument("--n-plot", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=150,
                    help="longer than the 2-skill eval: the policy may need to "
                         "probe BOTH blocked skills before finding the escape")
    ap.add_argument("--wall-radius", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.05, 0.06, 0.07, 0.08])
    ap.add_argument("--out", type=str, default="wall3_adaptation.png")
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
    sr = cfg.get("success_radius", 0.25)
    play = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    value_net, _, _ = wrap_value_norm(value_net, play, agent, cfg, nominal_start,
                                      0.0, device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)

    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(nominal_start), goal=tuple(goal),
                                  wall=wall, success_radius=sr,
                                  max_episode_steps=args.max_steps)

    rng = np.random.default_rng(args.seed)
    per_escape = max(1, args.n_trials // len(play))

    # build (wall_A, wall_B) pairs per escape skill, jointly validated
    trials = []          # (z_escape, [zA, zB], [wall_A, wall_B])
    for zc in play:
        za, zb = [z for z in play if z != zc]
        # walls live on each blocked skill's UNIQUE path segment: clear of BOTH
        # other skills' paths, so each wall visibly belongs to one skill
        wa = unique_walls(agent, env_fn, za, [zb, zc], rng,
                          radii=tuple(args.wall_radius), cap=4 * per_escape)
        wb = unique_walls(agent, env_fn, zb, [za, zc], rng,
                          radii=tuple(args.wall_radius), cap=4 * per_escape)
        rng.shuffle(wa); rng.shuffle(wb)
        kept = 0
        for wall_a, wall_b in zip(wa, wb):
            pair = [wall_a, wall_b]
            ok_a = not rollout_path(agent, env_fn(pair), za)[1]
            ok_b = not rollout_path(agent, env_fn(pair), zb)[1]
            ok_c = rollout_path(agent, env_fn(pair), zc)[1]
            if ok_a and ok_b and ok_c:
                trials.append((zc, [za, zb], pair))
                kept += 1
                if kept >= per_escape:
                    break
        print(f"[wall3] escape={zc}: blocked {za},{zb}  "
              f"candidates {len(wa)}x{len(wb)}  jointly-valid trials {kept}")

    n_t = len(trials)
    n_succ = n_via_escape = 0
    succ_by_escape = {z: [0, 0] for z in play}
    n_switches_succ = []
    picks = []
    for zc, blocked, pair in trials:
        cfg_b = BAMDPConfig(n_skills=n_skills,
                            skill_lengths=np.full(n_skills, float(args.max_steps)),
                            n_sub_episodes=10**9, schedule="budget",
                            continue_on_failure=True, expose_value=True,
                            expose_failing=True, certain_skill=True,
                            reset_on_switch=True)
        bamdp = SyntheticFailureBAMDP(lambda r, w=pair: env_fn(w), agent.disc, cfg_b,
                                      np.random.default_rng(1), device,
                                      value_net=value_net)
        bamdp.reset_meta(theta=np.zeros(n_skills))   # NO injected failures
        ep = rollout_record(model, bamdp, device)
        ep["walls"], ep["escape"], ep["blocked"] = pair, zc, blocked
        succ_by_escape[zc][1] += 1
        n_succ += int(ep["success"])
        succ_by_escape[zc][0] += int(ep["success"])
        if ep["success"]:
            n_via_escape += int(int(ep["skill"][-1]) == zc)
            n_switches_succ.append(len(ep["switches"]))
        if ep["success"] and len(ep["switches"]) >= 1 and len(picks) < args.n_plot:
            picks.append(ep)

    print(f"[wall3] {n_t} trials, 2 walls each (no injected failures):")
    print(f"  reached goal:        {n_succ}/{n_t} = {n_succ/max(n_t,1):.3f}")
    print(f"  ... via escape skill: {n_via_escape}/{max(n_succ,1)}")
    if n_switches_succ:
        print(f"  switches per success: mean {np.mean(n_switches_succ):.2f}  "
              f"(>=1 means it had to adapt)")
    for z in play:
        s, c = succ_by_escape[z]
        print(f"  escape skill {z}: {s}/{c} = {s/max(c,1):.3f}")

    if picks:
        cmap = plt.get_cmap("tab10")
        N = len(picks)
        fig, axes = plt.subplots(2, N, figsize=(3.8 * N, 6.6), squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 2]})
        for c, ep in enumerate(picks):
            xy, sk, val, tel = ep["xy"], ep["skill"], ep["value"], ep["teleports"]
            ax = axes[0, c]
            for (wcx, wcy, wrho) in ep["walls"]:
                ax.add_patch(plt.Circle((wcx, wcy), wrho, color="0.3", alpha=0.7,
                                        zorder=1))
            for t in range(len(xy) - 1):
                if tel[t + 1]:
                    continue
                ax.plot(xy[t:t + 2, 0], xy[t:t + 2, 1], color=cmap(sk[t]), lw=2,
                        zorder=2)
            ax.scatter(*xy[0], marker="*", c="k", s=200, zorder=5)
            ax.scatter(goal[0], goal[1], marker="X", c="k", s=150, zorder=5)
            ax.add_patch(plt.Circle(goal, sr, color="k", fill=False, ls="--",
                                    alpha=0.4))
            for sw in ep["switches"]:
                ax.scatter(*xy[sw], marker="*", c="lime", s=160, edgecolors="k",
                           zorder=6)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
            ax.grid(alpha=.3)
            ax.set_title(f"blocked {ep['blocked'][0]},{ep['blocked'][1]} → "
                         f"escape {ep['escape']} "
                         f"({len(ep['switches'])} switch"
                         f"{'es' if len(ep['switches']) != 1 else ''})", fontsize=10)
            axb = axes[1, c]
            axb.plot(val, color="0.5", lw=1)
            axb.scatter(range(len(val)), val, c=[cmap(z) for z in sk], s=16)
            for sw in ep["switches"]:
                axb.axvline(sw, color="k", lw=1.2, alpha=0.7)
            axb.set_ylim(-0.02, 1.02); axb.set_xlabel("timestep")
            if c == 0:
                axb.set_ylabel("value (normalized)")
            axb.grid(alpha=.3)
        for z in play:
            axes[0, 0].plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
        axes[0, 0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"3-skill WALL adaptation (2 walls, no injected failures) — "
                     f"{args.model}\ngrey = walls; lime ★ = switch; "
                     f"value stalls while stuck", fontsize=11)
        plt.tight_layout()
        out = os.path.join(run_dir, args.out)
        plt.savefig(out, dpi=130)
        print(f"[viz] saved {out}")


if __name__ == "__main__":
    main()
