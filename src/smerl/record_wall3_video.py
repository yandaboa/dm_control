"""Render MP4s of the 3-skill OOD wall environment: two skills' paths are
blocked by tar-pit walls placed on their UNIQUE path segments, and the policy —
with NO injected failures — probes, freezes, watches its value stall, switches,
and escapes via the third skill.

Same trial construction as eval_wall3.py (unique_walls placement, joint
validation), rendered like record_meta_video: left = arena with walls, fading
skill-colored trail (broken at teleport retries); right = exposed value V(s, z)
drawn live, lime markers at switches.

    python -m src.smerl.record_wall3_video --model bc_phase2_ros_r2.pt --n-videos 3
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.wall_env import WallPoint2DGoalEnv, rollout_path, unique_walls
from src.smerl.visualize_bamdp_switch import rollout_record


def stuck_skills(ep, min_run=3, eps=1e-6):
    """Skills that GENUINELY contacted a wall, per the env's own detection
    (info["stuck"], recorded by rollout_record): the skill committed at any
    stuck step. Falls back to the frozen-position heuristic for recordings
    that predate the stuck flag."""
    if "stuck" in ep and len(ep["stuck"]):
        return {int(z) for z, s in zip(ep["skill"], ep["stuck"]) if s}
    xy, sk, tel = ep["xy"], ep["skill"], ep["teleports"]
    out, run = set(), 0
    for t in range(1, len(xy)):
        if tel[t] or sk[t] != sk[t - 1]:
            run = 0
            continue
        if float(np.linalg.norm(xy[t] - xy[t - 1])) < eps:
            run += 1
            if run >= min_run:
                out.add(int(sk[t]))
        else:
            run = 0
    return out


def render_wall3(ep, play, goal, sr, start, out, fps, title):
    cmap = plt.get_cmap("tab10")
    xy, sk, val, tel = ep["xy"], ep["skill"].astype(int), ep["value"], ep["teleports"]
    switches = set(ep["switches"])
    T = len(xy)

    fig, (axA, axV) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle(title, fontsize=11)

    for (wcx, wcy, wrho) in ep["walls"]:
        axA.add_patch(plt.Circle((wcx, wcy), wrho, color="0.3", alpha=0.75, zorder=1))
    axA.scatter(*start, marker="*", c="k", s=220, zorder=5)
    axA.scatter(goal[0], goal[1], marker="X", c="k", s=160, zorder=5)
    axA.add_patch(plt.Circle(goal, sr, color="k", fill=False, ls="--", alpha=0.4))
    axA.set_xlim(-1, 1); axA.set_ylim(-1, 1); axA.set_aspect("equal")
    axA.grid(alpha=0.3)
    for z in play:
        axA.plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
    axA.legend(fontsize=8, loc="lower left")
    status = axA.set_title("", fontsize=10)

    axV.set_xlim(0, T + 1); axV.set_ylim(-0.02, 1.05)
    axV.set_xlabel("timestep"); axV.set_ylabel("value V(s, z)")
    axV.grid(alpha=0.3)

    trail_artists, val_artists = [], []

    def frame(i):
        for a in trail_artists:
            a.remove()
        trail_artists.clear()
        for t in range(i):                          # trail, broken at teleports
            if t + 1 >= T or tel[t + 1]:
                continue
            ln, = axA.plot(xy[t:t + 2, 0], xy[t:t + 2, 1],
                           color=cmap(sk[t]), lw=2.2, zorder=2,
                           alpha=0.35 + 0.65 * (t / max(i, 1)))
            trail_artists.append(ln)
        pt = axA.scatter(*xy[i], c=[cmap(sk[i])], s=90, edgecolors="k", zorder=6)
        trail_artists.append(pt)
        n_sw = sum(1 for s in switches if s <= i)
        status.set_text(f"committed: skill {sk[i]}   switches so far: {n_sw}")
        for a in val_artists:
            a.remove()
        val_artists.clear()
        seg = axV.scatter(np.arange(i + 1), val[:i + 1],
                          c=[cmap(z) for z in sk[:i + 1]], s=10, zorder=2)
        val_artists.append(seg)
        for s in switches:
            if s <= i:
                vl = axV.axvline(s, color="lime", lw=1.4, alpha=0.8, zorder=1)
                val_artists.append(vl)
        return trail_artists + val_artists

    sched = list(range(T)) + [T - 1] * fps          # hold the final frame 1s
    anim = FuncAnimation(fig, lambda k: frame(sched[k]), frames=len(sched),
                         interval=1000 / fps, blit=False)
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=2200))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_phase2_ros_r2.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="0,1,2")
    ap.add_argument("--n-videos", type=int, default=3,
                    help="one per escape skill, round-robin")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--min-switches", type=int, default=2,
                    help="keep episodes that switched at least this many times "
                         "before succeeding (it probed a blocked skill or two)")
    ap.add_argument("--require-both", action="store_true",
                    help="keep only episodes that GENUINELY froze against BOTH "
                         "walls (>=3 held positions on each blocked skill) "
                         "before escaping via the third")
    ap.add_argument("--wall-radius", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.05, 0.06, 0.07, 0.08])
    ap.add_argument("--out-dir", type=str, default="wall3_videos")
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

    out_dir = os.path.join(run_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_made = 0
    attempt = 0
    while n_made < args.n_videos and attempt < 12 * args.n_videos:
        zc = play[attempt % len(play)]              # rotate the escape skill
        attempt += 1
        za, zb = [z for z in play if z != zc]
        wa = unique_walls(agent, env_fn, za, [zb, zc], rng,
                          radii=tuple(args.wall_radius), cap=6)
        wb = unique_walls(agent, env_fn, zb, [za, zc], rng,
                          radii=tuple(args.wall_radius), cap=6)
        rng.shuffle(wa); rng.shuffle(wb)
        pair = None
        for wall_a, wall_b in zip(wa, wb):
            p = [wall_a, wall_b]
            if (not rollout_path(agent, env_fn(p), za)[1]
                    and not rollout_path(agent, env_fn(p), zb)[1]
                    and rollout_path(agent, env_fn(p), zc)[1]):
                pair = p
                break
        if pair is None:
            continue
        cfg_b = BAMDPConfig(n_skills=n_skills,
                            skill_lengths=np.full(n_skills, float(args.max_steps)),
                            n_sub_episodes=10**9, schedule="budget",
                            continue_on_failure=True, expose_value=True,
                            expose_failing=True, certain_skill=True,
                            reset_on_switch=True)
        bamdp = SyntheticFailureBAMDP(lambda r, w=pair: env_fn(w), agent.disc,
                                      cfg_b, np.random.default_rng(1), device,
                                      value_net=value_net)
        bamdp.reset_meta(theta=np.zeros(n_skills))   # NO injected failures
        ep = rollout_record(model, bamdp, device)
        ep["walls"] = pair
        hit = stuck_skills(ep) & {za, zb}
        print(f"[scan] blocked {za},{zb} escape {zc}: walls hit={sorted(hit)} "
              f"switches={len(ep['switches'])} success={ep['success']} "
              f"final={int(ep['skill'][-1])}")
        if not (ep["success"] and len(ep["switches"]) >= args.min_switches
                and int(ep["skill"][-1]) == zc):
            continue
        if args.require_both and hit != {za, zb}:
            continue
        out = os.path.join(out_dir, f"wall3_demo_{n_made}_blocked{za}{zb}"
                                    f"_escape{zc}_hit{''.join(map(str, sorted(hit)))}.mp4")
        print(f"[video] blocked {za},{zb} escape {zc}: hit walls {sorted(hit)}, "
              f"{len(ep['switches'])} switches, success -> {out}")
        render_wall3(ep, play, goal, sr, nominal_start, out, fps=12,
                     title=f"{args.model} — OOD walls (no injected failures): "
                           f"skills {za},{zb} blocked, finds {zc}\n"
                           f"grey = tar-pit walls on each blocked skill's unique "
                           f"path segment; lime = switch")
        n_made += 1
    print(f"[video] wrote {n_made} videos to {out_dir}")


if __name__ == "__main__":
    main()
