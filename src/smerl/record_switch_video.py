"""Render an MP4 demo of the transformer adapting in the wall env: it commits to a
skill, gets physically stuck in a freeze-on-contact wall, its exposed value flatlines,
and it switches to the escape skill and reaches the goal.

Left panel: the 2D arena, point mass animated with a fading trail colored by the
committed skill, the grey wall, start (*), and goal (X + radius). Right panel: the
exposed value over time, drawn as it happens, with a marker at the switch.

    python -m src.smerl.record_switch_video --model bc_logic_dec.pt --out switch_demo.mp4
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
from src.smerl.wall_env import WallPoint2DGoalEnv, valid_walls
from src.smerl.visualize_bamdp_switch import rollout_record


@torch.no_grad()
def model_start_skill(model, start, device):
    sid = model.id_of["state"]; D = model.max_dim
    sv = np.zeros(D, np.float32); sv[:len(start)] = start
    tid = torch.tensor([[sid]], device=device)
    h = model.backbone(model.embed(tid, torch.as_tensor(sv[None, None], device=device)),
                       torch.ones_like(tid, dtype=torch.float32))
    return int(model.head_logits("skill", h[0, -1]).argmax())


def pick_episode(model, agent, value_net, cfg, start, goal, sr, n_skills, walls,
                 force_start, max_steps, device, want_switch_after=12):
    """Roll the model on each wall (forced start), return the first clean episode
    that gets stuck, switches mid-path, and reaches the goal — the nicest demo."""
    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(start), goal=tuple(goal), wall=wall,
                                  success_radius=sr, max_episode_steps=max_steps)
    best = None
    for wall in walls:
        cfg_b = BAMDPConfig(n_skills=n_skills,
                            skill_lengths=np.full(n_skills, float(max_steps)),
                            n_sub_episodes=10**9, schedule="budget",
                            continue_on_failure=True, expose_value=True,
                            expose_failing=True, certain_skill=True,
                            reset_on_switch=True)
        bamdp = SyntheticFailureBAMDP(lambda r, w=wall: env_fn(w), agent.disc, cfg_b,
                                      np.random.default_rng(1), device, value_net=value_net)
        bamdp.reset_meta(theta=np.zeros(n_skills))
        ep = rollout_record(model, bamdp, device, force_first=force_start)
        ep["wall"] = wall
        if ep["switches"] and ep["success"]:
            sw = ep["switches"][0]
            if sw >= want_switch_after:          # a long, legible stall before switching
                return ep
            best = best or ep
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_logic_dec.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="1,2")
    ap.add_argument("--force-start", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--n-walls", type=int, default=60)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=int, default=18, help="extra frames held at the end")
    ap.add_argument("--switch-hold", type=int, default=7,
                    help="frames to linger on the switch moment")
    ap.add_argument("--out", type=str, default="switch_demo.mp4")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, start, _ = load_agent(os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    start = np.asarray(start, np.float32)
    n_skills = cfg["n_skills"]; sr = cfg.get("success_radius", 0.25)
    allowed = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    value_net, _, _ = wrap_value_norm(value_net, allowed, agent, cfg, start, 0.0,
                                      device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)

    z_block = args.force_start
    z_spare = [s for s in allowed if s != z_block][0]

    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(start), goal=tuple(goal), wall=wall,
                                  success_radius=sr, max_episode_steps=args.max_steps)
    walls = valid_walls(agent, env_fn, z_block, z_spare,
                        np.random.default_rng(args.seed), cap=args.n_walls)
    ep = pick_episode(model, agent, value_net, cfg, start, goal, sr, n_skills, walls,
                      z_block, args.max_steps, device)
    if ep is None:
        print("[video] no switch+success episode found"); return

    xy = ep["xy"]; sk = ep["skill"].astype(int); val = ep["value"]
    tel = ep["teleports"]; sw = ep["switches"][0]; wx, wy, wr = ep["wall"]
    T = len(xy)
    cmap = plt.get_cmap("tab10")
    c_block, c_spare = cmap(z_block), cmap(z_spare)
    # "stuck" steps: pre-switch frames where the point stopped moving (frozen in wall)
    stuck = np.zeros(T, bool)
    for t in range(1, sw):
        stuck[t] = np.linalg.norm(xy[t] - xy[t - 1]) < 5e-3
    first_stuck = int(np.argmax(stuck)) if stuck.any() else sw
    # frame schedule: linger on the switch, then hold the goal
    sched = []
    for k in range(T):
        sched.append(k)
        if k == sw:
            sched += [k] * args.switch_hold
    sched += [T - 1] * args.hold

    # ---- figure ----
    plt.rcParams.update({"font.size": 12})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 6.2),
                                   gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("white")

    # left: arena
    axL.add_patch(plt.Circle((wx, wy), wr, color="0.35", alpha=0.85, zorder=1))
    axL.text(wx, wy, "WALL", color="white", ha="center", va="center",
             fontsize=9, fontweight="bold", zorder=2)
    axL.scatter(*start, marker="*", c="k", s=320, zorder=3)
    axL.scatter(goal[0], goal[1], marker="X", c="k", s=200, zorder=3)
    axL.add_patch(plt.Circle(goal, sr, color="k", fill=False, ls="--", alpha=0.4))
    axL.text(goal[0], goal[1] + sr + 0.04, "goal", ha="center", fontsize=10)
    axL.set_xlim(-1, 1); axL.set_ylim(-1, 1); axL.set_aspect("equal")
    axL.grid(alpha=.25); axL.set_xticks([]); axL.set_yticks([])
    (trail,) = axL.plot([], [], lw=3, zorder=4)
    stuck_ring = axL.scatter([], [], s=620, facecolors="none", edgecolors="crimson",
                             linewidths=2.5, zorder=5)
    dot = axL.scatter([], [], s=160, zorder=6, edgecolors="k", linewidths=1.5)

    # right: value-over-time
    axR.set_xlim(0, T - 1); axR.set_ylim(-0.02, 1.02)
    axR.set_xlabel("timestep"); axR.set_ylabel("exposed value  V(s, skill)  (normalized)")
    axR.axhline(1.0, color="0.7", ls=":", lw=1); axR.grid(alpha=.25)
    axR.text(0.98, 0.96, "goal value", transform=axR.transAxes, ha="right",
             fontsize=9, color="0.5")
    (vline,) = axR.plot([], [], color="0.5", lw=1.5, zorder=1)
    vscat = axR.scatter([], [], s=22, zorder=2)
    sw_marker = axR.axvline(sw, color="crimson", lw=1.6, ls="--", alpha=0.0)  # revealed at switch
    axR.set_title("value climbs, then flatlines while stuck → switch", fontsize=12)

    # legend
    axL.plot([], [], color=c_block, lw=3, label=f"skill {z_block} (gets stuck)")
    axL.plot([], [], color=c_spare, lw=3, label=f"skill {z_spare} (escape)")
    axL.legend(loc="lower right", fontsize=10, framealpha=0.9)

    title = fig.suptitle("", fontsize=15, fontweight="bold")

    def frame(i):
        k = sched[i]
        on_switch = (k == sw) and (i > 0) and (sched[i - 1] == sw or sched[max(i-1,0)] != sw)
        is_switch_linger = (k == sw)
        # break the trail at the teleport so the jump isn't drawn as a line
        xs, ys = [], []
        for t in range(k + 1):
            if t > 0 and tel[t]:
                axL.plot(xs, ys, color=cmap(sk[t - 1]), lw=3, zorder=4, alpha=0.9)
                xs, ys = [], []
            xs.append(xy[t, 0]); ys.append(xy[t, 1])
        trail.set_data(xs, ys); trail.set_color(cmap(sk[k]))
        dot.set_offsets([xy[k]]); dot.set_color([cmap(sk[k])])

        # red ring while frozen in the wall (pre-switch)
        if stuck[k] and k < sw:
            stuck_ring.set_offsets([xy[k]])
        else:
            stuck_ring.set_offsets(np.empty((0, 2)))

        # reveal the value-panel switch line once we reach the switch
        sw_marker.set_alpha(0.8 if k >= sw else 0.0)

        if k < first_stuck:
            phase = f"committing to skill {z_block}…"
        elif k < sw:
            phase = f"skill {z_block} stuck — value flat, not reaching goal"
        elif is_switch_linger:
            phase = f"value stalled — switching to skill {z_spare}"
        elif k >= T - 1:
            phase = "goal reached"
        else:
            phase = f"skill {z_spare} (fresh retry) → reaching the goal"
        title.set_text(f"Adaptive skill switching (no injected failures)   |   {phase}")

        vline.set_data(range(k + 1), val[:k + 1])
        vscat.set_offsets(np.c_[range(k + 1), val[:k + 1]])
        vscat.set_color([cmap(z) for z in sk[:k + 1]])
        return trail, dot, stuck_ring, vline, vscat, sw_marker, title

    anim = FuncAnimation(fig, frame, frames=len(sched), interval=1000 / args.fps,
                         blit=False)
    out = os.path.join(run_dir, args.out)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400,
                          metadata={"title": "adaptive skill switching"})
    anim.save(out, writer=writer, dpi=130)
    plt.close(fig)
    print(f"[video] switch@{sw}  len={T}  success={ep['success']}  -> {out}")


if __name__ == "__main__":
    main()
