"""Render MP4s of BAMDP meta-episodes where the learner fails repeatedly, then
adapts and solves the task.

Rolls the model in the synthetic-failure BAMDP exactly like eval_adapt_multiep
(train-matched theta, terminate-on-switch), keeps meta-episodes that end in a
SUCCESSFUL sub-episode after at least ``--min-fails`` failed sub-episodes, and
renders each as a two-panel animation: the 2D arena (trail colored by the
committed skill, prior sub-episodes ghosted, teleports broken) and the exposed
value V(s, z) trace drawn as it happens (red ticks where the failure flag is on,
sub-episode boundaries dotted). The hidden theta is shown in the title so the
viewer knows which skills are secretly bad.

    python -m src.smerl.record_meta_video --model bc_phase2_ros_r2.pt --n-videos 3
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
from src.smerl.collect_trajectories import make_base_env_fn, wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.collect_demos import sample_theta
from src.smerl.adaptive_transformer import AdaptiveTransformer


@torch.no_grad()
def rollout_meta_record(model, bamdp, device, theta, n_sub, max_steps):
    """Like eval_adapt_multiep.rollout_meta but records xy per step."""
    at = AdaptiveTransformer(model, device)
    max_ctx = model.cfg.n_positions
    bamdp.reset_meta(theta=theta)
    subs = []
    for _ in range(n_sub):
        obs = bamdp.reset()
        active = None
        xy, sk, val, fail, tel = [], [], [], [], []
        terminated = truncated = False
        info = {}
        t = 0
        while not (terminated or truncated) and t < max_steps:
            if len(at._ids) + 3 > max_ctx:
                break
            s = obs["state"].astype(np.float32)
            at.update(s)
            z = at.sample_skill()
            tele = False
            if active is None:
                active = z; bamdp.set_active_skill(z)
            elif z != active:
                bamdp.switch_skill(z); active = z
                if bamdp.cfg.reset_on_switch:
                    obs = bamdp._observe()
                    s = obs["state"].astype(np.float32)
                    at.revise_state(s)
                    tele = True
            v = float(bamdp._v_obs)
            a = at.sample_action()
            at.push_value(v)
            xy.append(s[:2].copy()); sk.append(active); val.append(v)
            fail.append(bool(bamdp._failed)); tel.append(tele)
            obs, r, terminated, truncated, info = bamdp.step(a)
            t += 1
        if not sk:
            break
        subs.append({"xy": np.asarray(xy), "skills": np.asarray(sk),
                     "value": np.asarray(val), "failing": np.asarray(fail),
                     "teleports": np.asarray(tel),
                     "success": bool(info.get("is_success"))})
    return subs


def render_meta(subs, theta, play, goal, sr, start, out, fps, title):
    cmap = plt.get_cmap("tab10")
    # flatten with sub-episode ids
    sub_id = np.concatenate([np.full(len(s["xy"]), i)
                             for i, s in enumerate(subs)])
    xy = np.concatenate([s["xy"] for s in subs])
    sk = np.concatenate([s["skills"] for s in subs])
    val = np.concatenate([s["value"] for s in subs])
    fail = np.concatenate([s["failing"] for s in subs])
    tel = np.concatenate([s["teleports"] for s in subs])
    bounds = np.cumsum([len(s["xy"]) for s in subs])      # exclusive ends
    T = len(xy)

    fig, (axA, axV) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    theta_str = "  ".join(f"z{z}:{theta[z]:.2f}" for z in play)
    fig.suptitle(f"{title}\nhidden failure rates  {theta_str}   "
                 f"(>0.5 = bad skill)", fontsize=11)

    axA.scatter(*start, marker="*", c="k", s=220, zorder=5)
    axA.scatter(goal[0], goal[1], marker="X", c="k", s=160, zorder=5)
    axA.add_patch(plt.Circle(goal, sr, color="k", fill=False, ls="--", alpha=0.4))
    axA.set_xlim(-1, 1); axA.set_ylim(-1, 1)
    axA.set_aspect("equal"); axA.grid(alpha=0.3)
    for z in play:
        axA.plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
    axA.legend(fontsize=8, loc="lower left")
    status = axA.set_title("sub-ep #0", fontsize=10)

    axV.set_xlim(0, T + 1); axV.set_ylim(-0.02, 1.05)
    axV.set_xlabel("meta-episode timestep"); axV.set_ylabel("value V(s, z)")
    axV.grid(alpha=0.3)
    for b in bounds[:-1]:
        axV.axvline(b - 0.5, color="k", ls=":", lw=0.8, alpha=0.6)

    ghost_artists, trail_artists, val_artists = [], [], []

    def frame(i):
        cur = int(sub_id[i])
        # arena: ghost finished sub-eps once, live trail for the current one
        for a in trail_artists:
            a.remove()
        trail_artists.clear()
        while len(ghost_artists) < cur:                     # ghost the finished sub
            g = len(ghost_artists)
            seg = slice(0 if g == 0 else bounds[g - 1], bounds[g])
            ln, = axA.plot(xy[seg, 0], xy[seg, 1], color="0.82", lw=1.4, zorder=1)
            ghost_artists.append(ln)
        s0 = 0 if cur == 0 else bounds[cur - 1]
        for t in range(s0, i):                              # break trail at teleports
            if t + 1 >= T or tel[t + 1] or sub_id[t + 1] != cur:
                continue
            ln, = axA.plot(xy[t:t + 2, 0], xy[t:t + 2, 1],
                           color=cmap(sk[t]), lw=2.2, zorder=2)
            trail_artists.append(ln)
        pt = axA.scatter(*xy[i], c=[cmap(sk[i])], s=90, edgecolors="k", zorder=6)
        trail_artists.append(pt)
        hist = "  ".join(f"#{j}{'✓' if subs[j]['success'] else '✗'}"
                         for j in range(cur))
        status.set_text(f"sub-ep #{cur}  (done: {hist})" if hist
                        else f"sub-ep #{cur}")
        # value panel: draw incrementally
        for a in val_artists:
            a.remove()
        val_artists.clear()
        seg = axV.scatter(np.arange(i + 1), val[:i + 1],
                          c=[cmap(z) for z in sk[:i + 1]], s=10, zorder=2)
        val_artists.append(seg)
        if fail[:i + 1].any():
            fi = np.flatnonzero(fail[:i + 1])
            f = axV.scatter(fi, np.full(len(fi), 0.01), marker="|", c="r", s=40,
                            zorder=3)
            val_artists.append(f)
        return trail_artists + val_artists

    # schedule: every step, then hold the final frame
    sched = list(range(T)) + [T - 1] * fps
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
    ap.add_argument("--n-videos", type=int, default=3)
    ap.add_argument("--n-sub", type=int, default=4)
    ap.add_argument("--min-fails", type=int, default=2,
                    help="keep metas with >= this many FAILED sub-episodes before "
                         "the first success, ending in a success")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-metas", type=int, default=200,
                    help="sampling budget to find qualifying metas")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--theta-bad", type=float, default=0.95)
    ap.add_argument("--good-thr", type=float, default=0.15)
    ap.add_argument("--out-dir", type=str, default="meta_videos")
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
                                      args.radius, device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius,
                                   args.max_steps)
    cfg_b = BAMDPConfig(n_skills=n_skills,
                        skill_lengths=np.full(n_skills, float(args.max_steps)),
                        n_sub_episodes=10**9, schedule="budget",
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, plateau_prob=0.5,
                        certain_skill=True, reset_on_switch=True)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b,
                                  np.random.default_rng(args.seed), device,
                                  value_net=value_net)

    out_dir = os.path.join(run_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_made = 0
    for m in range(args.max_metas):
        theta, bad, good = sample_theta(rng, n_skills, play, 0.5, 0.5,
                                        args.theta_bad, args.good_thr)
        bamdp.rng = np.random.default_rng(args.seed + 1 + m)
        subs = rollout_meta_record(model, bamdp, device, theta, args.n_sub,
                                   args.max_steps)
        succ = [s["success"] for s in subs]
        if not succ or not succ[-1] or True not in succ:
            continue
        first_succ = succ.index(True)
        if first_succ < args.min_fails:                 # needs >= min_fails fails first
            continue
        n_bad = int(sum(theta[z] > 0.5 for z in play))
        out = os.path.join(out_dir, f"meta_demo_{n_made}_fails{first_succ}"
                                    f"_{n_bad}bad.mp4")
        print(f"[video] meta {m}: {''.join('✗' if not s else '✓' for s in succ)}  "
              f"({first_succ} fails first, {n_bad} bad skills) -> {out}")
        render_meta(subs, theta, play, goal, sr, nominal_start, out, args.fps,
                    f"{args.model} — meta-episode (fails {first_succ}x, then solves)")
        n_made += 1
        if n_made >= args.n_videos:
            break
    print(f"[video] wrote {n_made} videos to {out_dir}")


if __name__ == "__main__":
    main()
