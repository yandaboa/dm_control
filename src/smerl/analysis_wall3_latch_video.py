"""Spliced full-inference video + switch-decision probability bar chart for the
seed-11 attempt-23 double-hit wall3 episode continued past its first success
(same transformer context; env resets keep the walls).

Outputs:
  wall3_videos_bothhit/wall3_doublehit_full_inference.mp4
  wall3_doublehit_switch_probs_bar.png
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, "/mnt/storage/lti/dm_control")
os.chdir("/mnt/storage/lti/dm_control")

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.wall_env import WallPoint2DGoalEnv, rollout_path, unique_walls
from src.smerl.adaptive_transformer import AdaptiveTransformer

DEV = "cuda"
run_dir = "src/smerl/runs/smerl_lowalpha_ckpt"
MAX_STEPS, SEED, PLAY, EXTRA = 150, 11, [0, 1, 2], 100

agent, cfg, start, _ = load_agent(os.path.join(run_dir, "ckpt_step010000.pt"), DEV)
cfg["max_episode_steps"] = MAX_STEPS
goal = build_env(cfg).goal
start = np.asarray(start, np.float32)
n_skills = cfg["n_skills"]
sr = cfg.get("success_radius", 0.25)
vnet, _ = load_value_net(os.path.join(run_dir, "value_net_skill.pt"), DEV)
vnet, _, _ = wrap_value_norm(vnet, PLAY, agent, cfg, start, 0.0, DEV, n_skills)
model = load_bc(os.path.join(run_dir, "bc_phase2_ros_r2.pt"), DEV)

def env_fn(wall):
    return WallPoint2DGoalEnv(start=tuple(start), goal=tuple(goal), wall=wall,
                              success_radius=sr, max_episode_steps=MAX_STEPS)

rng = np.random.default_rng(SEED)
pair = None
for attempt in range(24):
    zc = PLAY[attempt % 3]
    za, zb = [z for z in PLAY if z != zc]
    wa = unique_walls(agent, env_fn, za, [zb, zc], rng, cap=6)
    wb = unique_walls(agent, env_fn, zb, [za, zc], rng, cap=6)
    rng.shuffle(wa); rng.shuffle(wb)
    pair = None
    for wall_a, wall_b in zip(wa, wb):
        p2 = [wall_a, wall_b]
        if (not rollout_path(agent, env_fn(p2), za)[1]
                and not rollout_path(agent, env_fn(p2), zb)[1]
                and rollout_path(agent, env_fn(p2), zc)[1]):
            pair = p2
            break
assert pair is not None and zc == 2
print(f"[splice] walls reproduced: blocked {za},{zb}, escape {zc}")

cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=np.full(n_skills, float(MAX_STEPS)),
                    n_sub_episodes=10**9, schedule="budget", continue_on_failure=True,
                    expose_value=True, expose_failing=True, certain_skill=True,
                    reset_on_switch=True)
bamdp = SyntheticFailureBAMDP(lambda r, w=pair: env_fn(w), agent.disc, cfg_b,
                              np.random.default_rng(1), DEV, value_net=vnet)
bamdp.reset_meta(theta=np.zeros(n_skills))

at = AdaptiveTransformer(model, DEV)
xy, sk, val, tel, stuck, probs = [], [], [], [], [], []
events = []                # (t, probs, chosen, prev_skill_or_None, sub_idx)
sub_bounds, sub_succ = [], []
extra_done, sub = 0, 0
with torch.no_grad():
    while True:
        obs = bamdp.reset()
        active = None
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            s = obs["state"].astype(np.float32)
            at.update(s)
            logits = model.head_logits("skill", at._last_hidden())
            p = torch.softmax(logits, -1).cpu().numpy()
            z = int(logits.argmax())
            at.sample_skill(force=z)
            tele = False
            if active is None:
                events.append((len(sk), p[PLAY].copy(), z, None, sub))
                active = z; bamdp.set_active_skill(z)
            elif z != active:
                events.append((len(sk), p[PLAY].copy(), z, active, sub))
                bamdp.switch_skill(z); active = z
                obs = bamdp._observe()
                s = obs["state"].astype(np.float32)
                at.revise_state(s)
                tele = True
            v = float(bamdp._v_obs)
            a = at.sample_action()
            at.push_value(v)
            xy.append(s[:2].copy()); sk.append(active); val.append(v)
            tel.append(tele); probs.append(p)
            obs, r, terminated, truncated, info = bamdp.step(a)
            stuck.append(bool(info.get("stuck", False)))
            if sub > 0:
                extra_done += 1
        sub_succ.append(bool(info.get("is_success", False)))
        sub_bounds.append(len(sk))
        sub += 1
        if extra_done >= EXTRA:
            break

xy = np.asarray(xy); sk = np.asarray(sk); val = np.asarray(val)
tel = np.asarray(tel); stuck = np.asarray(stuck)
T = len(sk)
sub_id = np.zeros(T, int)
for i, b in enumerate(sub_bounds[:-1]):
    sub_id[b:] = i + 1
print(f"[splice] {len(sub_bounds)} sub-eps, {T} steps, "
      f"{len(events)} skill decisions (incl. {len(sub_bounds)} sub-ep starts)")

# ---------- video ----------
cmap = plt.get_cmap("tab10")
fig, (axA, axV) = plt.subplots(1, 2, figsize=(11.5, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.25]})
fig.suptitle(f"bc_phase2_ros_r2 — continuous inference, two-wall env (no "
             f"injected failures)\nwalls block skills {za},{zb}; skill {zc} is "
             f"unobstructed", fontsize=11)
for (wcx, wcy, wrho) in pair:
    axA.add_patch(plt.Circle((wcx, wcy), wrho, color="0.3", alpha=0.75, zorder=1))
axA.scatter(*start, marker="*", c="k", s=220, zorder=5)
axA.scatter(goal[0], goal[1], marker="X", c="k", s=160, zorder=5)
axA.add_patch(plt.Circle(goal, sr, color="k", fill=False, ls="--", alpha=0.4))
axA.set_xlim(-1, 1); axA.set_ylim(-1, 1); axA.set_aspect("equal"); axA.grid(alpha=0.3)
for z in PLAY:
    axA.plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
axA.legend(fontsize=8, loc="lower left")
status = axA.set_title("", fontsize=10)
axV.set_xlim(0, T + 1); axV.set_ylim(-0.02, 1.05)
axV.set_xlabel("timestep"); axV.set_ylabel("value V(s, z)")
axV.grid(alpha=0.3)

ghosts, trail_artists, val_artists = [], [], []
def frame(i):
    cur = int(sub_id[i])
    for a in trail_artists:
        a.remove()
    trail_artists.clear()
    while len(ghosts) < cur:
        g = len(ghosts)
        seg = slice(0 if g == 0 else sub_bounds[g - 1], sub_bounds[g])
        ln, = axA.plot(xy[seg, 0], xy[seg, 1], color="0.85", lw=1.4, zorder=1.5)
        ghosts.append(ln)
    s0 = 0 if cur == 0 else sub_bounds[cur - 1]
    for t in range(s0, i):
        if t + 1 >= T or tel[t + 1] or sub_id[t + 1] != cur:
            continue
        ln, = axA.plot(xy[t:t + 2, 0], xy[t:t + 2, 1], color=cmap(sk[t]),
                       lw=2.2, zorder=2)
        trail_artists.append(ln)
    pt = axA.scatter(*xy[i], c=[cmap(sk[i])], s=90, edgecolors="k", zorder=6)
    trail_artists.append(pt)
    hist = "  ".join(f"#{j}{'✓' if sub_succ[j] else '✗'}" for j in range(cur))
    status.set_text(f"sub-ep #{cur}   committed: skill {sk[i]}"
                    + (f"   ({hist})" if hist else ""))
    for a in val_artists:
        a.remove()
    val_artists.clear()
    seg = axV.scatter(np.arange(i + 1), val[:i + 1],
                      c=[cmap(z) for z in sk[:i + 1]], s=10, zorder=2)
    val_artists.append(seg)
    if stuck[:i + 1].any():
        fi = np.flatnonzero(stuck[:i + 1])
        f = axV.scatter(fi, np.full(len(fi), 0.01), marker="|", c="r", s=40, zorder=3)
        val_artists.append(f)
    for b in sub_bounds[:-1]:                 # reset markers appear as they happen
        if b - 0.5 <= i:
            vl = axV.axvline(b - 0.5, color="k", ls=":", lw=1.0)
            val_artists.append(vl)
    return trail_artists + val_artists

FPS = 14
sched = list(range(T)) + [T - 1] * FPS    # one continuous take, hold last frame
anim = FuncAnimation(fig, lambda k: frame(sched[k]), frames=len(sched),
                     interval=1000 / FPS, blit=False)
out_mp4 = os.path.join(run_dir, "wall3_videos_bothhit",
                       "wall3_doublehit_full_inference.mp4")
anim.save(out_mp4, writer=FFMpegWriter(fps=FPS, bitrate=2200))
plt.close(fig)
print(f"[viz] saved {out_mp4}")

import sys as _sys
_sys.exit(0)   # bar chart already generated

n_ev = len(events)
fig, ax = plt.subplots(figsize=(max(8, 0.65 * n_ev), 4.5))
W = 0.27
for j, (t, p3, chosen, prev, subix) in enumerate(events):
    for k, z in enumerate(PLAY):
        b = ax.bar(j + (k - 1) * W, p3[k], width=W, color=cmap(z),
                   edgecolor="k" if z == chosen else "none",
                   linewidth=1.6 if z == chosen else 0)
    if prev is not None:                      # mark P(repeat the skill just left)
        k_prev = PLAY.index(prev)
        ax.plot(j + (k_prev - 1) * W, p3[k_prev] + 0.03, "rv", ms=6)
labels = [f"t{t}\nsub#{s}" + ("\nstart" if prev is None else f"\n{prev}→{c}")
          for (t, p3, c, prev, s) in events]
ax.set_xticks(range(n_ev)); ax.set_xticklabels(labels, fontsize=7)
for j, (t, p3, chosen, prev, subix) in enumerate(events):
    if prev is None and j > 0:
        ax.axvline(j - 0.5, color="k", ls=":", lw=1.0)
ax.set_ylim(0, 1.05); ax.set_ylabel("skill-head softmax at the decision")
ax.grid(alpha=0.3, axis="y")
ax.set_title(f"per-class probabilities at every skill decision (sub-ep starts + "
             f"switches)\nblack outline = chosen (argmax); red ▼ = prob of the "
             f"skill it just abandoned; walls block {za},{zb}", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(z)) for z in PLAY]
ax.legend(handles, [f"skill {z}" + ("  [escape]" if z == zc else "") for z in PLAY],
          fontsize=8, loc="upper left")
plt.tight_layout()
out_bar = os.path.join(run_dir, "wall3_doublehit_switch_probs_bar.png")
plt.savefig(out_bar, dpi=130)
print(f"[viz] saved {out_bar}")
for (t, p3, chosen, prev, subix) in events:
    print(f"  t={t:3d} sub#{subix} {'start' if prev is None else f'{prev}->{chosen}'}"
          f"  P(0)={p3[0]:.3f} P(1)={p3[1]:.3f} P(2)={p3[2]:.3f}")
