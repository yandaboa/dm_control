"""Replay the seed-11 double-hit wall3 episode, recording the skill-head softmax
at every decision step. Reproduces record_wall3_video's rng stream exactly
(unique_walls + shuffles) so the qualifying trial comes out identical."""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/storage/lti/dm_control")
os.chdir("/mnt/storage/lti/dm_control")

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.wall_env import WallPoint2DGoalEnv, rollout_path, unique_walls
from src.smerl.adaptive_transformer import AdaptiveTransformer
from src.smerl.record_wall3_video import stuck_skills

DEV = "cuda"
RUN = "runs/smerl_lowalpha_ckpt"
run_dir = os.path.join("src/smerl", RUN)
MAX_STEPS, SEED, PLAY = 150, 11, [0, 1, 2]

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

@torch.no_grad()
def rollout_probs(bamdp):
    at = AdaptiveTransformer(model, DEV)
    obs = bamdp.reset()
    probs, sk, val, stuck, tel_list, switches = [], [], [], [], [], []
    active = None
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        s = obs["state"].astype(np.float32)
        at.update(s)
        logits = model.head_logits("skill", at._last_hidden())
        p = torch.softmax(logits, -1).cpu().numpy()
        z = int(logits.argmax())
        at.sample_skill(force=z)          # identical to argmax sample_skill()
        if active is None:
            active = z; bamdp.set_active_skill(z)
        elif z != active:
            bamdp.switch_skill(z); switches.append(len(sk)); active = z
            if bamdp.cfg.reset_on_switch:
                obs = bamdp._observe()
                at.revise_state(obs["state"].astype(np.float32))
        v = float(bamdp._v_obs)
        a = at.sample_action()
        at.push_value(v)
        probs.append(p); sk.append(active); val.append(v)
        obs, r, terminated, truncated, info = bamdp.step(a)
        stuck.append(bool(info.get("stuck", False)))
    return {"probs": np.asarray(probs), "skill": np.asarray(sk),
            "value": np.asarray(val), "stuck": np.asarray(stuck),
            "switches": switches, "teleports": np.zeros(len(sk), bool),
            "xy": np.zeros((len(sk), 2)),
            "success": bool(info.get("is_success", False))}

# exact rng replication of record_wall3_video main loop (seed 11, n_videos 3,
# min_switches 2, require_both): iterate attempts until the qualifying episode
rng = np.random.default_rng(SEED)
attempt = 0
ep = None
while attempt < 36:
    zc = PLAY[attempt % 3]; attempt += 1
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
    if pair is None:
        continue
    cfg_b = BAMDPConfig(n_skills=n_skills,
                        skill_lengths=np.full(n_skills, float(MAX_STEPS)),
                        n_sub_episodes=10**9, schedule="budget",
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, certain_skill=True,
                        reset_on_switch=True)
    bamdp = SyntheticFailureBAMDP(lambda r, w=pair: env_fn(w), agent.disc, cfg_b,
                                  np.random.default_rng(1), DEV, value_net=vnet)
    bamdp.reset_meta(theta=np.zeros(n_skills))
    e = rollout_probs(bamdp)
    hit = {int(z) for z, s in zip(e["skill"], e["stuck"]) if s}
    print(f"[replay] attempt {attempt-1}: blocked {za},{zb} escape {zc} "
          f"hit={sorted(hit)} switches={len(e['switches'])} succ={e['success']}")
    if attempt - 1 == 23:                      # the recorded video's episode
        ep = e; ep["zc"], ep["za"], ep["zb"] = zc, za, zb
        break
assert ep is not None, "qualifying episode not reproduced"

P, sk, val, stuck = ep["probs"], ep["skill"], ep["value"], ep["stuck"]
T = len(sk)
cmap = plt.get_cmap("tab10")
fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
for z in PLAY:
    ax.plot(np.arange(T), P[:, z], color=cmap(z), lw=2,
            label=f"P(skill {z})" + ("  [escape]" if z == ep["zc"] else "  [blocked]"))
for s in ep["switches"]:
    ax.axvline(s, color="lime", lw=1.2, alpha=0.7)
in_stuck = np.flatnonzero(stuck)
if len(in_stuck):
    ax.fill_between(np.arange(T), 0, 1, where=stuck, color="0.85", zorder=0,
                    label="frozen in wall")
ax.set_ylabel("skill-head softmax"); ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
ax.set_title(f"bc_phase2_ros_r2 — seed-11 video episode (attempt 24): skill-head "
             f"probabilities per decision step (argmax policy)\n"
             f"blocked {ep['za']},{ep['zb']}; escape {ep['zc']}; "
             f"lime = switch, grey = frozen against a wall", fontsize=10)
axv.scatter(np.arange(T), val, c=[cmap(z) for z in sk], s=12)
axv.fill_between(np.arange(T), 0, 1, where=stuck, color="0.85", zorder=0)
axv.set_ylabel("V(s, z)"); axv.set_xlabel("timestep"); axv.set_ylim(-0.02, 1.02)
axv.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(run_dir, "wall3_video_ep_skill_probs.png")
plt.savefig(out, dpi=130)
print(f"[viz] saved {out}")
