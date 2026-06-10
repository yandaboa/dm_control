"""HG-DAgger intervention collection for the synthetic-failure BAMDP.

Generates the on-policy intervention data that teaches the multimodal transformer
to ADAPT — to abandon a skill whose value feedback has stagnated and commit to a
good one. One episode:

  1. Sample the latent theta (per-skill failure rates) from the prior.
  2. Sample the attempted skill z0 from the TRANSFORMER's skill head at s0
     (on-policy). The teacher then drives skill z0 (clean action targets, DART
     noise applied to the env).
  3. The BAMDP injects failure with hazard theta[z0] (certain_skill mode: the
     running skill is known, no discriminator). Failure is absorbing — the
     exposed value stagnates (plateau/decline).
  4. K ~ U{k_min..k_max} steps after the failing flag turns on, the EXPERT
     intervenes: switch to a good skill z_good (sampled from the top-2 success
     rates 1-theta), which clears the failure so the value recovers; the teacher
     drives z_good to the goal.
  5. Keep the episode iff it intervened AND ended in success.

The per-step active skill (z0 ... z0, z_good ... z_good) is stored so the merged
dataset teaches the skill token to switch when the value sequence stalls.

    python -m src.smerl.dagger_collect --model bc_multimodal.pt --n-keep 500
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.trajectory_store import TrajectoryWriter
from src.smerl.collect_trajectories import (make_base_env_fn,
                                            estimate_action_magnitude,
                                            wrap_value_norm)
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.adaptive_transformer import AdaptiveTransformer


def collect_dagger_episode(bamdp, agent, model, z_good, action_noise, device,
                           rng, k_min, k_max, obs_delay=2):
    """One on-policy intervention rollout. Returns (episode dict with a per-step
    ``skills`` array, kept flag). Returns (None, False) when the policy already
    starts on the good skill — there is no genuinely-good skill to switch TO, so
    the attempt cannot produce a switch demo."""
    obs = bamdp.reset()
    s0 = obs["state"]
    at = AdaptiveTransformer(model, device)
    at.update(s0)
    z0 = at.sample_skill(rng=rng)
    if z0 == z_good:
        return None, False
    bamdp.set_active_skill(z0)
    obs = bamdp._observe()

    states = [obs["state"].astype(np.float32)]
    values = [float(obs["value"][0])]
    failing = [bool(obs["failing"][0])]
    actions, rewards, skills = [], [], []
    active = z0
    intervened = False
    intervene_step = fail_step = None
    fail_mode = None
    k_fail = None
    wait = int(rng.integers(k_min, k_max + 1))
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        a = np.asarray(agent.act(obs["state"], z=active, deterministic=True),
                       dtype=np.float32)
        if action_noise > 0:
            a_apply = np.clip(a + rng.uniform(-action_noise, action_noise,
                                              size=a.shape).astype(np.float32),
                              -1.0, 1.0)
        else:
            a_apply = a
        nobs, r, terminated, truncated, info = bamdp.step(a_apply)
        actions.append(a)                        # clean teacher action = target
        skills.append(int(active))               # skill that produced this action
        rewards.append(float(r))
        if fail_step is None and info["failing"]:
            fail_step = len(actions)
            fail_mode = info["fail_mode"]
        switched = False
        if not intervened and info["failing"]:
            k_fail = 0 if k_fail is None else k_fail + 1
            if k_fail >= wait:
                bamdp.intervene(z_good)          # reset_on_switch -> teleports
                active = z_good
                intervened = True
                intervene_step = len(actions)
                switched = True
        # after a reset-on-switch teleport, the next recorded state is the fresh
        # start (so action a, taken under z0, "leads to" the retry start)
        obs = bamdp._observe() if switched else nobs
        states.append(obs["state"].astype(np.float32))
        values.append(float(obs["value"][0]))
        failing.append(bool(obs["failing"][0]))
    success = bool(info.get("is_success", False))
    keep = intervened and success
    # decoupled skill TARGET: flips to z_good obs_delay steps after failure onset
    # (while z0 is still executing, so the value keeps stalling in context)
    skills_arr = np.asarray(skills, dtype=np.int64)
    skill_target = skills_arr.copy()
    if intervened and fail_step is not None:
        sidx = max(0, fail_step - 1 + obs_delay)
        skill_target[sidx:] = int(z_good)
    ep = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "failing": np.asarray(failing, dtype=bool),
        "skills": skills_arr, "skill_target": skill_target,
        "z": int(z0), "theta": bamdp.theta.copy(), "success": success,
        "fail_step": fail_step, "fail_mode": fail_mode,
        "intervened": intervened, "intervene_step": intervene_step,
        "z_good": int(z_good),
    }
    return ep, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_multimodal.pt",
                    help="multimodal transformer to sample the attempted skill from")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--value-norm-skills", type=str, default=None,
                    help="comma list of skills to per-skill renormalize V(s,z)->[0,1]")
    ap.add_argument("--n-keep", type=int, default=500,
                    help="target # of kept (intervened + success) episodes")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="cap on attempts (default 20x n-keep)")
    ap.add_argument("--max-steps", type=int, default=100,
                    help="episode horizon (shared across both attempts under "
                         "reset_on_switch)")
    ap.add_argument("--skills", type=str, default="0,4",
                    help="comma list of the skills in play (one is made bad/"
                         "high-failure per episode, the other is z_good)")
    ap.add_argument("--reset-on-switch", action="store_true",
                    help="env teleports to a fresh start on a skill switch")
    ap.add_argument("--theta-bad", type=float, default=0.95,
                    help="failure rate assigned to the bad skill")
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--obs-delay", type=int, default=2,
                    help="steps after failure onset before the decoupled skill "
                         "TARGET flips to z_good (observation window)")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--beta-a", type=float, default=0.5)
    ap.add_argument("--beta-b", type=float, default=0.5)
    ap.add_argument("--decline-decay", type=float, default=0.95)
    ap.add_argument("--action-noise-frac", type=float, default=0.05,
                    help="DART noise (matches the base store); 0 disables")
    ap.add_argument("--calib-episodes", type=int, default=5)
    ap.add_argument("--out", type=str, default=None,
                    help="store dir (default <run>/trajectories_dagger)")
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
    model = load_bc(os.path.join(run_dir, args.model), device)
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius,
                                   args.max_steps)
    if args.value_norm_skills:
        ns = [int(s) for s in args.value_norm_skills.split(",")]
        value_net, _, _ = wrap_value_norm(value_net, ns, agent, cfg, nominal_start,
                                          args.radius, device, n_skills)

    skills = list(range(n_skills))
    action_noise = 0.0
    if args.action_noise_frac > 0:
        avg_mag = estimate_action_magnitude(base_env_fn, agent, skills,
                                            args.calib_episodes, rng)
        action_noise = args.action_noise_frac * avg_mag
        print(f"[dart] avg|action|={avg_mag:.4f} -> noise U(±{action_noise:.4f})")

    allowed = [int(s) for s in args.skills.split(",")]
    cfg_b = BAMDPConfig(n_skills=n_skills,
                        skill_lengths=np.full(n_skills, float(args.max_steps)),
                        n_sub_episodes=10**9, schedule="budget",
                        beta_a=args.beta_a, beta_b=args.beta_b,
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, plateau_prob=0.5,
                        decline_decay=args.decline_decay, certain_skill=True,
                        reset_on_switch=args.reset_on_switch)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b, rng, device,
                                  value_net=value_net)

    out_dir = args.out or os.path.join(run_dir, "trajectories_dagger")
    meta = {"run": cfg.get("run", args.run), "ckpt_file": args.ckpt_file,
            "obs_dim": obs_dim, "act_dim": act_dim, "n_skills": n_skills,
            "max_steps": args.max_steps, "goal": np.asarray(goal).tolist(),
            "radius": args.radius, "beta": [args.beta_a, args.beta_b],
            "dagger": True, "source_model": args.model,
            "k_window": [args.k_min, args.k_max],
            "action_noise": action_noise, "certain_skill": True,
            "obs_keys": ["state", "value", "failing"]}
    writer = TrajectoryWriter(out_dir, meta)

    from tqdm import tqdm
    max_attempts = args.max_attempts or 20 * args.n_keep
    kept = attempts = 0
    n_no_interv = n_interv_fail = 0
    switch_to = np.zeros(n_skills, dtype=int)
    pbar = tqdm(total=args.n_keep, desc="DAgger keep", unit="ep")
    while kept < args.n_keep and attempts < max_attempts:
        attempts += 1
        # 2-skill latent: one allowed skill is bad (high failure), rest are good
        bad = int(rng.choice(allowed))
        theta = np.full(n_skills, 0.02)
        theta[bad] = args.theta_bad
        bamdp.reset_meta(theta=theta)
        z_good = int(rng.choice([s for s in allowed if s != bad]))  # the other skill
        ep, keep = collect_dagger_episode(bamdp, agent, model, z_good,
                                          action_noise, device, rng,
                                          args.k_min, args.k_max, args.obs_delay)
        if keep:
            writer.add(ep)
            kept += 1
            switch_to[ep["z_good"]] += 1
            pbar.update(1)
        elif ep is None or not ep["intervened"]:   # started on the good skill, or never failed
            n_no_interv += 1
        else:
            n_interv_fail += 1
        pbar.set_postfix_str(f"attempts={attempts} keep-rate={kept/attempts:.2f} "
                             f"no-interv={n_no_interv}")
    pbar.close()
    path = writer.close()

    print(f"[dagger] kept {kept}/{args.n_keep} (of {attempts} attempts) -> {out_dir}")
    print(f"  discarded: no-intervention={n_no_interv}  "
          f"intervened-but-failed={n_interv_fail}")
    print(f"  switched-to-skill histogram: {switch_to.tolist()}")
    print(f"[manifest] {path}")


if __name__ == "__main__":
    main()
