"""Unified HG-DAgger demo collection for the synthetic-failure BAMDP.

One script generalizes base-demo and intervention collection across DAgger
iterations. Per meta-episode (one stored trajectory, sharing a single latent
``theta`` across consecutive sub-episodes):

  1. Sample ``theta`` with one play skill forced clearly-bad and one clearly-good,
     so a failure is always reachable.
  2. Sample ``takeover_t ~ U(takeover_range)`` and
     ``supervision_len ~ U(supervision_range)``.
  3. LEARNER phase: the current transformer picks skills on-policy (argmax, or
     sampled at ``--learner-temp``); the SMERL teacher supplies the action for
     the chosen skill; the BAMDP injects failures as usual. Sub-episodes that
     end (goal or truncation) reset and chain under the same theta, so the
     model's context accumulates observed successes/failures.
  4. TAKEOVER gate: only AFTER a failure has occurred, once ``t >= takeover_t``
     and ``>= delay`` steps have passed since the last failure, the expert takes
     over. No failure by ``--max-learner-steps`` -> end as a clean learner demo.
  5. EXPERT phase: an oracle commits to the best skill (argmin theta, re-picking
     if it is itself failing), driving through goal-resets for ``supervision_len``
     steps, then the meta-episode ends.

Supervision uses the original switching tricks: the skill TARGET is DECOUPLED (it
flips to z_good a few decaying-value tokens after failure onset, while z_bad still
executes — see ``--decouple-delay``), trained with switch upweighting + the logic
gate. ``expert_mask`` is still recorded as metadata. Failure injection is unchanged
from collect_trajectories / the BAMDP.

    python -m src.smerl.collect_demos --iteration 0 --model bc_logic_dec.pt \
        --takeover-range 5,15 --supervision-range 20,40 --n-episodes 500
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


def _range(s):
    lo, hi = (int(x) for x in s.split(","))
    return lo, hi


def sample_theta(rng, n_skills, play, beta_a, beta_b, bad_theta, good_thr, bad_skill=None):
    """Force one play skill clearly bad and one clearly good (so a failure is always
    reachable), the rest of the play skills ~Beta; non-play skills are bad. Returns
    (theta, bad_skill, good_skill)."""
    theta = np.full(n_skills, bad_theta, dtype=np.float64)
    theta[play] = rng.beta(beta_a, beta_b, size=len(play))
    bad = int(bad_skill) if bad_skill is not None else int(rng.choice(play))
    good = int(rng.choice([z for z in play if z != bad]))
    theta[bad] = rng.uniform(0.8, 0.95)
    theta[good] = rng.uniform(0.0, good_thr)
    return theta, bad, good


def expert_skill(theta, play, exclude=None):
    """Lowest-failure skill in play, optionally excluding one skill."""
    cand = [z for z in play if z != exclude] or list(play)
    return int(min(cand, key=lambda z: theta[z]))


def collect_meta_episode(model, agent, bamdp, device, rng, play, theta, *,
                         takeover_t, supervision_len, delay, max_learner_steps,
                         action_noise, learner_temp, decouple_delay=3):
    bamdp.reset_meta(theta=theta)
    at = AdaptiveTransformer(model, device) if model is not None else None
    z_good = expert_skill(theta, play)

    obs = bamdp.reset()
    s = obs["state"].astype(np.float32)
    active = int(rng.choice(play)) if model is None else None
    if active is not None:
        bamdp.set_active_skill(active)

    states, values, failing = [], [], []
    actions, skills, expert, rewards = [], [], [], []
    fail_step = fail_mode = intervene_step = None
    last_fail = None
    n_success = 0
    n_success_post = 0           # sub-ep successes AFTER the intervention
    in_expert = False
    expert_steps = 0
    t = 0

    while True:
        if not in_expert:                                  # LEARNER picks the skill
            if at is not None:
                at.update(s)
                z = at.sample_skill(rng=(rng if learner_temp > 0 else None),
                                    temperature=max(learner_temp, 1e-6))
            else:
                z = active
        else:                                              # EXPERT (oracle) picks
            z = expert_skill(theta, play,
                             exclude=active if bamdp._failed else None)

        if active is None:
            active = z; bamdp.set_active_skill(z)
        elif z != active:
            bamdp.switch_skill(z); active = z
            if bamdp.cfg.reset_on_switch:
                # teleported retry: the skill was decided on the old state, but the
                # action is taken (and the state recorded) from the fresh start
                s = bamdp._obs.astype(np.float32)
                if at is not None and not in_expert:
                    at.revise_state(s)
        v = float(bamdp._v_obs)
        if at is not None and not in_expert:
            at.skip_action(); at.push_value(v)

        a = np.asarray(agent.act(s, z=active, deterministic=True), dtype=np.float32)
        a_env = (np.clip(a + rng.uniform(-action_noise, action_noise, a.shape)
                         .astype(np.float32), -1.0, 1.0) if action_noise > 0 else a)

        states.append(s); values.append(v); failing.append(bool(bamdp._failed))
        actions.append(a); skills.append(int(active)); expert.append(in_expert)

        nobs, r, terminated, truncated, info = bamdp.step(a_env)
        rewards.append(float(r))
        if info["forced_failure"]:
            last_fail = t
            if fail_step is None:
                fail_step = t; fail_mode = info["fail_mode"]

        done_collecting = False
        if in_expert:
            expert_steps += 1
            done_collecting = expert_steps >= supervision_len
        elif last_fail is not None and t >= takeover_t and t - last_fail >= delay:
            in_expert = True; intervene_step = t          # only take over after a failure
        elif t >= max_learner_steps:
            done_collecting = True                         # no failure -> clean demo, end
        if done_collecting:
            s = nobs["state"].astype(np.float32)
            break

        if terminated or truncated:
            n_success += int(info["is_success"])
            n_success_post += int(in_expert and bool(info["is_success"]))
            obs = bamdp.reset()
            s = obs["state"].astype(np.float32)
            active = (expert_skill(theta, play) if in_expert
                      else (int(rng.choice(play)) if model is None else None))
            if active is not None:
                bamdp.set_active_skill(active)
        else:
            s = nobs["state"].astype(np.float32)
        t += 1

    states.append(s); values.append(float(bamdp._v_obs)); failing.append(bool(bamdp._failed))
    skills_arr = np.asarray(skills, dtype=np.int64)
    # decoupled skill TARGET: flip to z_good only after `decouple_delay` decaying-value
    # tokens past the first failure onset (value decays from fail_step+1), so the model
    # learns to switch once the stall is clearly visible — while z_bad still executes.
    skill_target = skills_arr.copy()
    if fail_step is not None:
        flip = min(fail_step + 1 + decouple_delay, len(skill_target))
        skill_target[flip:] = z_good
    return {
        "states": np.asarray(states, np.float32),
        "actions": np.asarray(actions, np.float32),
        "rewards": np.asarray(rewards, np.float32),
        "values": np.asarray(values, np.float32),
        "failing": np.asarray(failing, bool),
        "skills": skills_arr, "skill_target": skill_target,
        "expert_mask": np.asarray(expert, bool),
        "z": int(skills_arr[0]), "theta": bamdp.theta.copy(),
        "success": n_success > 0, "n_success": n_success,
        "n_success_post": n_success_post,
        "fail_step": fail_step, "fail_mode": fail_mode,
        "intervened": intervene_step is not None,
        "intervene_step": intervene_step, "z_good": z_good,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--value-norm-skills", type=str, default=None)
    ap.add_argument("--model", type=str, default=None,
                    help="transformer to sample on-policy learner skills from "
                         "(omit for iteration 0: learner commits to a random skill)")
    ap.add_argument("--iteration", type=int, default=0, help="DAgger iteration (recorded)")
    ap.add_argument("--skills", type=str, default="1,2", help="skills in play")
    ap.add_argument("--n-episodes", type=int, default=500,
                    help="episodes to store; with --keep-intervened-only this is the "
                         "target number of KEPT (intervened) episodes")
    ap.add_argument("--keep-intervened-only", action="store_true",
                    help="store ONLY meta-episodes that had a takeover (exclude clean "
                         "no-failure learner rollouts), so the store is 100% switches")
    ap.add_argument("--require-success", action="store_true",
                    help="additionally require >=1 sub-episode success AFTER the "
                         "intervention (the demo must show the correction working)")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="cap on attempts when filtering (default 8x n-episodes)")
    ap.add_argument("--takeover-range", type=_range, default=(5, 15),
                    help="lo,hi for the takeover timestep")
    ap.add_argument("--supervision-range", type=_range, default=(20, 40),
                    help="lo,hi for the number of expert steps after takeover")
    ap.add_argument("--delay-range", type=_range, default=(5, 10),
                    help="lo,hi steps to wait after the last failure before takeover")
    ap.add_argument("--max-learner-steps", type=int, default=80,
                    help="force takeover if the learner phase reaches this length")
    ap.add_argument("--decouple-delay", type=int, default=3,
                    help="decaying-value tokens to show before the decoupled skill "
                         "target flips to z_good (after failure onset)")
    ap.add_argument("--max-steps", type=int, default=100, help="per sub-episode horizon")
    ap.add_argument("--beta-a", type=float, default=0.5)
    ap.add_argument("--beta-b", type=float, default=0.5)
    ap.add_argument("--bad-theta", type=float, default=0.95,
                    help="failure rate for skills NOT in play")
    ap.add_argument("--good-thr", type=float, default=0.15,
                    help="upper bound on the enforced good skill's failure rate")
    ap.add_argument("--decline-decay", type=float, default=0.95)
    ap.add_argument("--reset-on-switch", action="store_true",
                    help="env-side terminate-on-switch: a skill switch teleports to "
                         "a fresh start (give up and retry) instead of continuing "
                         "in place; the meta step clock is kept")
    ap.add_argument("--learner-temp", type=float, default=0.0,
                    help="softmax temperature for on-policy skill sampling (0 = argmax)")
    ap.add_argument("--action-noise-frac", type=float, default=0.05)
    ap.add_argument("--calib-episodes", type=int, default=5)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=None)
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
    n_skills = cfg["n_skills"]
    play = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius, args.max_steps)
    if args.value_norm_skills:
        ns = [int(s) for s in args.value_norm_skills.split(",")]
        value_net, _, _ = wrap_value_norm(value_net, ns, agent, cfg, nominal_start,
                                          args.radius, device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device) if args.model else None

    action_noise = 0.0
    if args.action_noise_frac > 0:
        avg = estimate_action_magnitude(base_env_fn, agent, play, args.calib_episodes, rng)
        action_noise = args.action_noise_frac * avg
        print(f"[dart] avg|action|={avg:.4f} -> noise U(±{action_noise:.4f})")

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

    out_dir = args.out or os.path.join(run_dir, f"trajectories_dagger_iter{args.iteration}")
    meta = {"run": cfg.get("run", args.run), "ckpt_file": args.ckpt_file,
            "obs_dim": obs_dim, "act_dim": act_dim, "n_skills": n_skills,
            "max_steps": args.max_steps, "goal": np.asarray(goal).tolist(),
            "radius": args.radius, "iteration": args.iteration, "play_skills": play,
            "takeover_range": list(args.takeover_range),
            "supervision_range": list(args.supervision_range),
            "delay_range": list(args.delay_range), "source_model": args.model,
            "action_noise": action_noise, "certain_skill": True,
            "reset_on_switch": args.reset_on_switch,
            "require_success": args.require_success,
            "obs_keys": ["state", "value", "failing"]}
    writer = TrajectoryWriter(out_dir, meta)

    from tqdm import tqdm
    lengths, n_succ_eps, sub_succ = [], 0, 0
    kept = attempts = n_discard = n_discard_nosucc = 0
    filtering = args.keep_intervened_only or args.require_success
    max_attempts = args.max_attempts or args.n_episodes * (8 if filtering else 1)
    pbar = tqdm(total=args.n_episodes, desc=f"collect iter{args.iteration}", unit="ep")
    while kept < args.n_episodes and attempts < max_attempts:
        attempts += 1
        theta, _, _ = sample_theta(rng, n_skills, play, args.beta_a, args.beta_b,
                                   args.bad_theta, args.good_thr)
        takeover_t = int(rng.integers(args.takeover_range[0], args.takeover_range[1] + 1))
        sup_len = int(rng.integers(args.supervision_range[0], args.supervision_range[1] + 1))
        delay = int(rng.integers(args.delay_range[0], args.delay_range[1] + 1))
        ep = collect_meta_episode(model, agent, bamdp, device, rng, play, theta,
                                  takeover_t=takeover_t, supervision_len=sup_len,
                                  delay=delay, max_learner_steps=args.max_learner_steps,
                                  action_noise=action_noise, learner_temp=args.learner_temp,
                                  decouple_delay=args.decouple_delay)
        if args.keep_intervened_only and not ep["intervened"]:
            n_discard += 1
            continue
        if args.require_success and ep["n_success_post"] == 0:
            n_discard_nosucc += 1
            continue
        writer.add(ep); kept += 1; pbar.update(1)
        lengths.append(len(ep["actions"]))
        n_succ_eps += int(ep["success"])
        sub_succ += ep["n_success"]
    pbar.close()
    path = writer.close()

    print(f"[collect] kept {kept} meta-episodes of {attempts} attempts "
          f"({n_discard} non-intervened, {n_discard_nosucc} no-post-intervention-success "
          f"discarded) -> {out_dir}")
    print(f"  mean length={np.mean(lengths):.1f}  with-success={n_succ_eps}/{max(kept,1)}"
          f"  total sub-ep successes={sub_succ}")
    print(f"[manifest] {path}")


if __name__ == "__main__":
    main()
