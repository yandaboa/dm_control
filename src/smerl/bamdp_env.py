"""BAMDP wrapper: synthetic expert failures over per-skill success/failure rates.

Implements the construction in BAMDP.md. A *meta-episode* is a sequence of
sub-episodes that share a latent theta = (p_1, ..., p_K), the per-strategy
*failure* rates, sampled once per meta-episode from a Beta prior. The base env
resets normally between sub-episodes; only theta (and the meta bookkeeping)
persists.

Forced failures are injected via a per-strategy hazard. At each step the
discriminator reads the current state and emits weights w^(t) over the K
strategies; we mix *hazards* (not probabilities) and convert to a per-step
failure probability:

    h_i      = per-step hazard for strategy i (fixed or budget schedule)
    h^(t)    = sum_i w_i^(t) h_i^(t)
    f^(t)    = 1 - exp(-h^(t))          # Bernoulli failure draw

Two failure mechanics are supported:

  * Legacy (``continue_on_failure=False``): a forced failure ends the sub-episode
    early with is_success=False (the construction validated in bamdp_calibrate).
  * Continuing (``continue_on_failure=True``): a forced failure is an *absorbing*
    event that does NOT end the episode — the trajectory keeps running. Instead,
    the exposed task value (see below) decouples from the true state value at the
    failure point and either **plateaus** (holds flat) or **declines** (decays
    geometrically), chosen per-event 50/50. The episode then runs to truncation.

Task-value observation (``expose_value=True``). A task-oriented, skill-agnostic
state-value V(s) in [0,1] (``train_task_value.py``) is appended to the
observation. Before any failure it tracks V(s); once a failure is injected the
exposed value follows the plateau/decline rule above, so a doomed run is visible
as a value that stops climbing toward 1. When ``expose_failing=True`` a binary
"failing" flag (1 once the absorbing failure has fired, else 0) is appended too.
``info`` always carries ``value``, ``failing``, ``fail_mode`` and ``base_obs``.

This wrapper is policy-agnostic: it consumes primitive actions and infers the
active strategy from behavior via the discriminator. Drive it with a
skill-conditioned policy (e.g. the SMERL actor); feed that policy the *base*
6-dim observation (``info["base_obs"]``) when value-exposure is on, since the
actor was trained without the appended value/flag dims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smerl.smerl_sac import Discriminator


@dataclass
class BAMDPConfig:
    n_skills: int
    skill_lengths: np.ndarray          # n_i: avg episode length per strategy
    n_sub_episodes: int = 10           # sub-episodes per meta-episode
    beta_a: float = 0.5                # Beta prior over per-strategy failure rate
    beta_b: float = 0.5
    schedule: str = "budget"           # "budget" | "fixed"
    p_max: float = 0.999               # cap so H_i = -ln(1-p_i) stays finite
    r_rem_floor: float = 1.0           # floor on estimated remaining steps
    # --- continuing-failure + value-observation extensions ---
    continue_on_failure: bool = False  # forced failure is absorbing, non-terminating
    expose_value: bool = False         # append task value V(s) to the observation
    expose_failing: bool = False       # append a binary "failing" flag too
    plateau_prob: float = 0.5          # P(plateau) per failure event; else decline
    decline_decay: float = 0.95        # geometric decay/step for the decline mode
    certain_skill: bool = False        # skip the discriminator: hazard mix is a
                                       # one-hot on active_skill (the known running
                                       # skill), i.e. a discriminator certain (p=1)
    reset_on_switch: bool = False      # changing the active skill teleports the env
                                       # to a fresh start (keeping the step clock):
                                       # a switch = give up and retry from scratch


class SyntheticFailureBAMDP:
    """Meta-episode wrapper enforcing per-strategy failure rates on a base env.

    Usage (explicit meta/sub control)::

        bamdp = SyntheticFailureBAMDP(base_env_fn, disc, cfg, rng,
                                      value_net=value_net)
        bamdp.reset_meta()                 # sample theta
        obs = bamdp.reset()                # start a sub-episode
        obs, r, terminated, truncated, info = bamdp.step(action)
        # when terminated/truncated: call reset() to start the next sub-episode;
        # info["meta_done"] is True once n_sub_episodes have been consumed.
    """

    def __init__(self, base_env_fn, discriminator: Discriminator,
                 cfg: BAMDPConfig, rng: np.random.Generator | None = None,
                 device: str | torch.device = "cpu",
                 theta_override: np.ndarray | None = None,
                 value_net: nn.Module | None = None):
        self.base_env_fn = base_env_fn
        self.disc = discriminator
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.K = cfg.n_skills
        self.n_i = np.asarray(cfg.skill_lengths, dtype=np.float64)
        self._theta_override = theta_override
        self.value_net = value_net
        # skill-conditioned nets expose .n_skills and take V(s, z); the active
        # skill is set by the driver per sub-episode (we know which skill we run).
        self._value_skill = value_net is not None and hasattr(value_net, "n_skills")
        self.active_skill = 0
        if value_net is not None:
            value_net.eval()
        if cfg.expose_value and value_net is None:
            raise ValueError("expose_value=True requires a value_net")

        self.theta = None              # per-strategy failure rates p_i
        self.H_i = None                # total hazard budget -ln(1-p_i)
        self.sub_ep = 0
        self.env = None
        self._obs = None               # base observation (env-native)
        self._done = True
        # per-sub-episode hazard-budget state
        self._H_rem = None
        self._r_rem = None
        self._t = 0
        self._T_i = None               # time-integrated discriminator mass
        # per-sub-episode continuing-failure / value state
        self._failed = False           # absorbing failure flag
        self._fail_mode = None         # "plateau" | "decline"
        self._v_obs = 0.0              # currently exposed task value
        self._v_freeze = 0.0          # value captured at the failure instant
        self._k_since_fail = 0
        self._intervened = False       # expert took over -> stop injecting failures

    # ---- discriminator belief ----

    @torch.no_grad()
    def _weights(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)
        return F.softmax(self.disc(x), dim=-1).cpu().numpy()[0]

    # ---- task value ----

    @torch.no_grad()
    def _value(self, obs: np.ndarray) -> float:
        """Exposed value in [0,1]; 0 if no value net is attached. Task-oriented
        V(s), or skill-conditioned V(s, active_skill) for a skill-conditioned net."""
        if self.value_net is None:
            return 0.0
        x = torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)
        if self._value_skill:
            z = torch.tensor([int(self.active_skill)], device=self.device)
            v = self.value_net(x, z)
        else:
            v = self.value_net(x)
        return float(np.asarray(v.cpu()).reshape(-1)[0])

    # ---- observation assembly ----

    def _observe(self):
        """Observation for downstream consumers.

        Legacy (expose_value=False): the bare env array, fed straight to the
        skill policy / dynamics. With expose_value=True we instead return a
        **dict** with separate keys, so a consumer's signature makes explicit
        which mechanism reads which stream:

            "state"   : (obs_dim,) env observation -> skill policy / dynamics
            "value"   : (1,) task value V(s) (corrupted after failure)
            "failing" : (1,) binary absorbing-failure flag  [if expose_failing]

        The skill policy must be driven with obs["state"] (it was trained on the
        bare observation); the value/failing keys are for the BAMDP solver."""
        if not self.cfg.expose_value:
            return self._obs
        out = {"state": self._obs.astype(np.float32),
               "value": np.asarray([self._v_obs], dtype=np.float32)}
        if self.cfg.expose_failing:
            out["failing"] = np.asarray(
                [1.0 if self._failed else 0.0], dtype=np.float32)
        return out

    # ---- meta / sub episode control ----

    def reset_meta(self, theta: np.ndarray | None = None) -> None:
        """Begin a new meta-episode: (re)sample the latent failure rates."""
        if theta is not None:
            p = np.asarray(theta, dtype=np.float64)
        elif self._theta_override is not None:
            p = np.asarray(self._theta_override, dtype=np.float64)
        else:
            p = self.rng.beta(self.cfg.beta_a, self.cfg.beta_b, size=self.K)
        self.theta = np.clip(p, 0.0, self.cfg.p_max)
        self.H_i = -np.log1p(-self.theta)      # -ln(1 - p_i), exact near 0
        self.sub_ep = 0

    def reset(self):
        """Start the next sub-episode (sampling a new meta-episode if needed)."""
        if self.theta is None or self.sub_ep >= self.cfg.n_sub_episodes:
            self.reset_meta()
        self.env = self.base_env_fn(self.rng)
        obs, _ = self.env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        self._done = False
        self._t = 0
        self._H_rem = self.H_i.copy()
        self._r_rem = self.n_i.copy()
        self._T_i = np.zeros(self.K, dtype=np.float64)
        # reset continuing-failure / value state
        self._failed = False
        self._fail_mode = None
        self._k_since_fail = 0
        self._intervened = False
        self._v_obs = self._value(self._obs)
        self._v_freeze = self._v_obs
        return self._observe()

    def set_active_skill(self, z: int) -> None:
        """Set the running skill and refresh the exposed value to V(s, z). Used at
        sub-episode start once the driver has chosen which skill to attempt."""
        self.active_skill = int(z)
        self._v_obs = self._value(self._obs)
        self._v_freeze = self._v_obs

    def _restart_attempt(self) -> None:
        """reset_on_switch: teleport to a fresh start (new draw from base_env_fn)
        and clear the per-attempt failure/value/hazard state, but KEEP the meta
        step clock self._t — so the pre- and post-switch attempts share the one
        episode budget. The new attempt is a from-scratch retry of the new skill."""
        self.env = self.base_env_fn(self.rng)
        obs, _ = self.env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        self._failed = False
        self._fail_mode = None
        self._k_since_fail = 0
        self._H_rem = self.H_i.copy()
        self._r_rem = self.n_i.copy()
        self._v_obs = self._value(self._obs)
        self._v_freeze = self._v_obs

    def switch_skill(self, new_skill: int) -> None:
        """Agent-initiated skill switch at TEST time: re-evaluate failure under
        the new skill (clear the absorbing failure, refresh the exposed value),
        but KEEP injecting failures — the new skill may also be bad, so the agent
        must keep adapting until it finds a good one. Unlike ``intervene`` (the
        training-time expert takeover), this does not disable future failures."""
        if int(new_skill) == int(self.active_skill):
            return
        self.active_skill = int(new_skill)
        if self.cfg.reset_on_switch:
            self._restart_attempt()      # give up and retry the new skill from scratch
        else:
            self._failed = False
            self._fail_mode = None
            self._k_since_fail = 0
            self._v_obs = self._value(self._obs)

    # ---- expert intervention (HG-DAgger) ----

    def intervene(self, new_skill: int) -> None:
        """Expert takeover: switch the active skill, clear the absorbing failure
        so the exposed value recovers under the new skill, and stop injecting
        failures for the rest of the sub-episode. The exposed value is refreshed
        immediately to V(s, new_skill) so the recovery is visible at this step.
        With reset_on_switch the env also teleports to a fresh start (retry from
        scratch with the good skill)."""
        self.active_skill = int(new_skill)
        self._intervened = True
        if self.cfg.reset_on_switch:
            self._restart_attempt()
        else:
            self._failed = False
            self._fail_mode = None
            self._k_since_fail = 0
            self._v_obs = self._value(self._obs)

    # ---- hazard schedule ----

    def _per_strategy_hazard(self, w: np.ndarray) -> np.ndarray:
        if self.cfg.schedule == "fixed":
            return self.H_i / np.maximum(self.n_i, 1.0)
        # budget: spend remaining hazard over estimated remaining steps
        return self._H_rem / np.maximum(self._r_rem, self.cfg.r_rem_floor)

    def _spend_budget(self, w: np.ndarray, h_i: np.ndarray) -> None:
        if self.cfg.schedule != "budget":
            return
        # each strategy consumes budget / remaining-steps in proportion to its
        # current discriminator mass w_i (its share of "presence" this step)
        self._H_rem = np.maximum(self._H_rem - w * h_i, 0.0)
        self._r_rem = self._r_rem - w

    # ---- task-value bookkeeping under continuing failure ----

    def _update_value(self, next_obs: np.ndarray, just_failed: bool) -> None:
        """Update the exposed value. While healthy it tracks V(s); once the
        absorbing failure has fired it freezes and then plateaus or declines."""
        if not self._failed:
            self._v_obs = self._value(next_obs)
            return
        if just_failed:
            # capture the value at the failure instant, pick plateau/decline 50/50
            self._v_freeze = self._value(next_obs)
            self._k_since_fail = 0
            self._fail_mode = ("plateau"
                               if self.rng.random() < self.cfg.plateau_prob
                               else "decline")
            self._v_obs = self._v_freeze
            return
        self._k_since_fail += 1
        if self._fail_mode == "plateau":
            self._v_obs = self._v_freeze
        else:
            self._v_obs = self._v_freeze * (
                self.cfg.decline_decay ** self._k_since_fail)

    # ---- step ----

    def step(self, action):
        if self._done:
            raise RuntimeError("step() after sub-episode end; call reset() first")
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        # strategy belief w over the K skills: either the discriminator's softmax,
        # or (certain_skill) a one-hot on the known active skill -> hazard = h[active]
        if self.cfg.certain_skill:
            w = np.zeros(self.K, dtype=np.float64)
            w[int(self.active_skill)] = 1.0
        else:
            w = self._weights(next_obs)
        h_i = self._per_strategy_hazard(w)
        h_mixed = float(np.dot(w, h_i))
        f = 1.0 - np.exp(-h_mixed)
        self._T_i += w
        self._t += 1

        # only inject while healthy; a fired failure is absorbing; after an
        # expert intervention we stop injecting (the good skill runs to success)
        forced_failure = False
        just_failed = False
        if not self._failed and not self._intervened:
            self._spend_budget(w, h_i)
            forced_failure = bool(self.rng.random() < f)
            if forced_failure:
                self._failed = True
                just_failed = True

        env_success = bool(info.get("is_success", False))
        self._obs = next_obs
        self._update_value(next_obs, just_failed)

        if self.cfg.continue_on_failure:
            # absorbing failure does not end the episode; suppress goal success
            # and any bonus once failed, and run to our own truncation horizon.
            is_success = env_success and not self._failed
            if self._failed and env_success:
                reward -= self.env.success_bonus
            if self._failed:
                sub_terminated = False
                sub_truncated = self._t >= self.env.max_episode_steps
            else:
                sub_terminated = bool(terminated)
                sub_truncated = bool(truncated)
        else:
            # legacy: forced failure terminates the sub-episode early
            is_success = env_success
            if forced_failure:
                terminated = True
                is_success = False
                if env_success:
                    reward -= self.env.success_bonus
            sub_terminated = bool(terminated)
            sub_truncated = bool(truncated) and not forced_failure

        if sub_terminated or sub_truncated:
            self._done = True
            self.sub_ep += 1

        info = {
            **info,
            "is_success": is_success,
            "forced_failure": forced_failure,
            "failing": bool(self._failed),
            "intervened": bool(self._intervened),
            "active_skill": int(self.active_skill),
            "fail_mode": self._fail_mode,
            "value": float(self._v_obs),
            "base_obs": next_obs,
            "w": w,
            "h_mixed": h_mixed,
            "f": float(f),
            "theta": self.theta.copy(),
            "sub_episode": self.sub_ep,
            "sub_episode_step": self._t,
            "meta_done": self.sub_ep >= self.cfg.n_sub_episodes,
            "T_i": self._T_i.copy(),
        }
        return self._observe(), float(reward), sub_terminated, sub_truncated, info
