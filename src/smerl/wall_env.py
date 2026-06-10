"""Point2D env with a circular "wall" (freeze-on-contact) obstacle + a helper to
place a wall that sticks one skill's path while sparing another.

The wall is a circular region (cx, cy, radius). The first time the point mass
enters it, the point is frozen there permanently (velocity zeroed, position held),
regardless of the action. A rectangular wall is useless here — the closed-loop
goal-reaching skills simply steer around it — so we make the wall a tar pit that
cannot be escaped: a skill whose path runs through it gets genuinely stuck, its
state stops progressing, and the exposed value V(s,z) stagnates. That is the *real*
(physical) version of the synthetic failure we trained on. No failures are injected.
"""

from __future__ import annotations

import numpy as np

from src.smerl.point2d_env import Point2DGoalEnv


class WallPoint2DGoalEnv(Point2DGoalEnv):
    """Point2DGoalEnv + circular freeze-on-contact wall(s).

    ``wall`` is a single (cx, cy, radius) triple or a list of them — contact with
    ANY wall freezes the point."""

    def __init__(self, *args, wall=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall = (None if wall is None
                     else np.atleast_2d(np.asarray(wall, dtype=np.float32)))
        self._stuck = False

    def reset(self, *, seed=None, options=None):
        self._stuck = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        accel = action * self.max_accel
        if self._stuck:                           # frozen against the wall
            self._vel[:] = 0.0
        else:
            self._vel = self._vel + (accel - self.damping * self._vel) * self.dt
            new_pos = self._pos + self._vel * self.dt
            for i in range(2):                    # world bounds
                if new_pos[i] < -self.world_half_extent:
                    new_pos[i] = -self.world_half_extent; self._vel[i] = 0.0
                elif new_pos[i] > self.world_half_extent:
                    new_pos[i] = self.world_half_extent; self._vel[i] = 0.0
            if self.wall is not None:             # circular wall(s): freeze on contact
                d2 = ((new_pos[0] - self.wall[:, 0]) ** 2
                      + (new_pos[1] - self.wall[:, 1]) ** 2)
                if bool(np.any(d2 <= self.wall[:, 2] ** 2)):
                    self._stuck = True
                    self._vel[:] = 0.0
            self._pos = new_pos

        dist = float(np.linalg.norm(self._pos - self._fixed_goal))
        reward = -dist - self.action_cost * float(np.sum(action ** 2))
        terminated = dist < self.success_radius
        if terminated:
            reward += self.success_bonus
        self._steps += 1
        truncated = (not terminated) and self._steps >= self.max_episode_steps
        info = {"distance_to_goal": dist, "is_success": terminated,
                "goal": self.goal, "start": self.start, "wall": self.wall,
                "stuck": self._stuck}
        return self._get_obs(), float(reward), terminated, truncated, info


def rollout_path(agent, env, z):
    """Deterministic teacher path of skill z in ``env`` -> states [T+1, obs_dim]."""
    obs, _ = env.reset()
    states = [obs.copy()]
    terminated = truncated = False
    ok = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(
            agent.act(obs, z=z, deterministic=True))
        states.append(obs.copy())
        ok = bool(info.get("is_success", False))
    return np.asarray(states), ok


def valid_walls(agent, env_fn, z_block, z_spare, rng, radii=(0.05, 0.06, 0.07, 0.08),
                spare_margin=0.02, cap=80):
    """Circular freeze-on-contact walls on z_block's path that stick z_block (it
    can't reach the goal) but spare z_spare (its closest approach stays
    > radius + margin away). Candidate centers are z_block path points ordered by
    separation from z_spare's path (widest gap first — the only place a wall fits in
    the narrow corridor). env_fn(wall) builds a WallPoint2DGoalEnv(wall=...)."""
    A = rollout_path(agent, env_fn(None), z_block)[0][:, :2]
    B = rollout_path(agent, env_fn(None), z_spare)[0][:, :2]
    n = len(A)
    lo, hi = int(0.15 * n), int(0.85 * n)
    nn = np.argmin(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1), axis=1)
    sep = np.linalg.norm(A - B[nn], axis=1)
    order = [i for i in np.argsort(-sep) if lo <= i < hi]
    cands = []
    for idx in order:
        cx, cy = A[idx]
        dB = float(np.min(np.linalg.norm(B - A[idx], axis=1)))   # clearance to z_spare path
        for rho in radii:
            if rho + spare_margin < dB:           # geometric: wall can't touch z_spare
                cands.append((float(cx), float(cy), float(rho)))
    rng.shuffle(cands)
    valid = []
    for wall in cands:
        blk_goal = rollout_path(agent, env_fn(wall), z_block)[1]
        spr_goal = rollout_path(agent, env_fn(wall), z_spare)[1]
        if (not blk_goal) and spr_goal:
            valid.append(wall)
            if len(valid) >= cap:
                break
    return valid


def unique_walls(agent, env_fn, z_block, z_spares, rng,
                 radii=(0.05, 0.06, 0.07, 0.08), spare_margin=0.02, cap=80,
                 jitter=True):
    """Walls on the part of z_block's path UNIQUE to it: candidate centers are
    z_block path points ordered by clearance to ALL of z_spares' paths (most
    separated first), so the wall sits where only z_block travels — not on the
    shared trunk near the start. Each candidate must geometrically spare every
    path in z_spares; each is then verified to stick z_block while sparing all
    z_spares. ``jitter`` shuffles within the top-separation half so repeated
    calls don't return near-identical centers."""
    A = rollout_path(agent, env_fn(None), z_block)[0][:, :2]
    Bs = [rollout_path(agent, env_fn(None), z)[0][:, :2] for z in z_spares]
    n = len(A)
    lo, hi = int(0.15 * n), int(0.85 * n)
    sep = np.min(np.stack([np.min(np.linalg.norm(A[:, None, :] - B[None, :, :],
                                                 axis=-1), axis=1)
                           for B in Bs]), axis=0)          # clearance to NEAREST other path
    order = [i for i in np.argsort(-sep) if lo <= i < hi]
    if jitter and len(order) > 4:
        top = order[:max(4, len(order) // 2)]
        rng.shuffle(top)
        order = top + order[len(top):]
    cands = []
    for idx in order:
        cx, cy = A[idx]
        for rho in radii:
            if rho + spare_margin < sep[idx]:     # wall can't touch ANY spared path
                cands.append((float(cx), float(cy), float(rho)))
    valid = []
    for wall in cands:
        if rollout_path(agent, env_fn(wall), z_block)[1]:
            continue                              # must stick z_block
        if all(rollout_path(agent, env_fn(wall), z)[1] for z in z_spares):
            valid.append(wall)
            if len(valid) >= cap:
                break
    return valid
