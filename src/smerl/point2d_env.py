"""2D point-mass goal-reaching environment.

Initial state and goal are sampled once at environment construction
(via the optional ``seed`` argument) and stay fixed across ``reset()`` calls.
This is the single-task setting we want as a baseline before layering SMERL on top.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Point2DGoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        world_half_extent: float = 1.0,
        success_radius: float = 0.05,
        max_episode_steps: int = 200,
        dt: float = 0.05,
        max_accel: float = 2.0,
        damping: float = 0.5,
        action_cost: float = 0.01,
        success_bonus: float = 10.0,
        min_start_goal_dist: float = 0.6,
        start: tuple[float, float] | None = None,
        goal: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.world_half_extent = float(world_half_extent)
        self.success_radius = float(success_radius)
        self.max_episode_steps = int(max_episode_steps)
        self.dt = float(dt)
        self.max_accel = float(max_accel)
        self.damping = float(damping)
        self.action_cost = float(action_cost)
        self.success_bonus = float(success_bonus)

        # Observation: [x, y, vx, vy, gx, gy] — goal is included so the
        # network sees the task even though it stays constant for this env.
        high_pos = self.world_half_extent
        high_vel = 5.0  # generous bound on velocity, used only for the obs box
        obs_high = np.array(
            [high_pos, high_pos, high_vel, high_vel, high_pos, high_pos],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        if start is not None and goal is not None:
            # Use the explicitly provided start/goal.
            self._fixed_start = np.asarray(start, dtype=np.float32)
            self._fixed_goal = np.asarray(goal, dtype=np.float32)
            for arr, name in [(self._fixed_start, "start"),
                              (self._fixed_goal, "goal")]:
                if arr.shape != (2,):
                    raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
                if np.any(np.abs(arr) > self.world_half_extent):
                    raise ValueError(
                        f"{name}={arr.tolist()} is outside world bounds "
                        f"[-{self.world_half_extent}, {self.world_half_extent}]"
                    )
        else:
            # Sample start and goal once, deterministically from ``seed``.
            init_rng = np.random.default_rng(seed)
            for _ in range(1000):
                s = init_rng.uniform(
                    -self.world_half_extent, self.world_half_extent, size=2
                )
                g = init_rng.uniform(
                    -self.world_half_extent, self.world_half_extent, size=2
                )
                if np.linalg.norm(g - s) >= min_start_goal_dist:
                    break
            else:
                raise RuntimeError(
                    "Could not sample start/goal pair satisfying min distance."
                )
            self._fixed_start = s.astype(np.float32)
            self._fixed_goal = g.astype(np.float32)

        self._pos = self._fixed_start.copy()
        self._vel = np.zeros(2, dtype=np.float32)
        self._steps = 0

    @property
    def goal(self) -> np.ndarray:
        return self._fixed_goal.copy()

    @property
    def start(self) -> np.ndarray:
        return self._fixed_start.copy()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._pos, self._vel, self._fixed_goal]).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self._fixed_start.copy()
        self._vel = np.zeros(2, dtype=np.float32)
        self._steps = 0
        return self._get_obs(), {"goal": self.goal, "start": self.start}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        accel = action * self.max_accel

        # Semi-implicit Euler with linear damping. Position is clipped to the
        # world box; hitting a wall zeroes the normal velocity component so the
        # point can't accumulate energy against the boundary.
        self._vel = self._vel + (accel - self.damping * self._vel) * self.dt
        self._pos = self._pos + self._vel * self.dt
        for i in range(2):
            if self._pos[i] < -self.world_half_extent:
                self._pos[i] = -self.world_half_extent
                self._vel[i] = 0.0
            elif self._pos[i] > self.world_half_extent:
                self._pos[i] = self.world_half_extent
                self._vel[i] = 0.0

        dist = float(np.linalg.norm(self._pos - self._fixed_goal))
        reward = -dist - self.action_cost * float(np.sum(action**2))

        terminated = dist < self.success_radius
        if terminated:
            reward += self.success_bonus

        self._steps += 1
        truncated = (not terminated) and self._steps >= self.max_episode_steps

        info = {
            "distance_to_goal": dist,
            "is_success": terminated,
            "goal": self.goal,
            "start": self.start,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
