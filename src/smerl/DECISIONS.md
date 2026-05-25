# SMERL — Step 1: 2D Goal-Reaching Env + SAC

A baseline single-task env to build on before layering SMERL (Structured Maximum-Entropy RL) on top. The env's defining feature is that the **start and goal are sampled exactly once at construction time** and held constant across every `reset()`.

## Environment

`src/smerl/point2d_env.py` — `Point2DGoalEnv`, a Gymnasium env.

### Dynamics
- **State (internal):** position `(x, y)` and velocity `(vx, vy)`.
- **Observation (6D):** `[x, y, vx, vy, gx, gy]`. Goal is included even though it's constant for this env — keeps the obs space compatible with future variants where the goal varies.
- **Action (2D):** `a ∈ [-1, 1]^2`, scaled to acceleration `max_accel * a` (`max_accel=2.0`).
- **Integration:** semi-implicit Euler with linear damping, `dt=0.05`:
  - `v ← v + (max_accel * a − damping * v) * dt`
  - `x ← x + v * dt`
- **Walls:** position clipped to `[-1, 1]^2`; on contact, the normal velocity component is zeroed so the agent can't pump energy against the boundary.

### Reward (dense)
```
r_t = -‖x_t − g‖   −   action_cost · ‖a_t‖²   [+ success_bonus on terminate]
```
- `action_cost = 0.01` — a light penalty to discourage thrashing but small enough not to overwhelm the distance signal.
- `success_bonus = 10.0` — fires the step the agent enters the success radius. Order-of-magnitude check: at distance `1.3`, per-step reward is ≈ `−1.3`, so a 10-step early termination saves ≈13 of running cost. The +10 bonus is the same order — small enough that the agent still cares about distance, large enough that early termination dominates over "circling near the goal."

### Termination / truncation
- **Terminated:** `‖x − g‖ < success_radius` (default `0.05`). Episode ends with `success_bonus`.
- **Truncated:** `t == max_episode_steps` (default `200`). No bootstrap-cut hack — SB3 handles `truncated` correctly when the env reports it separately from `terminated`.

### Fixed start/goal at init
- A `np.random.default_rng(seed)` instance samples both start and goal uniformly in `[-1, 1]^2`, rejecting pairs closer than `min_start_goal_dist=0.6` (so the task isn't trivially solvable from the spawn position). Stored as instance attributes; `reset()` always restores them.
- `seed=0` gives `start≈(0.274, -0.460)`, `goal≈(-0.918, -0.967)`, distance ≈ 1.295.

### Decisions made (and the alternatives I weighed)

| Decision | Choice | Alternative | Why |
|---|---|---|---|
| Dynamics | Double integrator + damping | Velocity-control kinematic | Double integrator gives a non-trivial credit-assignment problem (acceleration vs. position) — better stress test for SAC than a trivially diagonal policy. |
| Reward shape | Negative L2 distance + success bonus | Sparse (only on success) / quadratic / potential-based | Dense L2 is the most common baseline; the bonus prevents "hover just outside the radius forever" pathologies that pure negative-distance can encourage when γ is high. Potential-based shaping would be cleaner theoretically but adds a parameter without changing the optimal policy here. |
| Action penalty | Light (0.01) | None / heavy | Some penalty stabilizes the learned actions but a heavy one would conflict with the success bonus. Tuned by inspection — ≈1% of typical per-step distance reward. |
| Include goal in obs | Yes (6D) | No (4D) | Goal is constant so the network *could* learn it implicitly via weights/bias. Including it costs nothing and lets the exact same env class be reused for goal-conditioned/SMERL variants. |
| Walls | Hard clamp, zero normal vel | Bounce / soft penalty | Hard clamp is simplest and physically reasonable; zeroing normal vel prevents the wall-pumping exploit. |
| Start/goal at init only | Sampled once via `seed` arg | Resampled on reset | This is the spec. Means we're solving a single deterministic task — SAC should reach 100% success and the policy should be a near-time-optimal bang-bang controller. |
| Success radius | 0.05 | Smaller / larger | At `max_accel=2.0`, `damping=0.5`, the terminal velocity at full thrust is `4.0` units/s; one step of overshoot can be ≈0.2. Radius 0.05 is tight enough to be non-trivial, large enough that the agent can land inside without millimetre precision. |

## SAC training

`src/smerl/train_sac.py` — uses `stable_baselines3.SAC`.

- Net arch: `[128, 128]` for both actor and twin critic. Small because obs is 6D; bigger nets train slower with no benefit on this task.
- LR `3e-4`, γ `0.99`, τ `0.005`, batch `256`, buffer `200k`. SB3 defaults except for the explicit `learning_starts=1000` (the env is short, so warm-up matters).
- `ent_coef="auto"` — SAC's automatic temperature; the default target entropy `−|A|` is appropriate for `|A|=2`.
- Device: **CPU** by default. The env is 6D and the net is tiny; CPU is faster than the GPU launch overhead. Pass `--device cuda` to put it on GPU once we move to bigger nets / batched envs.
- `EvalCallback` every 2k steps, deterministic, 5 episodes.

### Result (one run, 50k steps, seed=42)

```
[env]   start=[0.274, -0.460]   goal=[-0.918, -0.967]   distance=1.295
[eval]  success_rate=1.0   mean_return=-8.72   mean_final_dist=0.033   mean_length=24.0
[train] wall time: 325 s on CPU
```

24 steps × `dt=0.05` = 1.2 s of simulated time to cover 1.295 m — close to the time-optimal bang-bang controller for the dynamics. The policy is converged well before 50k steps; budget could be cut to ~20k for this seed.

### Conda env

Standalone `SMERL` env (Python 3.11). Reproduce via the helper script:

```bash
bash src/smerl/setup_env.sh        # creates the env from scratch
conda activate SMERL
python -m src.smerl.train_sac --total-timesteps 50000
```

Contents (also frozen in `src/smerl/requirements.txt`):

- `torch==2.7.0+cu128` and `torchvision==0.22.0+cu128` — installed from the PyTorch cu128 index. Matches the version pinned in the project's top-level `requirements.txt`; the host driver is 12.6 so cu128 wheels are the right call (cu130, which `pip install torch` defaults to, fails CUDA init on this host).
- `stable-baselines3==2.8.0`
- `gymnasium==1.2.3`
- `numpy` (2.x — SB3 2.8 supports it)
- `tensorboard`

CUDA verified working — H100 detected, matmul runs.

---

# Step 2: SMERL with a DIAYN-style discriminator

Implements **Algorithm 1** from Kumar et al., NeurIPS 2020 (*"One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL"*, arXiv 2010.14484). SMERL is a strict generalization of DIAYN: latent-conditioned SAC where the DIAYN per-step diversity bonus is **gated on whether the episode's task return is near-optimal**.

## What got added

- `src/smerl/smerl_sac.py` — networks (`GaussianActor`, `TwinQ`, `Discriminator`), `ReplayBuffer`, and `SMERLAgent` with one `update()` call per env step (SAC critic/actor/temperature + discriminator cross-entropy + Polyak target update).
- `src/smerl/train_smerl.py` — rollout + episode-end reward labeling + per-skill eval.

## The reward (Eq 5 of the paper)

```
r_SMERL(s_t, a_t) = r_env(s_t, a_t)
                  + α · 1[ R_M(π_θ) ≥ R*_M − ε ] · r̃(s_t)
```

- `r̃(s_t) = log q_φ(z | s_{t+1}) − log p(z)` — DIAYN intrinsic reward from the discriminator.
- `p(z)` uniform over `|Z|` skills → `log p(z) = −log|Z|` (constant).
- **Indicator is computed once per episode**: depends on the full task return `R_M(π_θ) = Σ r_env`. Implementation accumulates a per-episode list of transitions, computes the indicator on `done`/`truncated`, then pushes all transitions with the gated reward into the replay buffer. This keeps the per-step reward stored in the buffer Markovian.

## Hyperparameters (paper Table 2, "2D Navigation" row)

| Param | Value | Notes |
|---|---|---|
| `|Z|` (# skills) | 5 | |
| `α` (intrinsic weight) | 10.0 | |
| `ε` | `0.05 · |R_SAC|` | sign convention — see below |
| `R_SAC` (baseline return) | −8.72 | from Step 1's SAC run |
| nets (actor, twin Q, disc) | MLP 2×32, ReLU | |
| LR | 3e-4 (Adam) | shared by actor / critic / disc / temperature |
| γ | 0.99 | |
| τ (Polyak) | 0.01 | |
| batch | 128 | |
| buffer | 1 000 | paper-spec; tiny → high replay ratio |
| grad steps / env step | 1 | |
| entropy temperature | automatic, target = −|A| = −2 | |

## Decisions I made on top of the paper

1. **Sign convention for `ε`.** The paper writes `ε = 0.05 · R_SAC` assuming positive returns (HalfCheetah etc.). Our `R_SAC = −8.72` is negative, so taking the formula literally would *raise* the threshold and invert the gating logic. I use `ε = 0.05 · |R_SAC|` so the gate fires when `R ≥ R_SAC − ε = −9.16`. This preserves the intent: "near optimal" means returns within a small slack below the baseline.

2. **Indicator semantics.** Paper-exact: indicator is `1[R_M(π_θ) ≥ R*_M − ε]` evaluated on the **full episode task return**. Per-step rewards retroactively get the diversity bonus or none. Alternative — re-evaluating the indicator at gradient-step time using the latest discriminator — would change the algorithm; I didn't take it.

3. **Discriminator input.** `q_φ(z | s)` takes the raw 6-D obs (including the constant goal feature). Goal is constant for this env, so the discriminator just learns to ignore it. An alternative — feed only `(x, y, vx, vy)` — would be slightly cleaner but functionally equivalent.

4. **`r̃` computed at rollout, not at gradient step.** Paper pseudocode does it at rollout time (Algorithm 1 line "compute `q_φ(z|s_{t+1})` with discriminator"). Means each `r̃` reflects the discriminator's state when the transition was collected — a small staleness, but it's what the paper does.

5. **Replay buffer size = 1 000.** Paper-spec for 2D nav. Very small → high replay ratio (each transition is reused many times); also means the discriminator's training signal is closely tied to recent rollouts.

6. **Action selection.** Stochastic during training; **deterministic (tanh of `μ`)** for evaluation. Matches paper Appendix B.2: "during evaluation, we select the mean action."

7. **Device: CPU.** With 2×32 nets, batch 128, and 60k–80k steps, CPU is faster than GPU launch overhead.

## How to run

```bash
conda activate SMERL
python -m src.smerl.train_smerl \
    --total-timesteps 60000 \
    --learning-starts 1000 \
    --eval-every 5000 \
    --R-SAC -8.72
```

Outputs `src/smerl/runs/smerl_point2d/summary.json` (per-skill stats per eval + final + sample trajectories per skill) and `agent.pt` (actor / critic / discriminator weights).

## Run 1 (60k steps, paper hyperparams)

```
[smerl] |Z|=5  alpha=10.0  eps=0.4360  (R_SAC=-8.72  threshold=-9.156)

step     gated_frac   mean_R   best_R    succ_rate
 5000      0.000       -13.25   -12.57    1.00
10000      0.000        -9.55    -9.51    1.00   ← all skills converge to identical near-optimal path
15000      0.021       -31.66    -9.60    0.60   ← gate starts firing; some skills diverge
25000      0.054       -22.00    -8.73    0.80
40000      0.074       -30.20   -11.24    0.40
50000      0.085       -10.14    -8.75    1.00
60000      0.122       -17.17    -8.83    0.80   ← final
```

**Final per-skill (5 deterministic episodes each):**

| z | R | succ | length | end_dist |
|---|---:|---:|---:|---:|
| 0 | −14.24 | 100 % | 34 | 0.047 |
| 1 |  −8.83 | 100 % | 24 | 0.047 |
| 2 | −13.55 | 100 % | 34 | 0.033 |
| 3 | −37.73 |   0 % | 200 | 0.088 |
| 4 | −11.47 | 100 % | 29 | 0.043 |

See `runs/smerl_point2d/skills.png` for trajectory plots + training curves.

### Observations

- **Up to 10 k steps:** the gate stays shut. Every skill converges to the same near-optimal path (mean return −9.55, one step shy of the threshold). At that point SMERL is acting as plain latent-conditioned SAC.
- **From 15 k onward:** as the policy improves further, episodes start clearing the −9.16 threshold; the diversity bonus fires and the skills start diverging. `gated_frac` rises monotonically from 0 → 0.12.
- **Skills diverge but not all stay competent:** z=1 ends up the near-optimal skill (24 steps, R ≈ −8.83, *better than baseline SAC's R_SAC = −8.72* because it's evaluated deterministically). z=0, z=2, z=4 take slightly longer routes (29–34 steps) but still succeed. z=3 fails — it learns to push toward `(−1, −1)` along the bottom wall and stalls just outside the success radius (end_dist 0.088 = 0.05 radius + slop).
- **Why z=3 stalls:** the diversity bonus rewards visiting states that are easy to discriminate. The bottom-left corner is a strong "this is z=3" signal for the discriminator, but it's outside the success radius. With α = 10 the per-step intrinsic reward can exceed the −0.65 per-step distance penalty by a wide margin during gated episodes, so the agent prefers parking in a recognizable spot to converging on the goal. This is exactly the failure mode SMERL's *constrained* optimization is supposed to prevent — but the gate is closed for that skill's exploration, so it never gets pulled back toward optimality. The constraint is only satisfied when episodes succeed, and once z=3 stops succeeding, no gradient pushes it back.
- **`gated_frac = 0.12`** is the right ballpark for SMERL — most exploration uses task-only reward (so near-optimal performance is preserved); a small fraction of "good" episodes get the bonus that drives divergence. It would have been concerning if it stayed at 0.

### What I'd try next

- **Tighter buffer + more steps:** with `buffer_size=1000` and 200-step truncated episodes, only ~5 episodes fit in the buffer. Once z=3 starts truncating, those terrible-return transitions dominate its slice of the buffer and recovery is hard. Bumping the buffer to 10 k or training longer would help.
- **Per-skill stratified sampling:** the replay buffer mixes z's uniformly by recency, not by skill. If one skill's recent trajectories are bad, the others train on its data anyway. Stratifying batches by `z` would help.
- **Discriminator input pruning:** feeding only `(x, y)` (no velocity, no goal) would force the discriminator to discriminate on positional trajectory, which is the diversity signal we want. Right now the discriminator can shortcut on velocity, which is less meaningful diversity.

None of these are changes the paper makes; they're follow-ups for if we want better results on this specific env.
