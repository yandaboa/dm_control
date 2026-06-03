# BAMDP Experiment Proposal: Synthetic Expert Failures over Success Rates

## Motivation

We have multiple near-optimal experts in simulation. We want to construct BAMDPs in
which some experts succeed and others fail, so that a BAMDP solver can train on these
environments and learn to solve a downstream BAMDP. We assume access to a simulator,
a discriminator, and multiple strategies.

## Experiment Setup

We form BAMDPs over expert **failure rates**. In each environment, we sample a failure
rate $p_i$ for each strategy separately from a beta distribution. The BAMDP latent is
$\theta = (p_1, \dots, p_K)$.

> **Terminology note.** Throughout, $p_i$ denotes the *failure* probability of strategy
> $i$. If a success rate $s_i$ is preferred instead, substitute $p_i = 1 - s_i$.

## Forced-Failure Injection

During rollout in the environment, at each step the discriminator outputs a distribution $w^{(t)}$
over strategy probabilities. We inject forced failures via a **per-strategy hazard**.

For target failure rate $p_i$ and average episode length $n_i$, define the per-step
hazard:

$$h_i = -\tfrac{1}{n_i}\ln(1 - p_i).$$

At each step we mix **hazards** (not probabilities) using the discriminator weights,
then convert back to a per-step failure probability:

$$h^{(t)} = \sum_i w_i^{(t)} h_i, \qquad f^{(t)} = 1 - e^{-h^{(t)}}.$$

### Why mix in hazard space

Probabilities do not compose linearly across steps; what matters is cumulative
survival, $P(\text{survive}) = \prod_t (1 - f^{(t)})$. Working in log-survival turns
this product into a sum. Mixing hazards therefore yields a closed-form,
path-independent guarantee:

$$P(\text{survive}) = \prod_t (1 - f^{(t)})
= \exp\!\Big(-\sum_i h_i \, T_i\Big)
= \prod_i (1 - p_i)^{T_i / n_i}, \qquad T_i = \sum_t w_i^{(t)}.$$

Only the **time-integrated discriminator mass** $T_i$ matters, not the ordering of the
weights:

- **Pure, confidently-classified strategy $i$:** $T_i = n_i$, so the realized failure
  rate is exactly $p_i$ — the target.
- **Mixed trajectory:** receives the geometric interpolation
  $\prod_i (1-p_i)^{T_i/n_i}$, the principled blend of independent competing failure
  risks (independent risks add their hazards, not their probabilities).

Arithmetic mixing of the $f_i$ directly has no clean closed form and is
order-dependent, so it is avoided.

### Hazard budget (variance reduction)

To reduce variance in the cumulative rate, we replace the fixed-$n_i$ schedule with a
**hazard budget**: track a target cumulative hazard $H_i = -\ln(1 - p_i)$ per strategy
and spend the remainder evenly over the estimated remaining steps:

$$h_i^{(t)} = H_i^{\text{rem}} / \hat{n}_{\text{rem}}.$$

This self-corrects for episode-length variation. Without it, Jensen's inequality on the
convex map $(1-f)^N$ biases the realized failure rate slightly above target, since
$n_i$ is only the *average* length and failed episodes are truncated.

## Failure Mechanics and Rescue

When a forced failure triggers, it takes the form of early termination.

## Validation

Before trusting mixed rollouts, run each expert **in isolation** and confirm the
realized failure rate matches $p_i$. Pure-strategy calibration is the operational test
that the set $P = \{p_i\}$ is correctly enforced, since there is no ground-truth rate
for a genuinely mixed trajectory.

## Reference

The setup is naturally framed as meta-RL over a BAMDP with a learned belief; see the
VariBAD line of work (Zintgraf et al., 2020, *VariBAD: A Very Good Method for
Bayes-Adaptive Deep RL via Meta-Learning*). The discriminator here acts as a
hand-specified belief encoder over the strategy latent $\theta$.