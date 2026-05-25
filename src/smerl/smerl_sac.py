"""SMERL with a DIAYN-style discriminator on top of SAC.

Implements Algorithm 1 from Kumar et al., NeurIPS 2020
("One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL").

Per-step reward stored in the replay buffer (Eq 5):

    r_SMERL(s_t, a_t) = r_env(s_t, a_t)
                      + alpha * 1[ R_M(pi_theta) >= R*_M - eps ] * r_tilde(s_t)

where r_tilde(s_t) = log q_phi(z | s_{t+1}) - log p(z), p(z) uniform over |Z|.
The indicator is evaluated once per episode (depends on total task return).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- networks ----------------------------------


def mlp(sizes: Iterable[int], act: type[nn.Module] = nn.ReLU,
        out_act: type[nn.Module] | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    sizes = list(sizes)
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy conditioned on (state, one-hot z)."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, z_dim: int, act_dim: int,
                 hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        self.trunk = mlp([obs_dim + z_dim, *hidden], act=nn.ReLU)
        # nn.ReLU on the last hidden — append head linear layers
        last = hidden[-1]
        self.mu_head = nn.Linear(last, act_dim)
        self.log_std_head = nn.Linear(last, act_dim)
        # Make the trunk emit activations (add final ReLU manually below)
        self._final_act = nn.ReLU()

    def _features(self, obs: torch.Tensor, z_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z_onehot], dim=-1)
        return self._final_act(self.trunk(x))

    def forward(self, obs: torch.Tensor, z_onehot: torch.Tensor):
        h = self._features(obs, z_onehot)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor, z_onehot: torch.Tensor):
        """Reparameterised sample. Returns (action, log_prob, tanh-mean)."""
        mu, log_std = self(obs, z_onehot)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()
        a = torch.tanh(u)
        # SAC log-prob correction for tanh squash
        log_prob = normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, act_dim: int,
                 hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        in_dim = obs_dim + z_dim + act_dim
        self.q1 = mlp([in_dim, *hidden, 1], act=nn.ReLU)
        self.q2 = mlp([in_dim, *hidden, 1], act=nn.ReLU)

    def forward(self, obs: torch.Tensor, z_onehot: torch.Tensor,
                act: torch.Tensor):
        x = torch.cat([obs, z_onehot, act], dim=-1)
        return self.q1(x), self.q2(x)


class Discriminator(nn.Module):
    """q_phi(z | s): MLP over the raw state, outputs logits over |Z|."""

    def __init__(self, obs_dim: int, n_skills: int,
                 hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_skills], act=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ----------------------------- replay buffer -----------------------------


class ReplayBuffer:
    """Plain numpy circular buffer holding (s, a, r_smerl, s', done, z)."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.z = np.zeros((capacity,), dtype=np.int64)
        self.idx = 0
        self.size = 0

    def add(self, s, a, r, ns, d, z):
        i = self.idx
        self.obs[i] = s
        self.act[i] = a
        self.rew[i] = r
        self.next_obs[i] = ns
        self.done[i] = float(d)
        self.z[i] = int(z)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        out = {
            "obs": torch.as_tensor(self.obs[idxs], device=device),
            "act": torch.as_tensor(self.act[idxs], device=device),
            "rew": torch.as_tensor(self.rew[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=device),
            "done": torch.as_tensor(self.done[idxs], device=device),
            "z": torch.as_tensor(self.z[idxs], device=device),
        }
        return out


# ----------------------------- SMERL agent -------------------------------


@dataclass
class SMERLConfig:
    n_skills: int = 5
    alpha_div: float = 10.0          # diversity-reward weight
    eps_frac: float = 0.05           # eps = eps_frac * |R_SAC|
    R_SAC: float = -8.72             # baseline SAC return on this env

    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01                # Polyak coefficient
    batch_size: int = 128
    buffer_size: int = 1_000
    hidden: tuple[int, ...] = (32, 32)

    learning_starts: int = 1_000
    grad_steps_per_env_step: int = 1

    target_entropy: float | None = None  # default: -|A|


class SMERLAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SMERLConfig,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K = cfg.n_skills
        self.log_pz = -math.log(self.K)  # log p(z) for uniform p

        self.actor = GaussianActor(obs_dim, self.K, act_dim, cfg.hidden).to(device)
        self.critic = TwinQ(obs_dim, self.K, act_dim, cfg.hidden).to(device)
        self.critic_target = TwinQ(obs_dim, self.K, act_dim, cfg.hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.disc = Discriminator(obs_dim, self.K, cfg.hidden).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=cfg.lr)

        # Automatic entropy temperature
        target_entropy = cfg.target_entropy
        if target_entropy is None:
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy
        self.log_ent_coef = torch.zeros(1, requires_grad=True, device=device)
        self.ent_opt = torch.optim.Adam([self.log_ent_coef], lr=cfg.lr)

        self.replay = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim)

    def z_onehot(self, z: int | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(z, int):
            idx = torch.tensor([z], device=self.device, dtype=torch.long)
        elif isinstance(z, np.ndarray):
            idx = torch.as_tensor(z, device=self.device, dtype=torch.long)
        else:
            idx = z.to(self.device).long()
        return F.one_hot(idx, num_classes=self.K).float()

    # ---- intrinsic reward, computed from the current discriminator ----

    @torch.no_grad()
    def diversity_reward(self, next_obs_np: np.ndarray, z: int) -> float:
        obs_t = torch.as_tensor(next_obs_np[None, :], device=self.device,
                                dtype=torch.float32)
        logits = self.disc(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)
        log_qz_s = log_probs[0, int(z)].item()
        return float(log_qz_s - self.log_pz)

    # ---- action selection ----

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, z: int, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np[None, :], device=self.device,
                                dtype=torch.float32)
        z_oh = self.z_onehot(int(z))
        a, _, det = self.actor.sample(obs_t, z_oh)
        if deterministic:
            return det.cpu().numpy().squeeze(0)
        return a.cpu().numpy().squeeze(0)

    # ---- one SAC + discriminator gradient step ----

    def update(self) -> dict:
        if self.replay.size < self.cfg.batch_size:
            return {}
        batch = self.replay.sample(self.cfg.batch_size, self.device)
        obs, act, rew, next_obs, done, z = (
            batch["obs"], batch["act"], batch["rew"],
            batch["next_obs"], batch["done"], batch["z"],
        )
        z_oh = self.z_onehot(z)

        ent_coef = self.log_ent_coef.exp().detach()

        # ---- critic ----
        with torch.no_grad():
            a_next, logp_next, _ = self.actor.sample(next_obs, z_oh)
            q1_t, q2_t = self.critic_target(next_obs, z_oh, a_next)
            min_q_next = torch.min(q1_t, q2_t) - ent_coef * logp_next
            y = rew + self.cfg.gamma * (1.0 - done) * min_q_next

        q1, q2 = self.critic(obs, z_oh, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor ----
        a_pi, logp_pi, _ = self.actor.sample(obs, z_oh)
        q1_pi, q2_pi = self.critic(obs, z_oh, a_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (ent_coef * logp_pi - min_q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ---- entropy temperature ----
        ent_loss = -(self.log_ent_coef * (logp_pi.detach() + self.target_entropy)).mean()
        self.ent_opt.zero_grad(set_to_none=True)
        ent_loss.backward()
        self.ent_opt.step()

        # ---- discriminator ----
        disc_logits = self.disc(obs)
        disc_loss = F.cross_entropy(disc_logits, z)

        self.disc_opt.zero_grad(set_to_none=True)
        disc_loss.backward()
        self.disc_opt.step()

        # ---- target network Polyak update ----
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(),
                              self.critic_target.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        with torch.no_grad():
            disc_acc = (disc_logits.argmax(dim=-1) == z).float().mean().item()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "ent_coef": float(ent_coef.item()),
            "ent_loss": float(ent_loss.item()),
            "disc_loss": float(disc_loss.item()),
            "disc_acc": float(disc_acc),
            "mean_q": float(min_q_pi.mean().item()),
            "mean_logp_pi": float(logp_pi.mean().item()),
        }
