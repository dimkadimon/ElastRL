"""
sac.py — Soft Actor-Critic for the Elasto Mania RL environment.

Architecture and hyper-parameters follow TMRL's `SpinupSacAgent`
(tmrl/custom/custom_algorithms.py, https://github.com/trackmania-rl/tmrl), adapted to
a discrete action space (TMRL's LIDAR interface also uses discrete actions):

  * twin Q critics with a shared MLP trunk over (observation, one-hot action)
  * a categorical policy sampling one of the 9 key combinations
  * polyak-averaged target critics (tau = 0.995)
  * automatic entropy tuning (SAC v2), lr 1e-3, target entropy = 0.5 * log(n_actions)
  * gamma = 0.99, batch size 256, Adam with lr 3e-4 for actor/critic (TMRL defaults 1e-3
    proved too aggressive here; 3e-4 is the usual SAC value)

The replay buffer keeps observations as float32 and actions as int64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.ReLU, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [activation()]
    layers += [out_activation()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Discrete-action actor + twin Q critics with *separate* networks.

    This mirrors TMRL's `MLPActorCritic` (tmrl/custom/custom_models.py), where
    `SquashedGaussianMLPActor` and the two `MLPQFunction`s are independent MLPs and
    no parameters are shared between the policy and the value functions.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.n_actions = n_actions
        self.pi = mlp([obs_dim] + list(hidden) + [n_actions])
        self.q1 = mlp([obs_dim + n_actions] + list(hidden) + [1])
        self.q2 = mlp([obs_dim + n_actions] + list(hidden) + [1])
        self.reset_parameters()

    @staticmethod
    def _last_linear(seq):
        last = None
        for m in seq.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # small gain on the policy head -> near-uniform initial action distribution
        nn.init.orthogonal_(self._last_linear(self.pi).weight, gain=0.01)

    def logits(self, obs):
        return self.pi(obs)

    def q_values(self, obs, onehot):
        x = torch.cat([obs, onehot], dim=-1)
        return self.q1(x), self.q2(x)

    def q_all(self, obs):
        """Q values for *every* discrete action: (B, n_actions) (min of the twins)."""
        B = obs.shape[0]
        A = self.n_actions
        oh = torch.eye(A, device=obs.device, dtype=obs.dtype)[None, :, :].expand(B, A, A)
        obs_rep = obs[:, None, :].expand(B, A, obs.shape[1]).reshape(B * A, obs.shape[1])
        x = torch.cat([obs_rep, oh.reshape(B * A, A)], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return torch.min(q1, q2).reshape(B, A)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False, device="cpu"):
        o = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits = self.logits(o)
        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        return a.cpu().numpy()


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int = 400_000, seed: int = 0):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros(size, dtype=np.int64)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.size = size
        self.ptr = 0
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(self, obs, act, rew, next_obs, done):
        n = len(act)
        idx = (self.ptr + np.arange(n)) % self.size
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.ptr = (self.ptr + n) % self.size
        self.n = min(self.n + n, self.size)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.n, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])


@dataclass
class SAC:
    obs_dim: int
    n_actions: int
    device: str = "cpu"
    gamma: float = 0.99
    polyak: float = 0.995
    lr: float = 3e-4
    lr_alpha: float = 1e-3
    alpha_init: float = 0.01
    target_entropy: float = None  # None -> 0.5 * log(n_actions)
    hidden: tuple = (192, 192)
    batch_size: int = 256
    pi_loss_mode: str = "exact"  # "exact" (sum over all actions) or "sample" (cheaper MC estimate)
    # Lower bound on the entropy coefficient. With only ~1-2M environment steps the
    # policy must keep exploring; if alpha is allowed to decay to ~1e-4 (the textbook
    # floor) the agent commits early and can lock into a "do nothing" local optimum
    # on levels where the first obstacle is hard.
    alpha_min: float = 1e-4

    def __post_init__(self):
        self.model = ActorCritic(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.model_target = ActorCritic(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())
        for p in self.model_target.parameters():
            p.requires_grad_(False)

        self.opt_pi = torch.optim.Adam(self.model.pi.parameters(), lr=self.lr)
        self.opt_q = torch.optim.Adam(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()), lr=self.lr)
        self.log_alpha = torch.tensor(math.log(self.alpha_init), requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
        if self.target_entropy is None:
            self.target_entropy = 0.5 * math.log(self.n_actions)
        self.updates = 0

    @property
    def alpha(self):
        return float(self.log_alpha.exp().item())

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        return self.model.act(obs, deterministic=deterministic, device=self.device)

    # ---------------------------------------------------------------- learning
    def update(self, buf: ReplayBuffer) -> dict:
        obs, act, rew, next_obs, done = buf.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        next_t = torch.as_tensor(next_obs, device=self.device)
        act_t = torch.as_tensor(act, device=self.device)
        rew_t = torch.as_tensor(rew, device=self.device)
        done_t = torch.as_tensor(done, device=self.device)
        onehot = F.one_hot(act_t, self.n_actions).float()

        # ---- critic update: entropy-regularised Bellman backup over all actions ----
        with torch.no_grad():
            next_logits = self.model_target.logits(next_t)
            next_logp = F.log_softmax(next_logits, dim=-1)
            next_p = next_logp.exp()
            q_next = self.model_target.q_all(next_t)                  # (B,A) min of twins
            v_next = (next_p * (q_next - self.alpha * next_logp)).sum(dim=-1)
            target = rew_t + self.gamma * (1.0 - done_t) * v_next     # (B,)

        q1, q2 = self.model.q_values(obs_t, onehot)
        q_loss = F.mse_loss(q1.squeeze(-1), target) + F.mse_loss(q2.squeeze(-1), target)

        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()), 10.0)
        self.opt_q.step()

        # ---- actor update ----
        logits = self.model.logits(obs_t)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        if self.pi_loss_mode == "exact":
            with torch.no_grad():
                q_pi = self.model.q_all(obs_t)                        # (B,A) detached
            pi_loss = (p * (self.alpha * logp - q_pi)).sum(dim=-1).mean()
        else:
            # Single-sample score-function (REINFORCE) estimator of the same objective:
            #   J(pi) = E_a[alpha*log pi(a|s) - Q(s,a)]
            #   grad J = E_a[(alpha*log pi(a) + alpha - Q(s,a)) * grad log pi(a)]
            # (The naive "alpha*logpi(a_s) - Q(s,a_s)" with a detached Q is NOT a valid
            #  policy gradient: it just lowers the probability of every sampled action.)
            with torch.no_grad():
                a_s = torch.distributions.Categorical(logits=logits).sample()
                oh_s = F.one_hot(a_s, self.n_actions).float()
                q1s, q2s = self.model.q_values(obs_t, oh_s)
                q_s = torch.min(q1s, q2s).squeeze(-1)
                w_s = (self.alpha * logp.gather(1, a_s[:, None]).squeeze(-1)
                       + self.alpha - q_s).detach()
            logp_s = logp.gather(1, a_s[:, None]).squeeze(-1)
            pi_loss = (w_s * logp_s).mean()

        self.opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.pi.parameters(), 10.0)
        self.opt_pi.step()

        # ---- entropy coefficient (SAC v2) ----
        # J(alpha) = E[-alpha (log pi(a|s) + H_target)]  =>  dJ/dlog(alpha) = (H - H_target)
        # Entropy H here is *positive* (0 .. log n_actions), so the loss is
        # -(log_alpha * (target_entropy - H)) and alpha shrinks while H exceeds target.
        with torch.no_grad():
            ent = -(p * logp).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (self.target_entropy - ent))
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(self.alpha_min), math.log(0.5))

        with torch.no_grad():
            for tp, p_ in zip(self.model_target.parameters(), self.model.parameters()):
                tp.mul_(self.polyak).add_(p_, alpha=1.0 - self.polyak)

        self.updates += 1
        return dict(q_loss=float(q_loss.item()), pi_loss=float(pi_loss.item()),
                    entropy=float(ent.item()), alpha=self.alpha)

    # ------------------------------------------------------------------- io
    def save(self, path: str, extra: dict | None = None):
        torch.save(dict(model=self.model.state_dict(),
                        target=self.model_target.state_dict(),
                        log_alpha=self.log_alpha.detach(),
                        opt_pi=self.opt_pi.state_dict(),
                        opt_q=self.opt_q.state_dict(),
                        updates=self.updates,
                        extra=extra or {}), path)

    def load(self, path: str, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model_target.load_state_dict(ck["target"])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"])
        self.opt_pi.load_state_dict(ck["opt_pi"])
        self.opt_q.load_state_dict(ck["opt_q"])
        self.updates = ck.get("updates", 0)
        return ck.get("extra", {})
