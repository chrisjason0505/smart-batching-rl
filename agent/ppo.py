"""
agent/ppo.py

Highly optimised PPO (Proximal Policy Optimization) implementation designed
for fair comparison against the IMPALA agent on the CDN batching environment.

Implements all modern PPO tricks:
  - Generalised Advantage Estimation (GAE-λ)
  - Clipped surrogate objective
  - Value function clipping
  - Entropy bonus with cosine-annealing schedule
  - Advantage normalisation (per-minibatch)
  - Gradient clipping
  - Learning rate annealing
  - Mini-batch updates with multiple epochs
  - LSTM policy support (same architecture as IMPALA for fairness)

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.impala import LSTMActorCriticNet


# ─────────────────────────────────────────────────────────────────────────────
# GAE computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_gae(
    rewards:    torch.Tensor,   # [T]
    values:     torch.Tensor,   # [T]
    next_value: torch.Tensor,   # scalar
    dones:      torch.Tensor,   # [T]
    gamma:      float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for a single trajectory."""
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * not_done * last_gae

    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def compute_gae_batched(
    rewards:     torch.Tensor,   # [B, T]
    values:      torch.Tensor,   # [B, T]
    next_values: torch.Tensor,   # [B]
    dones:       torch.Tensor,   # [B, T]
    gamma:       float = 0.99,
    gae_lambda:  float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GAE for multiple trajectories [B, T]."""
    B, T = rewards.shape
    advantages = torch.zeros(B, T, device=rewards.device)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_values
        else:
            next_val = values[:, t + 1]

        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_val * not_done - values[:, t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """Proximal Policy Optimization agent with LSTM policy.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Number of actions.
    hidden : int
        Hidden layer width (matches IMPALA for fair comparison).
    lr : float
        Initial learning rate.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_eps : float
        PPO clipping epsilon.
    entropy_coeff : float
        Initial entropy bonus coefficient.
    entropy_coeff_final : float
        Final entropy coefficient (for annealing).
    value_coeff : float
        Value loss coefficient.
    max_grad_norm : float
        Max gradient norm for clipping.
    n_epochs : int
        Number of PPO update epochs per batch.
    n_minibatches : int
        Number of minibatches to split each batch into.
    clip_value : bool
        Whether to clip value function updates.
    target_kl : float | None
        If set, early-stop epoch if KL exceeds this.
    total_steps : int
        Total training steps (for LR / entropy annealing).
    """

    def __init__(
        self,
        obs_dim:            int   = 11,
        act_dim:            int   = 2,
        hidden:             int   = 128,
        lr:                 float = 3e-4,
        gamma:              float = 0.99,
        gae_lambda:         float = 0.95,
        clip_eps:           float = 0.2,
        entropy_coeff:      float = 0.01,
        entropy_coeff_final: float = 0.001,
        value_coeff:        float = 0.5,
        max_grad_norm:      float = 0.5,
        n_epochs:           int   = 4,
        n_minibatches:      int   = 4,
        clip_value:         bool  = True,
        target_kl:          float | None = 0.015,
        total_steps:        int   = 250_000,
    ):
        self.obs_dim            = obs_dim
        self.act_dim            = act_dim
        self.gamma              = gamma
        self.gae_lambda         = gae_lambda
        self.clip_eps           = clip_eps
        self.entropy_coeff      = entropy_coeff
        self.entropy_coeff_init = entropy_coeff
        self.entropy_coeff_final = entropy_coeff_final
        self.value_coeff        = value_coeff
        self.max_grad_norm      = max_grad_norm
        self.n_epochs           = n_epochs
        self.n_minibatches      = n_minibatches
        self.clip_value         = clip_value
        self.target_kl          = target_kl
        self.total_steps        = total_steps
        self.steps_done         = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Same network architecture as IMPALA for fair comparison
        self.net = LSTMActorCriticNet(obs_dim, act_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.lr_init = lr
        self._hx = None

    # ── Annealing ─────────────────────────────────────────────────────────

    def _anneal(self):
        """Update LR and entropy coefficient based on progress."""
        frac = min(1.0, self.steps_done / max(self.total_steps, 1))

        # Linear LR decay
        new_lr = self.lr_init * (1.0 - frac)
        for pg in self.optimizer.param_groups:
            pg["lr"] = max(new_lr, 1e-6)

        # Cosine entropy annealing
        import math
        self.entropy_coeff = (
            self.entropy_coeff_final
            + 0.5 * (self.entropy_coeff_init - self.entropy_coeff_final)
            * (1.0 + math.cos(math.pi * frac))
        )

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, hx=None):
        """Stochastic action + log-prob + value for rollout collection."""
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if hx is None:
                hx = self.net.initial_state(1, device=self.device)
            logits, value, hx_new = self.net(x, hx)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item(), hx_new

    def predict(self, obs: np.ndarray, hx=None) -> int:
        """Deterministic greedy action (evaluation)."""
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if hx is None:
                if self._hx is None:
                    self._hx = self.net.initial_state(1, device=self.device)
                hx = self._hx
            logits, _, hx_new = self.net(x, hx)
            self._hx = hx_new
        return int(logits.argmax(dim=-1).item())

    def reset_hidden(self):
        self._hx = None

    # ── Batched PPO update ────────────────────────────────────────────────

    def update_batched(self, batch: dict) -> dict:
        """PPO clipped objective update on batched trajectories [B, T].

        batch keys:
            obs:           [B, T, obs_dim]
            actions:       [B, T]
            rewards:       [B, T]
            log_probs:     [B, T]   (old log probs)
            values:        [B, T]   (old values)
            dones:         [B, T]
            bootstrap_obs: [B, obs_dim]
            initial_hx:    tuple or None
        """
        self._anneal()

        obs_t    = batch["obs"].to(self.device)
        acts_t   = batch["actions"].to(self.device)
        rews_t   = batch["rewards"].to(self.device)
        old_lp   = batch["log_probs"].to(self.device)
        old_vals = batch["values"].to(self.device)
        dones_t  = batch["dones"].to(self.device)
        boot_obs = batch["bootstrap_obs"].to(self.device)
        init_hx  = batch.get("initial_hx")

        B, T, _ = obs_t.shape

        # Compute bootstrap values
        with torch.no_grad():
            _, boot_vals, _ = self.net(boot_obs.unsqueeze(1))
            boot_vals = boot_vals.squeeze(1) * (1.0 - dones_t[:, -1])

        # Compute GAE
        advantages, returns = compute_gae_batched(
            rewards=rews_t,
            values=old_vals,
            next_values=boot_vals,
            dones=dones_t,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Flatten for minibatch updates
        obs_flat  = obs_t.reshape(B * T, -1)
        acts_flat = acts_t.reshape(B * T)
        old_lp_flat = old_lp.reshape(B * T)
        old_vals_flat = old_vals.reshape(B * T)
        adv_flat  = advantages.reshape(B * T)
        ret_flat  = returns.reshape(B * T)

        total_samples = B * T
        mb_size = max(total_samples // self.n_minibatches, 1)

        total_metrics = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "total_loss": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0,
        }
        n_updates = 0

        for epoch in range(self.n_epochs):
            perm = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, mb_size):
                end = min(start + mb_size, total_samples)
                idx = perm[start:end]

                mb_obs  = obs_flat[idx]
                mb_acts = acts_flat[idx]
                mb_old_lp = old_lp_flat[idx]
                mb_old_vals = old_vals_flat[idx]
                mb_adv  = adv_flat[idx]
                mb_ret  = ret_flat[idx]

                # Advantage normalisation (per-minibatch)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Forward pass (no LSTM unrolling for flattened minibatch)
                # Use encoder + LSTM step-by-step is expensive; use MLP-style
                mb_obs_seq = mb_obs.unsqueeze(1)  # [mb, 1, obs_dim]
                logits, values, _ = self.net(mb_obs_seq)
                logits = logits.squeeze(1)
                values = values.squeeze(1)

                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(mb_acts)
                entropy = dist.entropy()

                # Policy loss (clipped)
                log_ratio = new_lp - mb_old_lp
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                if self.clip_value:
                    v_clipped = mb_old_vals + torch.clamp(
                        values - mb_old_vals, -self.clip_eps, self.clip_eps
                    )
                    v_loss1 = F.mse_loss(values, mb_ret)
                    v_loss2 = F.mse_loss(v_clipped, mb_ret)
                    value_loss = torch.max(v_loss1, v_loss2)
                else:
                    value_loss = F.mse_loss(values, mb_ret)

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                total_metrics["policy_loss"] += policy_loss.item()
                total_metrics["value_loss"]  += value_loss.item()
                total_metrics["entropy"]     += entropy_loss.item()
                total_metrics["total_loss"]  += loss.item()
                total_metrics["approx_kl"]   += approx_kl
                total_metrics["clip_frac"]   += clip_frac
                n_updates += 1

            # Early stopping on KL
            if self.target_kl is not None:
                avg_kl = total_metrics["approx_kl"] / max(n_updates, 1)
                if avg_kl > self.target_kl:
                    break

        # Average metrics
        for k in total_metrics:
            total_metrics[k] /= max(n_updates, 1)

        self.steps_done += total_samples
        return total_metrics

    # ── Single-trajectory update (convenience) ────────────────────────────

    def update(self, rollout: dict) -> dict:
        """PPO update on a single rollout dict.

        rollout keys: obs, actions, rewards, log_probs, values, dones, next_obs
        """
        obs_t    = torch.FloatTensor(np.array(rollout["obs"])).to(self.device)
        acts_t   = torch.LongTensor(rollout["actions"]).to(self.device)
        rews_t   = torch.FloatTensor(rollout["rewards"]).to(self.device)
        old_lp   = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        old_vals = torch.FloatTensor(rollout["values"]).to(self.device)
        dones_t  = torch.FloatTensor(rollout["dones"]).to(self.device)

        with torch.no_grad():
            next_obs = rollout.get("next_obs")
            if next_obs is not None:
                if isinstance(next_obs, list):
                    next_obs = next_obs[-1]
                nx = torch.FloatTensor(np.array(next_obs)).unsqueeze(0).to(self.device)
                _, boot, _ = self.net(nx)
                boot = (boot * (1.0 - dones_t[-1])).squeeze()
            else:
                boot = torch.tensor(0.0, device=self.device)

        advantages, returns = compute_gae(
            rewards=rews_t, values=old_vals, next_value=boot,
            dones=dones_t, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        batch = {
            "obs": obs_t.unsqueeze(0),
            "actions": acts_t.unsqueeze(0),
            "rewards": rews_t.unsqueeze(0),
            "log_probs": old_lp.unsqueeze(0),
            "values": old_vals.unsqueeze(0),
            "dones": dones_t.unsqueeze(0),
            "bootstrap_obs": torch.FloatTensor(
                np.array(rollout.get("next_obs", rollout["obs"][-1]))
            ).unsqueeze(0).to(self.device),
        }
        return self.update_batched(batch)

    # ── Weight sync ───────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def set_weights(self, state_dict: dict):
        self.net.load_state_dict(state_dict)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "net":        self.net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "obs_dim":    self.obs_dim,
            "act_dim":    self.act_dim,
            "steps_done": self.steps_done,
        }, path)

    @classmethod
    def load(cls, path: str, **overrides):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            obs_dim=ckpt["obs_dim"],
            act_dim=ckpt["act_dim"],
            **overrides,
        )
        agent.net.load_state_dict(ckpt["net"])
        agent.steps_done = ckpt.get("steps_done", 0)
        return agent
