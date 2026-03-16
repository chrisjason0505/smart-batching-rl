"""
agent/impala.py

High-throughput IMPALA agent with LSTM temporal policy and batched V-trace.

Components
----------
ActorCriticNet          – legacy MLP (backward compat with old checkpoints)
LSTMActorCriticNet      – LSTM-based policy for temporal pattern learning
compute_vtrace          – single-trajectory V-trace (backward compat)
compute_vtrace_batched  – batched V-trace for learner [B, T, …]
IMPALAgent              – main agent class (supports MLP or LSTM mode)
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────────────

class LSTMActorCriticNet(nn.Module):
    """LSTM-based actor-critic for temporal pattern learning.

    Architecture::

        input(obs_dim) → LayerNorm → Linear(hidden) → ReLU
                                        ↓
                                     LSTM(hidden)
                                      ├→ policy_head → Linear(act_dim)
                                      └→ value_head  → Linear(1)
    """

    def __init__(self, obs_dim: int = 11, act_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.hidden_size = hidden
        self.obs_dim = obs_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, hx=None):
        """
        Parameters
        ----------
        x : [B, T, obs_dim] or [B, obs_dim]
        hx : tuple(h_0, c_0) each [1, B, hidden]  or None

        Returns
        -------
        logits : [B, T, act_dim] or [B, act_dim]
        values : [B, T] or [B]
        hx_new : tuple
        """
        squeeze_time = False
        if x.dim() == 2:
            x = x.unsqueeze(1)           # [B, 1, obs_dim]
            squeeze_time = True

        B, T, _ = x.shape
        encoded = self.encoder(x)         # [B, T, hidden]

        if hx is None:
            hx = self.initial_state(B, device=x.device)

        lstm_out, hx_new = self.lstm(encoded, hx)  # [B, T, hidden]

        logits = self.policy_head(lstm_out)            # [B, T, act_dim]
        values = self.value_head(lstm_out).squeeze(-1) # [B, T]

        if squeeze_time:
            logits = logits.squeeze(1)   # [B, act_dim]
            values = values.squeeze(1)   # [B]

        return logits, values, hx_new

    def initial_state(self, batch_size: int = 1, device=None):
        dev = device or next(self.parameters()).device
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=dev),
            torch.zeros(1, batch_size, self.hidden_size, device=dev),
        )


# ─────────────────────────────────────────────────────────────────────────────
# V-trace — batched [B, T]
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_vtrace_batched(
    behavior_log_probs: torch.Tensor,   # [B, T]
    target_log_probs:   torch.Tensor,   # [B, T]
    rewards:            torch.Tensor,   # [B, T]
    values:             torch.Tensor,   # [B, T]
    bootstrap_values:   torch.Tensor,   # [B]
    dones:              torch.Tensor,   # [B, T]
    discount:  float = 0.99,
    rho_bar:   float = 1.0,
    c_bar:     float = 1.0,
):
    """Batched V-trace for the learner.  All tensors are [B, T]."""
    B, T = rewards.shape
    not_done = 1.0 - dones

    log_rhos     = target_log_probs - behavior_log_probs
    rhos         = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs           = torch.clamp(rhos, max=c_bar)

    values_ext = torch.cat(
        [values, bootstrap_values.unsqueeze(1)], dim=1,
    )  # [B, T+1]

    deltas = clipped_rhos * (
        rewards + discount * not_done * values_ext[:, 1:] - values_ext[:, :-1]
    )

    vs_minus_v = torch.zeros(B, T + 1, device=rewards.device)
    for t in range(T - 1, -1, -1):
        vs_minus_v[:, t] = (
            deltas[:, t]
            + discount * not_done[:, t] * cs[:, t] * vs_minus_v[:, t + 1]
        )

    vs = values + vs_minus_v[:, :T]

    vs_next = torch.cat(
        [vs[:, 1:], bootstrap_values.unsqueeze(1)], dim=1,
    )
    advantages = clipped_rhos * (
        rewards + discount * not_done * vs_next - values
    )

    return vs, advantages


