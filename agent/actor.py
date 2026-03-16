"""
agent/actor.py

Parallel actor workers for high-throughput IMPALA training.

Each actor runs its own BatchingEnvV2 with an independent traffic pattern,
collects trajectory segments of ``unroll_length`` steps, and pushes them to
a shared thread-safe queue.  Actors periodically sync policy weights from the
learner via a shared dictionary — they *never* block waiting for updates.

Uses Python ``threading`` (not multiprocessing) to avoid Windows spawn issues
with PyTorch.  The GIL is largely released during NumPy / CUDA operations, so
we still get reasonable throughput.
"""

import threading
import queue
import time
import copy
import numpy as np
import torch

from env.batching_env_v2 import BatchingEnvV2
from agent.impala import LSTMActorCriticNet


# Traffic modes cycled across actors for diversity
_TRAFFIC_MODES = ("steady", "burst", "oscillating", "spike", "mixed")


def _make_env(actor_id: int, config: dict):
    """Create an env with a traffic mode determined by actor_id."""
    mode = _TRAFFIC_MODES[actor_id % len(_TRAFFIC_MODES)]
    return BatchingEnvV2(
        config=config,
        traffic_mode=mode,
        traffic_seed=actor_id * 1337,
    )


def actor_loop(
    actor_id:       int,
    config:         dict,
    obs_dim:        int,
    act_dim:        int,
    hidden:         int,
    unroll_length:  int,
    traj_queue:     queue.Queue,
    weight_store:   dict,            # {"weights": OrderedDict, "version": int}
    stop_event:     threading.Event,
    weight_sync_interval: int = 50,  # sync every N trajectories
    device_str:     str = "cpu",     # actors always run on CPU
):
    """Entry point for one actor thread.

    Collects trajectory segments and puts them on *traj_queue*.
    Periodically copies learner weights from *weight_store*.
    """
    device = torch.device(device_str)

    # Local network copy (CPU-only for actors)
    net = LSTMActorCriticNet(obs_dim, act_dim, hidden).to(device)
    net.eval()

    # Initial weight sync
    if "weights" in weight_store:
        net.load_state_dict(weight_store["weights"])

    env = _make_env(actor_id, config)
    obs, _ = env.reset(seed=actor_id)

    hx = net.initial_state(1, device=device)
    traj_count = 0
    local_version = weight_store.get("version", 0)
    total_steps = 0

    while not stop_event.is_set():
        # ── Collect one trajectory segment ────────────────────────────────
        obs_buf      = []
        action_buf   = []
        reward_buf   = []
        log_prob_buf = []
        done_buf     = []
        hx_init = (hx[0].clone(), hx[1].clone())

        for _ in range(unroll_length):
            with torch.no_grad():
                x = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits, _, hx_new = net(x, hx)
                dist = torch.distributions.Categorical(logits=logits)
                action   = dist.sample()
                log_prob = dist.log_prob(action)

            obs_buf.append(obs.copy())
            action_buf.append(action.item())
            log_prob_buf.append(log_prob.item())

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            reward_buf.append(reward)
            done_buf.append(float(done))

            obs = next_obs
            hx  = hx_new
            total_steps += 1

            if done:
                obs, _ = env.reset()
                hx = net.initial_state(1, device=device)

        # ── Package trajectory ────────────────────────────────────────────
        trajectory = {
            "obs":           np.array(obs_buf, dtype=np.float32),       # [T, obs_dim]
            "actions":       np.array(action_buf, dtype=np.int64),      # [T]
            "rewards":       np.array(reward_buf, dtype=np.float32),    # [T]
            "log_probs":     np.array(log_prob_buf, dtype=np.float32),  # [T]
            "dones":         np.array(done_buf, dtype=np.float32),      # [T]
            "bootstrap_obs": obs.copy(),                                # [obs_dim]
            "initial_hx_h":  hx_init[0].squeeze(0).numpy(),            # [1, hidden]
            "initial_hx_c":  hx_init[1].squeeze(0).numpy(),            # [1, hidden]
            "actor_id":      actor_id,
        }

        # Non-blocking put — drop if queue is full (actor never waits)
        try:
            traj_queue.put_nowait(trajectory)
        except queue.Full:
            pass  # learner is busy; skip this trajectory

        traj_count += 1

        # ── Periodic weight sync ─────────────────────────────────────────
        if traj_count % weight_sync_interval == 0:
            cur_version = weight_store.get("version", 0)
            if cur_version > local_version and "weights" in weight_store:
                try:
                    net.load_state_dict(weight_store["weights"])
                    local_version = cur_version
                except Exception:
                    pass  # stale dict; retry next time

    env.close()
