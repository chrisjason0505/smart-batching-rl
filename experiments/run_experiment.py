"""
experiments/run_experiment.py

Fair head-to-head experiment: IMPALA V3 vs PPO on the CDN batching
environment.  Both agents use:
  - Same LSTM architecture (LSTMActorCriticNet)
  - Same environment (BatchingEnvV2 at 1000 req/s)
  - Same total training steps
  - Same evaluation protocol

Produces:
  - Training curves (reward vs steps, reward vs wall-clock)
  - Stability / variance across 3 seeds per algorithm
  - Final evaluation comparison
  - Saved models and metrics JSON

Usage:
    python -m experiments.run_experiment
"""

import os
import sys
import time
import json
import queue
import threading
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.batching_env_v2 import BatchingEnvV2
from agent.impala_v3 import IMPALAV3Agent
from config import CONFIG
from agent.impala import LSTMActorCriticNet
from agent.ppo import PPOAgent
from agent.actor import actor_loop
def collate_trajectories(trajs: list[dict], device: torch.device) -> dict:
    """Stack trajectory dicts into batched tensors [B, T, …]."""
    batch = {
        "obs":           torch.FloatTensor(np.stack([t["obs"] for t in trajs])).to(device),
        "actions":       torch.LongTensor(np.stack([t["actions"] for t in trajs])).to(device),
        "rewards":       torch.FloatTensor(np.stack([t["rewards"] for t in trajs])).to(device),
        "log_probs":     torch.FloatTensor(np.stack([t["log_probs"] for t in trajs])).to(device),
        "dones":         torch.FloatTensor(np.stack([t["dones"] for t in trajs])).to(device),
        "bootstrap_obs": torch.FloatTensor(np.stack([t["bootstrap_obs"] for t in trajs])).to(device),
    }

    hx_h = np.stack([t["initial_hx_h"] for t in trajs])  # [B, 1, hidden]
    hx_c = np.stack([t["initial_hx_c"] for t in trajs])
    batch["initial_hx"] = (
        torch.FloatTensor(hx_h).permute(1, 0, 2).to(device),
        torch.FloatTensor(hx_c).permute(1, 0, 2).to(device),
    )
    return batch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_STEPS    = 200_000      # Scale up explicitly for convergence
EVAL_INTERVAL  = 20_000
EVAL_EPISODES  = 5
NUM_SEEDS      = 1            # Just 1 seed locally to be fast
NUM_ACTORS     = 4            # actor threads for IMPALA
UNROLL_LENGTH  = 32           # trajectory segment length
LEARNER_BATCH  = 128          # learner batch size (samples)
PPO_ROLLOUT    = 256          # PPO rollout length
OBS_DIM        = 11
ACT_DIM        = 2
HIDDEN         = 128

RESULTS_DIR = os.path.join(ROOT, "experiments", "results")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent, config, n_episodes=10, seed_offset=1000):
    """Run agent on BatchingEnvV2 and return mean/std reward."""
    env = BatchingEnvV2(config=config, traffic_mode="mixed", traffic_seed=9999)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden()
        total_r = 0.0
        done = False
        while not done:
            action = agent.predict(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# IMPALA V3 training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_impala_v3(seed: int, config: dict) -> dict:
    """Train IMPALA V3 agent and return metrics dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"  IMPALA V3 — Seed {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = IMPALAV3Agent(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN,
        lr=5e-4, lr_final=1e-5,
        entropy_coeff=0.015, entropy_coeff_final=0.001,
        value_coeff=0.5, max_grad_norm=10.0,
        rho_bar=1.0, c_bar=1.0,
        advantage_norm=True, reward_norm=True,
        aux_value_l2=1e-4, lag_penalty_coeff=0.1,
        total_steps=TOTAL_STEPS,
    )

    # Actor-learner setup
    max_queue = NUM_ACTORS * 4
    traj_queue = queue.Queue(maxsize=max_queue)
    weight_store: dict = {}
    stop_event = threading.Event()

    weight_store["weights"] = agent.get_weights()
    weight_store["version"] = 0

    actor_threads = []
    for aid in range(NUM_ACTORS):
        t = threading.Thread(
            target=actor_loop,
            kwargs=dict(
                actor_id=aid + seed * 100,
                config=config,
                obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN,
                unroll_length=UNROLL_LENGTH,
                traj_queue=traj_queue,
                weight_store=weight_store,
                stop_event=stop_event,
                weight_sync_interval=15,
                device_str="cpu",
            ),
            daemon=True,
            name=f"impala-actor-{aid}-s{seed}",
        )
        t.start()
        actor_threads.append(t)

    batch_trajs = max(LEARNER_BATCH // UNROLL_LENGTH, 1)
    consumed = 0
    n_updates = 0
    t_start = time.time()

    metrics = {
        "steps": [], "rewards_mean": [], "rewards_std": [],
        "wall_clock": [], "policy_loss": [], "value_loss": [],
        "entropy": [], "kl_approx": [],
    }

    next_eval = EVAL_INTERVAL

    try:
        while consumed < TOTAL_STEPS:
            # Drain trajectories
            trajs = []
            deadline = time.time() + 1.0
            while len(trajs) < batch_trajs and time.time() < deadline:
                try:
                    t = traj_queue.get(timeout=0.02)
                    trajs.append(t)
                except queue.Empty:
                    continue

            if not trajs:
                continue

            batch = collate_trajectories(trajs, agent.device)
            m = agent.update_batched(batch)

            steps_in_batch = len(trajs) * UNROLL_LENGTH
            consumed += steps_in_batch
            n_updates += 1

            # Publish weights
            weight_store["weights"] = agent.get_weights()
            weight_store["version"] = n_updates

            # Periodic evaluation
            if consumed >= next_eval:
                elapsed = time.time() - t_start
                mean_r, std_r = evaluate_agent(agent, config, EVAL_EPISODES)

                metrics["steps"].append(consumed)
                metrics["rewards_mean"].append(mean_r)
                metrics["rewards_std"].append(std_r)
                metrics["wall_clock"].append(elapsed)
                metrics["policy_loss"].append(m["policy_loss"])
                metrics["value_loss"].append(m["value_loss"])
                metrics["entropy"].append(m["entropy"])
                metrics["kl_approx"].append(m.get("kl_approx", 0.0))

                throughput = consumed / max(elapsed, 1e-6)
                print(f"  [{consumed:>8,}/{TOTAL_STEPS:,}] "
                      f"R={mean_r:>+10.1f} +/- {std_r:>7.1f} | "
                      f"PL={m['policy_loss']:.3f} | "
                      f"Ent={m['entropy']:.4f} | "
                      f"KL={m.get('kl_approx', 0):.4f} | "
                      f"Tput={throughput:,.0f} sps | "
                      f"t={elapsed:.1f}s", flush=True)

                next_eval = consumed + EVAL_INTERVAL

    finally:
        stop_event.set()
        for t in actor_threads:
            t.join(timeout=3.0)

    total_time = time.time() - t_start
    metrics["total_time"] = total_time
    metrics["final_throughput"] = consumed / max(total_time, 1e-6)

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"impala_v3_seed{seed}.pt")
    agent.save(model_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PPO training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(seed: int, config: dict) -> dict:
    """Train PPO agent (synchronous rollout collection) and return metrics."""
    print(f"\n{'='*60}", flush=True)
    print(f"  PPO — Seed {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = PPOAgent(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN,
        lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coeff=0.01, entropy_coeff_final=0.001,
        value_coeff=0.5, max_grad_norm=0.5,
        n_epochs=4, n_minibatches=4,
        clip_value=True, target_kl=0.015,
        total_steps=TOTAL_STEPS,
    )

    # Use multiple envs for parallel rollout collection (vectorised-like)
    n_envs = NUM_ACTORS
    envs = [
        BatchingEnvV2(
            config=config,
            traffic_mode=["steady", "burst", "oscillating", "spike", "mixed"][i % 5],
            traffic_seed=(seed * 100 + i) * 1337,
        )
        for i in range(n_envs)
    ]

    obs_all = []
    hx_all = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed * 100 + i)
        obs_all.append(o)
        hx_all.append(agent.net.initial_state(1, device=agent.device))

    consumed = 0
    n_updates = 0
    t_start = time.time()

    metrics = {
        "steps": [], "rewards_mean": [], "rewards_std": [],
        "wall_clock": [], "policy_loss": [], "value_loss": [],
        "entropy": [], "kl_approx": [],
    }

    next_eval = EVAL_INTERVAL
    rollout_len = PPO_ROLLOUT // n_envs

    while consumed < TOTAL_STEPS:
        # Collect rollouts from all envs
        all_obs, all_acts, all_rews, all_lps, all_vals, all_dones = \
            [], [], [], [], [], []
        all_boot_obs = []
        all_init_hx_h, all_init_hx_c = [], []

        for env_id in range(n_envs):
            obs_buf, act_buf, rew_buf, lp_buf, val_buf, done_buf = \
                [], [], [], [], [], []

            hx_init = (hx_all[env_id][0].clone(), hx_all[env_id][1].clone())
            all_init_hx_h.append(hx_init[0].squeeze(0).cpu().numpy())
            all_init_hx_c.append(hx_init[1].squeeze(0).cpu().numpy())

            obs = obs_all[env_id]
            hx = hx_all[env_id]

            for _ in range(rollout_len):
                action, lp, val, hx_new = agent.select_action(obs, hx)

                obs_buf.append(obs.copy())
                act_buf.append(action)
                lp_buf.append(lp)
                val_buf.append(val)

                next_obs, reward, terminated, truncated, info = envs[env_id].step(action)
                done = terminated or truncated
                rew_buf.append(reward)
                done_buf.append(float(done))

                obs = next_obs
                hx = hx_new

                if done:
                    obs, _ = envs[env_id].reset()
                    hx = agent.net.initial_state(1, device=agent.device)

            obs_all[env_id] = obs
            hx_all[env_id] = hx
            all_boot_obs.append(obs.copy())

            all_obs.append(np.array(obs_buf, dtype=np.float32))
            all_acts.append(np.array(act_buf, dtype=np.int64))
            all_rews.append(np.array(rew_buf, dtype=np.float32))
            all_lps.append(np.array(lp_buf, dtype=np.float32))
            all_vals.append(np.array(val_buf, dtype=np.float32))
            all_dones.append(np.array(done_buf, dtype=np.float32))

        # Build batch
        batch = {
            "obs":           torch.FloatTensor(np.stack(all_obs)).to(agent.device),
            "actions":       torch.LongTensor(np.stack(all_acts)).to(agent.device),
            "rewards":       torch.FloatTensor(np.stack(all_rews)).to(agent.device),
            "log_probs":     torch.FloatTensor(np.stack(all_lps)).to(agent.device),
            "values":        torch.FloatTensor(np.stack(all_vals)).to(agent.device),
            "dones":         torch.FloatTensor(np.stack(all_dones)).to(agent.device),
            "bootstrap_obs": torch.FloatTensor(np.stack(all_boot_obs)).to(agent.device),
        }

        # Build initial_hx
        hx_h = np.stack(all_init_hx_h)
        hx_c = np.stack(all_init_hx_c)
        batch["initial_hx"] = (
            torch.FloatTensor(hx_h).permute(1, 0, 2).to(agent.device),
            torch.FloatTensor(hx_c).permute(1, 0, 2).to(agent.device),
        )

        m = agent.update_batched(batch)

        steps_in_batch = n_envs * rollout_len
        consumed += steps_in_batch
        n_updates += 1

        # Periodic evaluation
        if consumed >= next_eval:
            elapsed = time.time() - t_start
            mean_r, std_r = evaluate_agent(agent, config, EVAL_EPISODES)

            metrics["steps"].append(consumed)
            metrics["rewards_mean"].append(mean_r)
            metrics["rewards_std"].append(std_r)
            metrics["wall_clock"].append(elapsed)
            metrics["policy_loss"].append(m["policy_loss"])
            metrics["value_loss"].append(m["value_loss"])
            metrics["entropy"].append(m["entropy"])
            metrics["kl_approx"].append(m.get("approx_kl", 0.0))

            throughput = consumed / max(elapsed, 1e-6)
            print(f"  [{consumed:>8,}/{TOTAL_STEPS:,}] "
                  f"R={mean_r:>+10.1f} +/- {std_r:>7.1f} | "
                  f"PL={m['policy_loss']:.3f} | "
                  f"Ent={m['entropy']:.4f} | "
                  f"CF={m.get('clip_frac', 0):.3f} | "
                  f"Tput={throughput:,.0f} sps | "
                  f"t={elapsed:.1f}s", flush=True)

            next_eval = consumed + EVAL_INTERVAL

    for env in envs:
        env.close()

    total_time = time.time() - t_start
    metrics["total_time"] = total_time
    metrics["final_throughput"] = consumed / max(total_time, 1e-6)

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"ppo_seed{seed}.pt")
    agent.save(model_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_full_experiment():
    """Run the full IMPALA V3 vs PPO comparison experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Use shorter episodes for faster training (10s episodes)
    config = {
        **CONFIG,
        "arrival_rate": 1000,
        "episode_ms": 10_000,           # 10s episodes instead of 60s
        "decision_interval_ms": 10,
    }

    all_results = {"impala_v3": {}, "ppo": {}}

    seeds = [42][:NUM_SEEDS]

    print("\n" + "=" * 60)
    print("  IMPALA V3 vs PPO — Fair Comparison Experiment")
    print(f"  Total steps: {TOTAL_STEPS:,} | Seeds: {seeds}")
    print(f"  Environment: BatchingEnvV2 @ 1000 req/s")
    print("=" * 60)

    # Train IMPALA V3 across seeds
    for seed in seeds:
        metrics = train_impala_v3(seed, config)
        all_results["impala_v3"][str(seed)] = metrics

    # Train PPO across seeds
    for seed in seeds:
        metrics = train_ppo(seed, config)
        all_results["ppo"][str(seed)] = metrics

    # Save raw results
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")

    # Convert to serialisable format
    serialisable = {}
    for algo, seeds_dict in all_results.items():
        serialisable[algo] = {}
        for s, m in seeds_dict.items():
            serialisable[algo][s] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in m.items()
            }

    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"\n  Raw results saved → {results_path}")

    # Generate plots
    generate_all_plots(all_results)

    # Print summary
    print_final_summary(all_results)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(results: dict):
    """Generate comprehensive comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor('#0d1117')

    colours = {"impala_v3": "#58a6ff", "ppo": "#f78166"}
    labels  = {"impala_v3": "IMPALA V3", "ppo": "PPO"}

    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # ── Plot 1: Reward vs Training Steps ──────────────────────────────────
    ax = axes[0, 0]
    for algo in ["impala_v3", "ppo"]:
        all_steps = []
        all_rewards = []
        for seed_key, m in results[algo].items():
            all_steps.append(m["steps"])
            all_rewards.append(m["rewards_mean"])

        # Interpolate to common x-axis
        if all_steps:
            min_len = min(len(s) for s in all_steps)
            steps_arr = np.array(all_steps[0][:min_len])
            rewards_arr = np.array([r[:min_len] for r in all_rewards])
            mean_r = rewards_arr.mean(axis=0)
            std_r = rewards_arr.std(axis=0)

            ax.plot(steps_arr, mean_r, color=colours[algo], label=labels[algo], linewidth=2)
            ax.fill_between(steps_arr, mean_r - std_r, mean_r + std_r,
                          alpha=0.2, color=colours[algo])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward vs Training Steps")
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ── Plot 2: Reward vs Wall-Clock Time ─────────────────────────────────
    ax = axes[0, 1]
    for algo in ["impala_v3", "ppo"]:
        all_wc = []
        all_rewards = []
        for seed_key, m in results[algo].items():
            all_wc.append(m["wall_clock"])
            all_rewards.append(m["rewards_mean"])

        if all_wc:
            min_len = min(len(s) for s in all_wc)
            wc_arr = np.array(all_wc[0][:min_len])
            rewards_arr = np.array([r[:min_len] for r in all_rewards])
            mean_r = rewards_arr.mean(axis=0)
            std_r = rewards_arr.std(axis=0)

            ax.plot(wc_arr, mean_r, color=colours[algo], label=labels[algo], linewidth=2)
            ax.fill_between(wc_arr, mean_r - std_r, mean_r + std_r,
                          alpha=0.2, color=colours[algo])

    ax.set_xlabel("Wall-Clock Time (s)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward vs Wall-Clock Time")
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ── Plot 3: Policy Loss ───────────────────────────────────────────────
    ax = axes[0, 2]
    for algo in ["impala_v3", "ppo"]:
        all_pl = []
        all_steps = []
        for seed_key, m in results[algo].items():
            all_steps.append(m["steps"])
            all_pl.append(m["policy_loss"])

        if all_pl:
            min_len = min(len(s) for s in all_pl)
            steps_arr = np.array(all_steps[0][:min_len])
            pl_arr = np.array([p[:min_len] for p in all_pl])
            mean_pl = pl_arr.mean(axis=0)

            ax.plot(steps_arr, mean_pl, color=colours[algo], label=labels[algo], linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss Convergence")
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ── Plot 4: Entropy ───────────────────────────────────────────────────
    ax = axes[1, 0]
    for algo in ["impala_v3", "ppo"]:
        all_ent = []
        all_steps = []
        for seed_key, m in results[algo].items():
            all_steps.append(m["steps"])
            all_ent.append(m["entropy"])

        if all_ent:
            min_len = min(len(s) for s in all_ent)
            steps_arr = np.array(all_steps[0][:min_len])
            ent_arr = np.array([e[:min_len] for e in all_ent])
            mean_ent = ent_arr.mean(axis=0)

            ax.plot(steps_arr, mean_ent, color=colours[algo], label=labels[algo], linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy Throughout Training")
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ── Plot 5: Stability (per-seed reward variance) ──────────────────────
    ax = axes[1, 1]
    for algo in ["impala_v3", "ppo"]:
        all_std = []
        all_steps = []
        for seed_key, m in results[algo].items():
            all_steps.append(m["steps"])
            all_std.append(m["rewards_std"])

        if all_std:
            min_len = min(len(s) for s in all_std)
            steps_arr = np.array(all_steps[0][:min_len])
            std_arr = np.array([s[:min_len] for s in all_std])
            mean_std = std_arr.mean(axis=0)

            ax.plot(steps_arr, mean_std, color=colours[algo], label=labels[algo], linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward Std (across eval episodes)")
    ax.set_title("Stability / Variance")
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ── Plot 6: Throughput Bar Chart ──────────────────────────────────────
    ax = axes[1, 2]
    throughputs = {}
    for algo in ["impala_v3", "ppo"]:
        tputs = [m["final_throughput"] for m in results[algo].values()]
        throughputs[algo] = (np.mean(tputs), np.std(tputs))

    bar_names = [labels[a] for a in throughputs]
    bar_means = [throughputs[a][0] for a in throughputs]
    bar_stds  = [throughputs[a][1] for a in throughputs]
    bar_colors = [colours[a] for a in throughputs]

    bars = ax.bar(bar_names, bar_means, yerr=bar_stds, capsize=8,
                  color=bar_colors, edgecolor='#30363d', linewidth=1.5)
    ax.set_ylabel("Steps / Second")
    ax.set_title("Training Throughput")
    ax.grid(True, alpha=0.15, color='#484f58', axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, bar_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,.0f}', ha='center', va='bottom', color='#c9d1d9',
                fontweight='bold')

    fig.suptitle("IMPALA V3 vs PPO — CDN Batching @ 1000 req/s",
                 fontsize=16, fontweight='bold', color='#e6edf3')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(RESULTS_DIR, "impala_vs_ppo_comparison.png")
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Comparison plots saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(results: dict):
    """Print a formatted comparison summary."""
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")

    for algo in ["impala_v3", "ppo"]:
        label = "IMPALA V3" if algo == "impala_v3" else "PPO"
        final_rewards = []
        final_times = []
        final_throughputs = []

        for seed_key, m in results[algo].items():
            if m["rewards_mean"]:
                final_rewards.append(m["rewards_mean"][-1])
            final_times.append(m["total_time"])
            final_throughputs.append(m["final_throughput"])

        print(f"\n  {label}:")
        if final_rewards:
            print(f"    Final Reward:    {np.mean(final_rewards):>+10.1f} ± {np.std(final_rewards):.1f}")
        print(f"    Training Time:   {np.mean(final_times):>10.1f}s ± {np.std(final_times):.1f}s")
        print(f"    Throughput:      {np.mean(final_throughputs):>10,.0f} ± {np.std(final_throughputs):,.0f} steps/s")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_experiment()
