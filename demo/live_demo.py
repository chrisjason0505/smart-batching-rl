"""
demo/live_demo.py

Real-time matplotlib animation of the trained IMPALA agent operating the
batching environment.  Shows four subplots:

    1. Queue size over time
    2. Oldest-request wait (ms) with SLA line
    3. Cumulative reward
    4. Action stream (Wait / Serve markers)

Usage (standalone):
    python -m demo.live_demo --model models/best/best_model.pt

Called from run_all.py:
    from demo.live_demo import build_demo
    fig, anim = build_demo(model_path, interval_ms=50)
    plt.show()
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def build_demo(model_path: str, interval_ms: int = 50, max_frames: int = 800):
    """Build a FuncAnimation for the trained agent.

    Parameters
    ----------
    model_path : str
        Path to a saved IMPALA .pt checkpoint.
    interval_ms : int
        Milliseconds between animation frames.
    max_frames : int
        Maximum number of frames (≈ env steps) to animate.

    Returns
    -------
    fig : matplotlib.figure.Figure
    anim : matplotlib.animation.FuncAnimation
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    from env.batching_env_v2 import BatchingEnvV2
    from agent.impala_v3 import IMPALAV3Agent
    from config import CONFIG

    env   = BatchingEnvV2(config=CONFIG, traffic_mode="steady")
    agent = IMPALAV3Agent.load(model_path)

    obs, _ = env.reset(seed=0)
    sla_ms = CONFIG["max_latency_ms"]

    # ── Data buffers ──────────────────────────────────────────────────────
    times:      list[int]   = []
    queue_sizes: list[int]  = []
    oldest_waits: list[float] = []
    cum_rewards: list[float] = []
    actions_log: list[int]  = []

    cumulative_reward = 0.0
    step_idx = 0

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("IMPALA Agent — Live Batching Demo", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#1e1e2f")

    for ax in axes.flat:
        ax.set_facecolor("#2b2b3d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#555")

    ax_q, ax_w, ax_r, ax_a = axes.flat
    ax_q.set_title("Queue Size")
    ax_q.set_ylabel("Pending")
    ax_w.set_title("Oldest Wait (ms)")
    ax_w.set_ylabel("ms")
    ax_r.set_title("Cumulative Reward")
    ax_r.set_ylabel("Reward")
    ax_a.set_title("Actions (0=Wait, 1=Serve)")
    ax_a.set_ylabel("Action")

    for ax in axes.flat:
        ax.set_xlabel("Step")

    line_q, = ax_q.plot([], [], color="#00d4ff", linewidth=1.2)
    line_w, = ax_w.plot([], [], color="#ff6b6b", linewidth=1.2)
    sla_line = ax_w.axhline(sla_ms, color="#ff0", linewidth=0.8, linestyle="--", label=f"SLA={sla_ms}ms")
    ax_w.legend(loc="upper left", fontsize=8, facecolor="#2b2b3d", edgecolor="#555", labelcolor="white")
    line_r, = ax_r.plot([], [], color="#4ecdc4", linewidth=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Animation functions ───────────────────────────────────────────────

    def init():
        line_q.set_data([], [])
        line_w.set_data([], [])
        line_r.set_data([], [])
        return line_q, line_w, line_r

    def animate(frame):
        nonlocal obs, cumulative_reward, step_idx

        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        done = terminated or truncated

        times.append(step_idx)
        queue_sizes.append(int(obs[0]))
        oldest_waits.append(float(obs[1]))
        cum_rewards.append(cumulative_reward)
        actions_log.append(action)

        obs = next_obs
        step_idx += 1

        if done:
            obs, _ = env.reset()

        # Update plots
        line_q.set_data(times, queue_sizes)
        line_w.set_data(times, oldest_waits)
        line_r.set_data(times, cum_rewards)

        # Action scatter (rebuild each frame for simplicity)
        ax_a.cla()
        ax_a.set_facecolor("#2b2b3d")
        ax_a.set_title("Actions (0=Wait, 1=Serve)")
        ax_a.title.set_color("white")
        ax_a.set_ylabel("Action")
        ax_a.yaxis.label.set_color("white")
        ax_a.set_xlabel("Step")
        ax_a.xaxis.label.set_color("white")
        ax_a.tick_params(colors="white")
        for spine in ax_a.spines.values():
            spine.set_color("#555")

        # Show last 200 actions
        window = 200
        t_win = times[-window:]
        a_win = actions_log[-window:]
        serve_t = [t for t, a in zip(t_win, a_win) if a == 1]
        serve_a = [1] * len(serve_t)
        wait_t  = [t for t, a in zip(t_win, a_win) if a == 0]
        wait_a  = [0] * len(wait_t)
        ax_a.scatter(wait_t, wait_a, color="#888", s=4, alpha=0.5, label="Wait")
        ax_a.scatter(serve_t, serve_a, color="#ff6b6b", s=8, alpha=0.8, label="Serve")
        ax_a.set_ylim(-0.3, 1.3)
        if t_win:
            ax_a.set_xlim(t_win[0], t_win[-1] + 1)
        ax_a.legend(loc="center right", fontsize=7, facecolor="#2b2b3d", edgecolor="#555", labelcolor="white")

        # Auto-scale other axes
        for ax, data_y in [(ax_q, queue_sizes), (ax_w, oldest_waits), (ax_r, cum_rewards)]:
            if data_y:
                ax.set_xlim(0, max(times[-1], 1))
                y_min, y_max = min(data_y), max(data_y)
                margin = max((y_max - y_min) * 0.1, 1)
                ax.set_ylim(y_min - margin, y_max + margin)

        return line_q, line_w, line_r

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=max_frames, interval=interval_ms, blit=False, repeat=False,
    )

    return fig, anim


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(ROOT, "experiments", "results", "impala_v3_seed42.pt")

    fig, anim = build_demo(args.model, interval_ms=50)
    plt.show()
