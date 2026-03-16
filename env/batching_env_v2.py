"""
env/batching_env_v2.py

Extended batching environment with 11-dimensional observations and
configurable traffic patterns for high-throughput IMPALA training.

Observation (11-dim float32)
----------------------------
 0  pending               – queue length
 1  oldest_wait_ms        – wait time of oldest request (ms)
 2  arrival_rate          – current effective arrival rate
 3  since_serve_ms        – time since last serve
 4  fill_ratio            – pending / max_batch  ∈ [0, 1]
 5  time_of_day           – normalised hour       ∈ [0, 1]
 6  rolling_arrival_rate  – EMA of recent instantaneous rates
 7  avg_interarrival      – mean inter-arrival (last 20 events)
 8  recent_avg_batch      – rolling mean of last 10 batch sizes
 9  latency_budget_ratio  – (max_lat − oldest_wait) / max_lat
10  burstiness            – CV of recent inter-arrival counts

Action  Discrete(2):  0 = Wait, 1 = Serve
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from collections import deque

from config import CONFIG
from env.traffic_generator import TrafficGenerator


class BatchingEnvV2(gymnasium.Env):
    """Extended batching environment with temporal features."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict | None = None,
        traffic_mode: str = "steady",
        traffic_seed: int | None = None,
    ):
        super().__init__()
        cfg = config or CONFIG

        # ── Config ────────────────────────────────────────────────────────
        self.max_batch_size    = cfg["max_batch_size"]
        self.max_latency_ms    = cfg["max_latency_ms"]
        self.base_arrival_rate = cfg["arrival_rate"]
        self.episode_ms        = cfg["episode_ms"]
        self.decision_interval = cfg["decision_interval_ms"]

        self.alpha         = cfg["alpha"]
        self.beta          = cfg["beta"]
        self.gamma_penalty = cfg["gamma"]
        self.sla_penalty   = cfg["sla_penalty"]

        self.peak_hours         = cfg["peak_hours"]
        self.peak_multiplier    = cfg["peak_multiplier"]
        self.offpeak_multiplier = cfg["offpeak_multiplier"]

        self._max_steps = self.episode_ms // self.decision_interval

        # ── Traffic generator ─────────────────────────────────────────────
        self.traffic_mode = traffic_mode
        self._traffic = TrafficGenerator(
            mode=traffic_mode,
            base_rate=self.base_arrival_rate,
            seed=traffic_seed,
        )

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(2)

        obs_high = np.array([
            self.max_batch_size,              # 0  pending
            float("inf"),                     # 1  oldest_wait_ms
            float("inf"),                     # 2  arrival_rate
            float("inf"),                     # 3  since_serve_ms
            1.0,                              # 4  fill_ratio
            1.0,                              # 5  time_of_day
            float("inf"),                     # 6  rolling_arrival_rate
            float("inf"),                     # 7  avg_interarrival
            float(self.max_batch_size),       # 8  recent_avg_batch
            1.0,                              # 9  latency_budget_ratio
            float("inf"),                     # 10 burstiness
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.zeros(11, dtype=np.float32),
            high=obs_high,
            dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────────
        self._queue: list[float] = []
        self._current_time_ms = 0.0
        self._last_serve_ms   = 0.0
        self._step_count      = 0
        self._start_hour      = 0.0
        self._served_latencies: list[float] = []
        self._rng = np.random.default_rng()

        # Temporal tracking
        self._arrival_times: deque = deque(maxlen=100)
        self._ema_rate = float(self.base_arrival_rate)
        self._recent_batch_sizes: deque = deque(maxlen=10)
        self._recent_arrivals: deque = deque(maxlen=20)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._traffic.reset(seed=seed)

        self._queue = []
        self._current_time_ms = 0.0
        self._last_serve_ms   = 0.0
        self._step_count      = 0
        self._served_latencies = []
        self._start_hour = float(self._rng.uniform(0.0, 24.0))

        self._arrival_times.clear()
        self._ema_rate = float(self.base_arrival_rate)
        self._recent_batch_sizes.clear()
        self._recent_arrivals.clear()

        return self._get_obs(), {}

    def step(self, action: int):
        reward = 0.0

        # ── 1. Process action ─────────────────────────────────────────────
        if action == 1 and len(self._queue) > 0:
            batch_size = min(len(self._queue), self.max_batch_size)
            served = self._queue[:batch_size]
            self._queue = self._queue[batch_size:]

            latencies = [self._current_time_ms - t for t in served]
            self._served_latencies.extend(latencies)
            mean_wait = float(np.mean(latencies))

            efficiency = self.alpha * (batch_size ** 1.5) / (self.max_batch_size ** 0.5)
            reward += efficiency - (self.beta * mean_wait)

            self._last_serve_ms = self._current_time_ms
            self._recent_batch_sizes.append(batch_size)

        elif action == 0 and len(self._queue) > 0:
            if len(self._queue) >= self.max_batch_size:
                reward -= self.gamma_penalty * 2.0

        elif action == 0 and len(self._queue) == 0:
            reward += 0.1

        # ── 2. SLA check ─────────────────────────────────────────────────
        if len(self._queue) > 0:
            oldest_wait = self._current_time_ms - self._queue[0]
            if oldest_wait > self.max_latency_ms:
                # Soft penalty proportional to lateness (scaled down to avoid explosions)
                overage = oldest_wait - self.max_latency_ms
                reward += (self.sla_penalty / 10.0) * (1.0 + min(overage / 1000.0, 5.0))

        # ── 3. Advance clock ─────────────────────────────────────────────
        self._current_time_ms += self.decision_interval
        self._step_count += 1

        # ── 4. Generate arrivals via traffic generator ────────────────────
        n_arrivals = self._traffic.generate(
            self._current_time_ms, self.decision_interval,
        )
        for _ in range(n_arrivals):
            t = self._current_time_ms - self._rng.uniform(0, self.decision_interval)
            self._queue.append(t)
            self._arrival_times.append(t)

        self._queue.sort()
        self._recent_arrivals.append(n_arrivals)

        # Update EMA rate
        interval_s = self.decision_interval / 1000.0
        instant_rate = n_arrivals / max(interval_s, 1e-6)
        self._ema_rate = 0.1 * instant_rate + 0.9 * self._ema_rate

        # ── 5. Termination ────────────────────────────────────────────────
        terminated = False
        truncated  = self._step_count >= self._max_steps

        info: dict = {}
        if terminated or truncated:
            info["mean_latency_ms"] = (
                float(np.mean(self._served_latencies))
                if self._served_latencies else 0.0
            )

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_hour(self) -> float:
        hours_elapsed = self._current_time_ms / 1_000.0 / 3_600.0
        return (self._start_hour + hours_elapsed) % 24.0

    def _effective_rate(self) -> float:
        hour = self._current_hour()
        lo, hi = self.peak_hours
        if lo <= hour < hi:
            return self.base_arrival_rate * self.peak_multiplier
        return self.base_arrival_rate * self.offpeak_multiplier

    def _get_obs(self) -> np.ndarray:
        pending = len(self._queue)
        oldest_wait = (
            self._current_time_ms - self._queue[0] if pending > 0 else 0.0
        )
        rate        = self._effective_rate()
        since_serve = self._current_time_ms - self._last_serve_ms
        fill_ratio  = min(pending / max(self.max_batch_size, 1), 1.0)
        time_of_day = self._current_hour() / 24.0

        # ── Extended features ─────────────────────────────────────────────
        rolling_rate = self._ema_rate

        # Average inter-arrival time (ms)
        if len(self._arrival_times) >= 2:
            arr_sorted = sorted(self._arrival_times)
            diffs = np.diff(arr_sorted)
            avg_interarrival = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        else:
            avg_interarrival = 0.0

        # Recent average batch size
        recent_batch = (
            float(np.mean(self._recent_batch_sizes))
            if self._recent_batch_sizes else 0.0
        )

        # Latency budget remaining
        latency_budget = max(
            0.0, (self.max_latency_ms - oldest_wait) / self.max_latency_ms
        )

        # Burstiness (coefficient of variation of recent arrival counts)
        if len(self._recent_arrivals) >= 2:
            arr = np.array(self._recent_arrivals, dtype=np.float64)
            burstiness = float(arr.std() / max(arr.mean(), 1e-6))
        else:
            burstiness = 0.0

        obs = np.array([
            pending, oldest_wait, rate, since_serve, fill_ratio, time_of_day,
            rolling_rate, avg_interarrival, recent_batch, latency_budget,
            burstiness,
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)
