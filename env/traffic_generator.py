"""
env/traffic_generator.py

Multi-pattern request traffic generator for simulating diverse arrival
processes.  Supports 5 traffic modes to expose the RL policy to realistic
batching conditions at up to ~10k req/s equivalent simulation rate.

Modes
-----
steady      – constant-rate Poisson arrivals
burst       – periodic 5× traffic spikes (200 ms every ~2 s)
oscillating – sinusoidal rate modulation (0.3×–2.5×)
spike       – sudden 10× spike for 200 ms then baseline
mixed       – randomly switches between the other four modes
"""

import numpy as np


class TrafficGenerator:
    """Generates request arrival counts for a given time interval.

    Parameters
    ----------
    mode : str
        One of ``'steady'``, ``'burst'``, ``'oscillating'``,
        ``'spike'``, ``'mixed'``.
    base_rate : float
        Base arrival rate in requests / second.
    seed : int | None
        RNG seed for reproducibility.
    """

    MODES = ("steady", "burst", "oscillating", "spike", "mixed")

    def __init__(
        self,
        mode: str = "steady",
        base_rate: float = 1000.0,
        seed: int | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")
        self.mode = mode
        self.base_rate = base_rate
        self._rng = np.random.default_rng(seed)

        # Mixed-mode bookkeeping
        self._mixed_sub = "steady"
        self._mixed_switch_ms = 0.0

        # Spike-mode bookkeeping
        self._spike_active = False
        self._spike_start = 0.0
        self._next_spike = self._rng.exponential(5000.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, current_time_ms: float, interval_ms: float) -> int:
        """Return the number of arrivals in *interval_ms* starting at
        *current_time_ms*."""
        rate = self._effective_rate(current_time_ms)
        expected = rate * (interval_ms / 1000.0)
        return int(self._rng.poisson(max(expected, 0.0)))

    def reset(self, seed: int | None = None):
        """Reset internal state (optionally with new seed)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mixed_sub = "steady"
        self._mixed_switch_ms = 0.0
        self._spike_active = False
        self._spike_start = 0.0
        self._next_spike = self._rng.exponential(5000.0)

    # ------------------------------------------------------------------
    # Rate computation
    # ------------------------------------------------------------------

    def _effective_rate(self, t_ms: float) -> float:
        if self.mode == "steady":
            return self.base_rate

        if self.mode == "burst":
            # 5× burst for 200 ms every ~2 000 ms
            phase = t_ms % 2000.0
            return self.base_rate * (5.0 if phase < 200.0 else 0.6)

        if self.mode == "oscillating":
            # Sinusoidal: period 3 s, multiplier ∈ [0.3, 2.5]
            phase = (t_ms / 3000.0) * 2.0 * np.pi
            return self.base_rate * (1.4 + 1.1 * np.sin(phase))

        if self.mode == "spike":
            return self._spike_rate(t_ms)

        if self.mode == "mixed":
            return self._mixed_rate(t_ms)

        return self.base_rate

    def _spike_rate(self, t_ms: float) -> float:
        if self._spike_active:
            if t_ms - self._spike_start > 200.0:
                self._spike_active = False
                self._next_spike = t_ms + self._rng.exponential(5000.0)
                return self.base_rate
            return self.base_rate * 10.0
        if t_ms >= self._next_spike:
            self._spike_active = True
            self._spike_start = t_ms
            return self.base_rate * 10.0
        return self.base_rate

    def _mixed_rate(self, t_ms: float) -> float:
        if t_ms >= self._mixed_switch_ms:
            self._mixed_sub = str(
                self._rng.choice(["steady", "burst", "oscillating", "spike"])
            )
            self._mixed_switch_ms = t_ms + self._rng.uniform(2000, 5000)
        saved = self.mode
        self.mode = self._mixed_sub
        rate = self._effective_rate(t_ms)
        self.mode = saved
        return rate
