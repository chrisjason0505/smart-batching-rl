# config.py
# Central configuration for the Dynamic Request Batching RL project

CONFIG = {
    # Environment
    "max_batch_size": 100,       # Maximum number of requests in a batch
    "max_latency_ms": 500,       # SLA: oldest request must not exceed this (ms)
    "arrival_rate": 1000,        # Base Poisson arrival rate (requests/second)
    "episode_ms": 60_000,        # Episode length in milliseconds (60 seconds)
    "decision_interval_ms": 10,  # How often the agent makes a decision (ms)

    # Reward shaping
    "alpha": 50.0,   # Reward multiplier for batch size (efficiency)
    "beta": 0.05,    # Penalty multiplier for oldest_wait_ms (latency)
    "gamma": 0.1,    # Penalty for idle (no requests, action=Wait)

    # SLA
    "sla_penalty": -50.0,  # Penalty when oldest request exceeds max_latency_ms

    # Traffic variation (time-of-day)
    "peak_hours": (8, 18),       # 08:00–18:00 → peak traffic window
    "peak_multiplier": 1.5,      # Lambda multiplier during peak hours
    "offpeak_multiplier": 0.5,   # Lambda multiplier during off-peak hours
}


# ───────────────────────────────────────────────────────────────────────────
# Experiment presets
# ───────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    "low_load": {
        **CONFIG,
        "arrival_rate": 5,
    },
    "standard": {
        **CONFIG,
        "arrival_rate": 10,
    },
    "high_load": {
        **CONFIG,
        "arrival_rate": 50,
    },
}


# ───────────────────────────────────────────────────────────────────────────
# IMPALA V2 training hyper-parameters
# ───────────────────────────────────────────────────────────────────────────

IMPALA_V2_CONFIG = {
    # Architecture
    "obs_dim":       11,
    "act_dim":       2,
    "lstm_hidden":   128,

    # Parallel actors
    "num_actors":    8,
    "unroll_length": 32,

    # Learner
    "learner_batch_size": 512,
    "lr":                 5e-4,
    "discount":           0.99,
    "entropy_coeff":      0.01,
    "value_coeff":        0.5,
    "max_grad_norm":      10.0,
    "rho_bar":            1.0,
    "c_bar":              1.0,

    # Actor → learner sync
    "actor_weight_sync_interval": 20,

    # Training budget
    "parallel_steps":  200_000,
    "finetune_steps":  50_000,
}


# ───────────────────────────────────────────────────────────────────────────
# DQN baseline hyper-parameters
# ───────────────────────────────────────────────────────────────────────────

DQN_CONFIG = {
    "obs_dim":       11,
    "act_dim":       2,
    "hidden":        128,
    "lr":            1e-3,
    "discount":      0.99,
    "buffer_size":   100_000,
    "batch_size":    64,
    "eps_start":     1.0,
    "eps_end":       0.05,
    "eps_decay":     10_000,
    "target_update": 1_000,
}


# ───────────────────────────────────────────────────────────────────────────
# PPO baseline hyper-parameters
# ───────────────────────────────────────────────────────────────────────────

PPO_CONFIG = {
    "obs_dim":             11,
    "act_dim":             2,
    "hidden":              128,
    "lr":                  3e-4,
    "gamma":               0.99,
    "gae_lambda":          0.95,
    "clip_eps":            0.2,
    "entropy_coeff":       0.01,
    "entropy_coeff_final": 0.001,
    "value_coeff":         0.5,
    "max_grad_norm":       0.5,
    "n_epochs":            4,
    "n_minibatches":       4,
    "clip_value":          True,
    "target_kl":           0.015,
    "total_steps":         250_000,
}


# ───────────────────────────────────────────────────────────────────────────
# IMPALA V3 (optimised) hyper-parameters
# ───────────────────────────────────────────────────────────────────────────

IMPALA_V3_CONFIG = {
    "obs_dim":             11,
    "act_dim":             2,
    "hidden":              128,
    "lr":                  5e-4,
    "lr_final":            1e-5,
    "discount":            0.99,
    "entropy_coeff":       0.015,
    "entropy_coeff_final": 0.001,
    "value_coeff":         0.5,
    "rho_bar":             1.0,
    "c_bar":               1.0,
    "max_grad_norm":       10.0,
    "advantage_norm":      True,
    "reward_norm":         True,
    "aux_value_l2":        1e-4,
    "lag_penalty_coeff":   0.1,
    "num_actors":          6,
    "unroll_length":       32,
    "learner_batch_size":  256,
    "total_steps":         250_000,
}


# ───────────────────────────────────────────────────────────────────────────
# CDN-specific environment config (1000 req/s)
# ───────────────────────────────────────────────────────────────────────────

CDN_CONFIG = {
    **CONFIG,
    "arrival_rate":     1000,        # 1000 req/s CDN simulation
    "max_batch_size":   100,
    "max_latency_ms":   500,
    "episode_ms":       60_000,
    "decision_interval_ms": 10,
}
