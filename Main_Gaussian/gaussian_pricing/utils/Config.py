# -*- coding: utf-8 -*-
"""
Global parameters for the Gaussian-feature online pricing benchmark.

SNR tiers, feature-sampling defaults, optimizer constraints, and schedule
precomputation used by gaussian_pricing.experiment.
"""
import math

# ========== Optimizer constraints ==========
W = 1e4
L0 = 0.1
C = 100

# ========== SNR tiers ==========
SNR_TIERS = ["low", "mid", "high"]
snr_configs = [
    {"name": "low",  "a_val": 0.5, "b_val": 5.0, "target_SNR": 1e-1},
    {"name": "mid",  "a_val": 0.5, "b_val": 5.0, "target_SNR": 1.0},
    {"name": "high", "a_val": 0.5, "b_val": 5.0, "target_SNR": 10.0},
]

# Samples for lambda_x calibration during SNR noise-scale calibration (n=1, d=3**6)
n_samples_for_snr_calib = max(5000, 3 * 3**6)

# ========== Feature sampling (truncated normal) ==========
feature_mu = 0
feature_sigma = 15


def get_p_and_precomputed(q_val, d_val):
    """
    Given q and feature dimension d_val, return conjugate p and schedule constants.
    When q=inf, p=1 (ell1).
    """
    if getattr(q_val, "__float__", None) is not None and math.isinf(float(q_val)):
        p_val = 1.0
    else:
        p_val = float(q_val) / (float(q_val) - 1)
    mfirst = math.ceil(d_val ** (1 / p_val - 1 / 2))
    exptime = math.ceil(d_val ** (1 / p_val - 1 / 2)) ** 2
    end_value = 2 * mfirst ** 2
    return p_val, exptime, end_value


def get_snr_config(snr_tier):
    """Return environment parameter dict for the given SNR tier."""
    for env in snr_configs:
        if env["name"] == snr_tier:
            return env
    return snr_configs[1]
