# -*- coding: utf-8 -*-
"""
Configuration for the q-sensitivity online pricing benchmark.

Covers optimizer constraints, SNR tier(s), feature-sampling defaults, the swept
regularizer geometry q in {2, 4, 1+ln(d)}, hyper-parameter grids, and per-mode
feature-generation settings (Gaussian l2-ball vs l4 heavy-tail).
"""
from __future__ import annotations

import math

# ========== Problem dimension (matches TOY_D in the experiment runner) ==========
D = 3 ** 6

# ========== Optimizer constraints ==========
W = 1e4
L0 = 0.1
C = 100

# ========== SNR ==========
SNR_TIERS = ["low"]

# Per feature mode (low tier only; mode-specific target SNR)
LOW_SNR_BY_MODE: dict[str, float] = {
    "Gaussian": 1.0,
    "l4_heavy_tail": 1e-3,
}

n_samples_for_snr_calib = max(5000, 3 * D)

# ========== Feature sampling ==========
feature_mu = 0
feature_sigma = 15
l4_feature_mean_shift = 0.05

# ========== Regularizer geometry sweep ==========
# Third tier q = 1 + ln(d) (natural log of feature dimension D). Conjugate p = q/(q-1).
Q_THEORY_LN: float = 1.0 + math.log(float(D))
Q_LIST: list[float] = [2.0, 4.0, Q_THEORY_LN]
Q_LABELS: list[str] = ["2", "4", "1+ln(d)"]

# Exploration length and eta/eps scaling use a fixed reference q (independent of swept q).
REFERENCE_Q: float = 2.0

# Hyper-parameter grids (unscaled; scaled by d**(1/p - 1/2) per q in the runner).
ETA_GRID: list[float] = [5e-3]
EPS_GRID: list[float] = [0.5]  # PGD only

# ========== Feature-generation modes ==========
FEATURE_MODES: list[str] = ["Gaussian", "l4_heavy_tail"]

# Gaussian / l2-ball features
GAUSSIAN_FIXED: dict = {
    "feature_exponent": 2.0,
    "a_val": 0.5,
    "b_val": 5.0,
    "c_val": float(C),
    "feature_mu": float(feature_mu),
    "feature_sigma": float(feature_sigma),
    "theta_seed": 4242,
    "theta_weak_lo": 0.001,
    "theta_weak_hi": 0.015,
    "theta_strong_frac": 0.15,
    "theta_strong_lo": 1.0,
    "theta_strong_hi": 2.0,
    "gamma_diag": 5.0,
    "price_lo": 30.0,
    "price_hi": 50.0,
    "n_price": 401,
    "n_price_experiment": 6,
}

# l4 heavy-tail features
L4_FIXED: dict = {
    "a_val": 40.0,
    "b_val": 50.0,
    "c_val": 0.25e4,
    "theta_sparse_indices": [0, 1, 2, 3, 4],
    "theta_sparse_values": [50.0, 20.0, 0.3, 0.2, 0.1],
    "gamma_lo": 0.1,
    "gamma_hi": 0.1,
    "price_hi_margin": 0.5,
    "n_price": 501,
    "n_price_experiment": 11,
    "pmax": None,
}


def get_p_and_precomputed(q_val, d_val):
    """
    Given q and feature dimension d_val, return conjugate p and schedule constants.
    When q=inf, p=1 (ell1).
    """
    if getattr(q_val, "__float__", None) is not None and math.isinf(float(q_val)):
        p_val = 1.0
    else:
        p_val = float(q_val) / (float(q_val) - 1)
    mfirst = 4 * math.ceil(d_val ** (1 / p_val - 1 / 2))
    exptime = 4 * math.ceil(d_val ** (1 / p_val - 1 / 2)) ** 2
    end_value = 2 * mfirst ** 2
    return p_val, exptime, end_value


def conjugate_p(q_val: float) -> float:
    """p = q / (q - 1); q = inf -> p = 1."""
    if math.isinf(float(q_val)):
        return 1.0
    return float(q_val) / (float(q_val) - 1.0)


def low_snr_for_mode(mode: str) -> float:
    """Target low-SNR for a feature mode."""
    if mode not in LOW_SNR_BY_MODE:
        raise ValueError(
            "Unknown feature mode {!r}; expected one of {}".format(mode, FEATURE_MODES)
        )
    return float(LOW_SNR_BY_MODE[mode])


def get_mode_config(mode: str) -> dict:
    """Return the fixed config dict for a feature mode."""
    if mode == "Gaussian":
        return GAUSSIAN_FIXED
    if mode == "l4_heavy_tail":
        return L4_FIXED
    raise ValueError("Unknown feature mode {!r}; expected one of {}".format(mode, FEATURE_MODES))
