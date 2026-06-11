"""
q-sensitivity benchmark for discrete online pricing: MOSEK vs PGD across the
regularizer geometry **q in {2, 4, 1+ln(d)}** (the third tier is the
dimension-dependent theory exponent), run under two feature-generation modes.

Feature modes (see config/Config.py):
  - "Gaussian"      : truncated-normal features with an l2-norm budget (l2-ball).
  - "l4_heavy_tail" : collinear + sparse-spike heavy-tailed features.

For a given feature mode the same feature trajectory X_{k,t} is shared by every
q; only the MLE / PGD regularizer (conjugate p = q/(q-1)) and the eta/eps scaling
change. The exploration schedule uses a fixed REFERENCE_Q, so differences across
q reflect the regularizer geometry alone.

Only the low-SNR tier is run, with a per-mode target SNR (Config.LOW_SNR_BY_MODE):
Gaussian (l2-ball) -> SNR=1.0, l4_heavy_tail (l4-ball) -> SNR=1e-3.

Outputs, per feature mode (low-SNR tier), written to results/<mode>/Low_SNR/:

  1. reward_by_method_param_t.csv             -> method, param, k, t, reward
  2. oracle_reward_by_t.csv                    -> k, t, oracle_reward
  3. solver_times.csv                         -> method, param, k, solve_idx, time_s
  4. cumulative_regret_by_method_param_t.csv  -> method, param, k, t, cumulative_regret
  5. mean_cumulative_regret_by_method_param_t.csv -> method, param, t, mean_cumulative_regret

It also saves a regret figure `q_sensitivity_regret.pdf` (mean cumulative regret,
one curve per (method, q, param); colors per method/q, markers per eta/eps).

The cumulative-regret CSVs (cumsum_t(oracle - reward)) are ready to plot directly.
`param` encodes q, e.g. "q=2,eta=0.005" (MOSEK) or "q=4,eta=0.005,eps=0.5" (PGD).

Dependencies: numpy, scipy, cvxpy + MOSEK (cvxMLEpnorm). Set QS_SMOKE=1 for a
fast smoke test.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np
from scipy.special import erf as _scipy_erf
from tqdm import tqdm

from ..config import Config as _cfg
from ..features.truncate_normal_sample_time_varying import (
    l4_heavy_tail_sample,
    truncate_normal_sample_time_varying,
)
from ..solvers.cvxMLEpnorm import cvxMLEpnorm
from ..solvers.Gradient_projected_Search import gradient_descent_projected_likelihood
from ..utils.randomP import randomP

# Repo root (two levels up from q_sensitivity/experiments/); outputs go to
# <repo_root>/results/ regardless of the current working directory.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Problem size (q-independent)
# ---------------------------------------------------------------------------
TOY_D = 3**6
TOY_T = 5000
TOY_K = 5

_n = 1
_d = TOY_D
W = _cfg.W
L0 = _cfg.L0

if os.environ.get("QS_SMOKE") == "1":
    TOY_T = 100
    TOY_K = 1

# ---------------------------------------------------------------------------
# Mutable globals configured per feature mode / per swept q
# ---------------------------------------------------------------------------
_feature_mode: str = "Gaussian"
_feature_exponent: float = 2.0
_feature_mu: float = float(_cfg.feature_mu)
_feature_sigma: float = float(_cfg.feature_sigma)
_a_val: float = 0.5
_b_val: float = 5.0
C: float = float(_cfg.C)

_theta0: np.ndarray = np.zeros((_n, _n * _d), dtype=np.float64)
_gamma0: np.ndarray = np.diag(np.full(_n, 5.0, dtype=np.float64))

PRICE_GRID: np.ndarray = np.linspace(30.0, 50.0, 401, dtype=np.float64)
PRICE_GRID_Experiment: np.ndarray = np.linspace(30.0, 50.0, 6, dtype=np.float64)
N_PRICE: int = int(PRICE_GRID.size)

# Exploration schedule from the reference q (fixed across the swept q).
_p_val: float = 2.0
_scale: float = 1.0
EXPERIMENT_TIME: int = 4
END_VALUE: int = 32

# Feature trajectory cache for the current mode (shared by every q and the oracle).
_XT_CACHE: dict[int, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Feature sampling & trajectory pre-computation (mode-aware, q-independent)
# ---------------------------------------------------------------------------
def _sample_xt_full(k_instance: int) -> np.ndarray:
    nd = _n * _d
    X_full = np.empty((int(TOY_T), nd), dtype=np.float64)
    for t in range(int(TOY_T)):
        Xt = np.zeros((_d, _n), dtype=np.float64)
        for i in range(_n):
            if _feature_mode == "l4_heavy_tail":
                Xt[:, i] = l4_heavy_tail_sample(
                    _d, int(k_instance), int(t), i,
                    mean_shift=float(_cfg.l4_feature_mean_shift),
                ).ravel()
            else:
                Xt[:, i] = truncate_normal_sample_time_varying(
                    _feature_exponent, C, _a_val, _b_val, _d,
                    int(k_instance), int(t), i,
                    mu=_feature_mu, sigma=_feature_sigma,
                ).ravel()
        X_full[t] = Xt.flatten()
    return X_full


def _collect_xt_trajectory(k_instance: int) -> tuple[np.ndarray, np.ndarray]:
    X_full = _XT_CACHE.get(int(k_instance))
    if X_full is None:
        X_full = _sample_xt_full(int(k_instance))
        _XT_CACHE[int(k_instance)] = X_full
    theta_flat = np.asarray(_theta0, dtype=np.float64).ravel()
    theta0_Xt_T = X_full @ theta_flat
    return X_full, theta0_Xt_T


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
def _expected_positive_part_vec(mu_mat: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.maximum(mu_mat, 0.0)
    s = float(sigma)
    z = mu_mat / s
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + _scipy_erf(z / math.sqrt(2.0)))
    return mu_mat * Phi + s * phi


def _oracle_expected_revenue_and_price_on_grid(
    theta0_Xt_scalar: float, noise_scale: float, price_grid: np.ndarray
) -> tuple[float, float]:
    pg = np.asarray(price_grid, dtype=np.float64).ravel()
    gamma0_scalar = float(np.asarray(_gamma0, dtype=np.float64).ravel()[0])
    mu_vec = float(theta0_Xt_scalar) - gamma0_scalar * pg
    exp_pos = _expected_positive_part_vec(mu_vec, float(noise_scale))
    rev_vec = pg * exp_pos
    j = int(np.argmax(rev_vec))
    return float(rev_vec[j]), float(pg[j])


def run_optimal_revenue_discrete(k_instance: int, noise_scale: float) -> np.ndarray:
    _, theta0_Xt_T = _collect_xt_trajectory(int(k_instance))
    Revenue_opt = np.zeros(int(TOY_T), dtype=np.float64)
    for t in range(int(TOY_T)):
        rev_t, _ = _oracle_expected_revenue_and_price_on_grid(
            float(theta0_Xt_T[t]), float(noise_scale), PRICE_GRID
        )
        Revenue_opt[t] = float(rev_t)
    return Revenue_opt


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------
def best_price_on_grid(
    theta: np.ndarray, gamma: np.ndarray, column_Xt: np.ndarray,
    price_grid: np.ndarray, n: int,
) -> np.ndarray:
    assert n == 1
    pg = np.asarray(price_grid, dtype=np.float64).ravel()
    theta_X = float(np.asarray(theta @ column_Xt, dtype=np.float64).ravel()[0])
    gamma_scalar = float(np.asarray(gamma, dtype=np.float64).ravel()[0])
    d_hat = theta_X - gamma_scalar * pg
    rev_vec = pg * np.maximum(d_hat, 0.0)
    best_j = int(np.argmax(rev_vec))
    return np.full((n, 1), float(pg[best_j]), dtype=np.float64)


# ---------------------------------------------------------------------------
# Generic online loop shared by MOSEK / PGD
# ---------------------------------------------------------------------------
def _run_online_loop(
    k_instance: int, noise_scale: float, *, retrain_fn,
    init_state: dict[str, Any] | None = None, desc: str = "online",
) -> tuple[np.ndarray, list[float]]:
    Revenue = np.zeros(TOY_T, dtype=np.float64)
    times: list[float] = []
    D: list[Any] = []
    P: list[Any] = []
    Column_X: list[Any] = []
    state: dict[str, Any] = dict(init_state) if init_state else {}
    exploration_countdown = 0

    X_full, theta0_Xt_T = _collect_xt_trajectory(int(k_instance))
    gamma_scalar = float(np.asarray(_gamma0, dtype=np.float64).ravel()[0])
    feat_dim = _n * _d

    for t in tqdm(range(TOY_T), desc=desc, leave=False):
        column_Xt = X_full[t].reshape(feat_dim, 1)
        theta0_Xt_scalar = float(theta0_Xt_T[t])

        if t <= END_VALUE:
            pt = randomP(PRICE_GRID_Experiment, _n).reshape(-1, 1)
        else:
            if math.isqrt(t) ** 2 == t:
                exploration_countdown = EXPERIMENT_TIME
            if exploration_countdown > 0:
                pt = randomP(PRICE_GRID_Experiment, _n).reshape(-1, 1)
                exploration_countdown -= 1
            else:
                theta_k = state.get("theta_k")
                gamma_k = state.get("gamma_k")
                assert theta_k is not None and gamma_k is not None
                pt = best_price_on_grid(theta_k, gamma_k, column_Xt, PRICE_GRID, _n)

        np.random.seed(int(k_instance + t))
        zt = np.random.normal(loc=0.0, scale=noise_scale, size=_n)
        pt_scalar = float(pt[0, 0])
        dt = np.maximum(
            np.array([theta0_Xt_scalar - gamma_scalar * pt_scalar], dtype=np.float64) + zt,
            0.0,
        )
        Revenue[t] = float(np.dot(pt.ravel(), dt))

        retrain_now = False
        is_initial = False
        if t <= END_VALUE:
            D.append(np.float32(dt))
            P.append(np.float32(pt))
            Column_X.append(np.float32(column_Xt))
            if t == END_VALUE:
                retrain_now = True
                is_initial = True
        else:
            if math.isqrt(t + 1) ** 2 == t + 1:
                D.append(np.float32(dt))
                P.append(np.float32(pt))
                Column_X.append(np.float32(column_Xt))
                retrain_now = True

        if retrain_now:
            t0 = time.time()
            updates = retrain_fn(D, P, Column_X, state, int(t), bool(is_initial))
            times.append(time.time() - t0)
            if updates:
                state.update(updates)
            if is_initial:
                exploration_countdown = 0

    return Revenue, times


def run_cvx_mosek_discrete(
    k_instance: int, eta_k_coeff: float, noise_scale: float,
) -> tuple[np.ndarray, list[float]]:
    def retrain_mosek(D, P, Column_X, state, t, is_initial):
        del t, is_initial, state
        explore_len = len(D)
        eta_k = float(eta_k_coeff) * (explore_len ** 0.25)
        theta_k, gamma_k = cvxMLEpnorm(D, P, Column_X, _d, _p_val, eta_k, L0, _n, W)
        return {"theta_k": theta_k, "gamma_k": gamma_k}

    return _run_online_loop(
        int(k_instance), float(noise_scale),
        retrain_fn=retrain_mosek, desc="MOSEK (k={})".format(k_instance),
    )


def run_gradient_descent_discrete(
    k_instance: int, epsilon_factor: float, eta_k_coeff: float,
    noise_scale: float, snr_tier: str = "mid",
) -> tuple[np.ndarray, list[float]]:
    init_state: dict[str, Any] = {
        "initial_point": np.full(_n * _n * _d + _n * _n, 0.1, dtype=np.float64),
    }

    def retrain_pgd(D, P, Column_X, state, t, is_initial):
        explore_len = len(D)
        eta_k = float(eta_k_coeff) * (explore_len ** 0.25)
        epsilon_k = float(epsilon_factor) * np.sqrt(explore_len)
        exptime_arg = explore_len if is_initial else int(t + 1)
        theta_k, gamma_k = gradient_descent_projected_likelihood(
            state["initial_point"],
            D, P, Column_X, epsilon_k, _d, _p_val, eta_k, L0, _n, W, exptime_arg,
            snr_tier=snr_tier,
        )
        return {
            "theta_k": theta_k,
            "gamma_k": gamma_k,
            "initial_point": np.concatenate((theta_k.ravel(), gamma_k.ravel())),
        }

    return _run_online_loop(
        int(k_instance), float(noise_scale),
        retrain_fn=retrain_pgd, init_state=init_state,
        desc="PGD (k={})".format(k_instance),
    )


# ---------------------------------------------------------------------------
# SNR calibration -> noise_scale (mode-aware feature sampling)
# ---------------------------------------------------------------------------
def _calibrate_signal_power(n_calib: int) -> tuple[float, float, float]:
    """SNR := ||theta||_2^2 * lambda_min(E[X X^T]) / sigma^2 (k=0, disjoint from runs)."""
    assert _n == 1
    N = int(n_calib)
    dim = _d * _n
    X_mat = np.empty((N, dim), dtype=np.float64)
    for i in range(N):
        Xt = np.zeros((_d, _n), dtype=np.float64)
        for ii in range(_n):
            if _feature_mode == "l4_heavy_tail":
                Xt[:, ii] = l4_heavy_tail_sample(
                    _d, 0, i, ii, mean_shift=float(_cfg.l4_feature_mean_shift)
                ).ravel()
            else:
                Xt[:, ii] = truncate_normal_sample_time_varying(
                    _feature_exponent, C, _a_val, _b_val, _d, 0, i, ii,
                    mu=_feature_mu, sigma=_feature_sigma,
                ).ravel()
        X_mat[i, :] = Xt.flatten()
    if not np.isfinite(X_mat).all():
        raise RuntimeError("_calibrate_signal_power: X_mat has non-finite entries")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Sigma_hat = (X_mat.T @ X_mat) / float(N)
    lam_min = max(float(np.linalg.eigvalsh(Sigma_hat)[0]), 0.0)
    theta_flat = np.asarray(_theta0, dtype=np.float64).ravel()
    theta_norm_sq = float(theta_flat @ theta_flat)
    signal_power = theta_norm_sq * lam_min
    return signal_power, theta_norm_sq, lam_min


# ---------------------------------------------------------------------------
# Feature-mode configuration (sets theta/gamma, prices, a/b/C, exploration)
# ---------------------------------------------------------------------------
def _build_theta_gamma(mode: str, cfg: dict) -> None:
    global _theta0, _gamma0
    nd = _n * _d
    if mode == "Gaussian":
        rng = np.random.default_rng(int(cfg["theta_seed"]))
        theta0 = rng.uniform(
            float(cfg["theta_weak_lo"]), float(cfg["theta_weak_hi"]), size=(_n, nd)
        ).astype(np.float64)
        hi_mask = rng.random(size=(_n, nd)) < float(cfg["theta_strong_frac"])
        if hi_mask.any():
            theta0[hi_mask] = rng.uniform(
                float(cfg["theta_strong_lo"]), float(cfg["theta_strong_hi"]),
                size=int(np.sum(hi_mask)),
            ).astype(np.float64)
        _theta0 = theta0
        _gamma0 = np.diag(np.full(_n, float(cfg["gamma_diag"]), dtype=np.float64))
    else:  # l4_heavy_tail: sparse theta
        theta = np.zeros((_n, nd), dtype=np.float64)
        idxs = [int(i) for i in cfg["theta_sparse_indices"]]
        vals = np.asarray(cfg["theta_sparse_values"], dtype=np.float64).ravel()
        for j, ix in enumerate(idxs):
            if not (0 <= ix < nd):
                raise ValueError("theta_sparse_indices[{}]={} out of range".format(j, ix))
            theta.ravel()[ix] = float(vals[j])
        _theta0 = theta
        _gamma0 = np.diag(
            np.linspace(float(cfg["gamma_lo"]), float(cfg["gamma_hi"]), _n, dtype=np.float64)
        )


def _build_price_grids(mode: str, cfg: dict) -> None:
    global PRICE_GRID, PRICE_GRID_Experiment, N_PRICE
    if mode == "Gaussian":
        lo = float(cfg["price_lo"])
        hi = float(cfg["price_hi"])
        n_f = max(int(cfg["n_price"]), 2)
        n_e = max(int(cfg["n_price_experiment"]), 2)
        PRICE_GRID = np.linspace(lo, hi, n_f, dtype=np.float64)
        PRICE_GRID_Experiment = np.linspace(lo, hi, n_e, dtype=np.float64)
    else:
        l1 = float(np.sum(np.abs(np.asarray(_theta0, dtype=np.float64))))
        g_min = float(np.min(np.diag(np.asarray(_gamma0, dtype=np.float64))))
        p_hi = l1 / (2.0 * max(g_min, 1e-12))
        hi = float(cfg["pmax"]) if cfg["pmax"] is not None else p_hi * (1.0 + float(cfg["price_hi_margin"]))
        n_f = max(int(cfg["n_price"]), 2)
        n_e = max(int(cfg["n_price_experiment"]), 2)
        PRICE_GRID = np.linspace(0.0, hi, n_f, dtype=np.float64)
        PRICE_GRID_Experiment = np.linspace(0.0, hi, n_e, dtype=np.float64)
    N_PRICE = int(PRICE_GRID.size)


def configure_feature_mode(mode: str) -> dict:
    """Set all mode-dependent globals (theta/gamma, prices, a/b/C, exploration)."""
    global _feature_mode, _feature_exponent, _feature_mu, _feature_sigma
    global _a_val, _b_val, C
    global EXPERIMENT_TIME, END_VALUE, _XT_CACHE

    cfg = _cfg.get_mode_config(mode)
    _feature_mode = mode
    _a_val = float(cfg["a_val"])
    _b_val = float(cfg["b_val"])
    C = float(cfg["c_val"])
    if mode == "Gaussian":
        _feature_exponent = float(cfg["feature_exponent"])
        _feature_mu = float(cfg["feature_mu"])
        _feature_sigma = float(cfg["feature_sigma"])

    _build_theta_gamma(mode, cfg)
    _build_price_grids(mode, cfg)

    # exploration schedule from the reference q (fixed across swept q)
    _, exptime, end_value = _cfg.get_p_and_precomputed(_cfg.REFERENCE_Q, _d)
    EXPERIMENT_TIME = int(exptime)
    END_VALUE = int(end_value)

    _XT_CACHE = {}  # X depends only on the mode; clear when switching modes
    set_solver_q(_cfg.REFERENCE_Q)
    return cfg


def set_solver_q(q_val: float) -> None:
    """Set the regularizer geometry (p and the eta/eps scaling) for one swept q."""
    global _p_val, _scale
    _p_val = float(_cfg.conjugate_p(q_val))
    _scale = float(_d ** (1.0 / _p_val - 1.0 / 2.0))


# ---------------------------------------------------------------------------
# Plotting: mean cumulative regret per (method, q, param)
# (style adapted from plot_q_sensitivity_with_markers_fixed.py)
# ---------------------------------------------------------------------------
# Color per (method, q): MOSEK = blue/orange/red, PGD = green/purple/brown.
_Q_PLOT_COLORS = {
    ("MOSEK", "2"): "#64B5F6",
    ("MOSEK", "4"): "#FFB74D",
    ("MOSEK", "1+ln(d)"): "#E57373",
    ("PGD", "2"): "#81C784",
    ("PGD", "4"): "#CE93D8",
    ("PGD", "1+ln(d)"): "#BCAAA4",
}
_Q_PLOT_ORDER = {"2": 0, "4": 1, "1+ln(d)": 2}
# PGD: cycle these per (q) group, ordered by (eta, eps); MOSEK uses eta-keyed
# markers (see _build_mosek_marker_map). Matches plot_q_sensitivity_with_markers_fixed.py.
_PGD_MARKERS = ["o", "^", "D", "*", ">"]


def _parse_q_from_param(param: str) -> str:
    for part in str(param).split(","):
        s = part.strip()
        if s.startswith("q="):
            return s.split("=", 1)[1].strip()
    return "?"


def _parse_eta_eps(param: str) -> tuple[float | None, float | None]:
    eta = eps = None
    for part in str(param).split(","):
        s = part.strip()
        if s.startswith("eta="):
            try:
                eta = float(s.split("=", 1)[1])
            except ValueError:
                eta = None
        elif s.startswith("eps="):
            try:
                eps = float(s.split("=", 1)[1])
            except ValueError:
                eps = None
    return eta, eps


def _q_label_for_legend(q_label: str) -> str:
    if q_label in ("1+ln(d)", "1+lnd", "1+logd"):
        return r"1+log$\kappa_\theta$"
    return q_label


def _legend_param_fragment(param: str) -> str:
    """Drop the leading q=... and render eta/eps as ring symbols (mathtext)."""
    parts_out: list[str] = []
    for part in str(param).split(","):
        s = part.strip()
        if s.startswith("q="):
            continue
        if s.startswith("eta="):
            parts_out.append(r"$\mathring{\eta}$=" + s[4:].strip())
        elif s.startswith("eps="):
            parts_out.append(r"$\mathring{\varepsilon}$=" + s[4:].strip())
        elif s:
            parts_out.append(s)
    return ",".join(parts_out)


def _plot_cumulative_regret(
    cum_curves: list[tuple[str, str, np.ndarray]], out_path: str
) -> None:
    """Plot mean (over k) cumulative regret, one line per (method, q, param)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip regret plot")
        return

    curves: list[dict[str, Any]] = []
    for method, param, cum in cum_curves:
        eta, eps = _parse_eta_eps(param)
        curves.append({
            "q": _parse_q_from_param(param),
            "method": str(method),
            "param": str(param),
            "eta": eta,
            "eps": eps,
            "mean": np.mean(np.asarray(cum, dtype=np.float64), axis=0),
        })
    if not curves:
        return

    def _sort_key(c: dict[str, Any]) -> tuple:
        m_order = 0 if c["method"] == "MOSEK" else 1 if c["method"] == "PGD" else 9
        return (
            _Q_PLOT_ORDER.get(c["q"], 9), m_order,
            float("inf") if c["eta"] is None else c["eta"],
            float("inf") if c["eps"] is None else c["eps"],
            c["param"],
        )

    curves.sort(key=_sort_key)

    # PGD markers: per q group, cycle _PGD_MARKERS ordered by (eta, eps).
    pgd_marker_map: dict[tuple[str, str], str] = {}
    pgd_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in curves:
        if c["method"] == "PGD":
            pgd_groups[c["q"]].append(c)
    for items in pgd_groups.values():
        items_sorted = sorted(
            items,
            key=lambda c: (
                1e18 if c["eta"] is None else c["eta"],
                1e18 if c["eps"] is None else c["eps"],
                c["param"],
            ),
        )
        for i, c in enumerate(items_sorted):
            pgd_marker_map[(c["q"], c["param"])] = _PGD_MARKERS[i % len(_PGD_MARKERS)]

    # MOSEK markers: eta=0.0005 -> circle, eta=0.005 -> square; else no marker.
    mosek_marker_map: dict[tuple[str, str], str] = {}
    for c in curves:
        if c["method"] != "MOSEK" or c["eta"] is None:
            continue
        if abs(float(c["eta"]) - 0.0005) < 1e-12:
            mosek_marker_map[(c["q"], c["param"])] = "o"
        elif abs(float(c["eta"]) - 0.005) < 1e-12:
            mosek_marker_map[(c["q"], c["param"])] = "s"

    T = int(curves[0]["mean"].shape[0])
    timesteps = np.arange(T)
    mk_step = max(1, T // 24)

    plt.figure(figsize=(12, 6))
    for idx, c in enumerate(curves):
        method = c["method"]
        color = _Q_PLOT_COLORS.get((method, c["q"]), "#90A4AE")

        marker = None
        markevery = None
        markersize = 0.0
        markerfacecolor = None
        markeredgewidth = 0.0
        if method == "PGD":
            marker = pgd_marker_map.get((c["q"], c["param"]), "o")
            markersize = 4.0
            markerfacecolor = color
            markeredgewidth = 0.8
            phase = (idx * 49) % mk_step if mk_step > 1 else 0
            markevery = slice(phase, None, mk_step) if mk_step > 1 else 1
        elif method == "MOSEK":
            marker = mosek_marker_map.get((c["q"], c["param"]))
            if marker is not None:
                markersize = 4.0
                markerfacecolor = color
                markeredgewidth = 0.6
                phase = (idx * 49) % mk_step if mk_step > 1 else 0
                markevery = slice(phase, None, mk_step) if mk_step > 1 else 1

        lw = 1.6 if method == "MOSEK" else 1.0
        label = "{} q={} ({})".format(
            method, _q_label_for_legend(c["q"]), _legend_param_fragment(c["param"])
        )
        plt.plot(
            timesteps, c["mean"], color=color, linestyle="-", linewidth=lw,
            marker=marker, markersize=markersize, markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth, markevery=markevery, label=label,
        )

    plt.xlabel("t", fontsize=24)
    plt.ylabel("Regret", fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(out_path))


# ---------------------------------------------------------------------------
# Run one feature mode x one SNR tier and write the three CSVs
# ---------------------------------------------------------------------------
def run_q_sensitivity(
    mode: str, snr_tier: str, signal_power: float, out_subdir: str,
) -> None:
    # Low-SNR target is feature-mode dependent: Gaussian (l2-ball) -> 1.0,
    # l4_heavy_tail (l4-ball) -> 1e-3 (see Config.LOW_SNR_BY_MODE).
    target_snr = float(_cfg.low_snr_for_mode(mode))
    noise_scale = float(math.sqrt(signal_power / target_snr))
    print(
        "[{} | SNR={}] target_SNR={:.4g} signal_power={:.6g} noise_scale={:.6g} "
        "price[{:.4g},{:.4g}] (n={})".format(
            mode, snr_tier, target_snr, signal_power, noise_scale,
            float(PRICE_GRID.min()), float(PRICE_GRID.max()), N_PRICE,
        )
    )

    out_dir = os.path.join(_here, "results", mode, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # ----- oracle reward per (k, t) -----
    rev_opt = np.zeros((TOY_K, TOY_T), dtype=np.float64)
    for k in range(TOY_K):
        rev_opt[k, :] = run_optimal_revenue_discrete(k + 1, noise_scale)

    path_oracle = os.path.join(out_dir, "oracle_reward_by_t.csv")
    with open(path_oracle, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "t", "oracle_reward"])
        for k in range(TOY_K):
            for t in range(TOY_T):
                w.writerow([k, t, float(rev_opt[k, t])])
    print("Saved {}".format(path_oracle))

    reward_rows: list[tuple[str, str, np.ndarray]] = []
    time_records: list[tuple[str, str, list[list[float]]]] = []

    for q_val, q_label in zip(_cfg.Q_LIST, _cfg.Q_LABELS):
        set_solver_q(q_val)
        scale = float(_scale)
        print("  q={} (p={:.6g}, scale={:.6g})".format(q_label, _p_val, scale))

        # ----- MOSEK -----
        for eta_u in _cfg.ETA_GRID:
            eta_s = float(eta_u) * scale
            param = "q={},eta={}".format(q_label, eta_u)
            rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
            tlist: list[list[float]] = []
            for k in range(TOY_K):
                r, tm = run_cvx_mosek_discrete(k + 1, eta_k_coeff=eta_s, noise_scale=noise_scale)
                rev[k, :] = r
                tlist.append(tm)
            reward_rows.append(("MOSEK", param, rev))
            time_records.append(("MOSEK", param, tlist))

        # ----- PGD -----
        for eta_u, eps_u in product(_cfg.ETA_GRID, _cfg.EPS_GRID):
            eta_s = float(eta_u) * scale
            eps_s = float(eps_u) * scale
            param = "q={},eta={},eps={}".format(q_label, eta_u, eps_u)
            rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
            tlist = []
            for k in range(TOY_K):
                r, tm = run_gradient_descent_discrete(
                    k + 1, eps_s, eta_k_coeff=eta_s, noise_scale=noise_scale, snr_tier=snr_tier
                )
                rev[k, :] = r
                tlist.append(tm)
            reward_rows.append(("PGD", param, rev))
            time_records.append(("PGD", param, tlist))

    # ----- reward per (method, param, k, t) -----
    path_reward = os.path.join(out_dir, "reward_by_method_param_t.csv")
    with open(path_reward, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "k", "t", "reward"])
        for method, param, rev in reward_rows:
            for k in range(TOY_K):
                for t in range(TOY_T):
                    w.writerow([method, param, k, t, float(rev[k, t])])
    print("Saved {}".format(path_reward))

    # ----- solver times per (method, param, k, solve_idx) -----
    path_times = os.path.join(out_dir, "solver_times.csv")
    with open(path_times, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "k", "solve_idx", "time_s"])
        for method, param, tlist in time_records:
            for k_idx, t_list in enumerate(tlist, start=1):
                for s_idx, t_s in enumerate(t_list, start=1):
                    w.writerow([method, param, k_idx, s_idx, float(t_s)])
    print("Saved {}".format(path_times))

    # ----- cumulative regret (ready for plotting regret curves) -----
    # Per-step regret = oracle_reward[t] - reward[t]; cumulative along time
    # (matches Toy_main: cumsum_t(oracle - reward)).
    cum_curves = [
        (method, param, np.cumsum(rev_opt - rev, axis=1)) for method, param, rev in reward_rows
    ]

    path_cum = os.path.join(out_dir, "cumulative_regret_by_method_param_t.csv")
    with open(path_cum, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "k", "t", "cumulative_regret"])
        for method, param, cum in cum_curves:
            for k in range(TOY_K):
                for t in range(TOY_T):
                    w.writerow([method, param, k, t, float(cum[k, t])])
    print("Saved {}".format(path_cum))

    # Mean over the K repetitions: directly plottable as one curve per (method, param).
    path_cum_mean = os.path.join(out_dir, "mean_cumulative_regret_by_method_param_t.csv")
    with open(path_cum_mean, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "t", "mean_cumulative_regret"])
        for method, param, cum in cum_curves:
            cum_mean = np.mean(cum, axis=0)
            for t in range(TOY_T):
                w.writerow([method, param, t, float(cum_mean[t])])
    print("Saved {}".format(path_cum_mean))

    # ----- regret figure (mean cumulative regret per (method, q, param)) -----
    fig_path = os.path.join(out_dir, "q_sensitivity_regret.pdf")
    _plot_cumulative_regret(cum_curves, fig_path)


if __name__ == "__main__":
    for _mode in _cfg.FEATURE_MODES:
        print("\n########## Feature mode: {} ##########".format(_mode))
        configure_feature_mode(_mode)
        # X (hence signal power) depends only on the feature mode; calibrate once.
        _signal_power, _theta_norm_sq, _lam_min = _calibrate_signal_power(
            int(_cfg.n_samples_for_snr_calib)
        )
        print(
            "[{}] ||theta||^2={:.6g} lambda_min(Sigma_X)={:.6g} signal_power={:.6g}".format(
                _mode, _theta_norm_sq, _lam_min, _signal_power
            )
        )
        for _tier in _cfg.SNR_TIERS:
            _subdir = "{}_SNR".format(str(_tier).capitalize())
            print("\n===== {} | SNR tier: {} (out=results/{}/{}) =====\n".format(_mode, _tier, _mode, _subdir))
            run_q_sensitivity(_mode, str(_tier), _signal_power, _subdir)
