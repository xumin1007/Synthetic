"""
Online learning for discrete pricing under an **Gaussian feature** geometry (q = 4):
benchmark of MOSEK / PGD / ILQX / SPO+ / AdaGrad.

This is a trimmed, open-source version of the toy experiment. Compared with the
full research script it keeps ONLY the data-generating process, the five online
estimators and the oracle, and it writes the following CSV files per SNR tier:

  1. reward_by_method_param_t.csv             -> reward of every (method, param, k, t)
  2. oracle_reward_by_t.csv                    -> oracle (paper-style conditional) reward per (k, t)
  3. solver_times.csv                         -> wall-clock solve time per (method, param, k, solve_idx)
  4. cumulative_regret_by_method_param_t.csv  -> cumsum_t(oracle - reward) per (method, param, k, t)
  5. mean_cumulative_regret_by_method_param_t.csv -> the above averaged over k (ready to plot)
  6. adagrad_solving_time.csv                 -> AdaGrad per-step update time per (param, k, t)

No holdout selection and no plotting are performed here; the cumulative-regret
CSVs are ready to be plotted directly, and any other metric can be reconstructed
downstream from the reward / oracle CSVs.

Setting
-------
- Feature X_t ~ truncated-normal with an l4-norm budget (q = 4), giving an
  "Gaussian" feature geometry. The MLE / PGD regularizer uses the conjugate
  exponent p = q / (q - 1) = 4/3.
- Linear mean demand: Q(p, X, theta) = theta0^T X - gamma0 * p.
- Oracle reward per step (paper definition, conditional on X_t):
      p_t*(X_t)      = argmax_{p in PRICE_GRID} p * E_eps[max(mu(p, X_t) + eps, 0)]
      OracleReward[t] = p_t*(X_t) * E_eps[max(mu(p_t*, X_t) + eps, 0)]
- Demand simulator truncates at 0: d_t = max(theta0^T X_t - gamma0 p_t + eps, 0).

Dependencies: numpy, scipy, torch, cvxpy + MOSEK (cvxMLEpnorm / Lasso),
gurobipy + pyepo (SPO+). Set Gaussian_SMOKE=1 for a fast smoke test.
"""

from __future__ import annotations

import csv
import math
import os
import time
import warnings
from typing import Any

import numpy as np
import torch
from scipy.special import erf as _scipy_erf
from tqdm import tqdm

warnings.filterwarnings("ignore", message="divide by zero encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in matmul", category=RuntimeWarning)

# Results are written to "<project root>/results"; the project root is the
# directory that contains this package.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .utils import Config as _cfg
from .algorithms.AdaGrad import AdaGradOnline
from .algorithms.cvxLasso_linear import cvxLasso_linear
from .algorithms.cvxMLEpnorm import cvxMLEpnorm
from .algorithms.Gradient_projected_Search import gradient_descent_projected_likelihood
from .algorithms.spo_plus import ToyNetMLP, fit_revenue_quadratic_ls, train_spoplus_on_history
from .utils.randomP import randomP
from .utils.truncate_normal_sample_time_varying import truncate_normal_sample_time_varying

# ---------------------------------------------------------------------------
# Problem size & discrete price grid
# ---------------------------------------------------------------------------
TOY_D = 3**6
TOY_T = 10000
TOY_K = 10

# Hyper-parameter grids (the open-source release sweeps the full grid; reward of
# every (method, param) combination is written out for downstream selection).
ETA_GRID = [5e-4, 5e-3, 5e-2]
EPS_GRID = [5e-2, 5e-1, 5e0, 5e1]
LAMBDA_GRID = [5e-4, 5e-3, 5e-2]
SPO_LR_GRID = [5e-5, 5e-4, 5e-3, 5e-2]
SPO_EPOCHS_GRID = [60]
SPO_L2_GRID = [1e-5]
SPO_HIDDEN = 64
ADAGRAD_LR_GRID = [1e-2, 1e-1, 1.0, 10.0]

TOY_DATA_SEED = 4242


PRICE_GRID = np.linspace(30.0, 50.0, 401, dtype=np.float64)
PRICE_GRID_Experiment = np.linspace(30.0, 50.0, 6, dtype=np.float64)
N_PRICE = int(PRICE_GRID.size)


# ---------------------------------------------------------------------------
# theta0 / gamma0 matched to the toy dimension
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(TOY_DATA_SEED)
_n = 1
_d = TOY_D
_theta0 = _rng.uniform(0.001, 0.015, size=(_n, _n * _d)).astype(np.float64)
_hi_mask = _rng.random(size=(_n, _n * _d)) < 0.15
_theta0[_hi_mask] = _rng.uniform(1.0, 2.0, size=int(np.sum(_hi_mask))).astype(np.float64)
_gamma0 = np.diag(np.full(_n, 5.0, dtype=np.float64)).astype(np.float64)

# Gaussian feature geometry: q = 4  =>  conjugate p = q / (q - 1) = 4/3.
_q = 4
_p_val, EXPERIMENT_TIME, END_VALUE = _cfg.get_p_and_precomputed(_q, _d)
C = _cfg.C
W = _cfg.W
L0 = _cfg.L0

_default_snr_cfg = _cfg.get_snr_config("mid")
_a_val = float(_default_snr_cfg["a_val"])
_b_val = float(_default_snr_cfg["b_val"])

_scale = float(_d ** (1 / _p_val - 1 / 2))
ETA_GRID_SCALED = [float(x * _scale) for x in ETA_GRID]
EPS_GRID_SCALED = [float(x * _scale) for x in EPS_GRID]

if os.environ.get("Gaussian_SMOKE") == "1":
    TOY_T = 100
    TOY_K = 1
    ETA_GRID = [5e-2]
    EPS_GRID = [5e-1]
    LAMBDA_GRID = [5e-2]
    SPO_LR_GRID = [5e-3]
    SPO_EPOCHS_GRID = [60]
    SPO_L2_GRID = [1e-4]
    ADAGRAD_LR_GRID = [5e-2, 1e-1]
    ETA_GRID_SCALED = [float(x * _scale) for x in ETA_GRID]
    EPS_GRID_SCALED = [float(x * _scale) for x in EPS_GRID]


# ---------------------------------------------------------------------------
# Feature sampling & trajectory pre-computation
# ---------------------------------------------------------------------------
def _sample_xt_timestep(k_instance: int, t: int) -> np.ndarray:
    Xt = np.zeros((_d, _n), dtype=np.float64)
    for i in range(_n):
        Xt[:, i] = truncate_normal_sample_time_varying(
            _q, C, _a_val, _b_val, _d, k_instance, t, i,
            mu=_cfg.feature_mu, sigma=_cfg.feature_sigma,
        ).ravel()
    return Xt


def _collect_xt_trajectory(k_instance: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    nd = _n * _d
    X_full = np.empty((int(T), nd), dtype=np.float64)
    for t in range(int(T)):
        X_full[t] = _sample_xt_timestep(int(k_instance), int(t)).flatten()
    theta_flat = np.asarray(_theta0, dtype=np.float64).ravel()
    if not np.isfinite(X_full).all():
        raise RuntimeError("_collect_xt_trajectory: X_full has non-finite entries")
    theta0_Xt_T = np.einsum("td,d->t", X_full, theta_flat, optimize=True)
    if not np.isfinite(theta0_Xt_T).all():
        raise RuntimeError("_collect_xt_trajectory: theta0_Xt_T has non-finite entries")
    return X_full, theta0_Xt_T


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
    _, theta0_Xt_T = _collect_xt_trajectory(int(k_instance), int(TOY_T))
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


def _list_elem_float(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64).ravel()[0])


# ---------------------------------------------------------------------------
# Generic online loop shared by MOSEK / ILQX / PGD
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

    X_full, theta0_Xt_T = _collect_xt_trajectory(int(k_instance), int(TOY_T))
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


def run_cvx_lasso_discrete(
    k_instance: int, lambda_factor: float, noise_scale: float,
) -> tuple[np.ndarray, list[float]]:
    def retrain_lasso(D, P, Column_X, state, t, is_initial):
        del t, is_initial, state
        theta_k, gamma_k = cvxLasso_linear(
            D, P, Column_X, _d, _n, W, lambda_factor=lambda_factor
        )
        return {"theta_k": theta_k, "gamma_k": gamma_k}

    return _run_online_loop(
        int(k_instance), float(noise_scale),
        retrain_fn=retrain_lasso, desc="ILQX (k={})".format(k_instance),
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


def run_adagrad_discrete(
    k_instance: int, lr: float, noise_scale: float,
) -> tuple[np.ndarray, list[float]]:
    Revenue = np.zeros(TOY_T, dtype=np.float64)
    times: list[float] = []
    adagrad = AdaGradOnline(n=_n, d=_d, W=W, lr=lr)
    theta_k: np.ndarray | None = None
    gamma_k: np.ndarray | None = None
    exploration_countdown = 0

    X_full, theta0_Xt_T = _collect_xt_trajectory(int(k_instance), int(TOY_T))
    gamma_scalar = float(np.asarray(_gamma0, dtype=np.float64).ravel()[0])
    feat_dim = _n * _d

    for t in tqdm(range(TOY_T), desc="AdaGrad (k={}, lr={})".format(k_instance, lr), leave=False):
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
                if theta_k is None or gamma_k is None:
                    pt = randomP(PRICE_GRID, _n).reshape(-1, 1)
                else:
                    pt = best_price_on_grid(theta_k, gamma_k, column_Xt, PRICE_GRID, _n)

        np.random.seed(int(k_instance + t))
        zt = np.random.normal(loc=0.0, scale=noise_scale, size=_n)
        pt_scalar = float(pt[0, 0])
        dt = np.maximum(
            np.array([theta0_Xt_scalar - gamma_scalar * pt_scalar], dtype=np.float64) + zt,
            0.0,
        )
        Revenue[t] = float(np.dot(pt.ravel(), dt))

        theta_k, gamma_k, update_time = adagrad.update(column_Xt, pt, dt)
        times.append(float(update_time))

    return Revenue, times


# ---------------------------------------------------------------------------
# SPO+ online wrapper (the SPO+ estimator core lives in spo_plus.py)
# ---------------------------------------------------------------------------
def run_spoplus_online(
    k_instance: int, noise_scale: float, lr: float, epochs: int, l2: float,
) -> tuple[np.ndarray, list[float]]:
    Revenue = np.zeros(TOY_T, dtype=np.float64)
    times: list[float] = []
    D: list[Any] = []
    P: list[Any] = []
    Column_X: list[Any] = []
    net: ToyNetMLP | None = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exploration_countdown = 0

    X_full, theta0_Xt_T = _collect_xt_trajectory(int(k_instance), int(TOY_T))
    gamma_scalar = float(np.asarray(_gamma0, dtype=np.float64).ravel()[0])
    feat_dim = _n * _d

    def _retrain(D, P, Column_X):
        beta_rev = fit_revenue_quadratic_ls(D, P, Column_X, _d, _n)
        cx_list = [np.asarray(cx, dtype=np.float32) for cx in Column_X]
        p_obs_list = [_list_elem_float(P[i]) for i in range(len(P))]
        r_obs_list = [_list_elem_float(P[i]) * _list_elem_float(D[i]) for i in range(len(D))]
        return train_spoplus_on_history(
            cx_list, p_obs_list, r_obs_list, beta_rev,
            price_grid=PRICE_GRID, exp_grid=PRICE_GRID_Experiment,
            n=_n, d=_d, n_price=N_PRICE,
            lr=lr, epochs=epochs, l2=l2, device=device, hidden=SPO_HIDDEN,
        )

    for t in tqdm(range(TOY_T), desc="SPO+ (k={})".format(k_instance), leave=False):
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
                if net is None:
                    pt = randomP(PRICE_GRID, _n).reshape(-1, 1)
                else:
                    net.eval()
                    with torch.no_grad():
                        xh = torch.from_numpy(
                            X_full[t].astype(np.float32, copy=True).reshape(1, -1)
                        ).to(device)
                        chat = net(xh)
                        j = int(torch.argmax(chat, dim=1).item())
                        pj = float(PRICE_GRID[j])
                    pt = np.full((_n, 1), pj, dtype=np.float64)

        np.random.seed(int(k_instance + t))
        zt = np.random.normal(loc=0.0, scale=noise_scale, size=_n)
        pt_scalar = float(pt[0, 0])
        dt = np.maximum(
            np.array([theta0_Xt_scalar - gamma_scalar * pt_scalar], dtype=np.float64) + zt,
            0.0,
        )
        Revenue[t] = float(np.dot(pt.ravel(), dt))

        if t <= END_VALUE:
            D.append(np.float32(dt))
            P.append(np.float32(pt))
            Column_X.append(np.float32(column_Xt))
            if t == END_VALUE:
                t0 = time.time()
                net = _retrain(D, P, Column_X)
                times.append(time.time() - t0)
                exploration_countdown = 0
        else:
            if math.isqrt(t + 1) ** 2 == t + 1:
                D.append(np.float32(dt))
                P.append(np.float32(pt))
                Column_X.append(np.float32(column_Xt))
                t0 = time.time()
                net = _retrain(D, P, Column_X)
                times.append(time.time() - t0)

    return Revenue, times


# ---------------------------------------------------------------------------
# SNR calibration -> noise_scale
# ---------------------------------------------------------------------------
def _calibrate_signal_power(
    a_val: float, b_val: float, c_val: float, n_calib: int,
) -> tuple[float, float, float]:
    """SNR := ||theta||_2^2 * lambda_min(E[X X^T]) / sigma^2."""
    assert _n == 1
    mu_use = float(_cfg.feature_mu)
    sigma_use = float(_cfg.feature_sigma)
    N = int(n_calib)
    dim = _d * _n
    X_mat = np.empty((N, dim), dtype=np.float64)
    for i in range(N):
        Xt = np.zeros((_d, _n), dtype=np.float64)
        for ii in range(_n):
            Xt[:, ii] = truncate_normal_sample_time_varying(
                _q, c_val, a_val, b_val, _d, 0, i, ii, mu=mu_use, sigma=sigma_use,
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


def _unscaled_from_scaled_grid(
    scaled_val: float, scaled_grid: list[float], unscaled_grid: list[float]
) -> float:
    arr = np.asarray(scaled_grid, dtype=np.float64)
    j = int(np.argmin(np.abs(arr - float(scaled_val))))
    return float(unscaled_grid[j])


# ---------------------------------------------------------------------------
# Run one SNR tier and write result CSVs
# ---------------------------------------------------------------------------
def run_Gaussian_comparison(snr_tier: str = "mid", out_subdir: str = "") -> None:
    global _a_val, _b_val, C

    cfg = _cfg.get_snr_config(snr_tier)
    _a_val = float(cfg["a_val"])
    _b_val = float(cfg["b_val"])
    if "C" in cfg:
        C = float(cfg["C"])
    target_snr = float(cfg["target_SNR"])

    signal_power, theta_norm_sq, lambda_min_Sigma_X = _calibrate_signal_power(
        _a_val, _b_val, C, n_calib=int(_cfg.n_samples_for_snr_calib)
    )
    noise_scale = float(math.sqrt(signal_power / target_snr))
    print(
        "[Gaussian q={} | SNR={}] a={:.4g} b={:.4g} C={:.4g} target_SNR={:.4g} "
        "||theta||^2={:.6g} lambda_min={:.6g} noise_scale={:.6g}".format(
            _q, snr_tier, _a_val, _b_val, C, target_snr,
            theta_norm_sq, lambda_min_Sigma_X, noise_scale,
        )
    )

    out_dir = os.path.join(_here, "results", out_subdir) if out_subdir \
        else os.path.join(_here, "results")
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

    # reward[(method, param)] = (K, T) array ; times[(method, param)] = list-of-lists per k
    reward_rows: list[tuple[str, str, np.ndarray]] = []
    time_records: list[tuple[str, str, list[list[float]]]] = []
    adagrad_time_records: list[tuple[str, list[list[float]]]] = []

    # ----- MOSEK -----
    for eta in ETA_GRID_SCALED:
        eta_u = _unscaled_from_scaled_grid(eta, ETA_GRID_SCALED, ETA_GRID)
        param = "eta={}".format(eta_u)
        rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
        tlist: list[list[float]] = []
        for k in range(TOY_K):
            r, tm = run_cvx_mosek_discrete(k + 1, eta_k_coeff=eta, noise_scale=noise_scale)
            rev[k, :] = r
            tlist.append(tm)
        reward_rows.append(("MOSEK", param, rev))
        time_records.append(("MOSEK", param, tlist))

    # ----- PGD -----
    for eta in ETA_GRID_SCALED:
        eta_u = _unscaled_from_scaled_grid(eta, ETA_GRID_SCALED, ETA_GRID)
        for eps in EPS_GRID_SCALED:
            eps_u = _unscaled_from_scaled_grid(eps, EPS_GRID_SCALED, EPS_GRID)
            param = "eta={},eps={}".format(eta_u, eps_u)
            rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
            tlist = []
            for k in range(TOY_K):
                r, tm = run_gradient_descent_discrete(
                    k + 1, eps, eta_k_coeff=eta, noise_scale=noise_scale, snr_tier=snr_tier
                )
                rev[k, :] = r
                tlist.append(tm)
            reward_rows.append(("PGD", param, rev))
            time_records.append(("PGD", param, tlist))

    # ----- ILQX -----
    for lam in LAMBDA_GRID:
        param = "lambda={}".format(lam)
        rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
        tlist = []
        for k in range(TOY_K):
            r, tm = run_cvx_lasso_discrete(k + 1, lambda_factor=lam, noise_scale=noise_scale)
            rev[k, :] = r
            tlist.append(tm)
        reward_rows.append(("ILQX", param, rev))
        time_records.append(("ILQX", param, tlist))

    # ----- SPO+ -----
    for lr in SPO_LR_GRID:
        for epochs in SPO_EPOCHS_GRID:
            for l2 in SPO_L2_GRID:
                param = "lr={},epochs={},l2={}".format(lr, epochs, l2)
                rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
                tlist = []
                for k in tqdm(range(TOY_K), desc="SPO+ {}".format(param), leave=True):
                    r, tm = run_spoplus_online(k + 1, noise_scale, lr=lr, epochs=epochs, l2=l2)
                    rev[k, :] = r
                    tlist.append(tm)
                reward_rows.append(("SPO+", param, rev))
                time_records.append(("SPO+", param, tlist))

    # ----- AdaGrad -----
    for lr in ADAGRAD_LR_GRID:
        param = "lr={}".format(lr)
        rev = np.zeros((TOY_K, TOY_T), dtype=np.float64)
        tlist = []
        for k in tqdm(range(TOY_K), desc="AdaGrad {}".format(param), leave=True):
            r, tm = run_adagrad_discrete(k + 1, lr=lr, noise_scale=noise_scale)
            rev[k, :] = r
            tlist.append(tm)
        reward_rows.append(("AdaGrad", param, rev))
        adagrad_time_records.append((param, tlist))

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

    # ----- AdaGrad solving times per (param, k, t) -----
    path_adagrad_times = os.path.join(out_dir, "adagrad_solving_time.csv")
    with open(path_adagrad_times, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "k", "t", "time_s"])
        for param, tlist in adagrad_time_records:
            for k_idx, t_list in enumerate(tlist):
                for t, t_s in enumerate(t_list):
                    w.writerow(["AdaGrad", param, k_idx, t, float(t_s)])
    print("Saved {}".format(path_adagrad_times))

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


def main() -> None:
    for _tier in _cfg.SNR_TIERS:
        _subdir = "{}_SNR".format(str(_tier).capitalize())
        print("\n========== Gaussian comparison | SNR tier: {} (out={}) ==========\n".format(_tier, _subdir))
        run_Gaussian_comparison(snr_tier=str(_tier), out_subdir=_subdir)


if __name__ == "__main__":
    main()
