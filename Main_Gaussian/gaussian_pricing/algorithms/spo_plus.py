"""
SPO+ estimator core (PyEPO) for the discrete-pricing benchmark.

Mirrors the single-file layout of the other estimators (MOSEK -> cvxMLEpnorm.py,
Lasso -> cvxLasso_linear.py, PGD -> Gradient_projected_Search.py): this module
holds everything SPO+-specific, namely

  - PriceChoiceModel : the PyEPO optimization model (pick one price level),
  - ToyNetMLP        : the prediction network feature -> price coefficients,
  - the doubly-robust (DR) revenue-label construction
    (fit_revenue_quadratic_ls / revenue_hat_row_quadratic / dr_c_row),
  - train_spoplus_on_history : one SPO+ retraining on the cumulative history.

The online loop that calls `train_spoplus_on_history` (the analogue of
`run_cvx_mosek_discrete`) stays in the main experiment script.

Dependencies: numpy, torch, gurobipy, pyepo.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import gurobipy as gp
from gurobipy import GRB

import pyepo
import pyepo.data.dataset as _pyepo_dataset_mod
from pyepo.model.grb import optGrbModel

DEFAULT_HIDDEN = 64


class PriceChoiceModel(optGrbModel):
    """Select one price level to maximize c^T w."""

    def __init__(self, n_price: int):
        self.n_price = n_price
        super().__init__()

    def _getModel(self):
        m = gp.Model()
        w = m.addVars(self.n_price, vtype=GRB.BINARY, name="w")
        m.addConstr(gp.quicksum(w[j] for j in range(self.n_price)) == 1)
        m.modelSense = GRB.MAXIMIZE
        return m, w


class ToyNetMLP(nn.Module):
    """Feature (dim n*d) -> predicted price coefficients (dim n_prices)."""

    def __init__(self, feat_dim: int, n_prices: int, hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_prices),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _list_elem_float(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64).ravel()[0])


_PI_LOOKUP_CACHE: dict[tuple[bytes, bytes], np.ndarray] = {}


def _build_pi_lookup(price_grid: np.ndarray, exp_grid: np.ndarray) -> np.ndarray:
    """Logging policy pi(j) = 1/M_exp on the experiment prices, else 0."""
    pg = np.asarray(price_grid, dtype=np.float64).ravel()
    eg = np.asarray(exp_grid, dtype=np.float64).ravel()
    key = (pg.tobytes(), eg.tobytes())
    cached = _PI_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached
    m_exp = int(eg.size)
    out = np.zeros(pg.size, dtype=np.float64)
    if m_exp > 0:
        diff = np.abs(pg[:, None] - eg[None, :])
        thr = 1e-8 + 1e-6 * np.abs(eg[None, :])
        any_match = (diff <= thr).any(axis=1)
        out[any_match] = 1.0 / float(m_exp)
    _PI_LOOKUP_CACHE[key] = out
    return out


def fit_revenue_quadratic_ls(
    D: list[Any], P: list[Any], Column_X: list[Any], d: int, n: int,
) -> np.ndarray:
    """Least-squares fit of R = p*theta^T X - gamma*p^2 on the cumulative history."""
    nd = n * d
    N = len(D)
    if N == 0:
        return np.zeros(nd + 1, dtype=np.float64)
    p_vec = np.fromiter((_list_elem_float(P[i]) for i in range(N)), dtype=np.float64, count=N)
    d_vec = np.fromiter((_list_elem_float(D[i]) for i in range(N)), dtype=np.float64, count=N)
    y = p_vec * d_vec
    X_mat = np.empty((N, nd), dtype=np.float64)
    for i in range(N):
        X_mat[i, :] = np.asarray(Column_X[i], dtype=np.float64).ravel()
    Phi = np.empty((N, nd + 1), dtype=np.float64)
    Phi[:, :nd] = X_mat * p_vec[:, None]
    Phi[:, nd] = p_vec * p_vec
    beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return np.asarray(beta, dtype=np.float64).ravel()


def revenue_hat_row_quadratic(
    beta: np.ndarray, column_X: np.ndarray, price_grid: np.ndarray, d: int, n: int,
) -> np.ndarray:
    """Predicted revenue \\hat R_j over the price grid from the quadratic LS fit."""
    assert n == 1
    nd = n * d
    x = np.asarray(column_X, dtype=np.float64).ravel()
    pg = np.asarray(price_grid, dtype=np.float64).ravel()
    linear = float(np.dot(np.asarray(beta[:nd], dtype=np.float64), x))
    quad_coef = float(beta[nd])
    out = pg * linear + quad_coef * (pg * pg)
    return out.astype(np.float32, copy=False)


def dr_c_row(
    r_hat_row: np.ndarray, p_obs: float, r_obs: float,
    price_grid: np.ndarray, exp_grid: np.ndarray,
) -> np.ndarray:
    """Doubly-robust label: c_j = \\hat R_j + 1[p_obs=p_j] (R_obs - \\hat R_j) / pi(j)."""
    pg = np.asarray(price_grid, dtype=np.float64).ravel()
    out = np.asarray(r_hat_row, dtype=np.float32).astype(np.float32, copy=True)
    match_mask = np.isclose(pg, float(p_obs), rtol=1e-6, atol=1e-8)
    if not match_mask.any():
        return out
    pi_vec = _build_pi_lookup(pg, np.asarray(exp_grid, dtype=np.float64).ravel())
    pi_pos = pi_vec > 1e-15
    correct_mask = match_mask & pi_pos
    if correct_mask.any():
        rh = np.asarray(r_hat_row, dtype=np.float64)[correct_mask]
        corrected = rh + (float(r_obs) - rh) / pi_vec[correct_mask]
        out[correct_mask] = corrected.astype(np.float32, copy=False)
    return out


def train_spoplus_on_history(
    column_X_list: list[np.ndarray],
    p_obs_list: list[float],
    r_obs_list: list[float],
    beta_revenue: np.ndarray,
    *,
    price_grid: np.ndarray,
    exp_grid: np.ndarray,
    n: int,
    d: int,
    n_price: int,
    lr: float,
    epochs: int,
    l2: float,
    device: torch.device,
    hidden: int = DEFAULT_HIDDEN,
) -> ToyNetMLP:
    """One SPO+ retraining: DR labels (\\hat R from quadratic LS, IPW with known
    pi = 1/M_exp) + PyEPO SPO+ training of ToyNetMLP. PyEPO stdout is suppressed."""
    n_samples = len(column_X_list)
    feat_dim = n * d
    c_mat = np.zeros((n_samples, n_price), dtype=np.float32)
    x_mat = np.zeros((n_samples, feat_dim), dtype=np.float32)
    for i, cx in enumerate(column_X_list):
        x_mat[i] = np.asarray(cx, dtype=np.float32).ravel()
        cx64 = np.asarray(cx, dtype=np.float64).reshape(-1, 1)
        r_hat_row = revenue_hat_row_quadratic(beta_revenue, cx64, price_grid, d, n)
        c_mat[i] = dr_c_row(r_hat_row, p_obs_list[i], r_obs_list[i], price_grid, exp_grid)

    optmodel = PriceChoiceModel(n_price=n_price)
    _tqdm_prev = _pyepo_dataset_mod.tqdm
    _pyepo_dataset_mod.tqdm = lambda seq, *args, **kwargs: seq
    try:
        with open(os.devnull, "w") as _devnull:
            with contextlib.redirect_stdout(_devnull):
                dataset = pyepo.data.dataset.optDataset(optmodel, x_mat, c_mat)
                loader = DataLoader(dataset, batch_size=min(64, max(1, n_samples)), shuffle=True)
                spop = pyepo.func.SPOPlus(optmodel, processes=1)
                net = ToyNetMLP(feat_dim, n_price, hidden=hidden).to(device)
                opt = torch.optim.Adam(net.parameters(), lr=lr)
                net.train()
                for _ in range(max(1, epochs)):
                    for xb, cb, wb, zb in loader:
                        xb = xb.to(device)
                        cb = cb.to(device)
                        wb = wb.to(device)
                        zb = zb.to(device)
                        cp = net(xb)
                        loss = spop(cp, cb, wb, zb)
                        if l2 > 0:
                            loss = loss + l2 * sum(p.pow(2).sum() for p in net.parameters())
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
    finally:
        _pyepo_dataset_mod.tqdm = _tqdm_prev
    return net
