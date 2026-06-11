# -*- coding: utf-8 -*-
"""
ILQX/Lasso estimation under linear demand: minimize MSE + lambda * ||vars||_1, s.t. ||vars||_1 <= W.
lambda_factor is the L1 penalty coefficient; tune via holdout selection.
"""
import cvxpy as cp
import numpy as np


def cvxLasso_linear(D, P, Column_X, d, n, W, lambda_factor=0.05):
    """
    ILQX/Lasso for linear demand. lambda_factor: L1 penalty coefficient (default 0.05).
    Tuning: pass different lambda_factor and select by holdout regret.
    """
    explore = len(D)
    t_safe = max(explore, 1)
    # Theory-aligned lambda scaling (optional); or grid-search lambda_factor directly
    lam_val = lambda_factor * (t_safe ** 0.25) * np.sqrt(np.log(d + 1) + np.log(t_safe + 1))

    vars = cp.Variable(n * n * d + n * n)
    theta = vars[:n * n * d].reshape((n, n * d), order="F")
    gamma = vars[n * n * d:].reshape((n, n), order="F")

    lt = 0
    for t in range(explore):
        pred = cp.reshape(theta @ Column_X[t] - gamma @ P[t], (n,), order="F")
        residual = np.reshape(D[t], (n,)) - pred
        lt += 0.5 * cp.sum_squares(residual)
    objective = lt + lam_val * cp.norm(vars, 1)
    constraints = [cp.norm(vars, 1) <= W]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
    except cp.error.SolverError:
        # When MOSEK fails (numerical/format issues), fall back to ECOS or SCS
        for solver in [cp.ECOS, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if vars.value is not None:
                    break
            except Exception:
                continue
        else:
            raise cp.error.SolverError("MOSEK failed and ECOS/SCS fallback also failed.")

    vars_opt = vars.value
    theta_opt = vars_opt[:n * n * d].reshape((n, n * d))
    gamma_opt = vars_opt[n * n * d:].reshape((n, n))
    return theta_opt, gamma_opt
