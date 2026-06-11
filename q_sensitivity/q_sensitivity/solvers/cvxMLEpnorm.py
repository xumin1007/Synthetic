from typing import Any

import cvxpy as cp
import numpy as np


def cvxMLEpnorm(D, P, Column_X, d, p, eta_k, L0, n, W):
    explore = len(D)  # number of exploration samples

    # Define the decision variable
    vars = cp.Variable((n * n * d + n * n))
    # Recover theta and gamma
    theta = vars[:n * n * d].reshape((n, n * d))  # (n, n*d) matrix
    gamma = vars[n * n * d:].reshape((n, n))    # (n, n) matrix

    # Objective function (vectorized):
    # The original per-t loop accumulated too many CVXPY sub-expressions (slow compile when q=inf)
    X_mat = np.column_stack([np.asarray(x, dtype=np.float64).reshape(n * d) for x in Column_X])  # (n*d, explore)
    P_mat = np.column_stack([np.asarray(pv, dtype=np.float64).reshape(n) for pv in P])           # (n, explore)
    D_mat = np.column_stack([np.asarray(dv, dtype=np.float64).reshape(n) for dv in D])           # (n, explore)
    pred_mat = theta @ X_mat - gamma @ P_mat
    lt = 0.5 * cp.sum_squares(D_mat - pred_mat)

    # Regularizer: when q=inf => p=1, use ||x||_1^2 to reduce canonicalization cost
    if np.isclose(float(p), 1.0):
        pnorm_sq = cp.square(cp.norm1(vars))
    else:
        pnorm_sq = cp.power(cp.sum(cp.abs(vars) ** p), 2.0 / float(p))
    objective = lt + 0.5 * eta_k * pnorm_sq

    # Constraints
    constraints = [
        W - cp.norm(vars, 1) >= 0,  # Constraint 1: l1-norm of x <= W
    ]

    # Define the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    # In high dimension with explore << n*d, MOSEK often fails numerically; try MOSEK -> CLARABEL -> SCS.
    solver_specs: list[tuple[Any, dict[str, Any]]] = [
        (
            cp.MOSEK,
            {
                "verbose": False,
                "mosek_params": {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-4,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
                },
            },
        ),
        (cp.SCS, {"verbose": False, "max_iters": int(2e5), "eps": 1e-5}),
    ]
    clarabel = getattr(cp, "CLARABEL", None)
    if clarabel is not None:
        solver_specs.insert(1, (clarabel, {"verbose": False}))

    last_exc: Exception | None = None
    vars_opt = None
    for solver, kwargs in solver_specs:
        try:
            problem.solve(solver=solver, **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
        vo = vars.value
        if vo is not None and np.all(np.isfinite(vo)):
            vars_opt = vo
            break
        last_exc = RuntimeError(
            "solver finished but no finite solution (status={})".format(problem.status)
        )

    if vars_opt is None:
        raise cp.error.SolverError(
            "cvxMLEpnorm: all solvers failed (d={}, explore={}, p={}, eta_k={}); last_error={!r}".format(
                d, explore, p, eta_k, last_exc
            )
        ) from (last_exc if isinstance(last_exc, Exception) else None)

    theta = vars_opt[:n * n * d].reshape((n, n * d))
    gamma = vars_opt[n * n * d:].reshape((n, n))
    return theta, gamma
