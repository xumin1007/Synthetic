import numpy as np

from ..utils.normcalculate import p_norm
from ..utils.project import project_l1


# Learning rate by SNR tier: lr = lr_coeff * (eta_k ** ETA_LR_EXP) / exptime / (d ** LR_DIM_SCALE_EXP)
# ETA_LR_EXP < 1 keeps lr from becoming too small at small eta_k and balances across eta_k values
ETA_LR_EXP = 0.5
SNR_TIERS = ['low', 'mid', 'high']
# In high dimensions, pure 1/d often yields steps that are too small (Toy d~729: lr~1e-6, zero projected increment); LR_DIM_SCALE_EXP<1 is a gentler dimension penalty.
# =1.0 matches the old /d exactly; default 0.75 scales lr up by roughly d^0.25 (e.g. d=729 ~5x).
LR_DIM_SCALE_EXP = 0.75
# Balance stability and speed: main bottleneck is lr too small; raise overall by an order of magnitude
LR_COEFF_BY_SNR = {'low': 1e-1, 'mid': 1e-1, 'high': 1e-1} #1e-1
# Slightly lower momentum to reduce oscillation near the boundary
MOMENTUM_BETA = 0.95
# Plateau decay: if no improvement for a while, lr *= LR_PLATEAU_DECAY (decrease only, never increase)
# Tuning goal: smoother lr decay; avoid hitting the floor too early and slowing late-stage convergence.
PLATEAU_PATIENCE = 100
LR_PLATEAU_DECAY = 0.97
MIN_LR = 1e-9 #3e-5
LOSS_IMPROVE_TOL = 1e-5
# Combined with absolute threshold: loss scales vary across batches/regularization; relative improvement triggers progress more reliably and reduces run-to-run speed variance
REL_LOSS_IMPROVE_FRAC = 1e-6
# Global gradient norm clip threshold (L2); suppresses early huge gradients that cause wall-hitting
MAX_GRAD_NORM = 1e6 
# Clip backoff: accumulate streak only on strong clipping (very small clip_scale).
# Smaller threshold is stricter (only huge raw norms count); together with clearing streak on loss improvement, reduces lr-trajectory variance across instances.
CLIP_BACKOFF_SCALE_THRESHOLD = 0.03
CLIP_BACKOFF_PATIENCE = 200
CLIP_BACKOFF_DECAY = 0.985
# For p!=1: allow gentle lr recovery when loss keeps improving without strong clipping (capped at a multiple of initial lr)
# Previously patience=20, growth=1.10, cap=20 needed ~620 improving iters to reach cap;
# within num_iters=1000 the cap was rarely hit, leaving many runs stuck at 1000 iters (grad ~30, epsilon not triggered).
# patience=5, growth=1.25, cap=50: ~50x lr_initial in ~85 iters, finishing step-size-limited PGD calls in hundreds of iters.
# Plateau (PATIENCE=100, 0.97x) and clip backoff still provide safety; lr is pulled back if it overshoots and causes oscillation or clipping.
LR_RECOVER_PATIENCE = 5
LR_RECOVER_GROWTH = 1.25
LR_RECOVER_MAX_MULT = 50.0

# p=1 (q=inf) stabilization: enabled only when p==1; does not affect p>1
P1_LR_BOOST = 1e0
P1_PLATEAU_PATIENCE = 900
P1_PLATEAU_DECAY = 0.98
P1_CLIP_BACKOFF_PATIENCE = 120
P1_CLIP_BACKOFF_DECAY = 0.92
P1_LR_RECOVER_PATIENCE = 80
P1_LR_RECOVER_GROWTH = 1.05
P1_MAX_LR_MULT = 20.0
P1_MIN_LR = 1e-10


def gradient_descent_projected_likelihood(vars0, D, P, Column_X, epsilon_k, d, p, eta_k, L0, n, W, exptime, snr_tier='mid'):
    is_p1 = abs(float(p) - 1.0) < 1e-12
    is_p2 = abs(float(p) - 2.0) < 1e-12
    plateau_patience = P1_PLATEAU_PATIENCE if is_p1 else PLATEAU_PATIENCE
    plateau_decay = P1_PLATEAU_DECAY if is_p1 else LR_PLATEAU_DECAY
    clip_backoff_patience = P1_CLIP_BACKOFF_PATIENCE if is_p1 else CLIP_BACKOFF_PATIENCE
    clip_backoff_decay = P1_CLIP_BACKOFF_DECAY if is_p1 else CLIP_BACKOFF_DECAY
    # Cache power exponents; non-integer ** in numpy is element-wise C pow (~100ns/element; ~70us at d=730).
    # Share one ** between p_norm and reg_grad via |v|^p = |v|^(p-1) * |v|.
    p_minus_one = float(p) - 1.0
    inv_p = 1.0 / float(p)

    # Stack D / P / Column_X into matrices once for shared use in objective / compute_gradient,
    # replacing the per-iter Python for t in range(explore) loop with a single matrix multiply.
    # D_mat: (N, n); P_mat: (N, n); X_mat: (N, n*d)
    nd = int(n) * int(d)
    N_samples = len(D)
    if N_samples > 0:
        D_mat = np.empty((N_samples, n), dtype=np.float64)
        P_mat = np.empty((N_samples, n), dtype=np.float64)
        X_mat = np.empty((N_samples, nd), dtype=np.float64)
        for _t in range(N_samples):
            D_mat[_t, :] = np.asarray(D[_t], dtype=np.float64).reshape(n)
            P_mat[_t, :] = np.asarray(P[_t], dtype=np.float64).reshape(n)
            X_mat[_t, :] = np.asarray(Column_X[_t], dtype=np.float64).reshape(nd)
    else:
        D_mat = np.zeros((0, n), dtype=np.float64)
        P_mat = np.zeros((0, n), dtype=np.float64)
        X_mat = np.zeros((0, nd), dtype=np.float64)

    def gradient_descent_projected( vars0, learning_rate, epsilon, num_iters):
        loss_history = []
        vars = vars0
        # Pre-allocated buffer instead of per-iter np.array(vars, copy=True) reallocation
        vars_prev = np.array(vars0, copy=True)
        velocity = np.zeros_like(vars0)  # momentum velocity
        lr_current = float(learning_rate)
        best_loss = np.inf
        iters_since_improvement = 0
        clip_trigger_count = 0
        clip_scale_sum = 0.0
        max_raw_grad_norm = 0.0
        clip_streak = 0
        improve_streak = 0
        lr_cap = float(learning_rate) * (P1_MAX_LR_MULT if is_p1 else LR_RECOVER_MAX_MULT)
        lr_initial = float(learning_rate)
        max_grad_norm_threshold = float(MAX_GRAD_NORM)
        # Cache previous ||vars|| to save one O(D) norm per iter in the stopping criterion
        last_vars_norm = float(np.sqrt(np.dot(vars_prev, vars_prev)))

        def _loss_improved(loss: float, best: float) -> bool:
            if not np.isfinite(best):
                return True
            scale = max(abs(float(best)), 1.0)
            need = max(float(LOSS_IMPROVE_TOL), float(REL_LOSS_IMPROVE_FRAC) * scale)
            return float(loss) < float(best) - need

        for iterth in range(num_iters):
            loss = objective(vars)
            loss_history.append(loss)
            # Plateau rule: if loss stagnates, decay lr only (no restart)
            if _loss_improved(loss, best_loss):
                best_loss = loss
                iters_since_improvement = 0
                improve_streak += 1
                # While objective decreases, do not let long strong-clipping streaks drive lr to zero (avoids inconsistent speed across data)
                clip_streak = 0
            else:
                iters_since_improvement += 1
                improve_streak = 0
            if iters_since_improvement >= plateau_patience:
                lr_floor = P1_MIN_LR if is_p1 else MIN_LR
                lr_current = max(lr_current * plateau_decay, lr_floor)
                iters_since_improvement = 0
            # Compute gradient
            grad = compute_gradient(vars)

            # Global gradient norm clipping: prevent early explosion that keeps updates flattened by projection
            # Use sqrt(g·g) instead of np.linalg.norm(g, 2) to save dispatch overhead
            raw_grad_norm = float(np.sqrt(np.dot(grad, grad)))
            max_raw_grad_norm = max(max_raw_grad_norm, raw_grad_norm)
            if raw_grad_norm > max_grad_norm_threshold:
                clip_scale = max_grad_norm_threshold / (raw_grad_norm + 1e-12)
                grad = grad * clip_scale
                clip_trigger_count += 1
                clip_scale_sum += clip_scale
                # Algebraically equivalent to np.linalg.norm(clipped_grad) (diff < 1e-18 vs original); saves one O(D) norm
                grad_norm = raw_grad_norm * clip_scale
                # Count streak only on strong clipping (grad far above threshold); avoids exponential lr decay when raw is chronically large
                thr = float(CLIP_BACKOFF_SCALE_THRESHOLD)
                if clip_scale < thr:
                    clip_streak += 1
                else:
                    clip_streak = 0
            else:
                clip_scale = 1.0
                grad_norm = raw_grad_norm
                clip_streak = 0

            # Back off learning rate after consecutive strong clipping
            if clip_streak >= clip_backoff_patience:
                lr_floor = P1_MIN_LR if is_p1 else MIN_LR
                lr_current = max(lr_current * clip_backoff_decay, lr_floor)
                clip_streak = 0

            # For p=1, if improvement continues without clipping, allow slow lr recovery to avoid being stuck at tiny steps
            if is_p1 and clip_scale >= 1.0 and improve_streak >= P1_LR_RECOVER_PATIENCE:
                lr_current = min(lr_current * P1_LR_RECOVER_GROWTH, lr_cap)
                improve_streak = 0
            # For p>1: gently increase lr on improvement without strong clipping to speed mid/late convergence
            if (
                not is_p1
                and clip_scale >= CLIP_BACKOFF_SCALE_THRESHOLD
                and improve_streak >= LR_RECOVER_PATIENCE
            ):
                lr_current = min(lr_current * LR_RECOVER_GROWTH, lr_initial * LR_RECOVER_MAX_MULT)
                improve_streak = 0

            # Check stopping condition
            if grad_norm < epsilon:
                break

            # Momentum update: velocity = beta * velocity + grad, then step with velocity
            velocity = MOMENTUM_BETA * velocity + grad
            vars_before_project = vars - lr_current * velocity

            # Project onto L1 norm constraint
            vars = project_l1(vars_before_project, W)

            # Stop if relative change in variable norm is below threshold; last_vars_norm refreshed at loop end,
            # reused here to save np.linalg.norm(vars_prev).
            # Threshold relaxed from 1e-8 to 1e-6: late-stage grad may not hit epsilon_k but vars barely move;
            # early-stop PGD calls that only fine-tune in the last 50-100 iters.
            diff = vars - vars_prev
            diff_norm = float(np.sqrt(np.dot(diff, diff)))
            if last_vars_norm >= 1e-10 and diff_norm / last_vars_norm < 1e-6:
                break
            # Reuse buffer via copyto instead of vars_prev = np.array(vars, copy=True)
            np.copyto(vars_prev, vars)
            last_vars_norm = float(np.sqrt(np.dot(vars_prev, vars_prev)))

            if ( iterth + 1) % 1000 == 0:
                # Every 1000 iters: check projection by printing delta before/after projection.
                delta_project_vars = vars_before_project - vars
                print(f"Iter {iterth + 1}: Delta Project Vars Norm = {np.linalg.norm(delta_project_vars)}")
                # Every 1000 iters: print loss, gradient norm, and current learning rate
                print(
                    f"Iter { iterth + 1}: Loss = {loss:.4f}, "
                    f"Gradient Norm = {grad_norm:.6f} (raw={raw_grad_norm:.6f}), "
                    f"lr = {lr_current:.2e}, clip_scale = {clip_scale:.3e}"
                )

        total_iters = max(len(loss_history), 1)
        clip_rate = clip_trigger_count / total_iters
        mean_clip_scale = (clip_scale_sum / clip_trigger_count) if clip_trigger_count > 0 else 1.0
        print(
            "[PGD-clip] iters={}, clip_triggered={} ({:.2%}), "
            "mean_clip_scale={:.3e}, max_raw_grad_norm={:.3e}".format(
                total_iters, clip_trigger_count, clip_rate, mean_clip_scale, max_raw_grad_norm
            )
        )
        return vars, loss_history

    # Each iter calls objective(vars) then compute_gradient(vars) sequentially; vars is unchanged between calls.
    # Cache shared intermediates (residual_mat, pnorm_val, abs_pow_p1, sign_vars) keyed by vars object identity:
    #   - objective fills cache on first entry; compute_gradient hits cache and skips matrix mul / power redo.
    #   - p_norm reuses abs_pow_p1 via |v|^p = |v|^(p-1) * |v|, cutting two non-integer ** per iter to one.
    _shared: dict = {
        "vars_obj": None,
        "residual_mat": None,
        "pnorm_val": 0.0,
        "abs_pow_p1": None,
        "sign_vars": None,
        "finite": True,
    }

    inv_N = 1.0 / float(max(int(N_samples), 1))
    # Pre-allocated buffer for data-term gradient writes; avoids D-sized np.concatenate allocation each iter.
    data_grad_buf = np.empty(int(n) * int(n) * int(d) + int(n) * int(n), dtype=np.float64)
    theta_grad_view = data_grad_buf[: n * n * d]
    gamma_grad_view = data_grad_buf[n * n * d:]

    def _refresh_shared(vars):
        if _shared["vars_obj"] is vars:
            return
        _shared["vars_obj"] = vars
        vars_finite = bool(np.isfinite(vars).all())
        _shared["finite"] = vars_finite
        if not vars_finite:
            _shared["residual_mat"] = None
            _shared["abs_pow_p1"] = np.zeros_like(vars)
            _shared["sign_vars"] = np.zeros_like(vars)
            _shared["pnorm_val"] = np.inf
            return
        if N_samples > 0:
            theta_v = vars[:n * n * d].reshape((n, n * d))
            gamma_v = vars[n * n * d:].reshape((n, n))
            pred_mat = X_mat @ theta_v.T - P_mat @ gamma_v.T
            _shared["residual_mat"] = D_mat - pred_mat
        else:
            _shared["residual_mat"] = None

        abs_vars = np.abs(vars)
        # For most q (=4, 1+logd), p_minus_one is non-integer; element-wise ** is the main per-iter cost.
        # Merge two ** into one via |v|^p = |v|^(p-1) * |v|.
        if is_p2:
            # p=2 specialization: |v|^(p-1) = |v|; sum(|v|^p) = v·v; skip abs and ** entirely.
            abs_pow_p1 = abs_vars
            sum_p = float(np.dot(vars, vars))
        elif is_p1:
            # p=1 specialization: |v|^0 = 1 (scalar broadcast, no D-sized array); sum(|v|^1) = sum(|v|).
            abs_pow_p1 = 1.0
            sum_p = float(abs_vars.sum())
        else:
            abs_pow_p1 = abs_vars ** p_minus_one
            sum_p = float(np.dot(abs_pow_p1, abs_vars))
        _shared["abs_pow_p1"] = abs_pow_p1
        _shared["sign_vars"] = np.sign(vars)
        _shared["pnorm_val"] = sum_p ** inv_p

    def objective(vars):  # Data term: sum of squared errors over samples (not averaged by N); regularizer unchanged.
        _refresh_shared(vars)
        if not _shared["finite"]:
            return 1e300
        pnorm_val = _shared["pnorm_val"]
        if N_samples == 0:
            return 0.5 * eta_k * pnorm_val ** 2
        residual_mat = _shared["residual_mat"]
        # np.einsum('ij,ij->', ...) avoids one intermediate array vs np.sum(a*a); small gain for (N,n), adds up as N grows.
        lt = 0.5 * float(np.einsum("ij,ij->", residual_mat, residual_mat))
        return lt + 0.5 * eta_k * pnorm_val ** 2

    def compute_gradient(vars):
        _refresh_shared(vars)
        if not _shared["finite"]:
            return np.zeros_like(vars, dtype=np.float64)
        pnorm_val = _shared["pnorm_val"]

        if N_samples > 0:
            # Vectorized per-t accumulation -> single matrix multiply
            #   grad_theta = sum_t outer(residual_t, -X_t) = -residual_mat.T @ X_mat
            #   grad_gamma = sum_t outer(residual_t,  p_t) =  residual_mat.T @ P_mat
            residual_mat = _shared["residual_mat"]
            # Write directly into pre-allocated buffer via dot(..., out=), replacing two D-sized allocations + copies from concatenate([theta, gamma]) / N with in-place matmul and scaling.
            np.dot(residual_mat.T, X_mat, out=theta_grad_view.reshape(n, n * d))
            np.dot(residual_mat.T, P_mat, out=gamma_grad_view.reshape(n, n))
            np.multiply(theta_grad_view, -inv_N, out=theta_grad_view)
            np.multiply(gamma_grad_view, inv_N, out=gamma_grad_view)
        else:
            data_grad_buf.fill(0.0)

        # Regularizer gradient: add epsilon in denominator to prevent blow-up when p_norm is tiny
        if p >= 2:
            reg_denom = max(pnorm_val ** (p - 2), 1e-10)
        else:
            reg_denom = pnorm_val ** (p - 2) + 1e-10
        # Reuse sign(v) and |v|^(p-1) from _refresh_shared; saves one abs+sign+** pass.
        reg_grad = (eta_k / reg_denom) * _shared["sign_vars"] * _shared["abs_pow_p1"]
        # Total gradient = normalized data term + regularizer term
        grad_vars = data_grad_buf + reg_grad
        return grad_vars
        

    lr_coeff = LR_COEFF_BY_SNR.get(snr_tier, 2.0)
    denom_dim = float(np.power(float(max(d, 1)), float(LR_DIM_SCALE_EXP)))
    # Use eta_k ** ETA_LR_EXP so lr does not become too small at small eta_k; more balanced convergence across eta_k
    learning_rate = np.maximum(lr_coeff * (eta_k ** ETA_LR_EXP) / float(exptime) / denom_dim, MIN_LR)
    if is_p1:
        learning_rate = float(max(learning_rate * P1_LR_BOOST, P1_MIN_LR))

    vars_opt, loss_history = gradient_descent_projected(vars0, learning_rate=learning_rate, epsilon=epsilon_k, num_iters=1000)

    theta = vars_opt[:n * n * d].reshape((n, n * d))
    gamma = vars_opt[n * n * d:].reshape((n, n))
    return theta, gamma