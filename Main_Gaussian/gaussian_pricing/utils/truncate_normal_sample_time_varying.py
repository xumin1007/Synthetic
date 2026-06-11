import atexit
from collections import defaultdict

import numpy as np
from scipy.stats import truncnorm

_DEFAULT_SIGMA = 0.1 / 3  # original default std; kept for backward compatibility

_Q_STATS: dict[str, dict[str, float]] = defaultdict(
    lambda: {
        "calls": 0.0,            # total calls (= accepted samples)
        "total_draws": 0.0,      # total draws (including resamples)
        "rejected_draws": 0.0,   # draws rejected by q-norm constraint
        "resampled_calls": 0.0,  # calls that required resampling (draws > 1)
    }
)


def _q_label(q) -> str:
    qf = float(q)
    if np.isinf(qf):
        return "inf"
    return "{:.10g}".format(qf)


def _seed_from_call_l4(d, k, t, i) -> int:
    """Salt distinct from _seed_from_call so L4-ball trajectories do not reuse the per-q RNG stream."""
    seed = (
        (int(d) * 1_000_003)
        ^ (int(k) * 97_003)
        ^ (int(t) * 9_739)
        ^ (int(i) * 389)
        ^ 0x4C340000
    ) & 0xFFFFFFFF
    return int(seed)


def _seed_from_call(q, d, k, t, i) -> int:
    """
    Stable seed from (q, d, k, t, i). Create one RNG at function entry;
    draw continuously inside the while loop (do not reset the same seed each round).
    """
    qf = float(q)
    q_code = 2147483647 if np.isinf(qf) else int(round(qf * 1_000_000))
    seed = (
        (int(d) * 1_000_003)
        ^ (int(k) * 97_003)
        ^ (int(t) * 9_739)
        ^ (int(i) * 389)
        ^ q_code
    ) & 0xFFFFFFFF
    return int(seed)


def _report_q_constraint_stats() -> None:
    if not _Q_STATS:
        return
    print("\n=== truncate_normal_sample_time_varying: q-constraint stats ===")
    print("{:<12} {:>10} {:>16} {:>16} {:>16} {:>16}".format(
        "q", "calls", "total_draws", "rejected_draws", "active_rate", "resample_rate"
    ))
    for q_lbl in sorted(_Q_STATS.keys(), key=lambda x: (x != "inf", x)):
        s = _Q_STATS[q_lbl]
        calls = max(s["calls"], 1.0)
        active_rate = s["resampled_calls"] / calls
        # average extra resamples per call
        resample_rate = s["rejected_draws"] / calls
        print("{:<12} {:>10.0f} {:>16.0f} {:>16.0f} {:>15.2%} {:>15.4f}".format(
            q_lbl, s["calls"], s["total_draws"], s["rejected_draws"], active_rate, resample_rate
        ))


atexit.register(_report_q_constraint_stats)


def truncate_normal_sample_time_varying(q, C, a, b, d, k, t, i, mu=0, sigma=None):
    """
    Draw a truncated normal random vector on [a, b] with q-norm <= C.
    Mean and variance are tunable: larger sigma (variance=sigma^2) raises feature scale
    and thus lambda_x (minimum eigenvalue of sample covariance).
    When components are approximately iid with variance sigma^2, lambda_x ~ sigma^2 in theory;
    under truncation and q-norm constraints, actual lambda_x is typically <= sigma^2.

    Parameters
    ----------
    q, C, a, b, d, k, t, i : as before
    mu : float, optional
        Mean; default 0.
    sigma : float, optional
        Standard deviation (variance=sigma^2). None uses original default 0.1/3 for compatibility.

    Returns
    -------
    r : np.ndarray
        d-dimensional vector satisfying the constraints.
    """

    lower, upper = a, b
    size = (d,)
    # Truncated normal: standardized interval (lower-mu)/sigma ~ (upper-mu)/sigma
    a_std = (lower - mu) / sigma
    b_std = (upper - mu) / sigma

    rng = np.random.default_rng(_seed_from_call(q=q, d=d, k=k, t=t, i=i))
    draws = 0
    rejected = 0
    while True:
        draws += 1
        x = truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma, size=size, random_state=rng)
        q_norm = np.linalg.norm(x.flatten(), ord=q)
        if q_norm <= C:
            break
        rejected += 1

    q_lbl = _q_label(q)
    s = _Q_STATS[q_lbl]
    s["calls"] += 1.0
    s["total_draws"] += float(draws)
    s["rejected_draws"] += float(rejected)
    if draws > 1:
        s["resampled_calls"] += 1.0
    return x


def l4_heavy_tail_sample(
    d,
    k,
    t,
    i,
    *,
    base_sigma=0.1,
    collinear_coef=0.9,
    coupling_sigma=0.05,
    true_nonzero_idx=0,
    spurious_idx=1,
    spike_prob=0.05,
    spike_lo=5.0,
    spike_hi=10.0,
    mean_shift=0.15,
):
    """
    l4_ball features: weak independent Gaussian base + two collinear dimensions
    + small high-frequency targeted spikes on the collinear dimension.

    - All: x ~ N(0, base_sigma^2) i.i.d.
    - Collinearity: x[spurious] = collinear_coef * x[true_nonzero] + N(0, coupling_sigma^2)
    - Spikes: with probability spike_prob, add +/- U(spike_lo, spike_hi) on collinear dim (magnitude 5-10, not extreme leverage)
    - Finally: add mean_shift to the whole vector so components have positive mean when mean_shift>0 (default 0.15)
    """
    d_i = int(max(int(d), 1))
    rng = np.random.default_rng(_seed_from_call_l4(d_i, int(k), int(t), int(i)))
    x = rng.normal(0.0, float(base_sigma), size=d_i)
    ti = int(true_nonzero_idx)
    sj = int(spurious_idx)
    need = max(ti, sj) + 1
    if d_i >= need and ti != sj:
        x[sj] = float(collinear_coef) * x[ti] + float(
            rng.normal(0.0, float(coupling_sigma))
        )
        if rng.random() < float(spike_prob):
            mag = float(rng.uniform(float(spike_lo), float(spike_hi)))
            sgn = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float64)))
            x[sj] += sgn * mag
    x = x + float(mean_shift)
    q_lbl = "l4+collinear"
    s = _Q_STATS[q_lbl]
    s["calls"] += 1.0
    s["total_draws"] += 1.0
    s["rejected_draws"] += 0.0
    return x
