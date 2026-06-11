# q-sensitivity: online pricing benchmark (MOSEK vs PGD, q = 2 / 4 / 1+ln(d))

Self-contained, open-source version of the q-sensitivity experiment. It compares
**MOSEK** and **PGD** across the regularizer geometry `q in {2, 4, 1+ln(d)}`
(the third tier is the dimension-dependent theory exponent, `ln` = natural log of
the feature dimension `d`; conjugate exponent `p = q/(q-1)`), under **two
feature-generation modes**:

| Mode             | Feature sampler                          | Geometry          |
|------------------|------------------------------------------|-------------------|
| `Gaussian`       | `truncate_normal_sample_time_varying`    | l2-ball (q = 2)   |
| `l4_heavy_tail`  | `l4_heavy_tail_sample`                    | l4 heavy-tail     |

For each feature mode the same feature trajectory `X_{k,t}` is shared by every
`q`; only the MLE / PGD regularizer (via `p`) and the `eta`/`eps` scaling change.
The exploration schedule uses a fixed `REFERENCE_Q`, so the differences across
`q` isolate the effect of the regularizer geometry.

> ILQX (Lasso) and SPO+ are omitted here because they do not depend on `q`.

## Outputs

Following the original `run_q_sensitivity.py`, the experiment is run only at the
**low SNR** tier, with the target SNR set per feature mode: Gaussian (l2-ball)
uses `SNR = 1.0` and l4_heavy_tail (l4-ball) uses `SNR = 1e-3` (see
`Config.LOW_SNR_BY_MODE`). Outputs are written per feature mode to
`results/<mode>/Low_SNR/`:

1. `reward_by_method_param_t.csv` — `method, param, k, t, reward`
2. `oracle_reward_by_t.csv` — `k, t, oracle_reward`
3. `solver_times.csv` — `method, param, k, solve_idx, time_s`
4. `cumulative_regret_by_method_param_t.csv` — `method, param, k, t, cumulative_regret`
5. `mean_cumulative_regret_by_method_param_t.csv` — `method, param, t, mean_cumulative_regret`

Plus a regret figure:

6. `q_sensitivity_regret.pdf` — mean (over `k`) cumulative regret, one curve per
   `(method, q, param)` (color per method/q, markers per `eta`/`eps`).

Cumulative regret is `cumsum_t(oracle - reward)`; the mean-over-`k` file is ready
to plot directly (one regret curve per `(method, param)`). `param` encodes the
swept `q`, e.g. `q=2,eta=0.005` (MOSEK) or `q=4,eta=0.005,eps=0.5` (PGD).

## Setup

```bash
pip install -r requirements.txt
```

`cvxMLEpnorm` (MOSEK runner) prefers the **MOSEK** solver with a valid license,
but transparently falls back to CLARABEL / SCS (bundled with cvxpy) when MOSEK is
unavailable. See `requirements.txt` for the (optional) MOSEK entry.

## Run

The code is packaged as `q_sensitivity`; run the experiment as a module from the
repository root (outputs are written to `<repo_root>/results/`):

```bash
# full run: both feature modes at their per-mode low SNR
python -m q_sensitivity.experiments.run_q_sensitivity_comparison

# fast smoke test (short horizon, single repetition)
QS_SMOKE=1 python -m q_sensitivity.experiments.run_q_sensitivity_comparison
```

## Configuration

- `q_sensitivity/config/Config.py` — all experiment settings: SNR tiers,
  optimizer constraints, feature-sampling defaults, the swept `q` grid,
  reference `q`, `eta`/`eps` grids, and per-mode fixed configs
  (`GAUSSIAN_FIXED`, `L4_FIXED`).
- `q_sensitivity/experiments/run_q_sensitivity_comparison.py` — problem size
  (`TOY_T`, `TOY_K`, `TOY_D`) and the experiment logic.

## Package layout

```
q_sensitivity/
├── config/        # Config.py
├── solvers/       # cvxMLEpnorm.py (MOSEK), Gradient_projected_Search.py (PGD)
├── features/      # truncate_normal_sample_time_varying.py (feature samplers)
├── utils/         # normcalculate.py, project.py, randomP.py
└── experiments/   # run_q_sensitivity_comparison.py (main script)
```

- `experiments/run_q_sensitivity_comparison.py` — main experiment script.
- `config/Config.py`, `solvers/cvxMLEpnorm.py`,
  `solvers/Gradient_projected_Search.py`,
  `features/truncate_normal_sample_time_varying.py`, `utils/randomP.py`,
  `utils/normcalculate.py`, `utils/project.py` — supporting modules.
