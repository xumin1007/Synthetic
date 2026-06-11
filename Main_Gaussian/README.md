# Gaussian feature: online pricing benchmark (MOSEK / PGD / ILQX / SPO+ / AdaGrad)

Self-contained, open-source version of the toy online-pricing experiment under an
**Gaussian feature geometry** (feature norm budget `q = 4`, conjugate exponent
`p = 4/3`). It compares five online estimators and writes **only** the raw results
needed to reconstruct any downstream metric (regret, holdout selection, etc.).

## Methods

Each estimator core lives in its own module:

| Name  | Estimator                                            | File                          |
|-------|-----------------------------------------------------|-------------------------------|
| MOSEK | p-norm regularized MLE solved with MOSEK            | `cvxMLEpnorm.py`              |
| PGD   | Projected gradient descent on the same likelihood   | `Gradient_projected_Search.py` |
| ILQX  | L1 (Lasso) regularized least squares                | `cvxLasso_linear.py`         |
| SPO+  | PyEPO SPO+ on doubly-robust revenue labels          | `spo_plus.py`                |
| AdaGrad | Recursive online least squares with L1 projection | `AdaGrad.py`                 |

Each method is swept over its full hyper-parameter grid; the reward of every
`(method, param, k, t)` is recorded so that holdout selection / regret can be
computed afterwards from the CSVs.

## Outputs

Running the script produces, per SNR tier, under `results/<Tier>_SNR/`:

1. `reward_by_method_param_t.csv` — `method, param, k, t, reward`
   (realized revenue of each method/parameter at every step `t` and repetition `k`)
2. `oracle_reward_by_t.csv` — `k, t, oracle_reward`
   (paper-style conditional oracle expected revenue per step)
3. `solver_times.csv` — `method, param, k, solve_idx, time_s`
   (wall-clock time of each retraining/solve)
4. `cumulative_regret_by_method_param_t.csv` — `method, param, k, t, cumulative_regret`
   (`cumsum_t(oracle - reward)`)
5. `mean_cumulative_regret_by_method_param_t.csv` — `method, param, t, mean_cumulative_regret`
   (averaged over the `K` repetitions; ready to plot one regret curve per `(method, param)`)
6. `adagrad_solving_time.csv` — `method, param, k, t, time_s`
   (AdaGrad per-step recursive update time; one record for every time step)

## Setup

```bash
pip install -r requirements.txt
```

`cvxMLEpnorm` and `cvxLasso_linear` require the **MOSEK** solver; SPO+ requires
**Gurobi** (via PyEPO). Both need valid licenses.

## Run

```bash
# full run (all three SNR tiers)
python run_Gaussian_comparison.py

# fast smoke test (short horizon, tiny grids)
Gaussian_SMOKE=1 python run_Gaussian_comparison.py
```

## Configuration

Problem size, price grid and hyper-parameter grids are defined at the top of
`run_Gaussian_comparison.py` (`TOY_T`, `TOY_K`, `PRICE_GRID`, `ETA_GRID`,
`EPS_GRID`, `LAMBDA_GRID`, `SPO_*`, `ADAGRAD_LR_GRID`). SNR tiers and feature-sampling parameters
come from `Config.py`.

## Project layout

```
opensource_Gaussian/
├── run_Gaussian_comparison.py   # thin launcher (keeps the run command stable)
├── requirements.txt
├── README.md
├── .gitignore
├── gaussian_pricing/            # importable package
│   ├── __init__.py
│   ├── experiment.py            # main experiment logic (run via the launcher)
│   ├── algorithms/              # the five online estimator cores
│   │   ├── __init__.py
│   │   ├── cvxMLEpnorm.py       #   MOSEK  (p-norm regularized MLE)
│   │   ├── Gradient_projected_Search.py  # PGD (projected gradient on the MLE)
│   │   ├── cvxLasso_linear.py   #   ILQX   (L1 / Lasso least squares)
│   │   ├── spo_plus.py          #   SPO+   (PyEPO predict-then-optimize)
│   │   └── AdaGrad.py           #   AdaGrad (recursive OLS + L1 projection)
│   └── utils/                   # data generation + shared math helpers
│       ├── __init__.py
│       ├── Config.py            #   SNR tiers, feature-sampling params, precompute
│       ├── truncate_normal_sample_time_varying.py  # truncated-normal features
│       ├── randomP.py           #   exploration price sampler
│       ├── normcalculate.py     #   p-norm helper
│       └── project.py           #   L1-ball projection
├── plotting/                    # figures from the result CSVs
│   ├── plot_best_curves.py
│   └── plot_final_regret_vs_eps.py
└── results/                     # experiment outputs land here (per-SNR subdirs)
```
