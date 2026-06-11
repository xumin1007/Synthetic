"""Plot final cumulative regret vs epsilon for the five algorithms.

Reads the minimal algorithm outputs produced by Toy_main.py:
    - reward CSV:  columns method,param,k,t,reward (all five methods merged)
    - oracle CSV:  columns k,t,oracle_reward

Cumulative regret is reconstructed on the fly as
    regret[method,param,k,t] = cumsum_t( oracle_reward[k,t] - reward[method,param,k,t] ).
The final (t = T-1) cumulative regret of every (method, param) is summarised into a
table, then plotted with PGD as a curve over epsilon (legend split by eta) and the
other methods as horizontal references across the epsilon range.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as _sp_stats  # type: ignore
except Exception:
    _sp_stats = None


ETA_MARKERS = ("^", "o", "D")
EPS_MARKERS = ("*", "^", "o", "D")
MOSEK_COLOR = "#172A3A"
PGD_COLOR_BY_ETA = {
    "0.0005": "#D55E00",
    "0.005": "#0072B2",
    "0.05": "#B8860B",
}
ILQX_COLOR = "#009E73"
SPO_COLOR = "#CC79A7"
ADAGRAD_COLOR = "#56B4E9"
ADAGRAD_LR_MARKERS = ("^", "o", "D", "+", "v")
ADAGRAD_LR_GRID = (1e-2, 1e-1, 1.0, 10.0)
_METHOD_SORT_ORDER = {"MOSEK": 0, "PGD": 1, "ILQX": 2, "SPO+": 3, "AdaGrad": 4}


@dataclass(frozen=True)
class CurveKey:
    method: str
    param: str


def _parse_param_value(param: str, key: str) -> float | None:
    for part in param.split(","):
        p = part.strip()
        if p.startswith(f"{key}="):
            try:
                return float(p.split("=", 1)[1])
            except Exception:
                return None
    return None


def _float_tag(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.12g}"


def _marker_for_eta(eta: float | None) -> str:
    etas = [5e-4, 5e-3, 5e-2]
    if eta is None:
        return ETA_MARKERS[0]
    j = int(np.argmin(np.abs(np.asarray(etas, dtype=np.float64) - float(eta))))
    return ETA_MARKERS[j % len(ETA_MARKERS)]


def _marker_for_eps(eps: float | None) -> str:
    epses = [5e-2, 5e-1, 5e0, 5e1]
    if eps is None:
        return EPS_MARKERS[0]
    j = int(np.argmin(np.abs(np.asarray(epses, dtype=np.float64) - float(eps))))
    return EPS_MARKERS[j % len(EPS_MARKERS)]


def _marker_for_lr_adagrad(lr: float | None) -> str:
    if lr is None:
        return ADAGRAD_LR_MARKERS[0]
    j = int(np.argmin(np.abs(np.asarray(ADAGRAD_LR_GRID, dtype=np.float64) - float(lr))))
    return ADAGRAD_LR_MARKERS[j % len(ADAGRAD_LR_MARKERS)]


def _curve_sort_key(ck: CurveKey) -> tuple[int, str]:
    order = _METHOD_SORT_ORDER.get(ck.method, 99)
    if ck.method == "SPO+":
        lr = _parse_param_value(ck.param, "lr")
        # SPO+ legend sorted by lr ascending (lr=1e-5 shown first)
        return (order, f"{float(lr) if lr is not None else float('inf'):020.12f}")
    return (order, ck.param)


def _build_style(method: str, param: str) -> dict[str, object]:
    eta = _parse_param_value(param, "eta")
    eps = _parse_param_value(param, "eps")
    lam = _parse_param_value(param, "lambda")
    lr = _parse_param_value(param, "lr")

    out: dict[str, object] = {"linewidth": 0.8, "alpha": 0.9, "markersize": 4.0}
    if method == "MOSEK":
        out["color"] = MOSEK_COLOR
        out["linestyle"] = "-"
        out["marker"] = _marker_for_eta(eta)
        return out
    if method == "PGD":
        out["color"] = PGD_COLOR_BY_ETA.get(_float_tag(eta), "#0072B2")
        out["linestyle"] = "--"
        out["marker"] = _marker_for_eps(eps)
        return out
    if method == "ILQX":
        out["color"] = ILQX_COLOR
        out["linestyle"] = "-"
        out["marker"] = _marker_for_eta(lam)
        return out
    if method == "SPO+":
        out["color"] = SPO_COLOR
        out["linestyle"] = "-"
        if lr is not None and math.isclose(lr, 1e-5, rel_tol=0.0, abs_tol=1e-12):
            out["marker"] = "*"
            out["markersize"] = 8.0
        else:
            out["marker"] = _marker_for_eta(lr)
        return out
    if method == "AdaGrad":
        out["color"] = ADAGRAD_COLOR
        out["linestyle"] = "-."
        out["marker"] = _marker_for_lr_adagrad(lr)
        return out
    out["color"] = "#333333"
    out["linestyle"] = "-"
    out["marker"] = "^"
    return out


def _critical_value_95(k_size: int) -> float:
    if k_size > 1 and _sp_stats is not None:
        try:
            return float(_sp_stats.t.ppf(0.975, k_size - 1))
        except Exception:
            pass
    return 1.959963984540054


def _read_reward_csv(path: str) -> dict[CurveKey, np.ndarray]:
    """Read a reward CSV (method,param,k,t,reward) into {CurveKey: (k, t) array}."""
    grouped: dict[CurveKey, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    max_k = -1
    max_t = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "param", "k", "t", "reward"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"reward CSV column mismatch: {path}")
        for row in reader:
            method = str(row["method"]).strip()
            param = str(row["param"]).strip()
            k = int(row["k"])
            t = int(row["t"])
            val = float(row["reward"])
            grouped[CurveKey(method, param)][k][t] = val
            max_k = max(max_k, k)
            max_t = max(max_t, t)
    if max_k < 0 or max_t < 0:
        raise ValueError(f"empty reward csv: {path}")
    k_size = max_k + 1
    t_size = max_t + 1
    out: dict[CurveKey, np.ndarray] = {}
    for ck, k_map in grouped.items():
        arr = np.full((k_size, t_size), np.nan, dtype=np.float64)
        for k, t_map in k_map.items():
            for t, val in t_map.items():
                arr[k, t] = val
        out[ck] = arr
    return out


def _read_oracle_csv(path: str) -> np.ndarray:
    """Read an oracle CSV (k,t,oracle_reward) into a (k, t) array."""
    grouped: dict[int, dict[int, float]] = defaultdict(dict)
    max_k = -1
    max_t = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"k", "t", "oracle_reward"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"oracle CSV column mismatch: {path}")
        for row in reader:
            k = int(row["k"])
            t = int(row["t"])
            grouped[k][t] = float(row["oracle_reward"])
            max_k = max(max_k, k)
            max_t = max(max_t, t)
    if max_k < 0 or max_t < 0:
        raise ValueError(f"empty oracle csv: {path}")
    arr = np.full((max_k + 1, max_t + 1), np.nan, dtype=np.float64)
    for k, t_map in grouped.items():
        for t, v in t_map.items():
            arr[k, t] = v
    return arr


def _cumulative_regret_curves(
    reward_csv: str,
    oracle_csv: str,
) -> dict[CurveKey, np.ndarray]:
    """Turn the merged reward CSV into per-curve cumulative regret arrays."""
    rewards = _read_reward_csv(reward_csv)
    oracle = _read_oracle_csv(oracle_csv)
    curves: dict[CurveKey, np.ndarray] = {}
    for ck, reward_arr in rewards.items():
        if reward_arr.shape != oracle.shape:
            raise ValueError(
                f"reward shape {reward_arr.shape} does not match oracle shape {oracle.shape}: {ck}"
            )
        curves[ck] = np.cumsum(oracle - reward_arr, axis=1)
    return curves


def save_regret_table(curves: dict[CurveKey, np.ndarray], out_path: str) -> None:
    """Write a CSV summarising the final cumulative regret for every (method, param) pair."""
    any_curve = next(iter(curves.values()))
    k_size = any_curve.shape[0]
    crit = _critical_value_95(k_size)

    rows: list[dict[str, object]] = []
    for ck in sorted(curves.keys(), key=_curve_sort_key):
        arr = curves[ck]
        final = arr[:, -1]  # shape (k_size,) — last time step
        mean = float(np.nanmean(final))
        std = float(np.nanstd(final, ddof=1)) if k_size > 1 else 0.0
        se = std / math.sqrt(k_size) if k_size > 0 else 0.0
        ci_half = crit * se
        rows.append(
            {
                "method": ck.method,
                "param": ck.param,
                "final_regret_mean": mean,
                "final_regret_std": std,
                "final_regret_95ci_lo": mean - ci_half,
                "final_regret_95ci_hi": mean + ci_half,
            }
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "param",
                "final_regret_mean",
                "final_regret_std",
                "final_regret_95ci_lo",
                "final_regret_95ci_hi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _method_color(method: str) -> str:
    if method == "MOSEK":
        return MOSEK_COLOR
    if method == "PGD":
        return PGD_COLOR_BY_ETA["0.005"]
    if method == "ILQX":
        return ILQX_COLOR
    if method == "SPO+":
        return SPO_COLOR
    if method == "AdaGrad":
        return ADAGRAD_COLOR
    return "#333333"


def plot_final_regret_vs_eps(table_csv: str, out_path: str, title: str) -> None:
    """
    Plot final regret from summary table:
    - PGD: x=eps, legend split by eta
    - methods without eps: horizontal reference lines across eps range
    """
    required = {"method", "param", "final_regret_mean", "final_regret_95ci_lo", "final_regret_95ci_hi"}
    pgd_by_eta: dict[float, list[tuple[float, float, float, float]]] = defaultdict(list)
    other_rows: list[tuple[str, str, float, float, float]] = []
    eps_values: list[float] = []

    with open(table_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"table CSV column mismatch: {table_csv}")
        for row in reader:
            method = str(row["method"]).strip()
            param = str(row["param"]).strip()
            mean = float(row["final_regret_mean"])
            lo = float(row["final_regret_95ci_lo"])
            hi = float(row["final_regret_95ci_hi"])
            eps = _parse_param_value(param, "eps")
            eta = _parse_param_value(param, "eta")
            if method == "PGD" and eps is not None and eta is not None:
                pgd_by_eta[eta].append((eps, mean, lo, hi))
                eps_values.append(eps)
            else:
                other_rows.append((method, param, mean, lo, hi))

    if not eps_values:
        raise ValueError(f"table CSV has no usable PGD eps values: {table_csv}")

    unique_eps = sorted(set(eps_values))
    x_min, x_max = unique_eps[0], unique_eps[-1]

    plt.figure(figsize=(14, 6.5))
    # PGD: lines over eps, legend by eta
    for eta in sorted(pgd_by_eta.keys()):
        items = sorted(pgd_by_eta[eta], key=lambda z: z[0])
        xs = np.asarray([it[0] for it in items], dtype=np.float64)
        ys = np.asarray([it[1] for it in items], dtype=np.float64)
        los = np.asarray([it[2] for it in items], dtype=np.float64)
        his = np.asarray([it[3] for it in items], dtype=np.float64)
        color = PGD_COLOR_BY_ETA.get(_float_tag(eta), "#0072B2")
        marker = _marker_for_eta(eta)
        yerr = np.vstack([ys - los, his - ys])
        plt.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt=f"-{marker}",
            color=color,
            linewidth=1.6,
            markersize=7.0,
            capsize=3.0,
            label=rf"AMLE-PGD ($\mathring{{\eta}}={eta:g}$)",
        )

    # Non-PGD: no eps axis meaning -> use horizontal references across eps range.
    # Add slight x-jitter for markers near x_max to reduce overlap.
    jitter_factors = (0.72, 0.80, 0.90, 0.95, 1.02)
    sorted_other = sorted(other_rows, key=lambda r: _curve_sort_key(CurveKey(r[0], r[1])))
    for idx, (method, param, mean, lo, hi) in enumerate(sorted_other):
        color = _method_color(method)
        style = _build_style(method, param)
        marker = style.get("marker", "o")
        markersize = float(style.get("markersize", 5.0))
        x_marker = x_max * jitter_factors[idx % len(jitter_factors)]
        ci_half = max(mean - lo, hi - mean)
        plt.hlines(mean, x_min, x_max, colors=color, linestyles="--", linewidth=1.5, alpha=0.75)
        plt.fill_between(
            [x_min, x_max],
            [mean - ci_half, mean - ci_half],
            [mean + ci_half, mean + ci_half],
            color=color,
            alpha=0.06,
            linewidth=0.0,
        )
        legend_label = f"{method} ({param})"
        if method == "MOSEK":
            eta = _parse_param_value(param, "eta")
            if eta is not None:
                legend_label = rf"AMLE-MOSEK ($\mathring{{\eta}}={eta:g}$)"
        if method == "ILQX":
            lam = _parse_param_value(param, "lambda")
            if lam is not None:
                legend_label = rf"{method} ($\mathring{{\lambda}}={lam:g}$)"
        if method == "SPO+":
            lr = _parse_param_value(param, "lr")
            if lr is not None:
                legend_label = f"{method} (lr={lr:g})"
        plt.plot(
            [x_marker],
            [mean],
            marker=marker,
            markersize=max(7.0, markersize),
            color=color,
            label=legend_label,
        )

    plt.xscale("log")
    plt.xticks(unique_eps, [f"{x:g}" for x in unique_eps])
    plt.xlabel(r"$\mathring{\varepsilon}$", fontsize=24)
    plt.ylabel("Regret", fontsize=24)
    plt.tick_params(axis="both", labelsize=20)
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute final regret from oracle reward; plot final regret vs eps (PGD grouped by eta, other methods as reference lines)"
    )
    parser.add_argument(
        "--reward-csv",
        default="results/Low_SNR/reward_by_method_param_t.csv",
        help="Merged reward CSV (MOSEK/PGD/ILQX/SPO+/AdaGrad)",
    )
    parser.add_argument(
        "--oracle-csv",
        default="results/Low_SNR/oracle_reward_by_t.csv",
        help="Oracle reward CSV (k,t,oracle_reward)",
    )
    parser.add_argument(
        "--table-csv",
        default="results/Low_SNR/final_regret_table.csv",
        help="Output summary table CSV path (final-step cumulative regret per method/param)",
    )
    parser.add_argument(
        "--eps-out",
        default="results/Low_SNR/final_regret_vs_eps.pdf",
        help="Output eps-view figure path (x=eps, y=final regret)",
    )
    parser.add_argument(
        "--eps-title",
        default="Low SNR: final regret vs eps (PGD eta legend + other method refs)",
        help="Eps-view figure title",
    )
    args = parser.parse_args()

    curves = _cumulative_regret_curves(args.reward_csv, args.oracle_csv)
    save_regret_table(curves=curves, out_path=args.table_csv)
    print(f"Saved regret table to: {args.table_csv}")
    plot_final_regret_vs_eps(table_csv=args.table_csv, out_path=args.eps_out, title=args.eps_title)
    print(f"Saved eps-view plot to: {args.eps_out}")


if __name__ == "__main__":
    main()
