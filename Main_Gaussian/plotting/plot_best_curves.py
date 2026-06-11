"""Plot the best-per-method cumulative regret curves for the five algorithms.

Reads the minimal algorithm outputs produced by Toy_main.py:
    - reward CSV:  columns method,param,k,t,reward (all five methods merged)
    - oracle CSV:  columns k,t,oracle_reward

Cumulative regret is reconstructed on the fly as
    regret[method,param,k,t] = cumsum_t( oracle_reward[k,t] - reward[method,param,k,t] ).
For each method the parameter with the lowest final mean cumulative regret is kept,
so the final figure shows exactly one (best) curve per method.
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


@dataclass(frozen=True)
class CurveKey:
    method: str
    param: str


_METHOD_ORDER = ["MOSEK", "PGD", "ILQX", "SPO+", "AdaGrad"]
_METHOD_COLOR = {
    "MOSEK": "#172A3A",
    "PGD": "#0072B2",
    "ILQX": "#009E73",
    "SPO+": "#CC79A7",
    "AdaGrad": "#56B4E9",
}


def _read_reward_csv(path: str) -> dict[CurveKey, np.ndarray]:
    """Read a reward CSV (method,param,k,t,reward) into {CurveKey: (k, t) array}."""
    grouped: dict[CurveKey, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    max_k = -1
    max_t = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        need = {"method", "param", "k", "t", "reward"}
        if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
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
        raise ValueError(f"reward CSV has no valid data: {path}")

    k_size, t_size = max_k + 1, max_t + 1
    out: dict[CurveKey, np.ndarray] = {}
    for ck, k_map in grouped.items():
        arr = np.full((k_size, t_size), np.nan, dtype=np.float64)
        for k, t_map in k_map.items():
            for t, v in t_map.items():
                arr[k, t] = v
        out[ck] = arr
    return out


def _read_oracle_csv(path: str) -> np.ndarray:
    """Read an oracle CSV (k,t,oracle_reward) into a (k, t) array."""
    grouped: dict[int, dict[int, float]] = defaultdict(dict)
    max_k = -1
    max_t = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        need = {"k", "t", "oracle_reward"}
        if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"oracle CSV column mismatch: {path}")
        for row in reader:
            k = int(row["k"])
            t = int(row["t"])
            grouped[k][t] = float(row["oracle_reward"])
            max_k = max(max_k, k)
            max_t = max(max_t, t)
    if max_k < 0 or max_t < 0:
        raise ValueError(f"oracle CSV has no valid data: {path}")
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


def _pick_best_by_method(curves: dict[CurveKey, np.ndarray]) -> dict[str, tuple[CurveKey, np.ndarray]]:
    best: dict[str, tuple[CurveKey, np.ndarray, float]] = {}
    for ck, arr in curves.items():
        score = float(np.nanmean(arr[:, -1]))
        prev = best.get(ck.method)
        if prev is None or score < prev[2]:
            best[ck.method] = (ck, arr, score)
    return {m: (v[0], v[1]) for m, v in best.items()}


def _parse_lr(param: str) -> float | None:
    for part in param.split(","):
        p = part.strip()
        if p.startswith("lr="):
            try:
                return float(p.split("=", 1)[1])
            except Exception:
                return None
    return None


def _critical_value_95(k_size: int) -> float:
    if k_size <= 1:
        return 0.0
    # Match existing script style; avoid extra scipy dependency
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(k_size, 1.96)


def _style_for_curve(method: str, param: str) -> dict[str, object]:
    style: dict[str, object] = {
        "color": _METHOD_COLOR.get(method, "#333333"),
        "linewidth": 1.8,
        "alpha": 0.95,
        "markersize": 4.0,
        "linestyle": "-",
    }
    if method == "PGD":
        style["linestyle"] = "--"
        style["marker"] = "o"
    elif method == "MOSEK":
        style["marker"] = "^"
    elif method == "ILQX":
        style["marker"] = "D"
    elif method == "AdaGrad":
        style["linestyle"] = "-."
        style["marker"] = "v"
    elif method == "SPO+":
        lr = _parse_lr(param)
        # SPO+ lr=1e-5 keeps purple color and uses * marker
        if lr is not None and math.isclose(lr, 1e-5, rel_tol=0.0, abs_tol=1e-12):
            style["marker"] = "*"
            style["markersize"] = 8.0
        else:
            style["marker"] = "s"
    return style


def _label(method: str, param: str) -> str:
    key_to_tex = {
        "eta": r"\mathring{\eta}",
        "eps": r"\mathring{\varepsilon}",
        "lambda": r"\mathring{\lambda}",
    }

    def _format_param_text(param_text: str) -> str:
        chunks: list[str] = []
        for part in param_text.split(","):
            p = part.strip()
            if "=" not in p:
                chunks.append(p)
                continue
            key, value = p.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in key_to_tex:
                chunks.append(f"${key_to_tex[key]}={value}$")
            else:
                chunks.append(f"{key}={value}")
        return ", ".join(chunks)

    if method == "SPO+":
        lr = _parse_lr(param)
        if lr is not None:
            return f"SPO+ (lr={lr:g})"
    if method == "AdaGrad":
        lr = _parse_lr(param)
        if lr is not None:
            return f"AdaGrad (lr={lr:g})"
    return f"{method} ({_format_param_text(param)})"


def plot_five_methods(
    reward_csv: str,
    oracle_csv: str,
    out_path: str,
    title: str,
) -> None:
    curves = _cumulative_regret_curves(reward_csv, oracle_csv)
    selected = _pick_best_by_method(curves)

    # Ensure the final figure has at most five method types (plot whatever is available)
    order = [m for m in _METHOD_ORDER if m in selected]
    if not order:
        raise ValueError("No plottable methods found.")

    k_size = next(iter(selected.values()))[1].shape[0]
    t_size = next(iter(selected.values()))[1].shape[1]
    xs = np.arange(t_size)
    crit = _critical_value_95(k_size)

    plt.figure(figsize=(12, 6))
    for idx, method in enumerate(order):
        ck, arr = selected[method]
        y = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0, ddof=1) / math.sqrt(k_size) if k_size > 1 else np.zeros_like(y)
        lo = y - crit * se
        hi = y + crit * se

        st = _style_for_curve(method, ck.param)
        st["markevery"] = (idx * 35, 320)
        color = st.get("color", "#333333")
        plt.fill_between(xs, lo, hi, color=color, alpha=0.08, linewidth=0)
        plt.plot(xs, y, label=_label(method, ck.param), **st)

    _axis_fs = 24
    plt.xlabel("t", fontsize=_axis_fs)
    plt.ylabel("Regret", fontsize=_axis_fs)
    plt.tick_params(axis="both", labelsize=_axis_fs)
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16, ncol=2, loc="upper left")
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read merged reward CSV, compute regret from oracle reward, plot best curves for five methods"
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
        "--out",
        default="results/Low_SNR/best_curves_five_methods.pdf",
        help="Output figure path",
    )
    parser.add_argument(
        "--title",
        default="Low SNR: MOSEK/PGD/ILQX/SPO+/AdaGrad cumulative regret",
        help="Figure title",
    )
    args = parser.parse_args()

    plot_five_methods(
        reward_csv=args.reward_csv,
        oracle_csv=args.oracle_csv,
        out_path=args.out,
        title=args.title,
    )
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
