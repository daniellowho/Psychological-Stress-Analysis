"""Evaluation, bootstrap, calibration and threshold utilities (numpy/sklearn/scipy only; no model training).

Shared by the stress (Step 3), emotion (Step 4) and stressor (Step 5) components. Binary helpers take the
probability of the POSITIVE class (`p`, shape (n,)). Temperature scaling fits ONE scalar on validation logits;
it never touches model weights.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

METRIC_KEYS = ("accuracy", "precision_stress", "recall_stress", "f1_stress", "f1_macro", "roc_auc", "pr_auc", "brier")


# ================================================================ probabilities
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def positive_proba(logits: np.ndarray, temperature: float = 1.0, positive: int = 1) -> np.ndarray:
    """P(positive class) from (n, 2) logits, optionally temperature-scaled."""
    return softmax(logits, temperature)[:, positive]


# ================================================================ point metrics
def binary_metrics(y_true: Sequence[int], p: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    """Accuracy, precision/recall/F1 of the positive (stress) class, macro-F1, ROC-AUC, PR-AUC (average precision),
    Brier score and the confusion matrix. AUCs are NaN when only one class is present."""
    from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, labels=[1], average=None, zero_division=0)
    two_classes = len(np.unique(y)) == 2
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum()); tp = int(((y == 1) & (yhat == 1)).sum())
    return {
        "n": int(len(y)), "prevalence": float(y.mean()) if len(y) else float("nan"), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision_stress": float(prec[0]), "recall_stress": float(rec[0]), "f1_stress": float(f1[0]),
        "f1_macro": float(f1_score(y, yhat, average="macro", labels=[0, 1], zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "predicted_positive_rate": float(yhat.mean()) if len(y) else float("nan"),
    }


def bootstrap_metrics(y_true: Sequence[int], p: Sequence[float], threshold: float = 0.5, n_resamples: int = 1000,
                      groups: Sequence[Any] | None = None, seed: int = 42, level: float = 0.95,
                      keys: Sequence[str] = METRIC_KEYS) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs. With `groups` (e.g. post_id) whole groups are resampled (cluster bootstrap), because
    segments of one post are not independent. Resamples where a metric is undefined (one class) are skipped for it."""
    y = np.asarray(y_true, dtype=int)
    pp = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    point = binary_metrics(y, pp, threshold)
    if groups is not None:
        g = pd.Series(np.asarray(groups)).astype(str)
        codes, uniq = pd.factorize(g)
        members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_resamples):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in pick])
        m = binary_metrics(y[idx], pp[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    a = (1 - level) / 2
    out = {}
    for k in keys:
        d = np.asarray(draws[k], dtype=float)
        d = d[~np.isnan(d)]
        out[k] = {"point": point[k], "lo": float(np.quantile(d, a)) if d.size else float("nan"),
                  "hi": float(np.quantile(d, 1 - a)) if d.size else float("nan"), "n_valid_resamples": int(d.size)}
    return out


def paired_bootstrap_gain(y_true: Sequence[int], p: Sequence[float], threshold_new: float, threshold_ref: float = 0.5,
                          metric: str = "f1_stress", n_resamples: int = 1000, groups: Sequence[Any] | None = None,
                          seed: int = 42, level: float = 0.95) -> dict[str, float]:
    """Gain metric(threshold_new) - metric(threshold_ref) on the SAME resampled rows (paired), with a percentile CI.
    Cluster resampling when `groups` is given."""
    y = np.asarray(y_true, dtype=int)
    pp = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is not None:
        codes, uniq = pd.factorize(pd.Series(np.asarray(groups)).astype(str))
        members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    gains = []
    for _ in range(n_resamples):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        gains.append(binary_metrics(y[idx], pp[idx], threshold_new)[metric] - binary_metrics(y[idx], pp[idx], threshold_ref)[metric])
    g = np.asarray(gains)
    a = (1 - level) / 2
    point = binary_metrics(y, pp, threshold_new)[metric] - binary_metrics(y, pp, threshold_ref)[metric]
    return {"gain": float(point), "lo": float(np.quantile(g, a)), "hi": float(np.quantile(g, 1 - a)),
            "share_resamples_gain_gt_0": float((g > 0).mean()), "metric": metric, "n_resamples": int(n_resamples)}


def mean_std(rows: Sequence[dict[str, Any]], keys: Sequence[str] = METRIC_KEYS) -> dict[str, dict[str, float]]:
    """Mean and sample std (ddof=1) of each metric across runs (e.g. seeds)."""
    df = pd.DataFrame([{k: r[k] for k in keys} for r in rows])
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=1)) if len(df) > 1 else 0.0, "n_runs": int(len(df))}
            for k in keys}


# ================================================================ calibration
def ece(y_true: Sequence[int], p: Sequence[float], n_bins: int = 15) -> float:
    """Expected calibration error of P(positive) with equal-width bins: sum_b (n_b/n) * |mean(y)_b - mean(p)_b|."""
    t = reliability_table(y_true, p, n_bins)
    n = t["n"].sum()
    return float((t["n"] / n * (t["frac_positive"] - t["mean_p"]).abs()).sum()) if n else float("nan")


def reliability_table(y_true: Sequence[int], p: Sequence[float], n_bins: int = 15) -> pd.DataFrame:
    """Per-bin count, mean predicted probability and observed positive fraction (empty bins dropped)."""
    y = np.asarray(y_true, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(pp, edges[1:-1], right=False), 0, n_bins - 1)
    df = pd.DataFrame({"bin": b, "y": y, "p": pp})
    t = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), frac_positive=("y", "mean")).reset_index()
    t["bin_lo"], t["bin_hi"] = edges[t["bin"]], edges[t["bin"] + 1]
    return t


def nll(logits: np.ndarray, y_true: Sequence[int], temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of temperature-scaled logits."""
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    y = np.asarray(y_true, dtype=int)
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y_true: Sequence[int], bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Temperature scaling (Guo et al., 2017): T = argmin NLL(logits / T) on VALIDATION data. Searches log T."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda lt: nll(logits, y_true, float(np.exp(lt))),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded", options={"xatol": 1e-5})
    return float(np.exp(res.x))


def choose_threshold(y_true: Sequence[int], p: Sequence[float], criterion: str = "max_f1_stress",
                     grid: Sequence[float] = (0.05, 0.95, 0.01)) -> dict[str, Any]:
    """Decision threshold chosen on VALIDATION. `max_f1_stress`: maximise F1 of the stress class; ties are broken
    towards 0.5 (least deviation from the default). `youden`: maximise TPR - FPR. Returns threshold + curve."""
    start, stop, step = grid
    ts = np.round(np.arange(start, stop + step / 2, step), 6)
    rows = []
    for t in ts:
        m = binary_metrics(y_true, p, float(t))
        tpr = m["recall_stress"]
        c = m["confusion"]
        fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
        rows.append({"threshold": float(t), "f1_stress": m["f1_stress"], "f1_macro": m["f1_macro"], "youden": tpr - fpr})
    curve = pd.DataFrame(rows)
    key = {"max_f1_stress": "f1_stress", "youden": "youden"}.get(criterion)
    if key is None:
        raise ValueError(f"unknown threshold criterion {criterion!r}")
    best = curve[key].max()
    cand = curve[np.isclose(curve[key], best)]
    chosen = float(cand.iloc[(cand["threshold"] - 0.5).abs().argsort().iloc[0]]["threshold"])
    return {"threshold": chosen, "criterion": criterion, "value_on_validation": float(best), "curve": curve}


def plot_reliability(curves: dict[str, tuple[Sequence[int], Sequence[float]]], n_bins: int = 15,
                     title: str = "Reliability (validation)", path: str | None = None):
    """Reliability diagram: observed stress fraction vs mean predicted probability per bin, one line per curve.
    Palette slots 1-2 of the default categorical palette; recessive grid; legend always shown."""
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=120)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#52514e", label="perfect calibration")
    for (name, (y, p)), c in zip(curves.items(), colors):
        t = reliability_table(y, p, n_bins)
        ax.plot(t["mean_p"], t["frac_positive"], marker="o", markersize=5, linewidth=2, color=c,
                label=f"{name} (ECE {ece(y, p, n_bins):.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(stress) in bin"); ax.set_ylabel("observed share labelled stress")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)                     # saved file is displayed instead (avoids duplicate figures / leaks)
    return fig


def metrics_by_group(df: pd.DataFrame, group_col: str, y_col: str = "label", p_col: str = "p",
                     threshold: float = 0.5, min_n: int = 1) -> pd.DataFrame:
    """Per-group metrics + error rates (aggregate only). Groups with n < min_n are reported with metrics blanked."""
    rows = []
    for g, d in df.groupby(group_col):
        m = binary_metrics(d[y_col], d[p_col], threshold)
        c = m["confusion"]
        row = {group_col: g, "n": m["n"], "prevalence": m["prevalence"], "accuracy": m["accuracy"],
               "f1_stress": m["f1_stress"], "roc_auc": m["roc_auc"],
               "false_positive_rate": c["fp"] / max(c["fp"] + c["tn"], 1),
               "false_negative_rate": c["fn"] / max(c["fn"] + c["tp"], 1)}
        if m["n"] < min_n:
            row = {k: (v if k in (group_col, "n") else np.nan) for k, v in row.items()}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

