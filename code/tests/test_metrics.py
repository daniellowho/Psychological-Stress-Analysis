"""Metric, bootstrap, calibration and threshold tests on synthetic numbers (no models)."""

import numpy as np
import pytest

from stress_signals.metrics import (binary_metrics, bootstrap_metrics, choose_threshold, ece, fit_temperature,
                                    mean_std, metrics_by_group, nll, positive_proba, reliability_table)


def test_binary_metrics_known_values():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.4, 0.2, 0.6, 0.8, 0.1]
    m = binary_metrics(y, p, 0.5)
    assert m["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision_stress"] == pytest.approx(2 / 3) and m["recall_stress"] == pytest.approx(2 / 3)
    assert m["f1_stress"] == pytest.approx(2 / 3) and m["f1_macro"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(8 / 9)


def test_binary_metrics_single_class_auc_nan():
    m = binary_metrics([1, 1, 1], [0.2, 0.7, 0.9])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_bootstrap_contains_point_and_cluster_mode():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    groups = np.repeat(np.arange(200), 2)
    for g in (None, groups):
        ci = bootstrap_metrics(y, p, 0.5, n_resamples=200, groups=g, seed=1)
        for k in ("f1_stress", "roc_auc", "accuracy"):
            assert ci[k]["lo"] <= ci[k]["point"] <= ci[k]["hi"]
    assert bootstrap_metrics(y, p, 0.5, 50, seed=3) == bootstrap_metrics(y, p, 0.5, 50, seed=3)   # deterministic


def test_ece_near_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert ece(y, p, 15) < 0.02
    assert ece(y, np.clip(p * 0.3 + 0.7, 0, 1), 15) > 0.2
    t = reliability_table(y, p, 10)
    assert t["n"].sum() == 20000 and len(t) == 10


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 30000)
    y = (rng.uniform(0, 1, z.size) < 1 / (1 + np.exp(-z))).astype(int)
    logits = np.stack([np.zeros_like(z), z * 3.0], axis=1)          # over-confident by a factor 3
    T = fit_temperature(logits, y)
    assert T == pytest.approx(3.0, rel=0.08)
    assert nll(logits, y, T) < nll(logits, y, 1.0)
    p = positive_proba(logits, T)
    assert ece(y, p) < ece(y, positive_proba(logits, 1.0))


def test_choose_threshold_criterion_and_tiebreak():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = choose_threshold(y, p, "max_f1_stress", (0.05, 0.95, 0.05))
    assert r["value_on_validation"] == pytest.approx(1.0)
    assert r["threshold"] == pytest.approx(0.5)             # every t in (0.2, 0.8] is perfect; closest to 0.5 wins
    with pytest.raises(ValueError):
        choose_threshold(y, p, "accuracy")


def test_mean_std_and_groups():
    ms = mean_std([{"f1_stress": 0.8}, {"f1_stress": 0.9}, {"f1_stress": 0.7}], keys=["f1_stress"])
    assert ms["f1_stress"]["mean"] == pytest.approx(0.8) and ms["f1_stress"]["std"] == pytest.approx(0.1)
    import pandas as pd
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 2, "label": [1, 0, 1, 0, 1, 0], "p": [0.9, 0.1, 0.4, 0.6, 0.7, 0.2]})
    t = metrics_by_group(df, "g", min_n=3)
    assert list(t["g"]) == ["a", "b"] and np.isnan(t.loc[1, "accuracy"]) and t.loc[0, "false_negative_rate"] == 0.5


def test_paired_bootstrap_gain():
    from stress_signals.metrics import paired_bootstrap_gain
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    p = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.12, 600), 0, 1)     # positives sit just above ~0.65, negatives ~0.35
    g = paired_bootstrap_gain(y, p, 0.5, 0.5, n_resamples=50)
    assert g["gain"] == 0 and g["lo"] == 0 and g["hi"] == 0       # same threshold -> exactly zero gain
    g2 = paired_bootstrap_gain(y, p, 0.45, 0.5, n_resamples=200, groups=np.arange(600) // 2)
    assert g2["lo"] <= g2["gain"] <= g2["hi"]
