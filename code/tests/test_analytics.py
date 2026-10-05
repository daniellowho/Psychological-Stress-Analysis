"""Step 10 tests: synthetic enriched frames only. No model is trained, fitted or loaded (BERTopic / LDA / embeddings are not exercised)."""

import numpy as np
import pandas as pd
import pytest

from stress_signals import analytics as A
from stress_signals.config import load_config

K = 50


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["analytics"]["emotions"]["bootstrap"]["n"] = 60
    return c


def _frame(n_days=70, per_day=40, seed=0, comm_p=None):
    """Two communities with different stress rates; community mix is the same every day unless comm_p says otherwise."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2020-01-06", periods=n_days):
        pa_ = (comm_p or {}).get(d, 0.5)
        comm = np.where(rng.random(per_day) < pa_, "a", "b")
        flag = rng.random(per_day) < np.where(comm == "a", 0.6, 0.2)
        rows.append(pd.DataFrame({"date": d, "source": "s", "community": comm, "stress_flag": flag}))
    df = pd.concat(rows, ignore_index=True)
    df["stress_prob"] = np.where(df["stress_flag"], 0.8, 0.2)
    return df


def test_wilson_known_values():
    lo, hi = A.wilson(np.array([5, 0, 10]), np.array([10, 10, 10]))
    assert lo[0] == pytest.approx(0.2366, abs=1e-3) and hi[0] == pytest.approx(0.7634, abs=1e-3)
    assert lo[1] == 0 and 0.27 < hi[1] < 0.28
    assert hi[2] == pytest.approx(1.0) and lo[2] == pytest.approx(0.7225, abs=1e-3)
    assert np.isnan(A.wilson(0, 0)[0])


def test_period_start():
    s = pd.Series(pd.to_datetime(["2020-01-08", "2020-01-12", "2020-02-29"]))
    assert A.period_start(s, "week").dt.strftime("%Y-%m-%d").tolist() == ["2020-01-06", "2020-01-06", "2020-02-24"]
    assert A.period_start(s, "month").dt.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2020-01-01", "2020-02-01"]
    assert A.period_start(s, "day").equals(s)


def test_stress_rate_suppression_and_rollups():
    df = _frame(per_day=60)                                  # ~30 per community per day (< K), 60 pooled (>= K)
    cells = A.stress_cells(df, "day")
    t = A.stress_rate_table(cells, K)
    assert {"ALL"} <= set(t["community"]) and set(t[t["community"] == "ALL"]["source"]) == {"s", "ALL"}
    sub = t[(t["community"] != "ALL")]
    assert sub["suppressed"].all() and sub["n_total"].isna().all() and sub["stress_rate"].isna().all()
    assert not t[t["community"] == "ALL"]["suppressed"].any()


def test_stress_rate_table_counts():
    df = _frame(per_day=200)
    t = A.stress_rate_table(A.stress_cells(df, "week"), K)
    allr = t[(t["source"] == "ALL") & (t["community"] == "ALL")]
    assert not allr["suppressed"].any()
    assert allr["n_total"].astype(int).sum() == len(df)
    assert (allr["ci_low"] <= allr["stress_rate"]).all() and (allr["stress_rate"] <= allr["ci_high"]).all()
    assert allr["mean_stress_prob"].between(0.2, 0.8).all()
    A.check_aggregate_only(t, K, ["n_total"])


def test_check_aggregate_only_rejects_ids_text_and_small_counts():
    with pytest.raises(ValueError):
        A.check_aggregate_only(pd.DataFrame({"record_id": ["a"], "n": [100]}), K)
    with pytest.raises(ValueError):
        A.check_aggregate_only(pd.DataFrame({"term": ["x" * 300]}), K)
    with pytest.raises(ValueError):
        A.check_aggregate_only(pd.DataFrame({"n": [10, 100]}), K, ["n"])
    A.check_aggregate_only(pd.DataFrame({"n": pd.array([pd.NA, 100], dtype="Int64")}), K, ["n"])


def test_adjusted_series_removes_composition_effect():
    # identical per-community rates all along, but community 'a' (high rate) share jumps from 20% to 80% in the second half
    days = pd.date_range("2020-01-06", periods=70)
    mix = {d: (0.2 if i < 35 else 0.8) for i, d in enumerate(days)}
    df = _frame(per_day=300, seed=1, comm_p=mix)
    cells = A.stress_cells(df, "week")
    w, ref = A.reference_weights(df, {"start": None, "end": None, "first_n_weeks": 4}, K)
    assert ref["start"] == "2020-01-06" and ref["end"] == "2020-02-02"
    assert w["weight_all"].sum() == pytest.approx(1.0) and w["weight_in_source"].sum() == pytest.approx(1.0)
    s = A.adjusted_series(cells, w, K, 0.8)
    a = s[s["scope"] == "ALL"].dropna(subset=["adjusted_rate"])
    early, late = a.iloc[:3], a.iloc[-3:]
    raw_jump = late["raw_rate"].mean() - early["raw_rate"].mean()
    adj_jump = late["adjusted_rate"].mean() - early["adjusted_rate"].mean()
    assert raw_jump > 0.15                                  # raw series moves because the mix moved
    assert abs(adj_jump) < 0.06                             # fixed weights: no real change -> adjusted barely moves
    assert (a["adj_ci_low"] <= a["adjusted_rate"]).all() and (a["adjusted_rate"] <= a["adj_ci_high"]).all()


def test_adjusted_series_needs_coverage():
    df = _frame(per_day=300)
    cells = A.stress_cells(df, "week")
    w, _ = A.reference_weights(df, {"start": None, "end": None, "first_n_weeks": 4}, K)
    cells = cells[~((cells["community"] == "a") & (cells["period"] == cells["period"].min()))]
    first = A.adjusted_series(cells, w, K, 0.8).query("scope == 'ALL'").sort_values("period").iloc[0]
    assert pd.isna(first["adjusted_rate"]) and not first["suppressed"]


def _emo_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    flag = rng.random(n) < 0.5
    comm = np.where(rng.random(n) < 0.5, "a", "b")
    sampled = flag | (rng.random(n) < 0.2)                   # unflagged sampled at 20% -> weight 5
    w = np.where(flag, 1.0, 5.0)
    df = pd.DataFrame({"date": pd.Timestamp("2020-03-02"), "source": "s", "community": comm, "stress_flag": flag, "stress_prob": np.where(flag, .8, .2),
                       "second_stage_scored": sampled, "sample_weight": np.where(sampled, w, np.nan)})
    df["emotion_p__sadness"] = np.where(rng.random(n) < np.where(flag, 0.6, 0.2), 0.9, 0.1)       # true lift 3
    df["emotion_p__joy"] = np.where(rng.random(n) < 0.3, 0.9, 0.1)                                  # true lift 1
    df["emotion_group__sadness"] = df["emotion_p__sadness"]
    df["stressor_flag__money"] = flag & (rng.random(n) < 0.5)
    df["stressor_flag__work"] = flag & (rng.random(n) < 0.4)
    df["stressor_top"] = np.where(df["stressor_flag__money"], "money", np.where(df["stressor_flag__work"], "work", "other_unclear"))
    return df


def test_emotion_lift_uses_weights_and_bootstrap(cfg):
    df = _emo_frame(n=20000)
    t = A.emotion_lift(df, cfg)
    r = t[(t["scope"] == "ALL") & (t["level"] == "label")].set_index("emotion")
    assert r.loc["sadness", "lift"] == pytest.approx(3.0, rel=0.2) and r.loc["joy", "lift"] == pytest.approx(1.0, rel=0.2)
    assert r.loc["sadness", "lift_ci_low"] < r.loc["sadness", "lift"] < r.loc["sadness", "lift_ci_high"]
    assert r.loc["sadness", "n_unflagged_weighted"] > 4 * r.loc["sadness", "n_unflagged_sampled"]       # weights applied
    assert not r["suppressed"].any() and r.loc["sadness", "community_agreement"] == 1.0
    assert set(t["scope"]) == {"ALL", "a", "b"} and "ekman" in set(t["level"])


def test_emotion_lift_unweighted_would_be_wrong(cfg):
    # prevalence in the unflagged group must be the WEIGHTED one: make weights extreme and compare with the unweighted value
    df = _emo_frame()
    u = df[df["second_stage_scored"] & ~df["stress_flag"]]
    u.loc[u.index[:50], "sample_weight"] = 1000.0
    df.loc[u.index, "sample_weight"] = u["sample_weight"]
    t = A.emotion_lift(df, cfg, per_community=False)
    got = t[(t["emotion"] == "sadness") & (t["level"] == "label")]["prev_unflagged"].iloc[0]
    w = u["sample_weight"].to_numpy()
    assert got == pytest.approx(((u["emotion_p__sadness"] >= .5) * w).sum() / w.sum(), rel=1e-3)


def test_emotion_lift_suppresses_small_groups(cfg):
    df = _emo_frame(n=4000)
    df = df[~((df["community"] == "a") & ~df["stress_flag"])]            # community a has no unflagged records
    t = A.emotion_lift(df, cfg)
    assert t[t["scope"] == "a"]["suppressed"].all() and t[t["scope"] == "a"]["lift"].isna().all()
    assert not t[t["scope"] == "b"]["suppressed"].any()


def test_cooccurrence(cfg):
    t = A.stressor_emotion_cooccurrence(_emo_frame(), cfg)
    r = t[(t["stressor"] == "money") & (t["emotion"] == "sadness") & (t["level"] == "label")].iloc[0]
    assert r["prev_given_stressor"] == pytest.approx(0.6, abs=0.06) and r["lift"] == pytest.approx(1.0, abs=0.12)    # stressor independent of emotion here
    assert r["ci_low"] < r["prev_given_stressor"] < r["ci_high"] and not r["suppressed"]
    small = A.stressor_emotion_cooccurrence(_emo_frame(n=100), cfg)
    assert small["suppressed"].any()


def test_stressor_share():
    df = _emo_frame(n=6000)
    df["date"] = pd.to_datetime("2020-03-02")
    t = A.stressor_share(df, "month", K)
    r = t[(t["community"] == "ALL") & (t["stressor"] == "money")].iloc[0]
    n = int((df["stress_flag"] & df["second_stage_scored"]).sum())
    assert r["n_stress_scored"] == n and r["share_flag"] == pytest.approx(0.5, abs=0.04)
    top = t[(t["community"] == "ALL")].set_index("stressor")["share_top"]
    assert top.sum() == pytest.approx(1.0)
    assert t["share_flag"].between(0, 1).all()
    A.check_aggregate_only(t, K, ["n_stress_scored"])


# ---------------------------------------------------------------- topics: pure counting logic (no model)
def _tc(shares, n_per=400, start="2020-01-06", gran_days=7, comm="a"):
    """shares: {topic_id: [share per period]} -> counts frame (period, community, topic_id, n); topic -1 takes the remainder."""
    n_per_periods = len(next(iter(shares.values())))
    rows = []
    for p in range(n_per_periods):
        used = 0
        for t, s in shares.items():
            c = int(round(s[p] * n_per))
            used += c
            rows.append((pd.Timestamp(start) + pd.Timedelta(days=gran_days * p), comm, t, c))
        rows.append((pd.Timestamp(start) + pd.Timedelta(days=gran_days * p), comm, -1, n_per - used))
    return pd.DataFrame(rows, columns=["date", "community", "topic_id", "n"])


ECFG = {"baseline_periods": 8, "recent_periods": 4, "min_count": 50, "growth_ratio": 1.5, "persistence_k": 2, "require_ci_above_baseline": True}
PCFG = {"share_threshold": 0.05, "min_fraction": 0.6, "min_periods": 8}


def test_emerging_and_persistent():
    flat = [0.10] * 12
    shares = {1: flat,                                                    # steady, 10%
              2: [0.02] * 8 + [0.02, 0.10, 0.12, 0.12],                   # jumps in the last 3 periods -> emerging
              3: [0.02] * 8 + [0.15, 0.02, 0.02, 0.02],                   # one-off spike (1 hot period) -> not emerging
              4: [0.0] * 8 + [0.06] * 4}                                  # new topic, baseline 0 -> floored baseline
    tc = _tc(shares)
    em = A.emerging_topics(tc, "week", ECFG, K).query("community == 'a'").set_index("topic_id")
    assert em.loc[2, "is_emerging"] and em.loc[4, "is_emerging"]
    assert not em.loc[1, "is_emerging"] and not em.loc[3, "is_emerging"]
    assert em.loc[3, "max_consecutive_hot"] == 1 and em.loc[2, "max_consecutive_hot"] >= 2
    assert em.loc[4, "baseline_below_k"] and pd.isna(em.loc[4, "n_baseline"])               # baseline 0 < K is never published
    assert em.loc[4, "share_baseline_floored"] == pytest.approx(K / (8 * 400))             # floored at K
    pe = A.persistent_topics(tc, "week", PCFG, K).query("community == 'a'").set_index("topic_id")
    assert pe.loc[1, "is_persistent"] and pe.loc[1, "fraction_above"] == 1.0
    assert not pe.loc[3, "is_persistent"] and not pe.loc[4, "is_persistent"]               # 4/12 periods above 5%
    st = A.topic_status(A.emerging_topics(tc, "week", ECFG, K), A.persistent_topics(tc, "week", PCFG, K))
    s = st.query("community == 'a'").set_index("topic_id")["status"]
    assert s[1] == "persistent" and s[2] in ("emerging", "both") and s[3] == "neither"


def test_emerging_needs_history_and_counts():
    tc = _tc({1: [0.1] * 6})                                              # fewer than R + B periods
    assert A.emerging_topics(tc, "week", ECFG, K).empty
    small = _tc({1: [0.1] * 12}, n_per=10)                                # window totals < K
    assert A.emerging_topics(small, "week", ECFG, K).empty


def test_topic_share_series_suppression():
    tc = _tc({1: [0.10] * 3, 2: [0.001] * 3}, n_per=1000)
    s = A.topic_share_series(tc, "week", K)
    t1, t2 = s[(s["community"] == "a") & (s["topic_id"] == 1)], s[(s["community"] == "a") & (s["topic_id"] == 2)]
    assert not t1["suppressed"].any() and t2["suppressed"].all() and t2["share"].isna().all()
    assert (s["topic_id"] != -1).all()
    A.check_aggregate_only(s, K, ["n_topic", "n_total"])


def test_assign_texts_nearest_centroid(monkeypatch):
    emb = {"t0": [1.0, 0.0], "t1": [0.0, 1.0], "near1": [0.1, 1.0], "far": [0.0, 0.0]}
    def fake_embed(texts, name, batch_size=64, device=None, show_progress=False):
        e = np.array([emb[t] for t in texts], dtype=np.float32)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.where(n == 0, 1, n)
    monkeypatch.setattr(A, "embed", fake_embed)
    ids, cent = np.array([10, 20]), np.array([[1, 0], [0, 1]], dtype=np.float32)
    topic, sim = A.assign_texts(["t0", "near1", "far"], ids, cent, "x", batch_size=2, min_similarity=0.5)
    assert topic.tolist() == [10, 20, -1] and sim[0] == pytest.approx(1.0)


def test_u01_is_deterministic_and_uniform():
    ids = pd.Series([f"r{i}" for i in range(20000)])
    u = A._u01(ids, 42)
    assert np.array_equal(u, A._u01(ids.iloc[::-1].reset_index(drop=True), 42)[::-1])
    assert 0.48 < u.mean() < 0.52 and not np.array_equal(u, A._u01(ids, 43))


def test_config_has_analytics_block(cfg):
    a = cfg["analytics"]
    assert a["topics"]["sample_size"] >= 50_000 and a["topics"]["embedding_model"] and a["topics"]["umap"] and a["topics"]["hdbscan"]
    assert a["gold_dir"] == "data/gold"
