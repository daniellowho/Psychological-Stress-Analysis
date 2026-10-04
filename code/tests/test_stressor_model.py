"""Step 5 helper tests. NOTHING here fits or updates weights: synthetic arrays, hand-set classifier weights, and a tiny
random encoder that only runs forward passes."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_signals import stressor_model as S
from stress_signals.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def tax(cfg):
    return S.load_taxonomy(cfg)


# ---------------------------------------------------------------- 5A taxonomy
def test_taxonomy_structure(tax):
    cats = S.category_ids(tax)
    assert cats == ["workplace_pressure", "academic_workload", "financial_concerns", "relationship_issues",
                    "health_concerns", "uncertainty", "overload", "societal_problems", "other_unclear"]
    assert {c for c, v in S.coverage(tax).items() if v == "zero_shot"} == {"uncertainty", "societal_problems"}
    m = S.sad_label_to_class(tax)
    assert set(m) == set(S.SAD_LABELS)
    assert m["Everyday Decision Making"] == "overload"                     # user decision, Step 5
    assert m["Emotional Turmoil"] == "emotional_turmoil"                   # non-stressor class, kept
    assert m["Family Issues"] == "relationship_issues/family" and m["Social Relationships"] == "relationship_issues/social"
    assert len(S.classifier_classes(tax)) == 9


def test_taxonomy_validation_catches_errors(tax):
    bad = copy.deepcopy(tax)
    bad["categories"][0]["sad_labels"] = ["School"]                       # School mapped twice, Work missing
    with pytest.raises(ValueError, match="SAD labels"):
        S.validate_taxonomy(bad)
    bad = copy.deepcopy(tax)
    bad["mendeley"] = bad["mendeley"][:-1]
    with pytest.raises(ValueError, match="Mendeley"):
        S.validate_taxonomy(bad)
    bad = copy.deepcopy(tax)
    bad["categories"][0]["paraphrases"] = bad["categories"][0]["paraphrases"][:2]
    with pytest.raises(ValueError, match="paraphrase"):
        S.validate_taxonomy(bad)


def test_mendeley_columns_match_real_file(cfg):
    p = Path(cfg["_root"]) / cfg["sources"]["mendeley_stress_indicators"]["local_path"]
    if not p.exists():
        pytest.skip("Mendeley CSV not on this machine")
    assert sorted(pd.read_csv(p, nrows=0).columns) == sorted(S.MENDELEY_COLUMNS)


def test_merge_class_probs(tax):
    classes = S.classifier_classes(tax)
    P = np.zeros((2, len(classes)))
    P[0, classes.index("relationship_issues/family")] = 0.3
    P[0, classes.index("relationship_issues/social")] = 0.2
    P[0, classes.index("emotional_turmoil")] = 0.5
    P[1, classes.index("workplace_pressure")] = 1.0
    M = S.merge_class_probs(P, tax)
    cats = S.category_ids(tax)
    assert M[0, cats.index("relationship_issues")] == pytest.approx(0.5)
    assert np.isnan(M[:, cats.index("uncertainty")]).all() and np.isnan(M[:, cats.index("societal_problems")]).all()
    assert np.nansum(M[0]) == pytest.approx(0.5)                          # emotional_turmoil mass is discarded
    assert M[1, cats.index("workplace_pressure")] == 1.0


def test_sad_split_partitions(cfg, tax):
    from stress_signals.preprocess import processed_path
    if not processed_path(cfg, "sad").exists():
        pytest.skip("processed SAD not built")
    df = S.load_sad(cfg, tax)
    sp = S.make_sad_split(df, cfg["stressor_model"]["sad"]["split"], 42)
    ids = [i for v in sp.values() for i in v]
    assert len(ids) == len(set(ids)) == len(df)
    assert set(df.loc[df["record_id"].isin(sp["test"]), "cls"]) == set(S.classifier_classes(tax))


# ---------------------------------------------------------------- inference unit
def test_sentence_windows_merge_and_cap():
    t = "Ok. My boss yelled at me again today. Rent is due!\nI can't sleep at all these days"
    w = S.sentence_windows(t, 1, 4)
    assert w == ["Ok. My boss yelled at me again today.", "Rent is due! I can't sleep at all these days"]
    assert S.sentence_windows("", 1, 4) == []
    assert S.sentence_windows("Hi.", 1, 4) == ["Hi."]
    many = " ".join(f"This is sentence number {i}." for i in range(50))
    capped = S.sentence_windows(many, 1, 4, max_windows=5)
    assert len(capped) == 5 and capped[0].endswith("0.") and capped[-1].endswith("49.")
    assert len(S.sentence_windows(many, 2, 4)) == 25


def test_aggregate_windows():
    Sw = np.array([[0.1, 0.9], [0.5, 0.2], [0.3, 0.4], [0.7, 0.0]])
    owner = np.array([0, 0, 0, 1])
    assert np.allclose(S.aggregate_windows(Sw, owner, 2, "max"), [[0.5, 0.9], [0.7, 0.0]])
    assert np.allclose(S.aggregate_windows(Sw, owner, 2, "mean")[0], [0.3, 0.5])
    assert np.allclose(S.aggregate_windows(Sw, owner, 2, "top2_mean")[0], [0.4, 0.65])
    assert np.allclose(S.aggregate_windows(Sw, owner, 2, "sum")[0], [0.9, 1.5])
    with pytest.raises(ValueError):
        S.aggregate_windows(Sw, owner, 3, "max")                          # post 2 has no windows


def test_keyword_regex_and_scorer(tax):
    rx = S.keyword_regex(["exam*", "laid off", "rent"])
    assert rx.search("My EXAMS start") and rx.search("I got laid   off") and rx.search("rent.")
    assert not rx.search("parenting") and not rx.search("current")        # whole words only
    ks = S.KeywordScorer(tax)
    out = ks.window_scores(["My boss fired me", "nothing here", ""])
    cats = S.category_ids(tax)
    assert out.shape == (3, len(cats)) and out[0, cats.index("workplace_pressure")] == 1 and out[1].sum() == 0
    assert ks.window_scores([]).shape == (0, len(cats))


# ---------------------------------------------------------------- embeddings (tiny random encoder, forward only)
def test_pool_hidden():
    import torch
    h = torch.tensor([[[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(S.pool_hidden(h, m, "mean"), torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(S.pool_hidden(h, m, "cls"), torch.tensor([[1.0, 1.0]]))


def test_embedder_order_and_norm():
    import transformers as tr
    try:
        tok = tr.AutoTokenizer.from_pretrained("distilroberta-base", local_files_only=True)
    except OSError:
        pytest.skip("distilroberta tokenizer not cached")
    import torch
    torch.manual_seed(0)
    conf = tr.RobertaConfig(vocab_size=tok.vocab_size, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=16, max_position_embeddings=40, pad_token_id=tok.pad_token_id)
    model = tr.AutoModel.from_config(conf)
    spec = S.EmbeddingSpec("tiny", "mean", True, 32, False, None)
    emb = S.Embedder(spec, "cpu", 2, model=model, tokenizer=tok)
    texts = ["a short one", "a much longer sentence with more words in it", "mid length text"]
    E = emb.encode(texts)
    assert E.shape == (3, 8) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-5)
    assert np.allclose(emb.encode(texts[::-1]), E[::-1], atol=1e-5)       # order preserved despite length sorting


def test_supervised_scorer_hybrid(tax):
    classes, cats = S.classifier_classes(tax), S.category_ids(tax)
    d = 4
    coef = np.zeros((len(classes), d))
    coef[classes.index("financial_concerns"), 0] = 10.0
    lr = S.lr_logits_fn({"coef": coef, "intercept": np.zeros(len(classes))})

    class FakeZS:
        def window_scores(self, windows, E):
            return np.full((len(windows), len(cats)), 0.33)

    sc = S.SupervisedScorer(tax, "embed_lr", lr, 1.0, FakeZS())
    E = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0]])
    out = sc.window_scores(["x", "y"], E)
    assert out[0, cats.index("financial_concerns")] > 0.99
    assert np.allclose(out[:, cats.index("uncertainty")], 0.33)
    assert sc.kinds[cats.index("uncertainty")] == "cosine" and sc.kinds[cats.index("overload")] == "probability"
    assert not np.isnan(out).any()


# ---------------------------------------------------------------- 5B gold set
def test_allocate_stratified():
    a = S.allocate_stratified({"a": 500, "b": 100, "c": 5}, 200, 20)
    assert sum(a.values()) == 200 and a["c"] == 5 and a["b"] >= 20 and all(a[k] <= v for k, v in {"a": 500, "b": 100, "c": 5}.items())
    with pytest.raises(ValueError):
        S.allocate_stratified({"a": 10}, 20, 5)


def _fake_dreaddit(n_posts=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    subs = ["anxiety", "ptsd", "homeless", "relationships", "stress"]
    for p in range(n_posts):
        for s in range(rng.integers(1, 3)):
            rows.append({"record_id": f"r{p}_{s}", "post_id": f"p{p}", "community": subs[p % 5],
                         "label": int(rng.random() < 0.7), "confidence": float(rng.choice([0.4, 0.6, 0.8, 1.0])),
                         "split": ["train", "validation", "test"][p % 3]})
    return pd.DataFrame(rows)


def test_sample_gold_rules(cfg):
    c = copy.deepcopy(cfg)
    c["stressor_model"]["gold"].update({"n_total": 100, "floor_per_subreddit": 10})
    dd = _fake_dreaddit()
    key = S.sample_gold(c, dd)
    src = dd.set_index("record_id").loc[key["record_id"]]
    assert len(key) == 100 and key["gold_id"].is_unique
    assert (src["label"] == 1).all() and (src["confidence"] >= 0.6).all()
    assert set(src["split"]) <= {"train", "validation"}                   # official test never sampled
    assert src["post_id"].is_unique                                       # one segment per post
    assert key["gold_split"].value_counts().to_dict() == {"dev": 50, "test": 50}
    assert "double_annotated" not in key.columns                          # single annotator (user decision)
    assert key.equals(S.sample_gold(c, dd))                               # deterministic


def test_read_sheet_validation(tmp_path, tax):
    cols = S.gold_columns(tax)
    df = pd.DataFrame({"gold_id": ["g1", "g2", "g3", "g4"], "text": ["t"] * 4, **{c: "" for c in cols}, "notes": ""})
    df.loc[0, "workplace_pressure"] = "1"
    df.loc[1, S.NONE_COL] = "1"
    df.loc[2, "overload"] = "maybe"
    df.loc[3, S.NONE_COL] = "1"
    df.loc[3, "uncertainty"] = "1"
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    lab, probs = S.read_sheet(p, cols)
    assert lab["annotated"].tolist() == [True, True, False, True]
    assert any("overload" in x for x in probs) and any("none_unclear together" in x for x in probs)


# ---------------------------------------------------------------- thresholds + evaluation
def test_tune_thresholds_kinds(cfg):
    t = cfg["stressor_model"]["thresholds"]
    Y = np.array([[1, 1], [1, 0], [0, 0], [0, 1]] * 5)
    Sc = np.array([[3, 0.9], [2, 0.2], [0, 0.1], [0, 0.8]] * 5, dtype=float)
    tab = S.tune_thresholds(Y, Sc, ["count", "cosine"], ["a", "b"], {**t, "low_support_rule": "tune"})
    assert tab.loc[0, "applied"] in (1.0, 2.0) and float(tab.loc[0, "applied"]).is_integer()
    assert 0.2 < tab.loc[1, "applied"] <= 0.8 and tab["f1_dev_at_tuned"].tolist() == [1.0, 1.0]
    low = S.tune_thresholds(Y[:4], Sc[:4], ["count", "cosine"], ["a", "b"], {**t, "low_support_rule": "keep_default"})
    assert low.loc[0, "low_support"] and low.loc[0, "applied"] == 1.0 and low.loc[0, "rule"] == "default_1_window"
    assert low.loc[1, "rule"] == "global_cosine" and low.loc[1, "applied"] == low.attrs["global_cosine_threshold"]


def test_global_cosine_threshold_pools_cosine_columns_only(cfg):
    t = cfg["stressor_model"]["thresholds"]
    Y = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]])
    Sc = np.array([[0.9, 0.6, 0.0], [0.1, 0.2, 0.9], [0.8, 0.1, 0.7], [0.2, 0.7, 0.1]])
    g = S.global_cosine_threshold(Y, Sc, ["cosine", "cosine", "probability"], t)
    assert 0.2 < g <= 0.6                                                    # separates the pooled cosine cells
    assert S.global_cosine_threshold(Y, Sc, ["probability"] * 3, t) is None


def test_no_agreement_code_path():
    assert not hasattr(S, "kappa_table") and not hasattr(S, "write_kappa")
    assert S.ANNOTATION_NOTE == "single annotator, no inter-annotator agreement"


def test_evaluate_split_none_rule(tax):
    L = len(S.category_ids(tax))
    Y = np.zeros((4, L), dtype=int)
    Y[0, 0] = 1
    Y[1, 2] = 1
    Ynone = np.array([0, 0, 1, 1])
    Sc = np.zeros((4, L))
    Sc[0, 0] = 0.9
    Sc[3, 1] = 0.9                                                          # false positive on a none row
    m = S.evaluate_split(Y, Ynone, Sc, [0.5] * L, tax)
    assert m["none_unclear"]["predicted_rate"] == 0.5
    assert m["none_unclear"]["precision"] == 0.5 and m["none_unclear"]["recall"] == 0.5
    assert m["f1_macro_supported"] == pytest.approx(0.5)                    # cats 0 (F1 1) and 2 (F1 0) have support
