"""Step 3 helper tests. NOTHING here trains or updates weights: models are built from a tiny random config and only
run forward passes (inference), and TrainingArguments are constructed but never passed to `train()`."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_signals.config import load_config
from stress_signals import stress_model as S


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_settings_resolve_model_options(cfg):
    s = S.settings(cfg)
    assert s["hf_id"] == "roberta-base" and s["label_map"] == {0: "not_stress", 1: "stress"}
    assert S.settings(cfg, "distilroberta-base")["hf_id"] == "distilroberta-base"
    assert S.settings(cfg, "deberta-v3-base")["hf_id"] == "microsoft/deberta-v3-base"
    with pytest.raises(ValueError):
        S.settings(cfg, "gpt-something")


def test_feature_columns_are_dropped_explicitly():
    df = pd.DataFrame({c: [1] for c in S.MODEL_INPUT_COLUMNS}
                      | {"lex_liwc_WC": [3], "lex_dal_avg_pleasantness": [1.0], "syntax_ari": [2.0],
                         "social_karma": [5], "sentiment": [0.1]})
    out = S.select_model_inputs(df)
    assert list(out.columns) == S.MODEL_INPUT_COLUMNS
    with pytest.raises(KeyError):
        S.select_model_inputs(df.drop(columns=["text_clean"]))


def test_confidence_weights():
    w = S.confidence_weights([0.0, 0.5, 1.0, np.nan], floor=0.5)
    assert np.all(w > 0) and w.mean() == pytest.approx(1.0)
    assert w[0] == pytest.approx(w[1]) and w[2] > w[1]


def test_precision_flags():
    assert S.precision_flags("auto", "roberta-base", cuda=False, bf16_supported=False) == {"fp16": False, "bf16": False}
    assert S.precision_flags("auto", "roberta-base", cuda=True, bf16_supported=False) == {"fp16": True, "bf16": False}
    assert S.precision_flags("auto", "microsoft/deberta-v3-base", True, False) == {"fp16": False, "bf16": False}
    assert S.precision_flags("auto", "microsoft/deberta-v3-base", True, True) == {"fp16": False, "bf16": True}
    with pytest.raises(ValueError):
        S.precision_flags("bf16", "roberta-base", True, False)


def test_training_args_match_installed_transformers(cfg, tmp_path):
    pytest.importorskip("transformers")
    compat = S.hf_compat()
    s = S.settings(cfg)
    args = S.build_training_args(s, tmp_path / "ck", seed=13, cuda=False)
    strategy = getattr(args, "eval_strategy", None) or getattr(args, "evaluation_strategy")
    assert str(getattr(strategy, "value", strategy)) == "epoch"
    assert str(getattr(args.save_strategy, "value", args.save_strategy)) == "epoch"
    assert args.load_best_model_at_end and args.metric_for_best_model == "f1_stress" and args.greater_is_better
    assert args.seed == 13 and not args.fp16 and args.remove_unused_columns is False
    assert args.num_train_epochs == s["epochs"] and args.warmup_ratio == pytest.approx(0.1)
    assert compat["tokenizer_key"] in ("processing_class", "tokenizer")


def test_weighted_cross_entropy():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 0])
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert S.weighted_cross_entropy(logits, labels).item() == pytest.approx(ce.mean().item())
    w = torch.tensor([1.0, 0.0, 3.0])
    assert S.weighted_cross_entropy(logits, labels, w).item() == pytest.approx(((ce * w).sum() / w.sum()).item())


@pytest.fixture(scope="module")
def tokenizer():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer not cached: {exc!r}")


def test_collator_pads_and_keeps_weights(tokenizer):
    enc = S.encode(tokenizer, ["short text", "a somewhat longer piece of text here"], 32)
    ds = S.EncodedDataset(enc, [0, 1], [0.5, 1.5])
    batch = S.WeightedCollator(tokenizer)([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2 and batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].tolist() == [0, 1] and batch["sample_weight"].tolist() == [0.5, 1.5]


def test_predict_logits_order_with_tiny_random_model(tokenizer):
    """Inference only (forward passes) on a 1-layer random model built from config: no download, no training."""
    tr = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    conf = tr.RobertaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=80, num_labels=2, pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(0)
    model = tr.AutoModelForSequenceClassification.from_config(conf)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    texts = ["one", "a much longer sentence with many more words in it", "mid length text", "x"]
    batched = S.predict_logits(model, tokenizer, texts, max_length=64, batch_size=3)
    single = np.concatenate([S.predict_logits(model, tokenizer, [t], 64, 1) for t in texts])
    assert batched.shape == (4, 2) and np.allclose(batched, single, atol=1e-4)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())   # weights untouched


def test_masking_and_random_control(cfg):
    s = S.settings(cfg)
    terms = s["shortcut"]["subreddit_names"] + s["shortcut"]["domain_terms"]
    texts = ["My boyfriend and the rent are killing me, see r/anxiety", "Nothing topical, just tired and stressed today"]
    masked, counts = S.mask_texts(texts, terms, "<mask>")
    assert counts == [3, 0]
    assert "boyfriend" not in masked[0] and "rent" not in masked[0] and "r/anxiety" not in masked[0]
    assert "stressed" in masked[1]                          # target-signal words are never masked
    ctrl = S.random_mask(texts, counts, "<mask>", seed=1, protect=S.mask_regex(terms))
    assert ctrl[0].count("<mask>") == 3 and ctrl[1] == texts[1]
    assert "boyfriend" in ctrl[0] or "rent" in ctrl[0]      # protected terms are not chosen by the control


def test_loso_split_holds_out_subreddit():
    mk = lambda n, c: pd.DataFrame({"community": c, "label": [0, 1] * (n // 2)})  # noqa: E731
    data = {"train": pd.concat([mk(4, "a"), mk(4, "b")]), "validation": pd.concat([mk(2, "a"), mk(2, "b")]),
            "test": pd.concat([mk(2, "a"), mk(2, "b")])}
    d = S.loso_data(data, "a")
    assert set(d["train"].community) == {"b"} and set(d["validation"].community) == {"b"}
    assert set(d["test"].community) == {"a"} and len(d["test"]) == 8


def test_select_final_seed_uses_validation_only():
    ev = {"seeds": {42: {"validation": {"all": {"f1_stress": 0.70}}, "test": {"all": {"f1_stress": 0.99}}},
                    13: {"validation": {"all": {"f1_stress": 0.80}}, "test": {"all": {"f1_stress": 0.10}}}}}
    assert S.select_final_seed(ev) == 13


def test_real_split_matches_file(cfg):
    p = Path(cfg["_paths"]["processed"]) / "dreaddit.parquet"
    if not p.exists():
        pytest.skip("processed Dreaddit not built")
    data = S.load_dreaddit(cfg)
    assert {k: len(v) for k, v in data.items()} == {"train": 2398, "validation": 415, "test": 715}
    assert list(data["train"].columns) == S.MODEL_INPUT_COLUMNS


def test_training_cells_carry_banner(cfg):
    calls = ("train_stress_seed(", "fit_baseline(", "train_loso(", "trainer.train(")
    for name in ("notebook.ipynb",):
        nb = json.loads((Path(cfg["_root"]) / "code" / name).read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if cell["cell_type"] == "code" and any(c in src for c in calls):
                assert S.TRAINING_BANNER in src, (name, src[:120])
                assert "H.run_step(" in src, (name, src[:120])        # training is always behind the run-once guard


def test_post_training_pipeline_offline_smoke(cfg, tmp_path, tokenizer):
    """evaluate -> select seed -> calibrate -> final report (once) -> bundle, on SYNTHETIC logits and a tiny RANDOM
    model saved with save_pretrained. No training, no download; writes only under tmp_path."""
    tr = pytest.importorskip("transformers")
    if not (Path(cfg["_paths"]["processed"]) / "dreaddit.parquet").exists():
        pytest.skip("processed Dreaddit not built")
    c = {**cfg, "_root": tmp_path, "paths": {**cfg["paths"], "models": "models"},
         "stress_model": {**cfg["stress_model"], "seeds": [42, 13],
                          "eval": {**cfg["stress_model"]["eval"], "bootstrap_resamples": 30}}}
    data = S.load_dreaddit(cfg)
    conf = tr.RobertaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=80, num_labels=2, pad_token_id=tokenizer.pad_token_id)
    rng = np.random.default_rng(0)
    for seed in (42, 13):
        d = S.run_dir_for(c, "roberta-base", seed)
        (d / "best_model").mkdir(parents=True)
        tr.AutoModelForSequenceClassification.from_config(conf).save_pretrained(d / "best_model")
        tokenizer.save_pretrained(d / "best_model")
        for split in ("validation", "test"):
            df = data[split]
            z = (df["label"].to_numpy() * 2 - 1) * 1.5 + rng.normal(0, 2, len(df))
            S._save_predictions(df, np.stack([np.zeros_like(z), 3 * z], 1), d / f"predictions_{split}.parquet")
        (d / "train_summary.json").write_text(json.dumps({"epochs_completed": 3, "transformers": "x"}))
        (d / "DONE.json").write_text("{}")
    ev = S.evaluate_runs(c, "roberta-base")
    assert set(ev["seeds"]) == {42, 13} and ev["summary"]["test"]["f1_stress"]["n_runs"] == 2
    seed = S.select_final_seed(ev)
    cal = S.calibrate(c, seed, "roberta-base")
    assert 1.5 < cal["temperature"] < 6 and cal["validation_after"]["ece"] <= cal["validation_before"]["ece"] + 1e-9
    lo, hi = cfg["stress_model"]["calibration"]["threshold_grid"][:2]
    assert lo - 1e-9 <= cal["threshold_tuned"] <= hi + 1e-9 and cal["threshold"] in (0.5, cal["threshold_tuned"])
    rep = S.final_test_report(c, seed, "roberta-base")
    assert rep["at_threshold_0.5"]["threshold"] == 0.5 and rep["at_threshold_tuned"]["threshold"] == cal["threshold_tuned"]
    assert rep["threshold"] == cal["threshold"]
    again = S.final_test_report(c, seed, "roberta-base")
    assert json.dumps(rep, default=str, sort_keys=True) == json.dumps(again, default=str, sort_keys=True)
    bundle = S.save_bundle(c, seed, "roberta-base")
    names = {p.name for p in bundle.iterdir()}
    assert {"config.json", "stress_config.json", "metrics.json", "model_card.md", "manifest.json"} <= names
    sc = json.loads((bundle / "stress_config.json").read_text())
    assert sc["threshold"]["fitted_on"] == "validation" and sc["calibration"]["temperature"] == pytest.approx(cal["temperature"])
    assert sc["threshold"]["applied"] in ("tuned", "0.5") and "tuned_value" in sc["threshold"]
    man = json.loads((bundle / "manifest.json").read_text())
    from stress_signals.utils import sha256_file
    assert all(sha256_file(bundle / k) == v["sha256"] for k, v in man["artifacts"].items())
    assert "NOT" in (bundle / "model_card.md").read_text() or "Not for diagnosis" in (bundle / "model_card.md").read_text()
    with pytest.raises(FileExistsError):
        S.save_bundle(c, seed, "roberta-base")


def test_decide_threshold_applies_only_clear_gains():
    rule = {"min_gain": 0.01, "require_ci_above_zero": True, "bootstrap_resamples": 200}
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    # scores shifted low: positives centred ~0.47 -> tuned 0.40 should clearly beat 0.5
    p = np.clip(0.30 + 0.17 * y + rng.normal(0, 0.05, 800), 0, 1)
    d = S.decide_threshold(y, p, 0.40, rule, groups=None)
    assert d["applied"] == "tuned" and d["applied_threshold"] == 0.40 and d["validation_gain_f1_stress"]["lo"] > 0
    # well-separated scores: 0.5 and 0.45 are equally perfect -> gain 0 -> keep 0.5
    p2 = np.where(y == 1, 0.9, 0.1)
    d2 = S.decide_threshold(y, p2, 0.45, rule)
    assert d2["applied"] == "0.5" and d2["applied_threshold"] == 0.5 and "not clear" in d2["reason"]
    assert S.decide_threshold(y, p2, 0.5, rule)["applied"] == "0.5"


def test_threshold_grid_restricted(cfg):
    assert cfg["stress_model"]["calibration"]["threshold_grid"][:2] == [0.40, 0.60]


def test_summarise_masking_uses_all_draws():
    res = pd.DataFrame([
        {"condition": "original", "draw_seed": None, "f1_stress": 0.80, "roc_auc": 0.85},
        {"condition": "topic_masked", "draw_seed": None, "f1_stress": 0.70, "roc_auc": 0.80},
        *[{"condition": "random_masked_control", "draw_seed": 42 + k, "f1_stress": f, "roc_auc": 0.84}
          for k, f in enumerate([0.79, 0.78, 0.77, 0.79, 0.76])]])
    t = S.summarise_masking(res).set_index("metric")
    assert t.loc["f1_stress", "n_control_draws"] == 5
    assert t.loc["f1_stress", "drop_control_max"] == pytest.approx(0.04) and t.loc["f1_stress", "drop_topic_masked"] == pytest.approx(0.10)
    assert bool(t.loc["f1_stress", "topic_drop_exceeds_all_control_draws"])
    assert cfg_draws() == 5


def cfg_draws():
    return load_config()["stress_model"]["shortcut"]["random_control_draws"]


def test_goemotions_overlap_flags(cfg):
    from stress_signals.preprocess import goemotions_overlap_path, goemotions_test_sensitivity_frames
    p = goemotions_overlap_path(cfg)
    if not p.exists():
        pytest.skip("overlap flags not built")
    f = pd.read_parquet(p)
    t = f[f["split"] == "test"]
    assert len(t) == 5427 and int(t["exclude_for_test_sensitivity"].sum()) == int((t["text_in_train"] | t["text_in_validation"]).sum())
    assert not f.loc[f["split"] != "test", "exclude_for_test_sensitivity"].any()
    pred = pd.DataFrame({"record_id": t["record_id"]})
    fr = goemotions_test_sensitivity_frames(pred, f)
    assert len(fr["all_test"]) - len(fr["test_without_cross_split_overlap"]) == int(t["exclude_for_test_sensitivity"].sum())


def test_senticnet_non_advert_flag(cfg):
    p = Path(cfg["_paths"]["processed"]) / "senticnet_validation.parquet"
    if not p.exists():
        pytest.skip("senticnet not built")
    d = pd.read_parquet(p)
    assert "Twitter_Non-Advert" not in set(d["community"])
    assert d.loc[d["community"] == "Twitter_Full", "is_non_advert"].notna().all()
    assert d.loc[d["community"] != "Twitter_Full", "is_non_advert"].isna().all()
