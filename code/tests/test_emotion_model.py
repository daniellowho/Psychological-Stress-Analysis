"""Step 4 helper tests. NOTHING here trains or updates weights: tiny random models built from a config run forward passes
(and, in the gradient-accumulation check, one backward pass with no optimizer step); metrics use synthetic arrays."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_signals import emotion_model as E
from stress_signals import metrics as M
from stress_signals.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def names(cfg):
    return E.label_names(cfg)


# ---------------------------------------------------------------- 4A data
def test_label_names_official_order(cfg, names):
    assert len(names) == 28 and names[0] == "admiration" and names[-1] == "neutral"
    assert cfg["models"]["emotion"]["num_labels"] == 28 == cfg["sources"]["goemotions"]["num_labels"]


def test_label_matrix_validates():
    m = E.label_matrix([[0], [1, 27], [5, 6, 7]], 28)
    assert m.shape == (3, 28) and m.dtype == np.int8 and m.sum(1).tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        E.label_matrix([[]], 28)
    with pytest.raises(ValueError):
        E.label_matrix([[28]], 28)


def test_kaggle_source_refused(cfg):
    c = {**cfg, "sources": {**cfg["sources"], "goemotions": {**cfg["sources"]["goemotions"], "use_kaggle": True}}}
    with pytest.raises(ValueError, match="UNVERIFIED"):
        E.resolve_source(c)


def test_real_goemotions_matches_split_and_audit(cfg, names):
    from stress_signals.preprocess import processed_path
    if not processed_path(cfg, "goemotions").exists():
        pytest.skip("processed GoEmotions not built")
    data = E.load_goemotions(cfg, "processed_official")
    assert {k: len(v) for k, v in data.items()} == {"train": 43410, "validation": 5426, "test": 5427}
    audit = Path(cfg["_paths"]["interim"]) / "audit_goemotions.json"
    if audit.exists():
        a = json.loads(audit.read_text(encoding="utf-8"))
        counts = a.get("per_label_counts") or a.get("label_counts")
        if isinstance(counts, dict) and "train" in counts:
            Y = E.label_matrix(data["train"]["labels"], 28)
            assert dict(zip(names, Y.sum(0).tolist())) == {k: int(v) for k, v in counts["train"].items()}
    Y = E.label_matrix(data["train"]["labels"], 28)
    assert Y.sum(0)[names.index("grief")] == 77 and Y.sum(0)[names.index("neutral")] == 14219   # Step 1 audit table


# ---------------------------------------------------------------- 4B training pieces (no training)
def test_pos_weights():
    Y = np.array([[1, 0], [0, 0], [0, 0], [0, 1]] * 25)
    w = E.pos_weights(Y, "inverse_sqrt", 10.0)
    assert w[0] == pytest.approx(np.sqrt(75 / 25)) and w[1] == pytest.approx(np.sqrt(75 / 25))
    assert np.all(E.pos_weights(np.eye(50)[:, :2].repeat(1, 0), "inverse", 3.0) <= 3.0)
    with pytest.raises(ValueError):
        E.pos_weights(np.zeros((4, 2)), "inverse_sqrt")


def test_installed_transformers_multilabel_loss_is_bce(cfg):
    """Confirms problem_type='multi_label_classification' in the INSTALLED transformers: BCEWithLogits (mean) on float
    labels, and that our bce_loss equals it. Forward pass on a tiny random model only."""
    tr = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    conf = tr.RobertaConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                            max_position_embeddings=24, num_labels=5, problem_type=cfg["emotion_model"]["problem_type"],
                            pad_token_id=1)
    model = tr.AutoModelForSequenceClassification.from_config(conf).eval()
    x = torch.randint(3, 64, (3, 6))
    y = torch.tensor([[1, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 1, 0, 0, 0]], dtype=torch.float)
    with torch.no_grad():
        out = model(input_ids=x, labels=y)
        ref = torch.nn.functional.binary_cross_entropy_with_logits(out.logits, y)
    assert model.config.problem_type == "multi_label_classification"
    assert out.loss.item() == pytest.approx(ref.item(), rel=1e-6)
    assert E.bce_loss(out.logits, y).item() == pytest.approx(ref.item(), rel=1e-6)
    pw = torch.tensor([2.0, 1, 1, 1, 1])
    assert E.bce_loss(out.logits, y, pw).item() > ref.item()


def test_ga_check_runs_and_weights_untouched():
    """The measured gradient-accumulation behaviour is a bool, and measuring it does not touch global RNG state."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    before = torch.random.get_rng_state().clone()
    E.trainer_divides_loss_by_ga.cache_clear()
    res = E.trainer_divides_loss_by_ga()
    assert isinstance(res, bool) and torch.equal(before, torch.random.get_rng_state())


@pytest.fixture(scope="module")
def tokenizer():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer not cached: {exc!r}")


def test_collator_float_labels(tokenizer):
    from stress_signals.stress_model import encode
    ds = E.MultiLabelDataset(encode(tokenizer, ["short", "a little longer text here"], 32), np.array([[1, 0, 1], [0, 1, 0]]))
    batch = E.MultiLabelCollator(tokenizer)([ds[0], ds[1]])
    assert batch["labels"].dtype.is_floating_point and batch["labels"].tolist() == [[1, 0, 1], [0, 1, 0]]
    assert batch["input_ids"].shape == batch["attention_mask"].shape


def test_training_args_for_emotion(cfg, tmp_path):
    pytest.importorskip("transformers")
    from stress_signals.stress_model import build_training_args
    s = E.settings(cfg)
    args = build_training_args(s, tmp_path / "ck", seed=42, cuda=False)
    assert args.metric_for_best_model == "macro_ap" and args.greater_is_better and args.load_best_model_at_end
    assert args.per_device_train_batch_size == 32 and args.warmup_ratio == pytest.approx(0.1)


def test_trainer_compute_metrics_keys():
    class P:  # minimal EvalPrediction stand-in
        predictions = np.array([[3.0, -3.0], [-3.0, 3.0], [2.0, 2.0]])
        label_ids = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = E.trainer_compute_metrics(P())
    assert set(m) == {"macro_ap", "f1_macro", "f1_micro", "roc_auc_macro"} and m["f1_macro"] == pytest.approx(1.0)


# ---------------------------------------------------------------- 4C metrics / thresholds
def test_multilabel_metrics_match_sklearn():
    from sklearn.metrics import f1_score, roc_auc_score
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 2, (400, 4))
    P = np.clip(Y * 0.3 + rng.random((400, 4)) * 0.7, 0, 1)
    m = M.multilabel_metrics(Y, P, 0.5)
    H = (P >= 0.5).astype(int)
    assert m["f1_macro"] == pytest.approx(f1_score(Y, H, average="macro", zero_division=0))
    assert m["f1_micro"] == pytest.approx(f1_score(Y, H, average="micro", zero_division=0))
    assert m["roc_auc_macro"] == pytest.approx(roc_auc_score(Y, P, average="macro"))
    assert M.fast_roc_auc(Y[:, 0], P[:, 0]) == pytest.approx(roc_auc_score(Y[:, 0], P[:, 0]))
    ties = np.array([0.5, 0.5, 0.2, 0.9])
    assert M.fast_roc_auc(np.array([1, 0, 0, 1]), ties) == pytest.approx(roc_auc_score([1, 0, 0, 1], ties))


def test_bootstrap_multilabel_contains_point():
    rng = np.random.default_rng(1)
    Y = rng.integers(0, 2, (300, 3))
    P = np.clip(Y * 0.4 + rng.random((300, 3)) * 0.6, 0, 1)
    ci = M.bootstrap_multilabel(Y, P, [0.5, 0.4, 0.6], n_resamples=100, seed=0, label_names=["a", "b", "c"])
    for k in ("f1_macro", "f1_micro", "roc_auc_macro"):
        assert ci[k]["lo"] <= ci[k]["point"] <= ci[k]["hi"]
    assert set(ci["per_label"]) == {"a", "b", "c"}


def test_threshold_tuning_finds_shifted_optimum():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    p1 = np.clip(0.15 + 0.2 * y + rng.normal(0, 0.05, 2000), 0, 1)       # positives ~0.35: best threshold well below 0.5
    p2 = np.where(y == 1, 0.9, 0.1)                                     # perfectly separated: tie -> closest to 0.5
    tab, curve = M.tune_label_thresholds(np.c_[y, y], np.c_[p1, p2], (0.05, 0.95, 0.01), ["low", "sep"])
    t = tab.set_index("label")
    assert 0.2 <= t.loc["low", "threshold"] <= 0.3 and t.loc["low", "f1_at_threshold"] > t.loc["low", "f1_at_0.5"]
    assert t.loc["sep", "threshold"] == pytest.approx(0.5)
    assert set(curve.columns) == {"threshold", "label", "f1"}


# ---------------------------------------------------------------- 4D calibration
def test_platt_and_temperature_improve_nll_and_keep_ranking():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, (3000, 2))
    z_true = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.2, y.shape)
    Z = np.c_[3.0 * z_true[:, 0], z_true[:, 1] + 2.0]           # label 0 over-confident, label 1 offset
    for method in ("platt", "temperature"):
        params = M.fit_label_calibration(Z, y, method, ["a", "b"])
        P0, P1 = M.sigmoid(Z), M.apply_label_calibration(Z, params)
        for j in range(2):
            assert M._binary_nll_logits(np.log(P1[:, j] / (1 - P1[:, j])), y[:, j]) <= \
                   M._binary_nll_logits(Z[:, j], y[:, j]) + 1e-9
            assert M.fast_roc_auc(y[:, j], P1[:, j]) == pytest.approx(M.fast_roc_auc(y[:, j], P0[:, j]))
        assert all(p["a"] > 0 for p in params)
    platt = M.fit_label_calibration(Z, y, "platt", ["a", "b"])
    assert abs(platt[1]["b"]) > 1.0                               # Platt removes the offset that temperature cannot
    with pytest.raises(ValueError):
        M.fit_label_calibration(Z, y, "isotonic")


# ---------------------------------------------------------------- 4E Ekman
def test_ekman_mapping_official_and_rules(cfg, names):
    if not (Path(cfg["_root"]) / cfg["emotion_model"]["ekman"]["mapping_file"]).exists():
        pytest.skip("official ekman_mapping.json not on this machine")
    mp = E.load_ekman_mapping(cfg, names)
    assert list(mp) == ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    assert sorted(m for ms in mp.values() for m in ms) == sorted(names)
    assert len(mp["joy"]) == 12 and mp["fear"] == ["fear", "nervousness"]
    mem = E.ekman_members(mp, names)
    rng = np.random.default_rng(4)
    P = rng.random((50, 28))
    pm, pn = E.ekman_probs(P, mem, "max"), E.ekman_probs(P, mem, "noisy_or")
    assert np.all(pm <= pn + 1e-12)                                # max is the lower bound
    assert np.allclose(pm[:, -1], P[:, names.index("neutral")])
    Yg = E.ekman_any(P > 0.9, mem)
    assert Yg.shape == (50, 7)
    with pytest.raises(ValueError):
        E.ekman_probs(P, mem, "sum")


def test_ekman_mapping_rejects_bad_files(cfg, names, tmp_path):
    bad = tmp_path / "ek.json"
    bad.write_text(json.dumps({"anger": ["anger", "joy"], "joy": ["joy"]}))
    c = {**cfg, "_root": Path("/"), "emotion_model": {**cfg["emotion_model"],
                                                      "ekman": {**cfg["emotion_model"]["ekman"], "mapping_file": str(bad)}}}
    with pytest.raises(ValueError, match="Ekman mapping invalid"):
        E.load_ekman_mapping(c, names)


# ---------------------------------------------------------------- 4F domain shift aggregation
def test_cluster_bootstrap_diff_and_k_min_suppression(cfg):
    rng = np.random.default_rng(5)
    n = 400
    lab = np.repeat([0, 1], n // 2)
    post = np.repeat(np.arange(n // 2), 2)                          # 2 segments per post, same label within a post
    P = np.clip(rng.random((n, 28)) * 0.3 + (lab[:, None] * 0.2) * (np.arange(28) == 14), 0, 1)   # fear higher if stress
    pred = pd.DataFrame({"label": lab, "post_id": post, "community": np.where(np.arange(n) < 380, "big", "tiny"),
                         "n_tokens": 100, "n_chunks": 1, **{f"p__{nm}": P[:, j] for j, nm in enumerate(E.label_names(cfg))}})
    mapping = E.load_ekman_mapping(cfg)
    out = E.summarise_domain_shift(pred, E.label_names(cfg), mapping, [0.5] * 28, "max", k_min=50, max_length=128,
                                   n_resamples=200)
    fear = next(r for r in out["labels"] if r["name"] == "fear")
    assert fear["diff_lo"] > 0 and fear["diff_mean_p"] == pytest.approx(0.2, abs=0.05)
    tiny = [r for r in out["by_community_ekman"] if r["community"] == "tiny"]
    assert tiny and all(r["suppressed"] and "mean_p__fear" not in r for r in tiny)
    assert "not the same as stress" in out["note"]


# ---------------------------------------------------------------- notebook guard
def test_emotion_training_cells_carry_banner(cfg):
    calls = ("train_emotion_seed(", "trainer.train(")
    nb = json.loads((Path(cfg["_root"]) / "code" / "notebook.ipynb").read_text(encoding="utf-8"))
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "code" and any(c in src for c in calls):
            hits += 1
            assert E.TRAINING_BANNER in src and "H.run_step(" in src, src[:120]
    assert hits >= 1


def test_step4_heavy_cells_are_run_once(cfg):
    nb = json.loads((Path(cfg["_root"]) / "code" / "notebook.ipynb").read_text(encoding="utf-8"))
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for step in ("4B.1", "4C.1", "4D.1", "4C.3", "4C.4", "4F.1", "4F.2", "4G"):
        cell = next(s for s in srcs if f"# Step {step}:" in s)
        assert "H.run_step(" in cell, step


# ---------------------------------------------------------------- post-training pipeline on SYNTHETIC logits
def test_post_training_pipeline_offline_smoke(cfg, names, tmp_path, tokenizer):
    """evaluate -> select seed -> calibrate -> thresholds -> final report (once) -> bundle, on SYNTHETIC logits and a
    tiny RANDOM 28-label model saved with save_pretrained. No training, no download; writes only under tmp_path."""
    tr = pytest.importorskip("transformers")
    if not (Path(cfg["_root"]) / cfg["emotion_model"]["ekman"]["mapping_file"]).exists():
        pytest.skip("official ekman_mapping.json not on this machine")
    c = {**cfg, "_root": tmp_path, "paths": {**cfg["paths"], "models": "models"},
         "storage": {"checkpoint_root": "ck"},
         "sources": {**cfg["sources"], "goemotions": {**cfg["sources"]["goemotions"],
                                                      "official_dir": str(Path(cfg["_root"]) / cfg["sources"]["goemotions"]["official_dir"])}},
         "_paths": {**cfg["_paths"]},
         "emotion_model": {**cfg["emotion_model"], "seeds": [42, 13],
                           "eval": {**cfg["emotion_model"]["eval"], "bootstrap_resamples": 20},
                           "ekman": {**cfg["emotion_model"]["ekman"],
                                     "mapping_file": str(Path(cfg["_root"]) / cfg["emotion_model"]["ekman"]["mapping_file"])}}}
    rng = np.random.default_rng(0)
    conf = tr.RobertaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=80, num_labels=28, pad_token_id=tokenizer.pad_token_id,
                            problem_type="multi_label_classification", id2label=dict(enumerate(names)),
                            label2id={n: i for i, n in enumerate(names)})
    prev = rng.random(28) * 0.15 + 0.01
    for seed in (42, 13):
        d = E.run_dir_for(c, "roberta-base", seed)
        (d / "best_model").mkdir(parents=True)
        tr.AutoModelForSequenceClassification.from_config(conf).save_pretrained(d / "best_model")
        tokenizer.save_pretrained(d / "best_model")
        for split, n in (("validation", 600), ("test", 600)):
            Y = (rng.random((n, 28)) < prev).astype(int)
            Y[Y.sum(1) == 0, 27] = 1
            Z = 2.5 * (2 * Y - 1) + rng.normal(0, 2.5, Y.shape) - 1.0
            ids = pd.DataFrame({"record_id": [f"{split}{i}" for i in range(n)], "split": split})
            E._save_predictions(ids, Y, Z, names, d / f"predictions_{split}.parquet")
        (d / "train_summary.json").write_text(json.dumps({"epochs_completed": 3, "transformers": "x"}))
        (d / "DONE.json").write_text("{}")
    ev = E.evaluate_runs(c, "roberta-base")
    assert set(ev["seeds"]) == {42, 13} and ev["summary"]["test"]["f1_macro"]["n_runs"] == 2
    seed = E.select_final_seed(ev)
    cal = E.calibrate(c, seed, "roberta-base")
    assert cal["validation_ece_after"]["pooled"] <= cal["validation_ece_before"]["pooled"] + 1e-9
    thr = E.tune_thresholds(c, seed, "roberta-base")
    assert len(thr["thresholds"]) == 28 and all(0.05 <= v <= 0.95 for v in thr["thresholds"].values())
    rep = E.final_test_report(c, seed, "roberta-base")
    assert rep["test_ci"]["f1_macro"]["lo"] <= rep["test"]["f1_macro"] <= rep["test_ci"]["f1_macro"]["hi"]
    assert rep["ekman"]["groups"][-1] == "neutral" and len(rep["ekman"]["test"]["per_group"]) == 7
    again = E.final_test_report(c, seed, "roberta-base")
    assert json.dumps(rep, sort_keys=True, default=str) == json.dumps(again, sort_keys=True, default=str)
    bundle = E.save_bundle(c, seed, "roberta-base")
    have = {p.name for p in bundle.iterdir()}
    assert {"config.json", "emotion_config.json", "thresholds.json", "ekman_mapping.json", "metrics.json",
            "model_card.md", "manifest.json"} <= have
    ec = json.loads((bundle / "emotion_config.json").read_text())
    assert ec["problem_type"] == "multi_label_classification" and len(ec["calibration"]["params"]) == 28
    assert ec["thresholds"]["fitted_on"].startswith("validation")
    man = json.loads((bundle / "manifest.json").read_text())
    from stress_signals.utils import sha256_file
    assert all(sha256_file(bundle / k) == v["sha256"] for k, v in man["artifacts"].items())
    card = (bundle / "model_card.md").read_text(encoding="utf-8")
    assert "not a stress detector" in card and "Not for diagnosis" in card
    with pytest.raises(FileExistsError):
        E.save_bundle(c, seed, "roberta-base")
