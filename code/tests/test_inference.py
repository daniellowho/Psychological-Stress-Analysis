"""Step 7 tests: unified inference. NOTHING here trains or updates weights: tiny randomly initialised models (built from a
config, never fitted) run forward passes; bundles are written to pytest temp folders, never to trained_models/.
The one test on the real saved bundles is OFF unless STRESS_SIGNALS_REAL_BUNDLE_TEST=1 (it loads two ~500 MB models)."""

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_signals import emotion_model as E
from stress_signals import inference as I
from stress_signals import stressor_model as SX
from stress_signals.config import load_config
from stress_signals.utils import write_json, write_manifest

NEUTRAL_EXAMPLES = [                       # hand-written, no real user text
    "The train to the city leaves at nine o'clock every morning.",
    "I made a pot of tea and read a chapter of my book.",
    "The library opens at ten and closes at six on weekdays.",
    "We planted tomatoes along the back fence this weekend.",
    "The meeting notes are saved in the shared folder.",
]


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["inference"] = {**c["inference"], "english_only": False}
    return c


@pytest.fixture(scope="module")
def tok():
    import transformers as tr
    try:
        return tr.AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"distilroberta tokenizer not cached locally: {exc!r}")


def _tiny_model(tok, num_labels: int, seed: int):
    import torch
    import transformers as tr
    torch.manual_seed(seed)
    conf = tr.RobertaConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=80, pad_token_id=tok.pad_token_id,
                            num_labels=num_labels)
    return tr.AutoModelForSequenceClassification.from_config(conf).eval()


def _seal(bundle: Path) -> None:
    files = [p for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "manifest.json"]
    write_manifest(bundle / "manifest.json", artifacts=files, seeds=[0], root=bundle)


def _write_stress(root: Path, tok, version: str = "v20260101") -> Path:
    d = root / "stress" / version
    d.mkdir(parents=True)
    _tiny_model(tok, 2, 1).save_pretrained(d)
    tok.save_pretrained(d)
    write_json(d / "stress_config.json", {
        "component": "stress", "max_length": 64, "positive_label": 1, "label_map": {"0": "not_stress", "1": "stress"},
        "calibration": {"method": "temperature_scaling", "temperature": 1.3}, "threshold": {"value": 0.5},
        "chunking": {"aggregation": "mean", "max_chunks_per_doc": 4, "max_tokens": 64, "stride": 16}})
    _seal(d)
    return d


def _write_emotion(root: Path, tok, cfg, version: str = "v20260101") -> Path:
    d = root / "emotion" / version
    d.mkdir(parents=True)
    names = E.label_names(cfg)
    _tiny_model(tok, len(names), 2).save_pretrained(d)
    tok.save_pretrained(d)
    mapping = E.load_ekman_mapping(cfg, names)
    write_json(d / "ekman_mapping.json", {"combine": "max", "groups": mapping})
    write_json(d / "emotion_config.json", {
        "component": "emotion", "max_length": 48, "label_map": {str(i): n for i, n in enumerate(names)},
        "calibration": {"method": "platt", "params": {n: {"a": 1.0, "b": 0.0} for n in names}},
        "ekman": {"combine": "max", "groups": list(mapping), "mapping_file": "ekman_mapping.json"},
        "chunking": {"aggregation": "mean", "max_chunks_per_doc": 4, "max_tokens": 48, "stride": 12}})
    _seal(d)
    return d


def _stressor_cfg(cfg, tax, method: str, kinds: dict[str, str], thr: dict[str, float], agg: str, **extra) -> dict:
    cov = SX.coverage(tax)
    return {"component": "stressor", "method": method, "taxonomy_file": "stressors_v1.yaml",
            "categories": [{"id": c, "coverage": cov[c], "score_kind": kinds[c], "lower_confidence": cov[c] == "zero_shot"}
                           for c in SX.category_ids(tax)],
            "thresholds": thr, "aggregation": agg, "inference_unit": {"windows": cfg["stressor_model"]["windows"]}, **extra}


def _write_stressor_keyword(root: Path, cfg, version: str = "v20260101") -> Path:
    d = root / "stressor" / version
    d.mkdir(parents=True)
    shutil.copyfile(SX.taxonomy_path(cfg), d / "stressors_v1.yaml")
    tax = SX.load_taxonomy(path=d / "stressors_v1.yaml")
    cats = SX.category_ids(tax)
    write_json(d / "stressor_config.json", _stressor_cfg(cfg, tax, "keyword", {c: "count" for c in cats}, {c: 1.0 for c in cats}, "sum"))
    _seal(d)
    return d


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, tok, cfg):
    root = tmp_path_factory.mktemp("artifacts")
    _write_stress(root, tok)
    _write_emotion(root, tok, cfg)
    _write_stressor_keyword(root, cfg)
    return root


@pytest.fixture(scope="module")
def pipe(artifacts, cfg):
    return I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg)


# ---------------------------------------------------------------- shape and ranges
def test_output_shape_and_columns(pipe):
    df = pipe.predict(NEUTRAL_EXAMPLES, batch_size=2)
    assert len(df) == 5 and list(df.columns) == pipe.output_columns()
    assert [c for c in df.columns if c.startswith("emotion_p__")].__len__() == 28
    assert [c for c in df.columns if c.startswith("emotion_group__")].__len__() == 7
    assert set(df["input_status"]) == {"ok"}
    assert (df["stress_version"] == "v20260101").all() and (df["emotion_version"] == "v20260101").all()
    assert df.index.tolist() == list(range(5))
    assert not any(c.startswith("tension_") for c in df.columns)          # off by default


def test_probabilities_in_unit_interval_and_flag_matches_threshold(pipe):
    df = pipe.predict(NEUTRAL_EXAMPLES)
    prob_cols = ["stress_prob_calibrated"] + [c for c in df.columns if c.startswith(("emotion_p__", "emotion_group__"))]
    arr = df[prob_cols].to_numpy(dtype=float)
    assert np.isfinite(arr).all() and (arr >= 0).all() and (arr <= 1).all()
    assert (df["stress_flag"].astype(bool) == (df["stress_prob_calibrated"] >= pipe.stress.threshold)).all()
    group_cols = [f"emotion_group__{g}" for g in pipe.emotion.groups]
    for g, members in zip(pipe.emotion.groups, pipe.emotion.members):
        want = df[[f"emotion_p__{pipe.emotion.names[m]}" for m in members]].max(axis=1)
        np.testing.assert_allclose(df[f"emotion_group__{g}"], want, atol=1e-12)
    assert set(group_cols) <= set(df.columns)


def test_no_text_or_ids_in_output(pipe):
    df = pipe.predict(NEUTRAL_EXAMPLES)
    joined = " ".join(df.astype(str).to_numpy().ravel())
    assert not any(s[:25] in joined for s in NEUTRAL_EXAMPLES)            # the input text is never echoed
    assert "text" not in " ".join(df.columns)


def test_empty_input_list(pipe):
    df = pipe.predict([])
    assert len(df) == 0 and list(df.columns) == pipe.output_columns()


# ---------------------------------------------------------------- determinism
def test_same_input_same_output(pipe):
    a = pipe.predict(NEUTRAL_EXAMPLES, batch_size=3)
    b = pipe.predict(NEUTRAL_EXAMPLES, batch_size=3)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_order_and_batch_size_invariance_within_tolerance(pipe):
    a = pipe.predict(NEUTRAL_EXAMPLES, batch_size=1)
    b = pipe.predict(NEUTRAL_EXAMPLES[::-1], batch_size=5).iloc[::-1].reset_index(drop=True)
    num = [c for c in a.columns if a[c].dtype.kind == "f"]
    np.testing.assert_allclose(a[num].to_numpy(), b[num].to_numpy(), atol=1e-4)


def test_duplicate_texts_get_identical_scores(pipe):
    df = pipe.predict([NEUTRAL_EXAMPLES[0]] * 3 + [NEUTRAL_EXAMPLES[1]], batch_size=2)
    num = [c for c in df.columns if df[c].dtype.kind == "f"]
    np.testing.assert_allclose(df.loc[0, num].to_numpy(dtype=float), df.loc[2, num].to_numpy(dtype=float), atol=1e-4)


# ---------------------------------------------------------------- edge cases
def test_edge_cases_return_nan_with_reason_and_never_crash(pipe):
    texts = [None, "", "   \n ", "😀😀😀🔥", "12345 !!! ???", float("nan"), 42, "https://example.org/x u/someone",
             "[removed]", "Ça va très bien aujourd'hui, merci beaucoup.", "x", NEUTRAL_EXAMPLES[0]]
    df = pipe.predict(texts)
    st = df["input_status"].tolist()
    assert st[0] == "none" and st[5] == "none" and st[6] == "non_string"
    assert st[1] == "empty" and st[2] == "empty"
    assert st[3] == "no_text" and st[4] == "no_text" and st[10] == "no_text"
    assert st[7] == "no_text" and st[8] == "empty_after_cleaning"            # only <URL>/<USER> placeholders left / only a [removed] marker
    bad = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert df.loc[bad, "stress_prob_calibrated"].isna().all()
    assert df.loc[bad].filter(regex="^emotion_(p|group)__").isna().all().all()
    assert df.loc[bad, "stress_flag"].isna().all()
    assert df.loc[bad, "stressor_top"].isna().all()
    assert df.loc[[9, 11], "stress_prob_calibrated"].notna().all()         # french is scored when english_only is off
    assert pipe.last_run["status_counts"]["none"] == 2


def test_very_long_text_is_truncated_and_scored(pipe):
    long = "The committee reviewed the schedule again. " * 2000            # ~84k characters
    df = pipe.predict([long, NEUTRAL_EXAMPLES[0]])
    assert df.loc[0, "input_status"] == "ok_truncated" and np.isfinite(df.loc[0, "stress_prob_calibrated"])
    assert df.loc[0, "n_chunks_stress"] <= 4 and df.loc[0, "n_chunks_emotion"] <= 4      # max_chunks_per_doc from the bundle
    assert df.loc[1, "input_status"] == "ok"


def test_non_english_filter(artifacts, cfg):
    pytest.importorskip("lingua")
    p = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg, english_only=True)
    df = p.predict(["Das Wetter ist heute sehr schön und wir gehen am Nachmittag im Park spazieren.",
                    "The weather is very nice today and we are going for a walk in the park this afternoon."])
    assert df["input_status"].tolist() == ["non_english", "ok"]
    assert np.isnan(df.loc[0, "stress_prob_calibrated"]) and np.isfinite(df.loc[1, "stress_prob_calibrated"])


# ---------------------------------------------------------------- stressor (keyword bundle; the method needs no weights)
def test_stressor_keyword_columns_flags_and_none(artifacts, cfg):
    pipe = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg, gate_stressors=False)
    tax = SX.load_taxonomy(cfg)
    cat = tax["categories"][0]
    kw = next(k for k in cat["keywords"] if not k.endswith("*") and " " not in k)
    df = pipe.predict([f"I keep thinking about {kw} all week long.", "Zzyzx quux flibber jabber wocky."])
    assert f"stressor_count__{cat['id']}" in df.columns and f"stressor_flag__{cat['id']}" in df.columns
    assert bool(df.loc[0, f"stressor_flag__{cat['id']}"]) and df.loc[0, "stressor_top"] != "none_unclear"
    assert not bool(df.loc[0, "stressor_none_unclear"])
    assert bool(df.loc[1, "stressor_none_unclear"]) and df.loc[1, "stressor_top"] == "none_unclear"


def test_stressor_gate_blanks_tags_when_not_stress_flagged(artifacts, cfg):
    kw = next(k for k in SX.load_taxonomy(cfg)["categories"][0]["keywords"] if not k.endswith("*") and " " not in k)
    text = f"I keep thinking about {kw} all week long."
    off = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg, gate_stressors=False).predict([text])
    on = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg, gate_stressors=True)
    on.stress.threshold = 2.0                                        # nothing can be stress-flagged
    df = on.predict([text])
    assert not bool(df.loc[0, "stress_flag"]) and df.loc[0, "stressor_top"] == "none_unclear" and bool(df.loc[0, "stressor_none_unclear"])
    assert not df.filter(like="stressor_flag__").iloc[0].any()
    cols = [c for c in df.columns if c.startswith("stressor_count__")]
    assert (df[cols].iloc[0] == off[cols].iloc[0]).all()             # raw scores untouched


def test_pipeline_without_stressor_bundle_omits_columns(tmp_path, tok, cfg):
    _write_stress(tmp_path, tok)
    _write_emotion(tmp_path, tok, cfg)
    p = I.StressSignalPipeline.from_artifacts(tmp_path, device="cpu", cfg=cfg)
    df = p.predict(NEUTRAL_EXAMPLES[:2])
    assert not p.has_stressor and not any(c.startswith("stressor_") and c != "stressor_version" for c in df.columns)
    assert df["stressor_version"].isna().all()


def test_stressor_embed_lr_hybrid_with_injected_embedder(tmp_path, tok, cfg):
    """Frozen-embedding + logistic-regression bundle built from RANDOM numbers (no fitting): checks merging, the zero-shot fill for
    categories SAD lacks, thresholds, and column naming. Not a statement about real accuracy."""
    import transformers as tr
    import torch
    _write_stress(tmp_path, tok)
    _write_emotion(tmp_path, tok, cfg)
    d = tmp_path / "stressor" / "v20260101"
    d.mkdir(parents=True)
    shutil.copyfile(SX.taxonomy_path(cfg), d / "stressors_v1.yaml")
    tax = SX.load_taxonomy(path=d / "stressors_v1.yaml")
    cats, cov, classes = SX.category_ids(tax), SX.coverage(tax), SX.classifier_classes(tax)
    rng = np.random.default_rng(0)
    dim = 8
    np.savez(d / "lr.npz", coef=rng.normal(size=(len(classes), dim)), intercept=np.zeros(len(classes)), classes=np.array(classes))
    protos = rng.normal(size=(len(cats), dim))
    np.save(d / "prototypes.npy", protos / np.linalg.norm(protos, axis=1, keepdims=True))
    kinds = {c: "probability" if cov[c] == "sad" else "cosine" for c in cats}
    write_json(d / "stressor_config.json", _stressor_cfg(cfg, tax, "embed_lr", kinds, {c: 0.5 if kinds[c] == "probability" else 0.0 for c in cats},
                                                         "max", calibration={"temperature": 1.0}))
    _seal(d)
    torch.manual_seed(0)
    emb_model = tr.AutoModel.from_config(tr.RobertaConfig(vocab_size=tok.vocab_size, hidden_size=dim, num_hidden_layers=1, num_attention_heads=2,
                                                          intermediate_size=16, max_position_embeddings=80, pad_token_id=tok.pad_token_id))
    embedder = SX.Embedder(SX.EmbeddingSpec("tiny", "mean", True, 32, False, None), "cpu", 2, model=emb_model, tokenizer=tok)
    p = I.StressSignalPipeline.from_artifacts(tmp_path, device="cpu", cfg=cfg, embedder=embedder)
    texts = ["I am worried about the deadline at the office this week.", "We walked along the river and had lunch outside."]
    df = p.predict(texts)
    pcols = [c for c in df.columns if c.startswith("stressor_p__")]
    ccols = [c for c in df.columns if c.startswith("stressor_cos__")]
    assert len(pcols) == sum(k == "probability" for k in kinds.values()) and len(ccols) == sum(k == "cosine" for k in kinds.values()) > 0
    assert ((df[pcols].to_numpy() >= 0) & (df[pcols].to_numpy() <= 1)).all()
    assert df["stressor_top"].notna().all()
    pd.testing.assert_frame_equal(df, p.predict(texts), check_exact=True)


# ---------------------------------------------------------------- versions, hashes, bundle manifest, cache
def test_latest_and_pinned_versions(tmp_path):
    for v in ("v20260101", "v20260301", "v20260201", "v20260301.bak-20260401T000000Z"):
        (tmp_path / "stress" / v).mkdir(parents=True)
        (tmp_path / "stress" / v / "manifest.json").write_text("{}", encoding="utf-8")
    assert I.list_versions(tmp_path, "stress") == ["v20260101", "v20260201", "v20260301"]
    assert I.resolve_version(tmp_path, "stress") == "v20260301"
    assert I.resolve_version(tmp_path, "stress", "v20260101") == "v20260101"
    assert I.resolve_version(tmp_path, "emotion") is None
    with pytest.raises(FileNotFoundError):
        I.resolve_version(tmp_path, "stress", "v19990101")


def test_missing_required_component_raises(tmp_path, cfg, tok):
    _write_stress(tmp_path, tok)
    with pytest.raises(FileNotFoundError, match="emotion"):
        I.StressSignalPipeline.from_artifacts(tmp_path, device="cpu", cfg=cfg)


def test_hash_mismatch_is_detected_at_load(artifacts, tmp_path, cfg):
    bad = tmp_path / "copy"
    shutil.copytree(artifacts, bad)
    cfg_file = bad / "stress" / "v20260101" / "stress_config.json"
    cfg_file.write_text(cfg_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(I.BundleIntegrityError, match="stress_config.json"):
        I.StressSignalPipeline.from_artifacts(bad, device="cpu", cfg=cfg)
    (bad / "stress" / "v20260101" / "stress_config.json").unlink()
    with pytest.raises(I.BundleIntegrityError, match="missing"):
        I.verify_bundle(bad / "stress" / "v20260101")


def test_bundle_manifest_roundtrip_and_tamper(artifacts, tmp_path, cfg):
    out = tmp_path / "bundle_manifest.json"
    m = I.write_bundle_manifest(artifacts, out, cfg=cfg)
    assert set(m["components"]) == {"stress", "emotion", "stressor"} and m["missing_optional_components"] == []
    assert I.versions_from_bundle_manifest(out) == {"stress": "v20260101", "emotion": "v20260101", "stressor": "v20260101"}
    I.check_bundle_manifest(artifacts, out)
    p = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg, versions=I.versions_from_bundle_manifest(out))
    assert p.bundle_info["bundle_id"] == m["bundle_id"]
    copy_root = tmp_path / "copy"
    shutil.copytree(artifacts, copy_root)
    mp = copy_root / "emotion" / "v20260101" / "manifest.json"
    mp.write_text(mp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(I.BundleIntegrityError):
        I.check_bundle_manifest(copy_root, out)


def test_get_pipeline_caches_per_process(artifacts, cfg):
    I.clear_pipeline_cache()
    a = I.get_pipeline(artifacts, "cpu", cfg)
    b = I.get_pipeline(artifacts, "cpu", cfg)
    assert a is b
    I.clear_pipeline_cache()
    assert I.get_pipeline(artifacts, "cpu", cfg) is not a
    I.clear_pipeline_cache()


def test_settings_defaults_fp16_off_and_legacy_flag():
    s = I.InferenceSettings.from_cfg({"inference": {"half_precision": False, "batch_size": 4}})
    assert s.use_fp16 is False and s.batch_size == 4
    assert I.InferenceSettings.from_cfg({}).use_fp16 is False
    assert I.InferenceSettings.from_cfg({"inference": {"use_fp16": True}}, batch_size=2).batch_size == 2


def test_project_config_has_fp16_off(cfg):
    assert cfg["inference"]["use_fp16"] is False and cfg["inference"]["batch_size"] >= 1


# ---------------------------------------------------------------- the real saved bundles (opt-in)
@pytest.mark.skipif(os.environ.get("STRESS_SIGNALS_REAL_BUNDLE_TEST") != "1", reason="set STRESS_SIGNALS_REAL_BUNDLE_TEST=1 (loads two 500 MB models)")
def test_real_bundles_smoke():
    cfg = load_config()
    root = Path(cfg["_root"]) / cfg["paths"]["models"]
    p = I.StressSignalPipeline.from_artifacts(root, cfg=cfg)
    df = p.predict(NEUTRAL_EXAMPLES, batch_size=cfg["inference"]["batch_size"])
    num = df[["stress_prob_calibrated"] + [c for c in df.columns if c.startswith(("emotion_p__", "emotion_group__"))]].to_numpy(dtype=float)
    assert len(df) == 5 and (num >= 0).all() and (num <= 1).all()
    if p.device == "cuda":
        assert p.last_run["peak_vram_gb"] < cfg["inference"]["max_vram_gb"]
    pd.testing.assert_frame_equal(df, p.predict(NEUTRAL_EXAMPLES, batch_size=cfg["inference"]["batch_size"]), check_exact=True)
