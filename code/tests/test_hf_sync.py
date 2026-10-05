"""Hugging Face sync tests: no network. `_pull` and the upload calls are replaced by fakes."""

import copy
from pathlib import Path

import pytest

from stress_signals import hf_sync as HF
from stress_signals.config import load_config


@pytest.fixture()
def cfg():
    return copy.deepcopy(load_config())


def test_pull_missing_covers_data_and_all_three_bundles(cfg, monkeypatch):
    seen = []
    monkeypatch.setattr(HF, "_pull", lambda repo, rtype, target, prefix, wanted: seen.append((repo, rtype, Path(target).as_posix())) or [])
    notes = HF.pull_missing(cfg)
    assert set(notes) == {"data", "model", "emotion", "stressor"} and all(n.startswith("nothing missing") for n in notes.values())
    by_repo = {r: t for r, _, t in seen}
    assert by_repo["Amrita-Vishwa-Ghopeetham/emotion-roberta-base"].endswith("trained_models/emotion/v20261004")
    assert by_repo["Amrita-Vishwa-Ghopeetham/stressor-distilroberta-base"].endswith("trained_models/stressor/v20261004")
    assert by_repo["Amrita-Vishwa-Ghopeetham/stress-roberta-base"].endswith("trained_models/stress/v20261004")


def test_pull_missing_can_pick_one_bundle_and_never_raises(cfg, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(HF, "_pull", boom)
    notes = HF.pull_missing(cfg, what=("stressor",))
    assert list(notes) == ["stressor"] and notes["stressor"].startswith("SKIPPED (ConnectionError)")
    cfg["huggingface"]["enabled"] = False
    assert "nothing pulled" in HF.pull_missing(cfg)["all"]


def test_readme_has_header_card_and_the_stressor_correction():
    card = "# Model card: x\n\n**Data.** Evaluation: hand-labelled gold set\n"
    r = HF.bundle_readme("stressor", card)
    assert r.startswith("---\nlanguage: en") and "base_model: distilroberta-base" in r and card.strip() in r
    assert "machine-labelled" in r and "prototypes.npy" in r and "all-MiniLM-L6-v2" in r
    assert HF.bundle_readme("emotion", card).count("---") == 2 and "go_emotions" in HF.bundle_readme("emotion", card)


def test_push_bundle_dry_run_lists_files_and_uploads_nothing(cfg, monkeypatch):
    called = []
    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", lambda *a, **k: called.append("create"))
    monkeypatch.setattr("huggingface_hub.HfApi.upload_folder", lambda *a, **k: called.append("upload"))
    out = HF.push_bundle(cfg, "stressor", dry_run=True)
    assert out["private"] is True and out["repo"].endswith("stressor-distilroberta-base") and "model.safetensors" in out["files"] and "README.md" in out["files"]
    assert out["verified_files"] >= 10 and called == []


def test_push_bundle_refuses_unknown_component_and_corrupt_bundle(cfg, tmp_path):
    with pytest.raises(KeyError):
        HF.push_bundle(cfg, "nonsense", dry_run=True)
    from stress_signals.inference import BundleIntegrityError
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text('{"artifacts": {"model.safetensors": {"type": "file", "sha256": "0"}}}', encoding="utf-8")
    (bad / "model.safetensors").write_bytes(b"x")
    (bad / "model_card.md").write_text("card", encoding="utf-8")
    cfg["_root"] = tmp_path
    cfg["huggingface"]["stressor_dir"] = "bad"
    with pytest.raises(BundleIntegrityError):
        HF.push_bundle(cfg, "stressor", dry_run=True)
