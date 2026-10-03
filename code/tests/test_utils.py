"""Utility tests: hashing, manifests, seeding, environment detection."""

import hashlib
from pathlib import Path

import pytest
import json
import random

from stress_signals.utils import detect_environment, get_logger, set_seed, sha256_file, write_manifest


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"stress_signals")
    assert sha256_file(p) == hashlib.sha256(b"stress_signals").hexdigest()


def test_manifest_written(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    out = tmp_path / "manifest.json"
    write_manifest(out, artifacts=[f, tmp_path / "missing"], seeds=[42], root=tmp_path)
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["artifacts"]["x.txt"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert m["artifacts"]["missing"]["type"] == "missing"
    assert m["seeds"] == [42] and "python" in m["library_versions"]


def test_seed_reproducible():
    set_seed(42)
    a = [random.random() for _ in range(3)]
    set_seed(42)
    assert a == [random.random() for _ in range(3)]


def test_logger_idempotent():
    n = len(get_logger("t").handlers)
    assert len(get_logger("t").handlers) == n


def test_detect_environment_never_raises():
    info = detect_environment()
    assert "python" in info and isinstance(info["warnings"], list)


def test_models_go_to_trained_models_on_both_kernels(monkeypatch):
    """No Google Drive: models live in <project root>/trained_models on whichever kernel runs."""
    import stress_signals.utils as U
    from stress_signals.config import load_config
    cfg = load_config()
    for colab, root in ((True, Path("/content/TA-BDA")), (False, Path(cfg["_root"]))):
        monkeypatch.setattr(U, "is_colab", lambda c=colab: c)
        c = {**cfg, "_root": root}
        assert U.models_root(c) == root / "trained_models"
        assert U.model_version_dir(c, "stress").parent == root / "trained_models" / "stress"
        assert "drive" not in U.models_root(c).as_posix().lower()


def test_checkpoints_outside_trained_models(monkeypatch):
    import stress_signals.utils as U
    from stress_signals.config import load_config
    cfg = load_config()
    for colab in (True, False):
        monkeypatch.setattr(U, "is_colab", lambda c=colab: c)
        ck = U.checkpoint_root(cfg)
        assert "trained_models" not in ck.parts and "drive" not in ck.as_posix().lower()


def test_config_rejects_drive_models_and_checkpoints_inside_trained_models():
    import copy
    from stress_signals.config import ConfigError, load_config, validate_config
    raw = {k: copy.deepcopy(v) for k, v in load_config().items() if not k.startswith("_")}
    for mutate in (lambda c: c["paths"].__setitem__("models", "G:/My Drive/TA-BDA Project/models"),
                   lambda c: c["storage"].__setitem__("local_checkpoint_root", "trained_models/checkpoints"),
                   lambda c: c["storage"].__setitem__("colab_checkpoint_root", "/content/drive/MyDrive/ck")):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ConfigError):
            validate_config(bad)


def test_package_never_deletes_files():
    """Guard for the 'nothing is ever deleted' rule: no delete calls anywhere in the package."""
    import re
    pkg = Path(__file__).resolve().parents[1] / "stress_signals"
    pat = re.compile(r"(rmtree|os\.remove|os\.unlink|\.unlink\(|\.rmdir\(|send2trash)")
    hits = [f"{p.name}:{i}" for p in pkg.glob("*.py") for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if pat.search(line) and not line.lstrip().startswith("#")]
    assert hits == [], hits


def test_colab_bundle_contains_code_and_training_data(tmp_path):
    import zipfile
    from stress_signals.config import load_config
    from stress_signals.utils import COLAB_BUNDLE_DATA, build_colab_bundle
    cfg = load_config()
    if not all((Path(cfg["_root"]) / r).is_file() for r in COLAB_BUNDLE_DATA):
        pytest.skip("processed Dreaddit not built")
    c = {**cfg, "_paths": {**cfg["_paths"], "artifacts": tmp_path}}
    out = build_colab_bundle(c)
    names = set(zipfile.ZipFile(out).namelist())
    assert "code/config.yaml" in names and "code/stress_signals/stress_model.py" in names and "code/notebook.ipynb" in names
    assert set(COLAB_BUNDLE_DATA) <= names
    assert not any("__pycache__" in n or n.startswith("data/raw") for n in names)
    assert out.stat().st_size < 5e6
