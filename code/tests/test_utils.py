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


def test_models_and_checkpoints_stay_local():
    """Models live in <project root>/trained_models; checkpoints outside it (the Trainer deletes old ones)."""
    import stress_signals.utils as U
    from stress_signals.config import load_config
    cfg = load_config()
    root = Path(cfg["_root"])
    assert U.models_root(cfg) == root / "trained_models"
    assert U.model_version_dir(cfg, "stress").parent == root / "trained_models" / "stress"
    ck = U.checkpoint_root(cfg)
    assert ck == root / "data" / "outputs" / "checkpoints" and "trained_models" not in ck.parts


def test_config_rejects_drive_models_and_checkpoints_inside_trained_models():
    import copy
    from stress_signals.config import ConfigError, load_config, validate_config
    raw = {k: copy.deepcopy(v) for k, v in load_config().items() if not k.startswith("_")}
    for mutate in (lambda c: c["paths"].__setitem__("models", "G:/My Drive/TA-BDA Project/models"),
                   lambda c: c["storage"].__setitem__("checkpoint_root", "trained_models/checkpoints"),
                   lambda c: c["inference"].__setitem__("device", "mps")):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ConfigError):
            validate_config(bad)


def test_select_device_prefers_nvidia_gpu_and_honours_config(monkeypatch):
    import torch
    from stress_signals.utils import device_label, select_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device() == "cuda" and select_device({"inference": {"device": "cuda"}}) == "cuda"
    assert select_device({"inference": {"device": "cpu"}}) == "cpu"            # config can force the CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device() == "cpu"
    assert select_device({"inference": {"device": "cuda"}}) == "cpu"           # forced GPU missing -> CPU, no crash
    assert device_label("cpu") == "cpu"


def test_raw_data_status_and_code_fingerprint(tmp_path):
    from stress_signals.config import load_config
    from stress_signals.utils import code_fingerprint, raw_data_status
    cfg = load_config()
    empty = {**cfg, "_root": tmp_path}
    assert raw_data_status(empty) == {k: False for k in ("sad", "mendeley", "senticnet", "zenodo_pilot")}
    (tmp_path / "code" / "stress_signals").mkdir(parents=True)
    (tmp_path / "code" / "tests").mkdir()
    (tmp_path / "code" / "config.yaml").write_text("a: 1")
    f = tmp_path / "code" / "stress_signals" / "x.py"
    f.write_text("x = 1")
    before = code_fingerprint(tmp_path)
    assert code_fingerprint(tmp_path) == before                               # stable
    f.write_text("x = 2")
    assert code_fingerprint(tmp_path) != before                               # any code change -> tests re-run


def test_package_never_deletes_files():
    """Guard for the 'nothing is ever deleted' rule: delete calls in the package are limited to the audited list below.
    (Until now this test passed vacuously: its regex began with a stray backspace character, so it matched nothing.)
    Every allowed call removes only the step's OWN scratch or staging files, never user data:
      external_validation.py  1  temp copy of the TensiStrength folder (a path without spaces)
      ingestion.py            3  partial part files of an interrupted input; the small files replaced by compaction; the lake of one
                                 corpus ONLY when ingest(force=True) is passed
      spark_jobs.py           4  its own payload / parity scratch files and the parity output folder
    A new delete call fails this test until it is reviewed and added here."""
    import re
    pkg = Path(__file__).resolve().parents[1] / "stress_signals"
    pat = re.compile(r"(rmtree|os\.remove|os\.unlink|\.unlink\(|\.rmdir\(|send2trash)")
    audited = {"external_validation.py": 1, "ingestion.py": 3, "spark_jobs.py": 4}
    found: dict[str, int] = {}
    for f in pkg.glob("*.py"):
        n = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if pat.search(line) and not line.lstrip().startswith("#"))
        if n:
            found[f.name] = n
    assert found == audited, f"delete calls found {found}, audited {audited}: review any new one, then update this list"


def test_gpu_supported_by_build_follows_cuda_compatibility():
    from stress_signals.utils import gpu_supported_by_build
    cu121 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]   # torch 2.5.1+cu121
    assert gpu_supported_by_build(6, 1, cu121)          # Quadro P1000
    assert gpu_supported_by_build(8, 9, cu121)          # RTX 40-series runs sm_86 code
    assert not gpu_supported_by_build(12, 0, cu121)     # RTX 50-series needs newer wheels
    assert gpu_supported_by_build(12, 0, cu121 + ["compute_90"])   # ... unless PTX can be JIT-compiled
    assert not gpu_supported_by_build(3, 5, cu121)
