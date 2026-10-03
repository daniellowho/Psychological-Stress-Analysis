"""Handoff log, run-once guard and Colab <-> PC transfer. No training: steps are tiny lambdas; writes only under tmp_path."""

import json
import zipfile
from pathlib import Path

import pytest

import stress_signals.handoff as H
from stress_signals.config import load_config
from stress_signals.utils import BUNDLE_HANDOFF_ARCNAME


@pytest.fixture(scope="module")
def base_cfg():
    return load_config()


def make_cfg(base, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    return {**base, "_root": root, "_paths": {**base["_paths"], "artifacts": root / "data" / "outputs"}}


def on_kernel(monkeypatch, colab: bool):
    monkeypatch.setattr(H, "is_colab", lambda: colab)


def test_step_runs_once_and_is_logged(base_cfg, tmp_path, monkeypatch):
    on_kernel(monkeypatch, False)
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    calls = []
    assert H.run_step(cfg, "3A.1/x", "demo", lambda: calls.append(1) or "ran") == "ran"
    assert H.run_step(cfg, "3A.1/x", "demo", lambda: calls.append(1) or "ran", skip_result="skipped") == "skipped"
    assert calls == [1]
    statuses = [e["status"] for e in H.load(cfg)["events"]]
    assert statuses == ["started", "done", "skipped"]
    ev = H.done_event(cfg, "3A.1/x")
    assert ev["kernel"] == "local" and ev["utc"].endswith("+00:00") and ev["duration_s"] is not None
    md = (H.models_root(cfg) / "HANDOFF.md").read_text(encoding="utf-8")
    assert "`3A.1/x`" in md and "**done**" in md
    assert md.index("## How this works") < md.index("## Current state per step") and "TRAIN_HERE" in md
    assert H.status_table(cfg).set_index("step").loc["3A.1/x", "state"] == "done"


def test_not_allowed_step_never_runs_but_done_one_still_skips(base_cfg, tmp_path, monkeypatch, capsys):
    on_kernel(monkeypatch, False)
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    assert H.run_step(cfg, "seed13", "t", lambda: pytest.fail("Run All must not train"), allow=False,
                      blocked_reason="switch it on", skip_result="no") == "no"
    assert "NOT RUN seed13: switch it on" in capsys.readouterr().out
    assert H.load(cfg)["events"] == []                                  # a blocked step leaves no trace
    H.run_step(cfg, "seed42", "t", lambda: 1)
    H.run_step(cfg, "seed42", "t", lambda: pytest.fail("done already"), allow=True)
    assert "SKIP seed42" in capsys.readouterr().out
    assert not H.ready(False, "3C.1", "needs seeds") and "NOT RUN 3C.1: needs seeds" in capsys.readouterr().out


def test_failed_step_is_logged_and_can_rerun(base_cfg, tmp_path, monkeypatch):
    on_kernel(monkeypatch, False)
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")

    def boom():
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError):
        H.run_step(cfg, "s", "t", boom)
    assert H.last_event(cfg, "s")["status"] == "failed" and "out of memory" in H.last_event(cfg, "s")["note"]
    assert H.run_step(cfg, "s", "t", lambda: 7) == 7


def test_existing_marker_counts_as_done(base_cfg, tmp_path, monkeypatch):
    on_kernel(monkeypatch, False)
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    marker = H.models_root(cfg) / "stress" / "runs" / "m" / "seed42" / "DONE.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert H.run_step(cfg, "seed42", "t", lambda: pytest.fail("must not run"), done_marker=marker, skip_result=1) == 1
    assert H.done_event(cfg, "seed42") is not None


def test_colab_results_roundtrip_blocks_local_rerun(base_cfg, tmp_path, monkeypatch):
    # --- Colab trains seed 42 and packs the results
    on_kernel(monkeypatch, True)
    colab = make_cfg(base_cfg, tmp_path / "content" / "TA-BDA")
    run = H.models_root(colab) / "stress" / "runs" / "roberta-base" / "seed42"

    def train():
        (run / "best_model").mkdir(parents=True)
        (run / "best_model" / "model.safetensors").write_bytes(b"w" * 100)
        (run / "DONE.json").write_text("{}")
        return run
    step = H.train_step_id("roberta-base", 42)
    H.run_step(colab, step, "train", train, done_marker=run / "DONE.json")
    loso = H.models_root(colab) / "stress" / "runs" / "loso" / "distilroberta-base_ptsd"
    (loso / "best_model").mkdir(parents=True)
    (loso / "best_model" / "model.safetensors").write_bytes(b"x")
    (loso / "DONE.json").write_text("{}")
    z = H.pack_colab_results(colab)
    assert z == tmp_path / "content" / "colab_results.zip"
    names = set(zipfile.ZipFile(z).namelist())
    assert "trained_models/stress/runs/roberta-base/seed42/best_model/model.safetensors" in names
    assert "trained_models/handoff.json" in names
    assert not any("/loso/" in n and "/best_model/" in n for n in names)        # LOSO weights left out by default
    assert "trained_models/stress/runs/loso/distilroberta-base_ptsd/DONE.json" in names

    # --- PC: a file that already exists locally must survive the import untouched
    on_kernel(monkeypatch, False)
    pc = make_cfg(base_cfg, tmp_path / "pc" / "TA-BDA")
    mine = H.models_root(pc) / "stress" / "runs" / "roberta-base" / "seed42" / "DONE.json"
    mine.parent.mkdir(parents=True)
    mine.write_text('{"local": true}')
    H.record(pc, "L0.1", "ran", "bootstrap")
    res = H.import_colab_results(pc, z)
    assert res["written"] >= 2 and res["kept_existing"] == 1 and res["events_merged"] >= 3
    assert json.loads(mine.read_text()) == {"local": True}
    assert (mine.parent / "best_model" / "model.safetensors").read_bytes() == b"w" * 100
    assert H.import_colab_results(pc, z)["written"] == 0                         # same zip twice: no-op

    # --- PC: the guard refuses to train seed 42 again and says it was done on Colab
    calls = []
    H.run_step(pc, step, "train", lambda: calls.append(1), done_marker=mine)
    assert calls == [] and H.done_event(pc, step)["kernel"] == "colab"
    assert H.last_event(pc, step)["status"] == "skipped" and H.last_event(pc, step)["kernel"] == "local"


def test_local_log_in_bundle_blocks_colab_rerun(base_cfg, tmp_path, monkeypatch):
    on_kernel(monkeypatch, False)
    pc = make_cfg(base_cfg, tmp_path / "pc" / "TA-BDA")
    H.run_step(pc, H.BASELINE_STEP, "baseline", lambda: None)
    # what colab_bundle.zip carries, unpacked on the server
    on_kernel(monkeypatch, True)
    colab = make_cfg(base_cfg, tmp_path / "content" / "TA-BDA")
    incoming = Path(colab["_root"]) / BUNDLE_HANDOFF_ARCNAME
    incoming.parent.mkdir(parents=True)
    incoming.write_bytes(H.log_path(pc).read_bytes())
    H.run_step(colab, H.BASELINE_STEP, "baseline", lambda: pytest.fail("Colab must not redo local work"))
    assert H.done_event(colab, H.BASELINE_STEP)["kernel"] == "local"


def test_import_ignores_files_outside_trained_models(base_cfg, tmp_path, monkeypatch):
    on_kernel(monkeypatch, False)
    pc = make_cfg(base_cfg, tmp_path / "pc" / "TA-BDA")
    z = tmp_path / "colab_results.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("code/stress_signals/utils.py", "evil")
        zf.writestr("trained_models/../../escape.txt", "evil")
        zf.writestr("trained_models/ok.txt", "ok")
    assert H.import_colab_results(pc, z)["written"] == 1
    assert (H.models_root(pc) / "ok.txt").read_text() == "ok"
    assert not (Path(pc["_root"]) / "code").exists() and not (tmp_path / "pc" / "escape.txt").exists()


def test_bundle_carries_local_handoff_log(base_cfg, tmp_path, monkeypatch):
    from stress_signals.utils import COLAB_BUNDLE_DATA, build_colab_bundle
    if not all((Path(base_cfg["_root"]) / r).is_file() for r in COLAB_BUNDLE_DATA):
        pytest.skip("processed Dreaddit not built")
    on_kernel(monkeypatch, False)
    models = tmp_path / "trained_models"
    c = {**base_cfg, "paths": {**base_cfg["paths"], "models": str(models)},
         "_paths": {**base_cfg["_paths"], "artifacts": tmp_path}}
    H.record(c, "3A.1/x", "done", "demo")
    names = set(zipfile.ZipFile(build_colab_bundle(c)).namelist())
    assert BUNDLE_HANDOFF_ARCNAME in names and not any(n.startswith("trained_models/") for n in names)
