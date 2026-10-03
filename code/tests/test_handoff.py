"""Handoff log + run-once guard. No training: steps are tiny lambdas; writes only under tmp_path."""

import json
from pathlib import Path

import pytest

import stress_signals.handoff as H
from stress_signals.config import load_config


@pytest.fixture(scope="module")
def base_cfg():
    return load_config()


def make_cfg(base, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    return {**base, "_root": root, "_paths": {**base["_paths"], "artifacts": root / "data" / "outputs"}}


def test_step_runs_once_and_is_logged(base_cfg, tmp_path, capsys):
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    calls = []
    assert H.run_step(cfg, "3A.1/x", "demo", lambda: calls.append(1) or "ran") == "ran"
    assert H.run_step(cfg, "3A.1/x", "demo", lambda: calls.append(1) or "ran", skip_result="skipped") == "skipped"
    assert calls == [1]
    assert "SKIP 3A.1/x: already done on" in capsys.readouterr().out
    assert [e["status"] for e in H.load(cfg)["events"]] == ["started", "done"]     # skips are not logged (no clutter)
    ev = H.done_event(cfg, "3A.1/x")
    assert ev["utc"].endswith("+00:00") and ev["duration_s"] is not None and ev["host"] and ev["device"]
    md = (H.models_root(cfg) / "HANDOFF.md").read_text(encoding="utf-8")
    assert "`3A.1/x`" in md and "**done**" in md and "machine (device)" in md
    assert md.index("## How this works") < md.index("## Current state per step")
    assert H.status_table(cfg).set_index("step").loc["3A.1/x", "state"] == "done"


def test_skip_loads_saved_result_and_force_reruns(base_cfg, tmp_path, capsys):
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    out = tmp_path / "evaluation.json"

    def compute():
        out.write_text(json.dumps({"f1": 0.8}))
        return {"f1": 0.8}
    load = lambda: json.loads(out.read_text())                                   # noqa: E731
    assert H.run_step(cfg, "3C.1", "eval", compute, done_marker=out, skip_result=load) == {"f1": 0.8}
    assert H.run_step(cfg, "3C.1", "eval", lambda: pytest.fail("must load, not recompute"),
                      done_marker=out, skip_result=load) == {"f1": 0.8}
    calls = []
    H.run_step(cfg, "3C.1", "eval", lambda: calls.append(1), done_marker=out, force=True)
    assert calls == [1] and "RE-RUN 3C.1" in capsys.readouterr().out
    assert H.last_event(cfg, "3C.1", ["started"])["note"] == "forced re-run"


def test_deleted_output_means_not_done(base_cfg, tmp_path, capsys):
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    marker = tmp_path / "DONE.json"
    H.run_step(cfg, "seed42", "t", lambda: marker.write_text("{}"), done_marker=marker)
    marker.unlink()                                       # the user moved/deleted the run folder
    calls = []
    H.run_step(cfg, "seed42", "t", lambda: calls.append(1), done_marker=marker)
    assert calls == [1] and "is gone, so it runs again" in capsys.readouterr().out


def test_not_allowed_step_never_runs(base_cfg, tmp_path, capsys):
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    assert H.run_step(cfg, "loso", "t", lambda: pytest.fail("must not run"), allow=False,
                      blocked_reason="optional", skip_result="no") == "no"
    assert "NOT RUN loso: optional" in capsys.readouterr().out
    assert H.load(cfg)["events"] == []
    assert not H.ready(False, "3C.1", "needs seeds") and "NOT RUN 3C.1: needs seeds" in capsys.readouterr().out


def test_failed_step_is_logged_and_can_rerun(base_cfg, tmp_path):
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")

    def boom():
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError):
        H.run_step(cfg, "s", "t", boom)
    assert H.last_event(cfg, "s")["status"] == "failed" and "out of memory" in H.last_event(cfg, "s")["note"]
    assert H.run_step(cfg, "s", "t", lambda: 7) == 7


def test_existing_marker_counts_as_done(base_cfg, tmp_path):
    cfg = make_cfg(base_cfg, tmp_path / "TA-BDA")
    marker = H.models_root(cfg) / "stress" / "runs" / "m" / "seed42" / "DONE.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert H.run_step(cfg, "seed42", "t", lambda: pytest.fail("must not run"), done_marker=marker, skip_result=1) == 1
    assert H.done_event(cfg, "seed42") is not None


def test_notebook_training_and_heavy_steps_are_run_once(base_cfg):
    """Single local notebook: no Colab, setup proves the NVIDIA GPU, and every heavy step goes through run_step."""
    nb = json.loads((Path(base_cfg["_root"]) / "code" / "notebook.ipynb").read_text(encoding="utf-8"))
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "colab" not in text.lower() and not (Path(base_cfg["_root"]) / "code" / "notebook_local.ipynb").exists()
    setup = next(s for s in srcs if "Step 0.0" in s)
    assert "nvidia-smi" in setup and "(x @ x)" in setup
    for step in ("1.2", "1.5", "2.2", "2.4", "2.5", "3A.1", "3A.2", "3B.1", "3C.1", "3D.1", "3D.2", "3E.1", "3E.2", "3F"):
        cell = next(s for s in srcs if f"# Step {step}" in s)
        assert "H.run_step(" in cell, step


def test_setup_cell_picks_the_right_torch_build(base_cfg):
    """Cell 0.0's torch choice per Windows laptop, taken from the notebook itself (nothing is installed)."""
    nb = json.loads((Path(base_cfg["_root"]) / "code" / "notebook.ipynb").read_text(encoding="utf-8"))
    setup = next("".join(c["source"]) for c in nb["cells"] if "# Step 0.0" in "".join(c["source"]))
    ns = {"PT": "https://download.pytorch.org/whl/"}
    exec(setup[setup.index("def _torch_plan"):setup.index("_PROBE =")], ns)
    plan = ns["_torch_plan"]
    gpu = lambda name, cap, drv: {"name": name, "cap": cap, "vram_gb": 4.0, "driver": drv}   # noqa: E731
    assert plan(None)[:3] == ("2.5.1", ns["PT"] + "cpu", False)                             # no NVIDIA GPU
    assert plan(gpu("Quadro P1000", 6.1, "555.99"))[:3] == ("2.5.1", ns["PT"] + "cu121", True)
    assert plan(gpu("RTX 4060 Laptop", 8.9, "560.94"))[:3] == ("2.5.1", ns["PT"] + "cu121", True)
    assert plan(gpu("RTX 3050", 8.6, "472.12"))[:3] == ("2.5.1", ns["PT"] + "cu118", True)     # old driver
    assert plan(gpu("RTX 5070 Laptop", 12.0, "576.02"))[:3] == ("2.7.1", ns["PT"] + "cu128", True)
    assert plan(gpu("GT 730", 3.5, "475.14"))[:3] == ("2.5.1", ns["PT"] + "cpu", False)        # too old for torch
