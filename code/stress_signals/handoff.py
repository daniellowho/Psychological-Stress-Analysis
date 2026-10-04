"""Handoff log + run-once guard: a timestamped record of every step the notebook ran (on which machine and device),
and a guard so that "Run All" never repeats finished work.

Files (both under <project root>/trained_models/, i.e. paths.models):
    handoff.json   machine-readable event log
    HANDOFF.md     the same log for people: one row per step (latest state) + the full event history, with timestamps

Usage in a notebook cell:
    run_step(CFG, "3B.1/roberta-base/seed42", "Fine-tune roberta-base, seed 42", lambda: SM.train_stress_seed(...),
             done_marker=<run dir>/"DONE.json", skip_result=<run dir>)
A step is DONE when the log has a `done` event for it AND its `done_marker` (if given) still exists. A done step is
skipped and `skip_result` is returned (a callable is called, e.g. to load the saved result). `force=True` runs it again.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .utils import device_label, get_logger, models_root, sha256_file, write_json

LOG = get_logger("stress_signals.handoff")
LOG_NAME, MD_NAME = "handoff.json", "HANDOFF.md"
DATA_FILES = ("data/processed/dreaddit.parquet", "data/processed/splits/dreaddit_split_v1.json")
_FINGERPRINT: dict[str, str] = {}

# Step ids of the training units (the same id always means the same work).
BASELINE_STEP = "3A.1/baseline_tfidf_lr"
BUNDLE_STEP = "3F/stress_bundle"


def train_step_id(model_choice: str, seed: int) -> str:
    return f"3B.1/{model_choice}/seed{seed}"


def loso_step_id(model_choice: str, subreddit: str) -> str:
    return f"3E.2/loso/{model_choice}_{subreddit}"


EMOTION_BUNDLE_STEP = "4G/emotion_bundle"


def emotion_train_step_id(model_choice: str, seed: int) -> str:
    return f"4B.1/emotion/{model_choice}/seed{seed}"


STRESSOR_LR_STEP = "5C.3/stressor_embed_lr"
STRESSOR_BUNDLE_STEP = "5D/stressor_bundle"


def stressor_train_step_id(model_choice: str, seed: int) -> str:
    return f"5C.4/stressor/{model_choice}/seed{seed}"


# ================================================================ log file
def log_path(cfg: dict[str, Any]) -> Path:
    return models_root(cfg) / LOG_NAME


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def load(cfg: dict[str, Any]) -> dict[str, Any]:
    p = log_path(cfg)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "events": []}


def _save(cfg: dict[str, Any], log: dict[str, Any]) -> None:
    log["events"].sort(key=lambda e: e["utc"])          # stable: same-second events keep their logging order
    write_json(log_path(cfg), log)
    write_markdown(cfg)


def data_fingerprint(cfg: dict[str, Any]) -> str | None:
    """sha256 over the Dreaddit parquet + split file the models train on (shows which data a step used)."""
    key = str(cfg["_root"])
    if key not in _FINGERPRINT:
        paths = [Path(cfg["_root"]) / f for f in DATA_FILES]
        if not all(p.is_file() for p in paths):
            return None
        _FINGERPRINT[key] = sha256_file(paths[0])[:16] + "-" + sha256_file(paths[1])[:16]
    return _FINGERPRINT[key]


def record(cfg: dict[str, Any], step: str, status: str, title: str = "", note: str = "",
           outputs: Iterable[str | Path] = (), duration_s: float | None = None) -> dict[str, Any]:
    """Append one event. status: started | done | failed."""
    root = Path(cfg["_root"])
    outs = []
    for o in outputs:
        o = Path(o)
        outs.append(o.relative_to(root).as_posix() if o.is_absolute() and o.is_relative_to(root) else o.as_posix())
    ev = {"id": uuid.uuid4().hex, "step": step, "title": title, "status": status, "host": platform.node(),
          "device": device_label(), "utc": _now_utc(), "duration_s": None if duration_s is None else round(duration_s, 1),
          "note": note, "outputs": outs, "data_fingerprint": data_fingerprint(cfg)}
    log = load(cfg)
    log["events"].append(ev)
    _save(cfg, log)
    return ev


# ================================================================ queries
def last_event(cfg: dict[str, Any], step: str, statuses: Iterable[str] | None = None) -> dict[str, Any] | None:
    want = set(statuses) if statuses else None
    evs = [e for e in load(cfg)["events"] if e["step"] == step and (want is None or e["status"] in want)]
    return evs[-1] if evs else None


def done_event(cfg: dict[str, Any], step: str) -> dict[str, Any] | None:
    """The latest `done` event for this step (None if it never finished)."""
    return last_event(cfg, step, ["done"])


def _when(ev: dict[str, Any]) -> str:
    t = _dt.datetime.fromisoformat(ev["utc"])
    return f"{t.astimezone().strftime('%Y-%m-%d %H:%M:%S')} local ({ev['utc']} UTC)"


def _where(ev: dict[str, Any]) -> str:
    return f"{ev.get('host', '?')} ({ev.get('device', '?')})"


# ================================================================ the guard
def run_step(cfg: dict[str, Any], step: str, title: str, fn: Callable[[], Any], *,
             done_marker: str | Path | None = None, skip_result: Any = None,
             allow: bool = True, blocked_reason: str = "", force: bool = False) -> Any:
    """Run `fn` ONCE. If the step is already done (log + `done_marker`, see module docstring) it is skipped and
    `skip_result` is returned (called first if it is callable, e.g. a loader of the saved result). `force=True` runs it
    again. A step that is not done runs only when `allow` is True (else `blocked_reason` is printed).
    A run is logged as started -> done (with duration) or failed."""
    marker = Path(done_marker) if done_marker else None
    done = done_event(cfg, step)
    if not force:
        if done is not None and (marker is None or marker.exists()):
            print(f"SKIP {step}: already done on {_where(done)} at {_when(done)}. Not running it again.")
            fp = data_fingerprint(cfg)
            if done.get("data_fingerprint") and fp and done["data_fingerprint"] != fp:
                print(f"   WARNING: it was done on different data (fingerprint {done['data_fingerprint']} vs {fp} now).")
            return skip_result() if callable(skip_result) else skip_result
        if done is not None:
            print(f"NOTE {step}: logged as done at {_when(done)}, but {marker} is gone, so it runs again.")
        elif marker is not None and marker.exists():
            print(f"SKIP {step}: its output already exists ({marker.name} in {marker.parent}). Logged as done.")
            record(cfg, step, "done", title, note="output found on disk (finished before it was logged)", outputs=[marker.parent])
            return skip_result() if callable(skip_result) else skip_result
    if not allow:
        print(f"NOT RUN {step}: {blocked_reason or 'not switched on'}")
        return skip_result() if callable(skip_result) else skip_result
    if force and done is not None:
        print(f"RE-RUN {step} (FORCE_RERUN): last done at {_when(done)}")
    record(cfg, step, "started", title, note="forced re-run" if force and done is not None else "")
    t0 = time.time()
    try:
        result = fn()
    except BaseException as exc:   # also KeyboardInterrupt: the log must show the step did not finish
        record(cfg, step, "failed", title, note=f"{type(exc).__name__}: {exc}"[:500], duration_s=time.time() - t0)
        raise
    outs = [result] if isinstance(result, Path) else [marker.parent] if marker is not None else []
    record(cfg, step, "done", title, outputs=outs, duration_s=time.time() - t0)
    print(f"DONE {step} on {device_label()} in {time.time() - t0:.0f}s")
    return result


def ready(ok: bool, step: str, why: str) -> bool:
    """For cells that need earlier results: False (with a printed reason) instead of an error, so "Run All" carries on."""
    if not ok:
        print(f"NOT RUN {step}: {why}")
    return bool(ok)


# ================================================================ status / markdown
def _latest_per_step(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest event per step, in first-seen order (the full history is listed separately)."""
    latest: dict[str, dict[str, Any]] = {}
    for e in events:
        latest[e["step"]] = e
    return latest


def status_table(cfg: dict[str, Any]):
    """One row per step: its current state, where and when."""
    import pandas as pd
    rows = [{"step": e["step"], "title": e["title"], "state": e["status"], "machine": _where(e), "when": _when(e),
             "duration_s": e["duration_s"], "note": e["note"]} for e in _latest_per_step(load(cfg)["events"]).values()]
    return pd.DataFrame(rows, columns=["step", "title", "state", "machine", "when", "duration_s", "note"])


HOW_IT_WORKS = [
    "## How this works (read first)", "",
    "**Run All does everything that is not done yet, and nothing twice.**", "",
    "- Every step that computes or writes something runs **once**. On later Run Alls it prints "
    "`SKIP <step>: already done on <machine> at <time>` and loads its saved result instead. Cells that only load or show "
    "things re-run (seconds).",
    "- **Training** (baseline 3A.1, seeds 3B.1) starts automatically for whatever is not done, on the laptop's NVIDIA GPU "
    "(CUDA) if it has one, else on the CPU. An interrupted seed resumes from its last epoch on the next Run All.",
    "- **Emotion model (Step 4)**: its seeds (4B.1) train the same way. Its evaluation (4C-4G) waits until every "
    "emotion seed is trained; the Dreaddit domain-shift check (4F) is inference only.",
    "- **Stressor model (Step 5)**: the SAD fits (5C.3 logistic regression, 5C.4 fine-tune seeds) train the same way. "
    "Everything that uses the Reddit gold set (5C.6 onward) waits until annotator 1's sheet is complete.",
    "- **Optional LOSO** (3E.2, ~10 extra trainings) runs only with `RUN_LOSO = True` in that cell.",
    "- **Evaluation** (3C–3F) waits until every seed is trained (`NOT RUN … missing [...]` until then), because the final "
    "test report is written only once and must not pick a final seed from a partial set.",
    "- **The test suite** (2.5) re-runs only after the code changed. **Re-run anything on purpose:** add its step number "
    "to `FORCE_RERUN` in cell 0.1, e.g. `FORCE_RERUN = {\"3C.1\"}`.",
    "- A step whose saved output was deleted counts as not done and runs again.", "",
]


def write_markdown(cfg: dict[str, Any]) -> Path:
    """(Re)write HANDOFF.md from handoff.json, also when nothing is logged yet (so the how-it-works notes are there)."""
    out = models_root(cfg) / MD_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(MD_NAME + ".tmp")
    tmp.write_text(render_markdown(load(cfg)), encoding="utf-8")
    os.replace(tmp, out)
    return out


def _cell(text: str | None) -> str:
    return (text or "").replace("|", "/").replace("\n", " ")


def render_markdown(log: dict[str, Any]) -> str:
    evs = log["events"]
    lines = ["# Handoff log: what has been done, where and when", "",
             f"_Regenerated {_now_utc()} UTC on {platform.node()}. Times are UTC. Source of truth: `handoff.json`._", "",
             *HOW_IT_WORKS,
             "## Current state per step", "",
             "| step | what | state | machine (device) | when (UTC) | duration | note |", "|---|---|---|---|---|---|---|"]
    for e in _latest_per_step(evs).values():
        dur = f"{e['duration_s']:.0f}s" if e.get("duration_s") is not None else ""
        lines.append(f"| `{e['step']}` | {e['title']} | **{e['status']}** | {_where(e)} | {e['utc']} | {dur} | {_cell(e['note'])} |")
    lines += ["", "## Full history (oldest first)", "", "| when (UTC) | machine (device) | step | event | note |",
              "|---|---|---|---|---|"]
    for e in evs:
        lines.append(f"| {e['utc']} | {_where(e)} | `{e['step']}` | {e['status']} | {_cell(e['note'])} |")
    return "\n".join(lines) + "\n"


def show(cfg: dict[str, Any]) -> None:
    tab = status_table(cfg)
    print(f"handoff log: {write_markdown(cfg)}  ({len(load(cfg)['events'])} events; how-it-works notes at the top)")
    if tab.empty:
        print("  (nothing recorded yet)")
        return
    try:
        from IPython.display import display
        display(tab)
    except ImportError:
        print(tab.to_string(index=False))
