"""Handoff between the Colab kernel and the local kernel: one timestamped log of what ran where, a guard that stops a
step from running twice (on either kernel), and the colab_results.zip transfer that replaces Google Drive.

Files (both under <project root>/trained_models/, i.e. paths.models):
    handoff.json   machine-readable event log (every event has a unique id, so merging logs is idempotent)
    HANDOFF.md     the same log for people: one row per step (latest state) + the full event history, with timestamps

How the two kernels learn about each other (no shared disk, no Drive):
    local -> Colab: build_colab_bundle() puts the local handoff.json into colab_bundle.zip as _handoff_import/handoff.json;
                    sync() merges it on the Colab server, so Colab skips steps the PC already finished.
    Colab -> local: pack_colab_results() zips trained_models/ (log included) into /content/colab_results.zip; after you
                    download it, import_colab_results() copies the files in (never overwriting) and merges the log, so the
                    local notebook skips steps Colab already finished.

Usage in a notebook cell:
    run_step(CFG, "3B.1/roberta-base/seed42", "Fine-tune roberta-base, seed 42", lambda: SM.train_stress_seed(...),
             done_marker=<run dir>/"DONE.json", skip_result=<run dir>)
A step counts as done when the log has a `done` event for it from ANY kernel, or when `done_marker` exists on disk.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .utils import BUNDLE_HANDOFF_ARCNAME, get_logger, is_colab, models_root, sha256_file, write_json

LOG = get_logger("stress_signals.handoff")
LOG_NAME, MD_NAME = "handoff.json", "HANDOFF.md"
DATA_FILES = ("data/processed/dreaddit.parquet", "data/processed/splits/dreaddit_split_v1.json")
STORED_SUFFIXES = (".safetensors", ".bin", ".pt", ".joblib", ".zip")   # already dense: zip them without compression
_FINGERPRINT: dict[str, str] = {}

# Step ids shared by notebook.ipynb (Colab) and notebook_local.ipynb: the SAME id means the SAME work on either kernel.
BASELINE_STEP = "3A.1/baseline_tfidf_lr"
BUNDLE_STEP = "3F/stress_bundle"

# Why a training cell did not train (printed by run_step when allow=False).
COLAB_ONLY = ("notebook.ipynb trains on the Colab kernel only. To train on this PC use notebook_local.ipynb (switch it on in "
              "TRAIN_HERE, cell L0.5). If Colab already trained it, import colab_results.zip (cell 3.0d).")
NOT_IN_TRAIN_HERE = ("not done on either kernel as far as this PC knows, and not switched on in TRAIN_HERE (cell L0.5). "
                     "If Colab trained it, import colab_results.zip (cell L0.3); to train it here, add it to TRAIN_HERE.")


def train_step_id(model_choice: str, seed: int) -> str:
    return f"3B.1/{model_choice}/seed{seed}"


def loso_step_id(model_choice: str, subreddit: str) -> str:
    return f"3E.2/loso/{model_choice}_{subreddit}"


# ================================================================ log file
def kernel_name() -> str:
    return "colab" if is_colab() else "local"


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
    """sha256 over the Dreaddit parquet + split file both kernels train on (proves they used the same data)."""
    key = str(cfg["_root"])
    if key not in _FINGERPRINT:
        paths = [Path(cfg["_root"]) / f for f in DATA_FILES]
        if not all(p.is_file() for p in paths):
            return None
        _FINGERPRINT[key] = sha256_file(paths[0])[:16] + "-" + sha256_file(paths[1])[:16]
    return _FINGERPRINT[key]


def record(cfg: dict[str, Any], step: str, status: str, title: str = "", note: str = "",
           outputs: Iterable[str | Path] = (), duration_s: float | None = None) -> dict[str, Any]:
    """Append one event. status: started | done | failed | skipped | ran | imported | packed."""
    root = Path(cfg["_root"])
    outs = []
    for o in outputs:
        o = Path(o)
        outs.append(o.relative_to(root).as_posix() if o.is_absolute() and o.is_relative_to(root) else o.as_posix())
    ev = {"id": uuid.uuid4().hex, "step": step, "title": title, "status": status, "kernel": kernel_name(),
          "host": platform.node(), "utc": _now_utc(), "duration_s": None if duration_s is None else round(duration_s, 1),
          "note": note, "outputs": outs, "data_fingerprint": data_fingerprint(cfg)}
    log = load(cfg)
    log["events"].append(ev)
    _save(cfg, log)
    return ev


def merge_events(cfg: dict[str, Any], events: Iterable[dict[str, Any]]) -> int:
    """Add events not yet in the log (matched by id). Returns how many were new."""
    log = load(cfg)
    have = {e["id"] for e in log["events"]}
    new = [e for e in events if e.get("id") not in have]
    if new:
        log["events"].extend(new)
        _save(cfg, log)
    return len(new)


def sync(cfg: dict[str, Any]) -> int:
    """Colab: merge the local log that arrived inside colab_bundle.zip. Safe to call repeatedly (and locally)."""
    incoming = Path(cfg["_root"]) / BUNDLE_HANDOFF_ARCNAME
    if not incoming.is_file():
        return 0
    n = merge_events(cfg, json.loads(incoming.read_text(encoding="utf-8")).get("events", []))
    if n:
        LOG.info("merged %d handoff events from the local PC (colab_bundle.zip)", n)
    return n


# ================================================================ queries
def last_event(cfg: dict[str, Any], step: str, statuses: Iterable[str] | None = None) -> dict[str, Any] | None:
    want = set(statuses) if statuses else None
    evs = [e for e in load(cfg)["events"] if e["step"] == step and (want is None or e["status"] in want)]
    return evs[-1] if evs else None


def done_event(cfg: dict[str, Any], step: str) -> dict[str, Any] | None:
    """The first `done` event for this step from ANY kernel (None if it never finished)."""
    evs = [e for e in load(cfg)["events"] if e["step"] == step and e["status"] == "done"]
    return evs[0] if evs else None


def _when(ev: dict[str, Any]) -> str:
    t = _dt.datetime.fromisoformat(ev["utc"])
    return f"{t.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} ({ev['utc']} UTC)"


# ================================================================ the guard
def run_step(cfg: dict[str, Any], step: str, title: str, fn: Callable[[], Any], *,
             done_marker: str | Path | None = None, skip_result: Any = None,
             allow: bool = True, blocked_reason: str = "") -> Any:
    """Run `fn` once across BOTH kernels. Skips (and returns `skip_result`) when the log already has a `done` event for
    `step` from either kernel, or `done_marker` exists. If it is not done yet, it runs ONLY when `allow` is True
    (notebooks pass an explicit opt-in, so "Run All" never starts training by accident); otherwise it prints
    `blocked_reason` and returns `skip_result`. A run is logged as started -> done (with duration) or failed."""
    sync(cfg)
    here = kernel_name()
    marker = Path(done_marker) if done_marker else None
    done = done_event(cfg, step)
    if done is not None:
        where = "HERE" if done["kernel"] == here else f"on {done['kernel'].upper()}"
        print(f"SKIP {step}: already done {where} at {_when(done)}. Not running it again.")
        if marker is not None and not marker.exists():
            if done["kernel"] == "colab" and here == "local":
                print("   Its files are not on this PC yet: download colab_results.zip and run the import cell "
                      "(notebook 3.0d / notebook_local L0.3).")
            elif done["kernel"] == "local" and here == "colab":
                print("   Its files stay on the PC; nothing to do on Colab.")
            else:
                print(f"   WARNING: the log says done but {marker} is missing.")
        fp = data_fingerprint(cfg)
        if done.get("data_fingerprint") and fp and done["data_fingerprint"] != fp:
            print(f"   WARNING: it was done on different data (fingerprint {done['data_fingerprint']} vs {fp} here).")
        record(cfg, step, "skipped", title, note=f"already done on {done['kernel']} at {done['utc']}")
        return skip_result
    if marker is not None and marker.exists():
        print(f"SKIP {step}: its output already exists ({marker.name} in {marker.parent}). Logged as done.")
        record(cfg, step, "done", title, note="output found on disk (finished before it was logged)", outputs=[marker.parent])
        return skip_result
    if not allow:
        print(f"NOT RUN {step}: {blocked_reason or 'not switched on for this kernel'}")
        return skip_result
    started = last_event(cfg, step, ["started"])
    if started is not None and started["kernel"] != here:
        print(f"NOTE {step}: {started['kernel']} STARTED this at {_when(started)} but never logged it as done "
              f"(crash/disconnect?). Running it here on {here}.")
    record(cfg, step, "started", title)
    t0 = time.time()
    try:
        result = fn()
    except BaseException as exc:   # also KeyboardInterrupt: the log must show the step did not finish
        record(cfg, step, "failed", title, note=f"{type(exc).__name__}: {exc}"[:500], duration_s=time.time() - t0)
        raise
    outs = [result] if isinstance(result, Path) else [marker.parent] if marker is not None else []
    record(cfg, step, "done", title, outputs=outs, duration_s=time.time() - t0)
    print(f"DONE {step} on {here} in {time.time() - t0:.0f}s")
    return result


def ready(ok: bool, step: str, why: str) -> bool:
    """For cells that need earlier results: False (with a printed reason) instead of an error, so "Run All" carries on."""
    if not ok:
        print(f"NOT RUN {step}: {why}")
    return bool(ok)


def log_ran(cfg: dict[str, Any], step: str, title: str, note: str = "") -> None:
    """For repeatable, read-only steps (evaluation, plots): logged for the record, never skipped."""
    record(cfg, step, "ran", title, note=note)


# ================================================================ status / markdown
def status_table(cfg: dict[str, Any]):
    """One row per step: its current state, where and when."""
    import pandas as pd
    rows: dict[str, dict[str, Any]] = {}
    for e in load(cfg)["events"]:
        r = rows.setdefault(e["step"], {"step": e["step"], "title": e["title"], "state": "", "kernel": "", "when": "",
                                        "duration_s": None, "note": ""})
        if e["status"] in ("skipped",):
            continue
        if r["state"] == "done" and e["status"] in ("ran", "started", "failed"):
            continue                                    # a later re-run attempt does not undo a finished step
        r.update(state=e["status"], kernel=e["kernel"], when=_when(e), duration_s=e["duration_s"], note=e["note"],
                 title=e["title"] or r["title"])
    return pd.DataFrame(list(rows.values()), columns=["step", "title", "state", "kernel", "when", "duration_s", "note"])


HOW_IT_WORKS = [
    "## How this works (read first)", "",
    "**Run All is safe. Training never starts by accident.**", "",
    "- **`notebook.ipynb`**: training cells (3A.1, 3B.1, 3E.2) train **only on the Colab kernel**. "
    "On the local kernel they print `NOT RUN …: notebook.ipynb trains on the Colab kernel only…`.",
    "- **`notebook_local.ipynb`**: nothing trains until you switch it on in cell **L0.5**:",
    "  `TRAIN_HERE = {\"baseline\": False, \"seeds\": [], \"loso\": False}` (e.g. `\"seeds\": [42]`). "
    "That cell also previews what Run All would do now (\"done on colab at … → skip\" / \"WILL TRAIN HERE\" / "
    "\"not switched on → skip\").",
    "- **Done always wins.** A step marked **done** below, on EITHER kernel, is skipped by both notebooks even if it is "
    "switched on. The cell says where and when it ran.",
    "- **Missing Colab results:** if Colab trained something that is not imported yet, the message says to import "
    "`colab_results.zip` instead of retraining.",
    "- **Evaluation cells** (3A.2, 3C.\\*, 3D.\\*, 3E.1, 3E.3, 3F) print e.g. `NOT RUN 3C.1: … missing [13, 2024]` instead "
    "of failing, so Run All carries on. They wait for **every** seed, because 3D.2 writes the final test report only once "
    "and must not pick a final seed from a partial set.", "",
    "**Moving results between kernels:** PC → Colab: rebuild `colab_bundle.zip` (cell 3.0c / L9), upload it, run 0.0 "
    "(it carries this log). Colab → PC: cell 3G writes `/content/colab_results.zip`; download it to the project root, "
    "`data/outputs/` or Downloads, and run 3.0d / L0.3. The import never overwrites or deletes files, and it merges the logs.", "",
]


def write_markdown(cfg: dict[str, Any]) -> Path:
    """(Re)write HANDOFF.md from handoff.json, also when nothing is logged yet (so the how-it-works notes are there)."""
    out = models_root(cfg) / MD_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(MD_NAME + ".tmp")
    tmp.write_text(render_markdown(load(cfg)), encoding="utf-8")
    os.replace(tmp, out)
    return out


def render_markdown(log: dict[str, Any]) -> str:
    evs = log["events"]
    latest: dict[str, dict[str, Any]] = {}
    for e in evs:
        if e["status"] == "skipped":
            continue
        cur = latest.get(e["step"])
        if cur is not None and cur["status"] == "done" and e["status"] in ("ran", "started", "failed"):
            continue
        latest[e["step"]] = e
    lines = ["# Handoff log: what has been done, where and when", "",
             f"_Regenerated {_now_utc()} UTC by the {kernel_name()} kernel. Times are UTC. Source of truth: `handoff.json`._", "",
             *HOW_IT_WORKS,
             "## Current state per step", "",
             "| step | what | state | kernel | when (UTC) | duration | note |", "|---|---|---|---|---|---|---|"]
    for e in latest.values():
        dur = f"{e['duration_s']:.0f}s" if e.get("duration_s") is not None else ""
        lines.append(f"| `{e['step']}` | {e['title']} | **{e['status']}** | {e['kernel']} | {e['utc']} | {dur} | "
                     f"{(e['note'] or '').replace('|', '/')} |")
    lines += ["", "## Full history (oldest first)", "", "| when (UTC) | kernel | step | event | note |", "|---|---|---|---|---|"]
    for e in evs:
        lines.append(f"| {e['utc']} | {e['kernel']} | `{e['step']}` | {e['status']} | {(e['note'] or '').replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def show(cfg: dict[str, Any]) -> None:
    sync(cfg)
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


# ================================================================ Colab -> PC transfer
def colab_results_path(cfg: dict[str, Any]) -> Path:
    """Colab: /content/colab_results.zip (next to where 'Upload to Colab' puts the bundle)."""
    return Path(cfg["_root"]).parent / cfg["storage"].get("colab_results_zip", "colab_results.zip")


def pack_colab_results(cfg: dict[str, Any], include_loso_weights: bool = False) -> Path:
    """COLAB: zip trained_models/ (runs, predictions, logs, handoff log) into /content/colab_results.zip for download.
    LOSO model weights are left out by default (evaluation only needs their predictions; ~330 MB each)."""
    src = models_root(cfg)
    if not src.is_dir():
        raise FileNotFoundError(f"{src} does not exist: nothing has been trained on this server yet")
    record(cfg, "3G/pack", "packed", "Pack trained_models into colab_results.zip")
    out = colab_results_path(cfg)
    tmp = out.with_suffix(".zip.tmp")
    root = Path(cfg["_root"])
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if not f.is_file() or f.suffix == ".tmp":
                continue
            rel = f.relative_to(root).as_posix()
            if not include_loso_weights and "/loso/" in rel and "/best_model/" in rel:
                continue
            z.write(f, rel, compress_type=zipfile.ZIP_STORED if f.suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED)
            n += 1
    os.replace(tmp, out)
    print(f"{out}: {n} files, {out.stat().st_size / 1e6:.1f} MB")
    return out


def find_colab_results(cfg: dict[str, Any]) -> Path | None:
    """LOCAL: newest colab_results*.zip in the project root, data/outputs/ or ~/Downloads."""
    name = Path(cfg["storage"].get("colab_results_zip", "colab_results.zip"))
    dirs = [Path(cfg["_root"]), Path(cfg["_paths"]["artifacts"]), Path.home() / "Downloads"]
    hits = [p for d in dirs if d.is_dir() for p in d.glob(f"{name.stem}*{name.suffix}") if p.is_file()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def import_colab_results(cfg: dict[str, Any], zip_path: str | Path | None = None) -> dict[str, Any]:
    """LOCAL: copy colab_results.zip into the project. Files that already exist are NEVER overwritten (and nothing is
    deleted); the Colab handoff log is MERGED into the local one. Importing the same zip twice is a no-op."""
    zp = Path(zip_path) if zip_path else find_colab_results(cfg)
    if zp is None or not zp.is_file():
        print("no colab_results.zip found (looked in the project root, data/outputs/ and Downloads). "
              "On Colab run cell 3G, download /content/colab_results.zip, put it in one of those folders, re-run.")
        return {"zip": None, "written": 0, "kept_existing": 0, "events_merged": 0}
    digest = sha256_file(zp)
    prev = [e for e in load(cfg)["events"] if e["status"] == "imported" and digest in (e.get("note") or "")]
    if prev:
        print(f"{zp.name} was already imported at {_when(prev[-1])}; nothing to do.")
        return {"zip": zp, "written": 0, "kept_existing": 0, "events_merged": 0}
    root = Path(cfg["_root"]).resolve()
    models = models_root(cfg).resolve()
    written = kept = 0
    events: list[dict[str, Any]] = []
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            dest = (root / info.filename).resolve()
            if not dest.is_relative_to(models):          # only trained_models/ may come from Colab
                LOG.warning("ignoring %s (outside trained_models/)", info.filename)
                continue
            if dest.name == LOG_NAME and dest.parent == models:
                events = json.loads(z.read(info).decode("utf-8")).get("events", [])
                continue
            if dest.name == MD_NAME and dest.parent == models:
                continue                                  # regenerated from the merged log
            if dest.exists():
                kept += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            with z.open(info) as src, tmp.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            os.replace(tmp, dest)
            written += 1
    merged = merge_events(cfg, events)
    record(cfg, "import/colab_results", "imported", "Import colab_results.zip",
           note=f"{zp.name} sha256={digest}: {written} files written, {kept} already present, {merged} log events merged")
    print(f"imported {zp}: {written} files written, {kept} already present (kept), {merged} Colab log events merged")
    return {"zip": zp, "written": written, "kept_existing": kept, "events_merged": merged}
