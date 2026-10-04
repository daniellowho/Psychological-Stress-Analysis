"""Step 9: PySpark distributed NLP inference over the Step 8 Parquet lake. INFERENCE ONLY: nothing here trains or updates weights.

Design (and why it differs from "read the lake with Spark"):
  * Spark is the SCHEDULER. The driver lists the calendar dates (UTC) that still need scoring and ships a tiny DataFrame of dates;
    `mapInPandas` hands each date to a task. The task reads that date's rows straight from the lake with pyarrow, scores them and
    writes the enriched Parquet itself. No row data goes through the JVM. Reason: Spark's own file reader/writer needs Hadoop's
    Windows binaries (winutils.exe / hadoop.dll), which this machine does not have; reading the lake failed with
    `NativeIO$Windows.access0`. pyarrow in the task has no such need, and on a cluster the same code works with a shared path.
  * One StressSignalPipeline per worker process, loaded lazily (inference.get_pipeline cache), never at import time, never on the driver.
    `inference_mode`:
      in_process  the Spark Python worker imports torch and runs the models itself (the normal design; Linux / cluster).
      subprocess  the Spark worker stays free of torch and starts ONE long-lived inference process that it reuses for every date
                  (default on Windows: on this machine a JVM-started worker crashed on `import torch`, while a process started
                  from that worker imported it fine. Cause not found.) The child loads the pipeline once, so the cost is the same.
  * Two-stage inference: the stress model scores every record; the emotion and stressor models run only on stress-flagged records plus a
    seeded random sample of the rest (sample decided by hash(seed, record_id), so it is independent of batching and partitioning).
    `sample_weight` = 1 (flagged), 1/sample_rate (sampled unflagged), null (not sampled).
  * Restartable and idempotent: output is one file per source/date; a `_done/date=...json` marker is written last. A date whose marker
    matches the current bundle and settings is skipped; anything else is recomputed and its file replaced atomically.
  * Privacy: the enriched output has NO text column. With `spark.keep_text_for_topics: true` the cleaned text goes to a SEPARATE
    restricted folder that must never be loaded to Postgres or the dashboard.

Compatibility: PySpark 3.5.x runs on Java 8, 11 or 17 (Spark 3.5 docs; Java 8 is deprecated there) and Python 3.8+. Check with
`python -m stress_signals.spark_jobs env` (prints the pyspark and Java versions, Java bitness, torch/CUDA, memory).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from .utils import get_logger, sha256_file, write_json

LOG = get_logger("stress_signals.spark_jobs")
RESULT_TAG = "@@RESULT@@ "
SUMMARY_COLS = ["date", "status", "rows_in", "rows_scored", "rows_second_stage", "seconds", "error"]
SUMMARY_SCHEMA = "date string, status string, rows_in long, rows_scored long, rows_second_stage long, seconds double, error string"


# ================================================================ settings, paths
def spark_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["spark"]


def enriched_root(cfg: dict[str, Any], corpus: str) -> Path:
    return Path(cfg["_root"]) / spark_settings(cfg)["enriched_dir"] / corpus


def restricted_root(cfg: dict[str, Any], corpus: str) -> Path:
    return Path(cfg["_root"]) / spark_settings(cfg)["restricted_dir"] / corpus


def resolve_mode(cfg: dict[str, Any]) -> str:
    m = spark_settings(cfg).get("inference_mode", "auto")
    return ("subprocess" if sys.platform == "win32" else "in_process") if m == "auto" else m


def resolve_workers(cfg: dict[str, Any], device: str | None = None) -> int:
    w = spark_settings(cfg).get("workers", "auto")
    if w != "auto":
        return max(1, int(w))
    if device is None:
        from .utils import select_device
        device = select_device(cfg)
    return 1 if device == "cuda" else 2


def _jsonable(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(cfg, default=str))


def make_payload(cfg: dict[str, Any], corpus: str, *, lake_path: str | Path, out_root: str | Path | None = None,
                 artifacts_dir: str | Path | None = None, bundle_manifest: str | Path | None = None, device: str | None = None) -> dict[str, Any]:
    """Everything a worker needs, as plain JSON (shipped by file, so workers never load config.yaml or touch global state)."""
    from .inference import versions_from_bundle_manifest
    from .utils import models_root
    sp = spark_settings(cfg)
    mpath = Path(bundle_manifest) if bundle_manifest else Path(cfg["_root"]) / cfg["inference"]["bundle_manifest"]
    versions = versions_from_bundle_manifest(mpath) if mpath.is_file() else None
    ts = sp["two_stage"]
    ab = lambda x: str(Path(x).resolve())          # workers and the inference child run in another working directory: every path absolute
    p = {"cfg": _jsonable(cfg), "corpus": corpus, "lake_path": ab(lake_path), "out_root": ab(out_root or enriched_root(cfg, corpus)),
         "restricted_root": ab(restricted_root(cfg, corpus)), "keep_text": bool(sp.get("keep_text_for_topics", False)),
         "artifacts_dir": ab(artifacts_dir or models_root(cfg)), "versions": versions, "device": device or cfg["inference"].get("device", "auto"),
         "batch_size": int(cfg["inference"]["batch_size"]), "rows_per_call": int(sp.get("rows_per_call", 512)),
         "two_stage": bool(ts.get("enabled", True)), "sample_rate": float(ts.get("sample_rate", 0.15)) if ts.get("enabled", True) else 1.0,
         "seed": int(ts.get("seed", cfg["primary_seed"])), "num_threads": int(cfg["inference"].get("num_threads_cpu", 4))}
    p["settings_hash"] = hashlib.sha256(json.dumps({k: p[k] for k in ("two_stage", "sample_rate", "seed", "keep_text", "versions")},
                                                   sort_keys=True).encode()).hexdigest()[:12]
    return p


# ================================================================ environment check and Spark session
def _java_info() -> dict[str, Any]:
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=20)
        txt = (r.stderr or r.stdout).strip()
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "error": repr(exc)}
    import re
    m = re.search(r'version "(\d+)(?:\.(\d+))?', txt)
    major = (int(m.group(2)) if m and m.group(1) == "1" else int(m.group(1))) if m else None
    return {"found": True, "major": major, "text": txt.splitlines()[0] if txt else "", "bits": 64 if "64-Bit" in txt else 32,
            "java_home": os.environ.get("JAVA_HOME")}


def check_environment() -> dict[str, Any]:
    """pyspark / Java / Python / torch / memory facts + a compatibility verdict. Nothing is started except `java -version`."""
    import pyspark
    java = _java_info()
    out: dict[str, Any] = {"pyspark": pyspark.__version__, "python": platform.python_version(), "java": java, "platform": platform.platform(),
                           "cpu": platform.processor(), "cpu_count": os.cpu_count(), "problems": [], "notes": []}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_gb_total"], out["ram_gb_free"] = round(vm.total / 1e9, 1), round(vm.available / 1e9, 1)
    except ImportError:
        out["ram_gb_total"] = None
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            out["gpu"], out["gpu_arch_list"] = torch.cuda.get_device_name(0), torch.cuda.get_arch_list()
            out["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as exc:  # noqa: BLE001
        out["torch"] = f"not importable: {exc!r}"
    major = int(pyspark.__version__.split(".")[0]), int(pyspark.__version__.split(".")[1])
    if major[0] == 3 and major[1] == 5:
        if not java.get("found"):
            out["problems"].append("java not found on PATH: install Java 17 (64-bit recommended) or set JAVA_HOME")
        elif java["major"] not in (8, 11, 17):
            out["problems"].append(f"Java {java['major']} is not one of the versions Spark 3.5 supports (8, 11, 17)")
        elif java["major"] == 8:
            out["notes"].append("Java 8 works with Spark 3.5 but is deprecated there; Java 17 is the better choice")
        if java.get("found") and java.get("bits") == 32:
            out["notes"].append("32-bit Java: the JVM heap is limited to about 1.4 GB (spark.driver_memory must stay at 1g); install a 64-bit JDK 17 for more")
    if sys.platform == "win32":
        out["notes"].append("Windows: Spark cannot read/write local files through Hadoop without winutils.exe; this module avoids that (tasks use pyarrow)")
    return out


def build_spark(cfg: dict[str, Any], workers: int | None = None, ship_package: bool = False, **extra_conf: str):
    """SparkSession for local mode: `local[<workers>]`, memory / shuffle partitions / Arrow from config. Python workers use THIS interpreter."""
    from pyspark.sql import SparkSession
    sp = spark_settings(cfg)
    n = workers or resolve_workers(cfg)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    code_dir = str(Path(__file__).resolve().parent.parent)           # workers must find `stress_signals` even when it is not pip-installed
    os.environ["PYTHONPATH"] = os.pathsep.join([code_dir] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != code_dir])
    extra_conf = {"spark.executorEnv.PYTHONPATH": os.environ["PYTHONPATH"], **extra_conf}
    b = (SparkSession.builder.master(f"local[{n}]").appName(sp.get("app_name", "stress_signals"))
         .config("spark.ui.enabled", "false").config("spark.driver.memory", str(sp.get("driver_memory", "1g")))
         .config("spark.sql.shuffle.partitions", str(sp.get("shuffle_partitions", 8)))
         .config("spark.sql.execution.arrow.pyspark.enabled", str(bool(sp.get("arrow", True))).lower())
         .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(sp.get("arrow_max_records_per_batch", 256)))
         .config("spark.sql.session.timeZone", "UTC").config("spark.python.worker.reuse", "true")
         .config("spark.task.maxFailures", "1"))
    for k, v in extra_conf.items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    if ship_package:
        spark.sparkContext.addPyFile(str(package_zip(cfg)))   # cluster only: local mode uses the installed package (addPyFile needs winutils on Windows)
    return spark


def package_zip(cfg: dict[str, Any]) -> Path:
    """Zip of the stress_signals package for `SparkContext.addPyFile` (cluster use). Built next to the interim caches."""
    import zipfile
    src = Path(__file__).resolve().parent
    out = Path(cfg["_paths"]["interim"]) / "stress_signals_pkg.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*.py")):
            z.write(f, Path("stress_signals") / f.relative_to(src))
    return out


# ================================================================ lake access (pyarrow, no Hadoop)
def open_lake(lake_path: str | Path) -> pds.Dataset:
    p = Path(lake_path)
    if p.is_file():
        return pds.dataset(str(p), format="parquet")
    return pds.dataset(str(p), format="parquet", partitioning="hive", ignore_prefixes=["_", "."])


def lake_dates(lake_path: str | Path) -> pd.Series:
    """Rows per UTC calendar date (index = 'YYYY-MM-DD'). Reads only created_at."""
    cnt: dict[str, int] = {}
    for b in open_lake(lake_path).to_batches(columns=["created_at"], batch_size=500_000):
        d = b.to_pandas()["created_at"].dt.strftime("%Y-%m-%d")
        for k, v in d.value_counts().items():
            cnt[k] = cnt.get(k, 0) + int(v)
    return pd.Series(cnt, dtype="int64").sort_index()


def done_marker(out_root: str | Path, date: str) -> Path:
    return Path(out_root) / "_done" / f"date={date}.json"


def is_done(payload: dict[str, Any], date: str, bundle_id: str | None = None) -> bool:
    mp = done_marker(payload["out_root"], date)
    if not mp.is_file():
        return False
    m = json.loads(mp.read_text(encoding="utf-8"))
    ok = m.get("settings_hash") == payload["settings_hash"] and all((Path(payload["out_root"]) / f).is_file() for f in m.get("files", []))
    return ok and (bundle_id is None or m.get("bundle_id") == bundle_id)


def pending_dates(payload: dict[str, Any], dates: Sequence[str], bundle_id: str | None, force: bool = False) -> list[str]:
    return list(dates) if force else [d for d in dates if not is_done(payload, d, bundle_id)]


# ================================================================ the unit of work: one date
def enriched_schema(columns: Sequence[str]) -> pa.Schema:
    fields = []
    for c in columns:
        if c == "created_at":
            t = pa.timestamp("us", tz="UTC")
        elif c == "date":
            t = pa.date32()
        elif c == "inference_ts":
            t = pa.timestamp("us", tz="UTC")
        elif c in ("stress_flag", "stressor_none_unclear", "second_stage_scored") or c.startswith("stressor_flag__"):
            t = pa.bool_()
        elif c.startswith("n_chunks_"):
            t = pa.int32()
        elif c.startswith(("emotion_p__", "emotion_group__", "stressor_cos__", "stressor_p__", "stressor_count__", "tension_")) or c in ("stress_prob", "sample_weight"):
            t = pa.float64()
        else:
            t = pa.string()
        fields.append(pa.field(c, t))
    return pa.schema(fields)


def read_date(lake_path: str | Path, date: str) -> pd.DataFrame:
    lo = pd.Timestamp(date, tz="UTC")
    flt = (pds.field("created_at") >= lo.to_pydatetime()) & (pds.field("created_at") < (lo + pd.Timedelta(days=1)).to_pydatetime())
    ds = open_lake(lake_path)
    names = ds.schema.names
    cols = [c for c in ("record_id", "source", "community", "created_at", "text_clean") if c in names]
    df = ds.to_table(filter=flt, columns=cols).to_pandas()
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df.sort_values("record_id", kind="stable").reset_index(drop=True)


def enrich_frame(pipe: Any, df: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    """Score a lake frame in `rows_per_call` slices (memory bound) and return the enriched frame (no text)."""
    parts = []
    step = max(1, int(payload["rows_per_call"]))
    for i in range(0, len(df), step):
        sl = df.iloc[i:i + step]
        texts, ids = sl["text_clean"].tolist(), sl["record_id"].tolist()
        if payload["two_stage"]:
            r = pipe.predict_two_stage(texts, ids, payload["sample_rate"], payload["seed"], payload["batch_size"])
        else:
            r = pipe.predict(texts, payload["batch_size"])
            r["second_stage_scored"] = r["input_status"].isin(["ok", "ok_truncated"]).to_numpy()
            r["sample_weight"] = np.where(r["second_stage_scored"], 1.0, np.nan)
        r.index = sl.index
        parts.append(r)
    res = pd.concat(parts) if parts else pd.DataFrame()
    out = df[["record_id", "created_at", "source", "community"]].copy()
    out["date"] = out["created_at"].dt.date
    out = pd.concat([out, res.rename(columns={"stress_prob_calibrated": "stress_prob"})], axis=1)
    out["inference_ts"] = pd.Timestamp.now(tz="UTC")
    return out


def process_date(pipe: Any, payload: dict[str, Any], date: str) -> dict[str, Any]:
    """Score one UTC date and write its enriched Parquet (one file per source) + the restricted text file (if enabled) + the done marker."""
    t0 = time.time()
    df = read_date(payload["lake_path"], date)
    out_root = Path(payload["out_root"])
    files: list[str] = []
    n_scored = n_second = 0
    if len(df):
        enr = enrich_frame(pipe, df, payload)
        n_scored = int(enr["input_status"].isin(["ok", "ok_truncated"]).sum())
        n_second = int(enr["second_stage_scored"].sum())
        schema = enriched_schema(list(enr.columns))
        for src, g in enr.groupby("source", sort=True):
            rel = f"source={src}/date={date}/part-00000.parquet"
            _atomic_parquet(out_root / rel, pa.Table.from_pandas(g.reset_index(drop=True), schema=schema, preserve_index=False))
            files.append(rel)
        if payload["keep_text"]:
            rr = Path(payload["restricted_root"])
            for src, g in df.groupby("source", sort=True):
                _atomic_parquet(rr / f"source={src}/date={date}/part-00000.parquet",
                                pa.Table.from_pandas(g[["record_id", "text_clean"]].reset_index(drop=True), preserve_index=False))
    marker = {"date": date, "rows_in": int(len(df)), "rows_scored": n_scored, "rows_second_stage": n_second, "files": files,
              "bundle_id": pipe.bundle_info["bundle_id"], "settings_hash": payload["settings_hash"], "sample_rate": payload["sample_rate"],
              "two_stage": payload["two_stage"], "seed": payload["seed"], "device": pipe.device, "seconds": round(time.time() - t0, 3),
              "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "text_written_to_restricted": bool(payload["keep_text"])}
    write_json(done_marker(out_root, date), marker)           # LAST: its presence means the date is complete
    return {"date": date, "status": "done", "rows_in": len(df), "rows_scored": n_scored, "rows_second_stage": n_second,
            "seconds": marker["seconds"], "error": ""}


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


# ================================================================ workers: in-process pipeline or a long-lived inference child
_PAYLOADS: dict[str, dict[str, Any]] = {}
_CHILDREN: dict[str, "_InferenceChild"] = {}


def load_payload(path: str) -> dict[str, Any]:
    if path not in _PAYLOADS:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = p["cfg"]
        cfg["_root"] = Path(cfg["_root"])
        cfg["_paths"] = {k: Path(v) for k, v in cfg["_paths"].items()}
        _PAYLOADS[path] = p
    return _PAYLOADS[path]


def pipeline_for(payload: dict[str, Any]) -> Any:
    """The per-process pipeline (inference.get_pipeline cache): loaded on first use, then reused for every date in this process."""
    from .inference import get_pipeline
    try:
        import torch
        torch.set_num_threads(max(1, int(payload["num_threads"])))
    except Exception:  # noqa: BLE001
        pass
    return get_pipeline(payload["artifacts_dir"], payload["device"], payload["cfg"], payload["versions"])


class _InferenceChild:
    """One long-lived `python -m stress_signals.spark_jobs serve` process per Spark worker. It loads the pipeline once and then answers
    one JSON request per line on stdin with one tagged JSON line on stdout. It exits when its stdin closes (the worker ended)."""

    def __init__(self, payload_path: str):
        code_dir = str(Path(__file__).resolve().parent.parent)
        env = {**os.environ, "PYTHONPATH": code_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
        log_dir = Path(load_payload(payload_path)["cfg"]["_paths"]["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / f"spark_inference_child_{os.getpid()}.log", "ab")
        self.proc = subprocess.Popen([sys.executable, "-m", "stress_signals.spark_jobs", "serve", payload_path], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env, cwd=code_dir)

    def request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            if line.startswith(RESULT_TAG):
                return json.loads(line[len(RESULT_TAG):])
        raise RuntimeError(f"inference child exited (code {self.proc.poll()}); see {self.log.name}")


def run_unit(payload_path: str, mode: str, date: str) -> dict[str, Any]:
    payload = load_payload(payload_path)
    try:
        if mode == "in_process":
            return process_date(pipeline_for(payload), payload, date)
        child = _CHILDREN.get(payload_path)
        if child is None or child.proc.poll() is not None:
            child = _CHILDREN[payload_path] = _InferenceChild(payload_path)
        return child.request({"op": "process", "date": date})
    except Exception as exc:  # noqa: BLE001 - one bad date must not stop the job; it is reported and retried on the next run
        return {"date": date, "status": "error", "rows_in": 0, "rows_scored": 0, "rows_second_stage": 0, "seconds": 0.0, "error": repr(exc)[:500]}


def make_worker(payload_path: str, mode: str):
    """The mapInPandas function. Imports nothing heavy: in `subprocess` mode the Spark worker never loads torch."""
    def work(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in batches:
            rows = [run_unit(payload_path, mode, d) for d in pdf["date"].tolist()]
            yield pd.DataFrame(rows, columns=SUMMARY_COLS)
    return work


def serve(payload_path: str) -> None:
    """Child loop: load the pipeline once (lazy), then process requests from stdin until it closes."""
    proto = sys.stdout
    sys.stdout = sys.stderr                        # library prints must not corrupt the protocol
    payload = load_payload(payload_path)
    pipe = None
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        try:
            if pipe is None:
                pipe = pipeline_for(payload)
            res = process_date(pipe, payload, msg["date"]) if msg["op"] == "process" else {"status": "unknown_op"}
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res = {"date": msg.get("date"), "status": "error", "rows_in": 0, "rows_scored": 0, "rows_second_stage": 0, "seconds": 0.0,
                   "error": repr(exc)[:500]}
        proto.write(RESULT_TAG + json.dumps(res) + "\n")
        proto.flush()


# ================================================================ the job
def run_job(cfg: dict[str, Any], corpus: str | None = None, *, lake_path: str | Path | None = None, out_root: str | Path | None = None,
            force: bool = False, dates: Sequence[str] | None = None, max_dates: int | None = None, spark: Any = None, mode: str | None = None,
            device: str | None = None, artifacts_dir: str | Path | None = None, bundle_manifest: str | Path | None = None) -> dict[str, Any]:
    """Score every pending date of the lake with Spark. Returns a summary and writes `<out_root>/_job_summary.json`.
    Re-running skips dates whose marker matches the current bundle and settings (so it is idempotent and resumable)."""
    from .ingestion import corpus_spec, lake_root
    name = (corpus or "custom") if lake_path else corpus_spec(cfg, corpus)[0]
    lake = Path(lake_path) if lake_path else lake_root(cfg, name)
    payload = make_payload(cfg, name, lake_path=lake, out_root=out_root, artifacts_dir=artifacts_dir, bundle_manifest=bundle_manifest, device=device)
    mode = mode or resolve_mode(cfg)
    bundle_id = _bundle_id(payload)
    all_dates = list(dates) if dates is not None else lake_dates(lake).index.tolist()
    todo = pending_dates(payload, all_dates, bundle_id, force)
    n_skipped = len(all_dates) - len(todo)                 # finished earlier (same bundle and settings)
    if max_dates:
        todo = todo[:int(max_dates)]
    LOG.info("spark job: %d dates in the lake, %d pending (mode=%s, two_stage=%s, rate=%s)", len(all_dates), len(todo), mode, payload["two_stage"], payload["sample_rate"])
    results: list[dict[str, Any]] = []
    t0 = time.time()
    if todo:
        scratch = Path(cfg["_paths"]["interim"]) / "spark_payloads"
        scratch.mkdir(parents=True, exist_ok=True)
        ppath = scratch / f"payload_{payload['settings_hash']}_{os.getpid()}.json"
        ppath.write_text(json.dumps(payload), encoding="utf-8")
        own = spark is None
        spark = spark or build_spark(cfg, resolve_workers(cfg, device))
        try:
            n_tasks = max(1, int(spark.sparkContext.defaultParallelism))
            per_job = int(spark_settings(cfg).get("dates_per_job", 14))
            worker = make_worker(str(ppath), mode)
            for i in range(0, len(todo), per_job):
                chunk = todo[i:i + per_job]
                sdf = spark.createDataFrame(pd.DataFrame({"date": chunk})).repartition(min(len(chunk), n_tasks))
                got = sdf.mapInPandas(worker, SUMMARY_SCHEMA).toPandas()
                results += got.to_dict("records")
                LOG.info("spark job: %d/%d dates finished (%d rows)", min(i + per_job, len(todo)), len(todo), int(got["rows_in"].sum()))
        finally:
            if own:
                spark.stop()
            ppath.unlink(missing_ok=True)
    summary = {"corpus": name, "lake": str(lake), "out_root": payload["out_root"], "mode": mode, "bundle_id": bundle_id,
               "dates_in_lake": len(all_dates), "dates_skipped_done": n_skipped,
               "dates_run": len(todo), "dates_error": [r["date"] for r in results if r["status"] != "done"],
               "errors": {r["date"]: r["error"] for r in results if r["status"] != "done"},
               "rows_in": int(sum(r["rows_in"] for r in results)), "rows_scored": int(sum(r["rows_scored"] for r in results)),
               "rows_second_stage": int(sum(r["rows_second_stage"] for r in results)), "wall_seconds": round(time.time() - t0, 2),
               "two_stage": payload["two_stage"], "sample_rate": payload["sample_rate"], "seed": payload["seed"],
               "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    write_json(Path(payload["out_root"]) / "_job_summary.json", summary)
    return summary


def _bundle_id(payload: dict[str, Any]) -> str | None:
    """bundle_id of the pinned versions, read from the bundle manifest (no model is loaded on the driver)."""
    mp = Path(payload["cfg"]["_root"]) / payload["cfg"]["inference"]["bundle_manifest"]
    if mp.is_file():
        return json.loads(mp.read_text(encoding="utf-8")).get("bundle_id")
    return None


def read_enriched(out_root: str | Path) -> pd.DataFrame:
    """All enriched rows under out_root as one frame (hive columns source/date included)."""
    part = pds.partitioning(pa.schema([("source", pa.string()), ("date", pa.date32())]), flavor="hive")   # same types as the file columns
    return pds.dataset(str(out_root), format="parquet", partitioning=part, ignore_prefixes=["_", "."]).to_table().to_pandas()


# ================================================================ parity: Spark job vs the plain local pipeline
def enrich_local(payload_path: str, out_path: str) -> None:
    """The plain local pipeline on the whole sample in ONE call (used by parity_check in its own short-lived process)."""
    payload = load_payload(payload_path)
    pipe = pipeline_for(payload)
    sample = pd.read_parquet(payload["lake_path"]).sort_values("record_id", kind="stable").reset_index(drop=True)
    enrich_frame(pipe, sample, {**payload, "rows_per_call": len(sample) or 1}).to_parquet(out_path)


def parity_check(cfg: dict[str, Any], sample_path: str | Path, *, atol: float | None = None, out_root: str | Path | None = None,
                 mode: str | None = None, device: str | None = None, artifacts_dir: str | Path | None = None, spark: Any = None) -> dict[str, Any]:
    """Run the 1-day sample through (a) the local pipeline in one process, one call, and (b) the Spark job; compare the outputs.
    Exact: record ids, stage-2 selection, sample_weight, input_status, stress_flag, stressor_top, stage-2 flags. Within `atol`: every
    probability / score column. Differences come only from batch composition (padding), the same effect the Step 7 test bounds at 1e-4."""
    atol = float(atol if atol is not None else spark_settings(cfg)["parity"]["atol"])
    out_root = Path(out_root or Path(cfg["_paths"]["interim"]) / "spark_parity")
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)
    payload = make_payload(cfg, "parity", lake_path=sample_path, out_root=out_root, device=device, artifacts_dir=artifacts_dir)
    scratch = Path(cfg["_paths"]["interim"]) / "spark_payloads"
    scratch.mkdir(parents=True, exist_ok=True)
    ppath, lpath = scratch / f"parity_payload_{os.getpid()}.json", scratch / f"parity_local_{os.getpid()}.parquet"
    ppath.write_text(json.dumps(payload), encoding="utf-8")
    # the local reference runs in a SHORT-LIVED process: its models (and GPU / RAM) are released before the Spark worker loads its own copy
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "stress_signals.spark_jobs", "enrich-local", str(ppath), str(lpath)], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parent.parent),
                       env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})
    t_local = time.time() - t0
    if r.returncode != 0 or not lpath.is_file():
        raise RuntimeError("local reference run failed: " + (r.stderr or "")[-800:])
    local = pd.read_parquet(lpath)
    ppath.unlink(missing_ok=True)
    lpath.unlink(missing_ok=True)
    t0 = time.time()
    summary = run_job(cfg, "parity", lake_path=sample_path, out_root=out_root, force=True, spark=spark, mode=mode, device=device,
                      artifacts_dir=artifacts_dir)
    t_spark = time.time() - t0
    res_err: dict[str, Any] = {"n_local": len(local), "n_spark": 0, "atol": atol, "seconds_local": round(t_local, 2),
                               "seconds_spark_job": round(t_spark, 2), "spark_errors": summary["errors"]}
    if summary["errors"] or not (out_root / "_done").exists():
        return {**res_err, "ok": False, "problems": ["the Spark job reported errors: " + json.dumps(summary["errors"])]}
    sp = read_enriched(out_root).sort_values("record_id", kind="stable").reset_index(drop=True)
    local = local.sort_values("record_id", kind="stable").reset_index(drop=True)
    res: dict[str, Any] = {"n_local": len(local), "n_spark": len(sp), "atol": atol, "seconds_local": round(t_local, 2),
                           "seconds_spark_job": round(t_spark, 2), "spark_errors": summary["errors"], "problems": []}
    if len(local) != len(sp) or not (local["record_id"].to_numpy() == sp["record_id"].to_numpy()).all():
        res["problems"].append("record ids differ between local and Spark")
        res["ok"] = False
        return res
    skip = {"inference_ts", "date", "source"}
    num_cols = [c for c in local.columns if c not in skip and pd.api.types.is_float_dtype(local[c])]
    exact_cols = [c for c in local.columns if c not in skip and c not in num_cols and c != "created_at"]
    max_abs = {}
    for c in num_cols:
        a, b = local[c].to_numpy(dtype=float), sp[c].to_numpy(dtype=float)
        if not (np.isnan(a) == np.isnan(b)).all():
            res["problems"].append(f"{c}: NaN pattern differs")
            continue
        m = ~np.isnan(a)
        d = float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0
        max_abs[c] = d
        if d > atol:
            res["problems"].append(f"{c}: max |diff| {d:.2e} > atol {atol:.0e}")
    for c in exact_cols:
        a, b = local[c].astype(object).where(local[c].notna(), None).tolist(), sp[c].astype(object).where(sp[c].notna(), None).tolist()
        bad = sum(1 for x, y in zip(a, b) if x != y)
        if bad:
            res["problems"].append(f"{c}: {bad} rows differ")
    res.update({"max_abs_diff": max(max_abs.values()) if max_abs else 0.0, "worst_column": max(max_abs, key=max_abs.get) if max_abs else None,
                "columns_compared": len(num_cols) + len(exact_cols), "stage2_scored": int(local["second_stage_scored"].sum()),
                "flagged": int(local["stress_flag"].fillna(False).astype(bool).sum()), "ok": not res["problems"]})
    return res


# ================================================================ benchmark and int8 accuracy check
def sample_posts(cfg: dict[str, Any], corpus: str | None, n: int, seed: int = 42, lake_path: str | Path | None = None) -> pd.DataFrame:
    """n lake rows chosen deterministically (hash of seed + record_id), spread over the whole lake. Local use only: it returns text."""
    from .ingestion import corpus_spec, lake_root
    name, _ = corpus_spec(cfg, corpus)
    ds = open_lake(lake_path or lake_root(cfg, name))
    total = ds.count_rows()
    keep = min(1.0, 1.15 * n / max(total, 1))
    frames = []
    for b in ds.to_batches(columns=["record_id", "source", "community", "created_at", "text_clean"], batch_size=100_000):
        t = b.to_pandas()
        from .inference import unit_hash
        u = unit_hash(t["record_id"].tolist(), seed + 7919)       # a different salt than the stage-2 sample, or the two samples would be correlated
        frames.append(t[u < keep].assign(_u=u[u < keep]))
    df = pd.concat(frames).sort_values("_u", kind="stable").head(n).drop(columns="_u")
    return df.sort_values("record_id", kind="stable").reset_index(drop=True)


def hardware_description() -> dict[str, Any]:
    d = {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()}
    try:
        import psutil
        d["ram_gb_total"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda"] = torch.cuda.is_available()
        if d["cuda"]:
            d["gpu"] = torch.cuda.get_device_name(0)
            d["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as exc:  # noqa: BLE001
        d["torch"] = repr(exc)
    return d


def benchmark(cfg: dict[str, Any], corpus: str | None = None, n: int | None = None, *, device: str | None = None, two_stage: bool = True,
              project_rows: Sequence[int] | None = None, lake_path: str | Path | None = None) -> dict[str, Any]:
    """MEASURE the throughput of the pipeline in this process (no Spark overhead) on `n` posts of YOUR lake and project the runtime for the
    corpus sizes in `project_rows`. The projection is linear (rows / measured rows per second) and assumes the corpus resembles the
    sample (text length, share of stress-flagged posts, same hardware, nothing else using the GPU/CPU). It is a planning number, not a promise."""
    from .inference import StressSignalPipeline, versions_from_bundle_manifest
    from .utils import models_root
    sp = spark_settings(cfg)
    n = int(n or sp["benchmark"]["n_posts"])
    posts = sample_posts(cfg, corpus, n, cfg["primary_seed"], lake_path)
    mp = Path(cfg["_root"]) / cfg["inference"]["bundle_manifest"]
    t0 = time.time()
    pipe = StressSignalPipeline.from_artifacts(models_root(cfg), device or cfg["inference"].get("device", "auto"), cfg,
                                               versions_from_bundle_manifest(mp) if mp.is_file() else None)
    t_load = time.time() - t0
    ts = sp["two_stage"]
    bs, step = int(cfg["inference"]["batch_size"]), int(sp.get("rows_per_call", 512))
    warm = pipe.predict(posts["text_clean"].head(8).tolist(), bs)          # warm-up (kernels, caches) is not timed
    del warm
    s0 = s1 = s2 = 0.0
    n2 = 0
    t0 = time.time()
    for i in range(0, len(posts), step):
        sl = posts.iloc[i:i + step]
        if two_stage and ts.get("enabled", True):
            pipe.predict_two_stage(sl["text_clean"].tolist(), sl["record_id"].tolist(), float(ts["sample_rate"]), int(ts["seed"]), bs)
        else:
            pipe.predict(sl["text_clean"].tolist(), bs)
        s0 += pipe.last_run["seconds_prepare"]
        s1 += pipe.last_run["seconds_stress"]
        s2 += pipe.last_run["seconds_second_stage"]
        n2 += pipe.last_run["n_second_stage"]
    wall = time.time() - t0
    rps = len(posts) / wall
    sizes = list(project_rows or sp["benchmark"]["project_rows"])
    out = {"n_posts": int(len(posts)), "wall_seconds": round(wall, 2), "records_per_second": round(rps, 2), "seconds_load_models": round(t_load, 1),
           "seconds_prepare_clean_language_id": round(s0, 2), "seconds_stress_stage": round(s1, 2), "seconds_second_stage": round(s2, 2), "second_stage_share_scored": round(n2 / max(len(posts), 1), 3),
           "two_stage": bool(two_stage and ts.get("enabled", True)), "sample_rate": ts.get("sample_rate"), "device": pipe.device,
           "batch_size": bs, "hardware": hardware_description(),
           "projection_hours": {f"{s:,} posts": round(s / rps / 3600, 2) for s in sizes},
           "projection_note": "linear projection from this sample: assumes similar text length, similar share of stress-flagged posts, the same hardware "
                              "and no other load. Spark orchestration adds start-up time per job; more workers only help on CPU."}
    return out


def int8_accuracy_check(cfg: dict[str, Any], texts: Sequence[str]) -> dict[str, Any]:
    """OPTIONAL CPU int8 path: compare fp32 and dynamically quantised scores on `texts` BEFORE using int8. Reports the largest and mean absolute
    probability difference per model and how many stress flags change. Decide the acceptable loss yourself; thresholds were fitted on fp32."""
    from .inference import StressSignalPipeline, versions_from_bundle_manifest
    from .utils import models_root
    mp = Path(cfg["_root"]) / cfg["inference"]["bundle_manifest"]
    v = versions_from_bundle_manifest(mp) if mp.is_file() else None
    a = StressSignalPipeline.from_artifacts(models_root(cfg), "cpu", cfg, v, cpu_int8=False)
    ra = a.predict(list(texts)).copy()
    a = None
    b = StressSignalPipeline.from_artifacts(models_root(cfg), "cpu", cfg, v, cpu_int8=True)
    rb = b.predict(list(texts))
    emo = [c for c in ra.columns if c.startswith("emotion_p__")]
    d_stress = (ra["stress_prob_calibrated"] - rb["stress_prob_calibrated"]).abs()
    d_emo = (ra[emo] - rb[emo]).abs().to_numpy()
    return {"n": len(ra), "stress_prob_max_abs_diff": float(d_stress.max()), "stress_prob_mean_abs_diff": float(d_stress.mean()),
            "stress_flag_changes": int((ra["stress_flag"].astype(bool) != rb["stress_flag"].astype(bool)).sum()),
            "emotion_p_max_abs_diff": float(np.nanmax(d_emo)), "emotion_p_mean_abs_diff": float(np.nanmean(d_emo)),
            "note": "int8 is CPU only; accept it only if these differences are tolerable for your use. Not a substitute for re-checking thresholds."}


# ================================================================ command line
def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Step 9 Spark inference (inference only)")
    ap.add_argument("cmd", choices=["env", "run", "serve", "benchmark", "parity", "enrich-local"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("arg2", nargs="?")
    ap.add_argument("--corpus")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-dates", type=int)
    ap.add_argument("--n", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.arg)
        return
    if a.cmd == "enrich-local":
        enrich_local(a.arg, a.arg2)
        return
    from .config import load_config
    cfg = load_config()
    if a.cmd == "env":
        print(json.dumps(check_environment(), indent=2, default=str))
    elif a.cmd == "run":
        print(json.dumps(run_job(cfg, a.corpus, force=a.force, max_dates=a.max_dates), indent=2, default=str))
    elif a.cmd == "benchmark":
        print(json.dumps(benchmark(cfg, a.corpus, a.n), indent=2, default=str))
    else:
        print(json.dumps(parity_check(cfg, a.arg), indent=2, default=str))


if __name__ == "__main__":
    main()
