"""Step 9 tests: tiny synthetic bundles (from test_inference), synthetic text only. No model is trained.
Spark-dependent tests are skipped when pyspark or a working Java is missing."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_signals import inference as I
from stress_signals import spark_jobs as SJ
from stress_signals.config import load_config

import test_inference as TI                                              # tiny-bundle helpers (tests/ is on sys.path under pytest)
from test_inference import artifacts, cfg, pipe, tok  # noqa: F401  (module-scoped fixtures)

pytest.importorskip("pyspark")


def _java_ok() -> bool:
    try:
        return subprocess.run(["java", "-version"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


needs_spark = pytest.mark.skipif(not _java_ok(), reason="no working Java for Spark")


def _day_frame(n=40, day="2020-01-06", source="synthetic"):
    ts = pd.date_range(f"{day} 00:30", periods=n, freq="31min", tz="UTC")
    texts = [("I am so worried about money and my job today, everything feels like too much. " if i % 3 == 0 else
              "The train leaves at nine and the library opens at ten on weekdays. ") + f"Note number {i}." for i in range(n)]
    return pd.DataFrame({"record_id": [f"r{i:04d}" for i in range(n)], "source": source, "community": "testsub", "created_at": ts,
                         "text_clean": texts, "n_chars": [len(t) for t in texts], "n_tokens_est": [len(t.split()) for t in texts], "lang": "en"})


@pytest.fixture()
def env(tmp_path, artifacts, cfg):
    c = load_config()
    c["_root"] = tmp_path
    c["_paths"] = {**c["_paths"], **{k: tmp_path / k for k in ("interim", "logs", "reports", "manifests")}}
    c["inference"] = {**c["inference"], "english_only": False, "device": "cpu", "batch_size": 4, "bundle_manifest": "bundle_manifest.json"}
    c["spark"] = {**c["spark"], "workers": 1, "rows_per_call": 16, "driver_memory": "1g",
                  "two_stage": {"enabled": True, "sample_rate": 0.5, "seed": 7}}
    sample = tmp_path / "sample.parquet"
    _day_frame().to_parquet(sample)
    return c, artifacts, sample


# ---------------------------------------------------------------- two-stage design (no Spark)
def test_two_stage_selection_weights_and_determinism(pipe):
    df = _day_frame(60)
    a = pipe.predict_two_stage(df["text_clean"], df["record_id"], 0.5, 7, 8)
    b = pipe.predict_two_stage(df["text_clean"].tolist()[::-1], df["record_id"].tolist()[::-1], 0.5, 7, 3).iloc[::-1].reset_index(drop=True)
    assert (a["second_stage_scored"].to_numpy() == b["second_stage_scored"].to_numpy()).all()      # independent of order and batching
    flagged = a["stress_flag"].fillna(False).astype(bool).to_numpy()
    scored = a["second_stage_scored"].to_numpy()
    assert scored[flagged].all()                                                                    # every flagged text gets stage 2
    w = a["sample_weight"].to_numpy()
    assert (w[flagged] == 1.0).all() and np.allclose(w[~flagged & scored], 2.0) and np.isnan(w[~scored]).all()
    emo = [c for c in a.columns if c.startswith("emotion_p__")]
    assert a.loc[scored, emo].notna().all().all() and a.loc[~scored, emo].isna().all().all()       # stage 2 columns only where scored
    assert a["stress_prob_calibrated"].notna().all()                                                # stage 1 on everything
    assert a.loc[~scored, "stressor_top"].isna().all()


def test_two_stage_rate_one_scores_everything_and_rate_is_validated(pipe):
    df = _day_frame(12)
    a = pipe.predict_two_stage(df["text_clean"], df["record_id"], 1.0, 1)
    assert a["second_stage_scored"].all() and (a["sample_weight"] == 1.0).all()
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            pipe.predict_two_stage(df["text_clean"], df["record_id"], bad, 1)


def test_unit_hash_is_stable_and_uniform_enough():
    keys = [f"id{i}" for i in range(4000)]
    u = I.unit_hash(keys, 42)
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.03
    assert np.array_equal(u, I.unit_hash(keys, 42)) and not np.array_equal(u[:50], I.unit_hash(keys[:50], 43))


def test_int8_quantisation_runs_on_cpu_and_is_not_broken(artifacts, cfg):
    ra = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg).predict(TI.NEUTRAL_EXAMPLES)
    q = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg).quantize_int8()
    rb = q.predict(TI.NEUTRAL_EXAMPLES)
    assert q.bundle_info["int8"] and np.isfinite(rb["stress_prob_calibrated"]).all()
    assert (ra["stress_prob_calibrated"] - rb["stress_prob_calibrated"]).abs().max() < 0.2          # tiny random models: loose, just not broken


# ---------------------------------------------------------------- the unit of work (no Spark)
def test_process_date_is_text_free_idempotent_and_setting_sensitive(env):
    cfg, artifacts, sample = env
    out = Path(cfg["_root"]) / "out"
    payload = SJ.make_payload(cfg, "t", lake_path=sample, out_root=out, artifacts_dir=artifacts)
    pipe = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg)
    bid = pipe.bundle_info["bundle_id"]
    assert not SJ.is_done(payload, "2020-01-06", bid)
    r = SJ.process_date(pipe, payload, "2020-01-06")
    assert r["status"] == "done" and r["rows_in"] == 40 and SJ.is_done(payload, "2020-01-06", bid)
    df = SJ.read_enriched(out)
    assert len(df) == 40 and "text_clean" not in df.columns and "text" not in df.columns
    assert {"record_id", "created_at", "date", "source", "community", "stress_prob", "stress_flag", "sample_weight", "inference_ts",
            "stress_version", "emotion_version", "stressor_version"} <= set(df.columns)
    assert not (Path(cfg["_root"]) / cfg["spark"]["restricted_dir"]).exists()                        # keep_text_for_topics is off
    other = {**cfg, "spark": {**cfg["spark"], "two_stage": {"enabled": True, "sample_rate": 0.9, "seed": 7}}}
    assert not SJ.is_done(SJ.make_payload(other, "t", lake_path=sample, out_root=out, artifacts_dir=artifacts), "2020-01-06")   # new rate: recompute
    assert not SJ.is_done(payload, "2020-01-06", "some-other-bundle")                                                          # new bundle: recompute


def test_keep_text_goes_to_a_separate_restricted_path_only(env):
    cfg, artifacts, sample = env
    cfg["spark"]["keep_text_for_topics"] = True
    out = Path(cfg["_root"]) / "out"
    payload = SJ.make_payload(cfg, "t", lake_path=sample, out_root=out, artifacts_dir=artifacts)
    pipe = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg)
    SJ.process_date(pipe, payload, "2020-01-06")
    assert "text_clean" not in SJ.read_enriched(out).columns
    rr = Path(payload["restricted_root"])
    assert rr.exists() and not str(rr).startswith(str(out)) and "text_clean" in pd.read_parquet(next(rr.rglob("*.parquet"))).columns


def test_empty_date_writes_a_marker_and_no_rows(env):
    cfg, artifacts, sample = env
    out = Path(cfg["_root"]) / "out"
    payload = SJ.make_payload(cfg, "t", lake_path=sample, out_root=out, artifacts_dir=artifacts)
    pipe = I.StressSignalPipeline.from_artifacts(artifacts, device="cpu", cfg=cfg)
    r = SJ.process_date(pipe, payload, "2021-05-05")
    assert r["rows_in"] == 0 and SJ.done_marker(out, "2021-05-05").exists()


def test_lake_dates_and_enriched_schema_types(env):
    cfg, artifacts, sample = env
    assert SJ.lake_dates(sample).to_dict() == {"2020-01-06": 40}
    s = SJ.enriched_schema(["stress_prob", "stress_flag", "created_at", "date", "emotion_p__joy", "stressor_top", "n_chunks_stress"])
    assert [str(f.type) for f in s] == ["double", "bool", "timestamp[us, tz=UTC]", "date32[day]", "double", "string", "int32"]


def test_environment_check_reports_versions():
    e = SJ.check_environment()
    assert e["pyspark"].startswith("3.") and "java" in e and isinstance(e["problems"], list)


# ---------------------------------------------------------------- Spark itself
@needs_spark
def test_spark_job_matches_local_pipeline_within_tolerance(env):
    cfg, artifacts, sample = env
    res = SJ.parity_check(cfg, sample, mode="subprocess", artifacts_dir=artifacts)
    assert res["ok"], res["problems"]
    assert res["n_local"] == res["n_spark"] == 40 and res["max_abs_diff"] <= res["atol"] and not res["spark_errors"]


@needs_spark
def test_spark_job_is_idempotent_and_skips_finished_dates(env):
    cfg, artifacts, sample = env
    out = Path(cfg["_root"]) / "out"
    s1 = SJ.run_job(cfg, "t", lake_path=sample, out_root=out, mode="subprocess", artifacts_dir=artifacts)
    assert s1["dates_run"] == 1 and s1["rows_in"] == 40 and not s1["dates_error"]
    f = out / "source=synthetic" / "date=2020-01-06" / "part-00000.parquet"
    first = f.stat().st_mtime_ns
    s2 = SJ.run_job(cfg, "t", lake_path=sample, out_root=out, mode="subprocess", artifacts_dir=artifacts)
    assert s2["dates_run"] == 0 and s2["dates_skipped_done"] == 1 and f.stat().st_mtime_ns == first
    assert SJ.run_job(cfg, "t", lake_path=sample, out_root=out, mode="subprocess", artifacts_dir=artifacts, force=True)["dates_run"] == 1
