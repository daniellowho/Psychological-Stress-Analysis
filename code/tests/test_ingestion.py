"""Step 8 ingestion tests: synthetic posts only (no real user text), no model weights touched, no network."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

zstandard = pytest.importorskip("zstandard")

from stress_signals import ingestion as G
from stress_signals.config import load_config


def _ts(y, m, d, h=12):
    return int(pd.Timestamp(year=y, month=m, day=d, hour=h, tz="UTC").timestamp())


GOOD = "I have been worrying about my rent and my job every single night this week, it is exhausting."


def _post(i, sub="anxiety", ts=None, title="Rent worries", body=GOOD, **kw):
    o = {"id": f"p{i}", "subreddit": sub, "created_utc": ts or _ts(2019, 3, 5), "title": title, "selftext": body,
         "author": "someone_secret", "permalink": "/r/x/comments/abc", **kw}
    return json.dumps(o)


def _uniq(n):
    return _post(n, title=f"T{n}", body=GOOD + f" Post number {n} is unique.")


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c["_root"] = str(tmp_path)
    c["_paths"] = {**c["_paths"], **{k: str(tmp_path / k) for k in ("lake", "reports", "manifests", "processed")}}
    for k in ("reports", "manifests", "processed", "reddit"):
        (tmp_path / k).mkdir()
    c["sources"]["main_corpus"] = {**c["sources"]["main_corpus"], "local_dir": "reddit", "subreddits": ["anxiety", "legaladvice"],
                                   "excluded_subreddits": ["suicidewatch"]}
    c["ingestion"] = {**c["ingestion"], "active_corpus": "main", "language_filter": False, "chunk_rows": 3, "flush_rows": 4,
                      "target_rows_per_file": 100, "bot_authors": ["automoderator"]}
    return c


def _dump(cfg, lines, name="r_anxiety_submissions.zst"):
    (Path(cfg["_root"]) / "reddit" / name).write_bytes(zstandard.ZstdCompressor().compress(("\n".join(lines) + "\n").encode("utf-8")))


def test_reddit_adapter_filters_and_counts_every_drop(cfg):
    lines = [
        _post(1),                                                                                       # kept
        _post(2, sub="Anxiety", title="Another", body=GOOD + " Different text here so it is not a duplicate."),   # community is case-insensitive
        _post(3, sub="cooking"), _post(4, sub="suicidewatch"),                                          # not in communities / excluded
        _post(5, ts=_ts(2018, 12, 31)), _post(6, ts=_ts(2023, 1, 1)),                                   # outside DATE_RANGE
        _post(7, body="[removed]"), _post(8, body="[deleted]"), _post(9, body="   "),
        _post(10, removed_by_category="moderator"),
        _post(11, author="AutoModerator"),
        "{not json", json.dumps({"id": "x", "subreddit": "anxiety"}),                                   # malformed / missing timestamp
        json.dumps({"id": "y", "subreddit": "anxiety", "created_utc": _ts(2019, 3, 5)}),               # missing selftext
        _post(12, title="Third", body=GOOD + " A third distinct post."),                                # kept
    ]
    _dump(cfg, lines)
    man = G.ingest(cfg, "main")
    d = man["drops_by_reason"]
    assert d["not_in_communities"] == 2 and d["outside_date_range"] == 2
    assert d["removed_or_deleted"] == 2 and d["empty_selftext"] == 1 and d["removed_post"] == 1 and d["bot_account"] == 1
    assert d["malformed_json"] == 1 and d["missing_timestamp"] == 1 and d["missing_field"] == 1
    assert man["balanced"] and man["rows_in"] == len(lines) and man["rows_out"] == 3


def test_lake_layout_schema_privacy_and_utc(cfg):
    _dump(cfg, [_post(1, ts=_ts(2019, 3, 5)), _post(2, ts=_ts(2019, 4, 6), title="Second", body=GOOD + " More words about April."),
                _post(3, ts=_ts(2020, 1, 31, 23), title="Third", body=GOOD + " January end.")])
    man = G.ingest(cfg, "main")
    assert set(man["partitions"]) == {"source=reddit_submission/year=2019/month=03", "source=reddit_submission/year=2019/month=04",
                                      "source=reddit_submission/year=2020/month=01"}
    f = next(G.lake_root(cfg, "main").glob("source=*/year=2019/month=03/*.parquet"))
    assert pq.read_schema(f).equals(G.LAKE_SCHEMA, check_metadata=False)
    assert pq.ParquetFile(f).metadata.row_group(0).column(0).compression.lower() == "zstd"
    df = G.read_lake(cfg, "main").to_table().to_pandas()
    assert not [c for c in df.columns if c.lower() in {"author", "user", "username", "permalink", "url"}]
    assert "someone_secret" not in "".join(df["text_clean"]) and str(df["created_at"].dt.tz) == "UTC"
    assert df["created_at"].max().hour == 23 and df["created_at"].max().month == 1      # UTC kept, not shifted


def test_title_and_selftext_are_joined_and_cleaned(cfg):
    _dump(cfg, [_post(1, title="Title here", body="Body text. Mail me a@b.org or see https://x.example/p and u/someone please.")])
    G.ingest(cfg, "main")
    t = G.read_lake(cfg, "main").to_table().to_pandas()["text_clean"].iloc[0]
    assert t.startswith("Title here Body text.") and "a@b.org" not in t and "https://" not in t and "u/someone" not in t


def test_dedupe_across_files_keeps_first_and_is_counted(cfg):
    _dump(cfg, [_post(1), _post(2, title="Other", body=GOOD + " And more.")], name="a_first.zst")
    _dump(cfg, [_post(3), _post(4, title="New", body=GOOD + " Brand new content to read.")], name="b_second.zst")
    man = G.ingest(cfg, "main")
    assert man["drops_by_reason"]["exact_duplicate_text"] == 1 and man["rows_out"] == 3 and man["balanced"]


def test_resume_skips_finished_inputs_and_redoes_changed_ones(cfg):
    _dump(cfg, [_post(1), _post(2, title="Other", body=GOOD + " And more.")], name="a.zst")
    sha1 = {k: v["sha256"] for k, v in G.ingest(cfg, "main")["partitions"].items()}
    m2 = G.ingest(cfg, "main")                                                  # nothing changed: identical lake
    assert {k: v["sha256"] for k, v in m2["partitions"].items()} == sha1 and m2["rows_out"] == 2
    _dump(cfg, [_post(1), _post(5, title="Fifth", body=GOOD + " Fifth post.")], name="a.zst")        # input changed
    m3 = G.ingest(cfg, "main")
    assert m3["rows_out"] == 2 and m3["balanced"]


def test_compaction_merges_parts_and_keeps_rows(cfg):
    _dump(cfg, [_uniq(i) for i in range(1, 8)])                                 # flush_rows 4 -> several part files
    cfg["ingestion"]["compact"] = False
    m = G.ingest(cfg, "main", force=True)
    cfg["ingestion"]["compact"] = True
    m2 = G.ingest(cfg, "main", force=True)
    assert sum(v["files"] for v in m["partitions"].values()) >= 2
    assert sum(v["files"] for v in m2["partitions"].values()) == 1 and m2["rows_out"] == m["rows_out"] == 7


def test_data_quality_suppresses_small_cells_and_has_no_text(cfg):
    cfg["privacy"]["k_min"] = 3
    _dump(cfg, [_uniq(i) for i in range(1, 5)] + [_post(9, ts=_ts(2019, 5, 1), title="Lonely", body=GOOD + " The lonely day.")])
    G.ingest(cfg, "main")
    res = G.data_quality(cfg, "main")
    rep = Path(cfg["_paths"]["reports"])
    csv = pd.read_csv(rep / "data_quality_main" / "volume_by_day_community.csv")
    assert csv["suppressed"].sum() == 1 and csv.loc[csv["suppressed"], "n"].isna().all() and res["rows"] == 5
    md = (rep / "data_quality_main.md").read_text(encoding="utf-8")
    assert "rent" not in md.lower() and "balanced" in md


def test_day_sample_and_sampling_frame(cfg):
    _dump(cfg, [_uniq(i) for i in range(1, 4)] + [_post(9, ts=_ts(2019, 5, 1), title="Other day", body=GOOD + " other.")])
    G.ingest(cfg, "main")
    p = G.make_day_sample(cfg, "main")
    assert p.name == "main_2019-03-05.parquet" and len(pd.read_parquet(p)) == 3
    assert len(pd.read_parquet(G.make_day_sample(cfg, "main", "2019-05-01"))) == 1
    txt = G.write_sampling_frame(cfg).read_text(encoding="utf-8")
    assert "suicidewatch" in txt and "Selection bias" in txt and "No individual-level analysis" in txt


def test_no_input_files_is_a_clear_error(cfg):
    with pytest.raises(FileNotFoundError, match="Download them yourself"):
        G.ingest(cfg, "main")


def test_generic_adapter_reads_only_mapped_columns(cfg, tmp_path):
    d = tmp_path / "generic"
    d.mkdir()
    pd.DataFrame({"id": ["a", "b"], "subreddit": ["X", "X"], "created_utc": [_ts(2019, 3, 5), _ts(2019, 3, 6)],
                  "text": [GOOD, GOOD + " Second one."], "author": ["secret1", "secret2"]}).to_csv(d / "f.csv", index=False)
    cfg["sources"]["generic_ingest"] = {"local_dir": "generic", "files": ["*.csv"], "format": "csv", "created_format": "epoch_s",
                                        "columns": {"id": "id", "community": "subreddit", "created": "created_utc", "text": "text", "text2": None},
                                        "source": "generic", "default_community": "u"}
    cfg["ingestion"]["corpora"]["gen"] = {"adapter": "generic", "date_range": None}
    man = G.ingest(cfg, "gen")
    df = G.read_lake(cfg, "gen").to_table().to_pandas()
    assert man["rows_out"] == 2 and set(df["community"]) == {"x"} and "author" not in df.columns
    cfg["sources"]["generic_ingest"]["columns"]["text2"] = "author"
    with pytest.raises(AssertionError, match="author"):
        G.build_adapter(cfg, "gen")


def test_zenodo_adapter_whitelists_columns_and_reads_pre_before_post(cfg, tmp_path):
    z = tmp_path / "z"
    z.mkdir()
    for win in ("pre", "post"):                                                 # the same text appears in both windows
        pd.DataFrame({"subreddit": ["anxiety"], "author": ["secret"], "date": ["2019/05/01"], "post": [GOOD], "feat": [1.0]}
                     ).to_csv(z / f"anxiety_{win}_features_tfidf_256.csv", index=False)
    cfg["sources"]["zenodo_pilot"] = {**cfg["sources"]["zenodo_pilot"], "local_dir": "z", "subreddits": ["anxiety"]}
    cfg["ingestion"]["corpora"]["pilot"] = {"adapter": "zenodo_pilot", "date_range": None}
    man = G.ingest(cfg, "pilot")
    assert man["rows_in"] == 2 and man["rows_out"] == 1 and man["drops_by_reason"]["exact_duplicate_text"] == 1
    assert list(man["per_input"])[0].startswith("anxiety_pre")


def test_inspect_file_reports_field_names_only(tmp_path):
    p = tmp_path / "x.zst"
    p.write_bytes(zstandard.ZstdCompressor().compress((_post(1) + "\n").encode("utf-8")))
    r = G.inspect_file(p)
    assert "selftext" in r["fields"] and not r["missing_needed"] and "someone_secret" not in json.dumps(r)
