"""Step 8: ingestion to a partitioned Parquet data lake. NO training, NO scraping: only files you downloaded yourself.

    SourceAdapter (zenodo_pilot | reddit_submissions | generic)  ->  raw frame (record_id, community, created_at, text_raw)
      -> Step 2 cleaning / bot / length / language / exact-dedupe filters (every drop counted by reason)
      -> lake/<corpus>/source=<s>/year=<YYYY>/month=<MM>/part-*.parquet   (explicit schema, zstd)
      -> ingestion manifest (rows in / out, per input file and per partition, sha256 of every file)
      -> data quality report, sampling-frame report, 1-day sample for the Step 9 parity test.

The SAME code runs the pilot and the main corpus; only `ingestion.active_corpus` (and the paths in `sources.*`) change.
Author / username fields are never kept. The Reddit adapter reads `author` for ONE purpose, comparing it to the bot-account
list, and discards it on the same line; it never reaches a DataFrame. Resumable per input file (state under lake/<corpus>/_state/).

Check before using real data: Reddit's terms of use, the dataset licence (Zenodo record 3941387; Arctic Shift terms) and your
institution's ethics requirements. Field names of the Reddit dump are UNVERIFIED until a real file has been read (see RedditSubmissionAdapter).
"""

from __future__ import annotations

import abc
import glob
import hashlib
import io
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from .preprocess import (
    LanguageIdentifier, assert_privacy_safe, clean_text, dedupe_key, is_bot_text, n_tokens_estimate, stable_record_id,
)
from .privacy import author_columns
from .utils import get_logger, library_versions, sha256_file, write_json

LOG = get_logger("stress_signals.ingestion")

RAW_COLUMNS = ["record_id", "community", "created_at", "text_raw"]
LAKE_SCHEMA = pa.schema([
    ("record_id", pa.string()), ("source", pa.string()), ("community", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")), ("text_clean", pa.string()),
    ("n_chars", pa.int32()), ("n_tokens_est", pa.int32()), ("lang", pa.string()),
])
REMOVED_MARKERS = ("[removed]", "[deleted]", "[removed by reddit]")


# ================================================================ config helpers
def ingestion_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["ingestion"]


def corpus_spec(cfg: dict[str, Any], corpus: str | None = None) -> tuple[str, dict[str, Any]]:
    """(corpus name, its config block). `corpus` defaults to ingestion.active_corpus."""
    ing = ingestion_settings(cfg)
    name = corpus or ing["active_corpus"]
    if name not in ing["corpora"]:
        raise KeyError(f"ingestion.corpora has no {name!r} (have {sorted(ing['corpora'])})")
    return name, ing["corpora"][name]


def lake_root(cfg: dict[str, Any], corpus: str | None = None) -> Path:
    name, _ = corpus_spec(cfg, corpus)
    return Path(cfg["_paths"]["lake"]) / name


def date_range(cfg: dict[str, Any], corpus: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """DATE_RANGE as UTC [start, end) timestamps, or None. 'YYYY-MM' bounds: the end month is INCLUDED."""
    _, spec = corpus_spec(cfg, corpus)
    dr = spec.get("date_range")
    if not dr:
        return None
    lo = pd.Timestamp(str(dr["start"]), tz="UTC")
    hi = pd.Timestamp(str(dr["end"]), tz="UTC")
    hi = hi + (pd.offsets.MonthBegin(1) if len(str(dr["end"])) <= 7 else pd.Timedelta(days=1))
    return lo, hi


# ================================================================ source adapters
class SourceAdapter(abc.ABC):
    """Yields frames with RAW_COLUMNS (UTC created_at, text_raw not yet cleaned). Never yields an author/user column.
    `stats` (a Counter owned by the caller, one per input file) receives `rows_read` and `drop:<reason>` counts."""

    name = "base"
    source = "base"

    def __init__(self, cfg: dict[str, Any], corpus: str | None = None):
        self.cfg = cfg
        self.corpus = corpus

    @abc.abstractmethod
    def inputs(self) -> list[Path]:
        """Input files in a stable order (the order decides which duplicate is kept: the first seen)."""

    @abc.abstractmethod
    def iter_chunks(self, path: Path, chunk_rows: int, stats: Counter) -> Iterator[pd.DataFrame]:
        """Stream one input file as frames of at most `chunk_rows` rows."""

    @staticmethod
    def _frame(rows: dict[str, list]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=RAW_COLUMNS)

    @staticmethod
    def _frame_epoch(rows: dict[str, list]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=RAW_COLUMNS)
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s", utc=True)
        return df


class ZenodoPilotAdapter(SourceAdapter):
    """Zenodo Reddit Mental Health pilot CSVs (<subreddit>_<window>_features_tfidf_256.csv). Reads ONLY the whitelisted columns
    (subreddit, date, post): `author` and the ~100 feature columns are never loaded. Windows are read `pre` before `post`, so
    the global exact dedupe keeps the earlier window's copy of any text that appears in both (e.g. the overlapping 2019 data)."""

    name = "zenodo_pilot"

    def __init__(self, cfg: dict[str, Any], corpus: str | None = None):
        super().__init__(cfg, corpus)
        self.src = cfg["sources"]["zenodo_pilot"]
        self.cols = list(self.src["load_columns"])
        if author_columns(self.cols, cfg):
            raise AssertionError("zenodo load_columns must not include author fields")
        self.source = "zenodo_pilot"

    def inputs(self) -> list[Path]:
        zdir = Path(self.cfg["_root"]) / self.src["local_dir"]
        files = []
        for sub in self.src["subreddits"]:
            for win in self.src["windows"]:
                p = zdir / self.src["file_pattern"].format(subreddit=sub, window=win)
                if p.exists():
                    files.append(p)
        return files

    def iter_chunks(self, path: Path, chunk_rows: int, stats: Counter) -> Iterator[pd.DataFrame]:
        s = self.src
        for raw in pd.read_csv(path, usecols=self.cols, dtype=str, keep_default_na=False, encoding="utf-8", chunksize=chunk_rows):
            stats["rows_read"] += len(raw)
            created = pd.to_datetime(raw[s["date_column"]], format="%Y/%m/%d", utc=True, errors="coerce")
            sub = raw["subreddit"].str.lower()
            ids = [stable_record_id("zenodo", f"{c}|{d}|{hashlib.sha1(t.encode('utf-8')).hexdigest()}")
                   for c, d, t in zip(raw["subreddit"], raw[s["date_column"]], raw[s["text_column"]])]   # CSV case, as in Step 2 ids
            yield pd.DataFrame({"record_id": ids, "community": sub, "created_at": created, "text_raw": raw[s["text_column"]]})


class RedditSubmissionAdapter(SourceAdapter):
    """zstandard-compressed ndjson Reddit SUBMISSION dumps (one JSON object per line), streamed line by line: a file is never
    loaded whole. Also reads plain .ndjson / .jsonl. Filters to COMMUNITIES and DATE_RANGE; text = title + selftext.

    Field names used: id, subreddit, created_utc, title, selftext, removed_by_category, author (bot check only, then discarded).
    UNVERIFIED against a real Arctic Shift / Pushshift file: when you have one, run `python -m stress_signals.ingestion inspect <file>`
    and compare. A missing field is counted as a drop (`drop:missing_field`), never a crash."""

    name = "reddit_submissions"
    source = "reddit_submission"

    def __init__(self, cfg: dict[str, Any], corpus: str | None = None):
        super().__init__(cfg, corpus)
        self.src = cfg["sources"]["main_corpus"]
        self.communities = {c.lower() for c in self.src["subreddits"]} - {c.lower() for c in self.src.get("excluded_subreddits", [])}
        self.window = date_range(cfg, corpus)
        self.bots = {b.lower() for b in cfg["ingestion"].get("bot_authors", [])}

    def inputs(self) -> list[Path]:
        root = Path(self.cfg["_root"]) / self.src["local_dir"]
        files: list[str] = []
        for pat in self.src.get("file_globs", ["*.zst", "*.ndjson", "*.jsonl"]):
            files += glob.glob(str(root / "**" / pat), recursive=True)
        return sorted(Path(f) for f in set(files))

    @staticmethod
    def open_lines(path: Path) -> Iterator[str]:
        """Text lines of a .zst / plain file without loading it. Reddit dumps need a big zstd window (2**31)."""
        if path.suffix == ".zst":
            import zstandard
            with open(path, "rb") as fh, zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(fh) as rd:
                yield from io.TextIOWrapper(rd, encoding="utf-8", errors="replace", newline="\n")
        else:
            with open(path, "r", encoding="utf-8", errors="replace", newline="\n") as fh:
                yield from fh

    def iter_chunks(self, path: Path, chunk_rows: int, stats: Counter) -> Iterator[pd.DataFrame]:
        lo_hi = None if self.window is None else (int(self.window[0].timestamp()), int(self.window[1].timestamp()))
        rows: dict[str, list] = {c: [] for c in RAW_COLUMNS}
        for line in self.open_lines(path):
            line = line.strip()
            if not line:
                continue
            stats["rows_read"] += 1
            try:
                o = json.loads(line)
            except ValueError:
                stats["drop:malformed_json"] += 1
                continue
            if not isinstance(o, dict):
                stats["drop:malformed_json"] += 1
                continue
            sub = str(o.get("subreddit") or "").lower()
            if sub not in self.communities:
                stats["drop:not_in_communities"] += 1
                continue
            try:
                ts = int(float(o["created_utc"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                stats["drop:missing_timestamp"] += 1
                continue
            if lo_hi and not (lo_hi[0] <= ts < lo_hi[1]):
                stats["drop:outside_date_range"] += 1
                continue
            if str(o.get("author") or "").lower() in self.bots:      # the only use of `author`; the value is not kept
                stats["drop:bot_account"] += 1
                continue
            if o.get("removed_by_category"):
                stats["drop:removed_post"] += 1
                continue
            body = o.get("selftext")
            if not isinstance(body, str):
                stats["drop:missing_field"] += 1
                continue
            if body.strip().lower() in REMOVED_MARKERS:
                stats["drop:removed_or_deleted"] += 1
                continue
            if not body.strip():
                stats["drop:empty_selftext"] += 1
                continue
            nid = o.get("id")
            if not nid:
                stats["drop:missing_field"] += 1
                continue
            rows["record_id"].append(stable_record_id(self.source, nid))
            rows["community"].append(sub)
            rows["created_at"].append(ts)
            rows["text_raw"].append(f"{o.get('title') or ''}\n\n{body}".strip())
            if len(rows["record_id"]) >= chunk_rows:
                yield self._frame_epoch(rows)
                rows = {c: [] for c in RAW_COLUMNS}
        if rows["record_id"]:
            yield self._frame_epoch(rows)


class GenericAdapter(SourceAdapter):
    """Any CSV / JSONL you downloaded. Config `sources.generic_ingest`: files (globs under local_dir), format (csv | jsonl),
    columns {id, community, created, text, text2?}, created_format (epoch_s | epoch_ms | iso | a strptime pattern),
    source (name), default_community. ONLY the mapped columns are read, so author/user columns are never loaded."""

    name = "generic"

    def __init__(self, cfg: dict[str, Any], corpus: str | None = None):
        super().__init__(cfg, corpus)
        self.g = cfg["sources"]["generic_ingest"]
        self.source = self.g.get("source", "generic")
        self.map = self.g["columns"]
        bad = author_columns(self.map.values(), cfg)
        if bad:
            raise AssertionError(f"generic_ingest.columns must not map author fields: {bad}")

    def inputs(self) -> list[Path]:
        root = Path(self.cfg["_root"]) / self.g["local_dir"]
        return sorted(Path(f) for pat in self.g["files"] for f in glob.glob(str(root / pat), recursive=True))

    def _created(self, s: pd.Series) -> pd.Series:
        fmt = self.g.get("created_format", "iso")
        if fmt in ("epoch_s", "epoch_ms"):
            return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s" if fmt == "epoch_s" else "ms", utc=True, errors="coerce")
        return pd.to_datetime(s, utc=True, errors="coerce", **({} if fmt == "iso" else {"format": fmt}))

    def iter_chunks(self, path: Path, chunk_rows: int, stats: Counter) -> Iterator[pd.DataFrame]:
        m = self.map
        want = [v for v in m.values() if v]
        if self.g["format"] == "csv":
            it = pd.read_csv(path, usecols=want, dtype=str, keep_default_na=False, encoding="utf-8", chunksize=chunk_rows)
        else:
            def _jsonl() -> Iterator[pd.DataFrame]:
                buf: list[dict] = []
                for line in RedditSubmissionAdapter.open_lines(path):
                    if not line.strip():
                        continue
                    try:
                        o = json.loads(line)
                        buf.append({k: ("" if o.get(k) is None else str(o.get(k))) for k in want})
                    except (ValueError, AttributeError):
                        stats["rows_read"] += 1
                        stats["drop:malformed_json"] += 1
                    if len(buf) >= chunk_rows:
                        yield pd.DataFrame(buf)
                        buf = []
                if buf:
                    yield pd.DataFrame(buf)
            it = _jsonl()
        for raw in it:
            stats["rows_read"] += len(raw)
            text = raw[m["text"]] if not m.get("text2") else raw[m["text"]].str.cat(raw[m["text2"]], sep="\n\n")
            comm = raw[m["community"]].str.lower() if m.get("community") else self.g.get("default_community", "unknown")
            ids = raw[m["id"]].map(lambda x: stable_record_id(self.source, x)) if m.get("id") else \
                text.map(lambda t: stable_record_id(self.source, hashlib.sha1(t.encode("utf-8")).hexdigest()))
            yield pd.DataFrame({"record_id": ids, "community": comm, "created_at": self._created(raw[m["created"]]), "text_raw": text})


ADAPTERS: dict[str, type[SourceAdapter]] = {"zenodo_pilot": ZenodoPilotAdapter, "reddit_submissions": RedditSubmissionAdapter,
                                            "generic": GenericAdapter}


def build_adapter(cfg: dict[str, Any], corpus: str | None = None) -> SourceAdapter:
    name, spec = corpus_spec(cfg, corpus)
    if spec["adapter"] not in ADAPTERS:
        raise KeyError(f"unknown adapter {spec['adapter']!r}; have {sorted(ADAPTERS)}")
    return ADAPTERS[spec["adapter"]](cfg, name)


# ================================================================ chunk processing (Step 2 filters at scale)
def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.sha1(dedupe_key(text).encode("utf-8")).digest()[:8], "little", signed=True)


def process_chunk(raw: pd.DataFrame, cfg: dict[str, Any], source: str, window: tuple[pd.Timestamp, pd.Timestamp] | None,
                  lid: LanguageIdentifier | None, seen: set[int], new: set[int], stats: Counter) -> pd.DataFrame:
    """Raw frame -> lake frame. Every dropped row is counted in `stats` as `drop:<reason>`; `edits:*` counts cleaning edits."""
    df = raw
    n0 = len(df)
    df = df[df["created_at"].notna()]
    stats["drop:missing_timestamp"] += n0 - len(df)
    if window is not None and len(df):
        n0 = len(df)
        df = df[(df["created_at"] >= window[0]) & (df["created_at"] < window[1])]
        stats["drop:outside_date_range"] += n0 - len(df)
    if df.empty:
        return pd.DataFrame(columns=[f.name for f in LAKE_SCHEMA])
    edits: Counter = Counter()
    df = df.assign(text_clean=[clean_text(t, cfg, edits) for t in df["text_raw"]]).drop(columns="text_raw")
    for k, v in edits.items():
        stats[f"edits:{k}"] += v
    n0 = len(df)
    df = df[df["text_clean"].str.len() > 0]
    stats["drop:empty_or_removed_after_cleaning"] += n0 - len(df)
    n0 = len(df)
    df = df[~df["text_clean"].map(lambda t: is_bot_text(t, cfg))]
    stats["drop:bot_text"] += n0 - len(df)
    n0 = len(df)
    min_tok = int(cfg["preprocess"]["min_tokens"])
    df = df[df["text_clean"].map(n_tokens_estimate) >= min_tok]
    stats[f"drop:shorter_than_{min_tok}_tokens"] += n0 - len(df)
    if lid is not None and len(df):
        codes, _ = lid.detect(df["text_clean"].tolist())
        df = df.assign(lang=codes)
        n0 = len(df)
        df = df[df["lang"] == "en"]
        stats["drop:non_english"] += n0 - len(df)
    else:
        df = df.assign(lang=None)
    n0 = len(df)
    keep = []
    for t in df["text_clean"]:
        h = _hash64(t)
        if h in seen:
            keep.append(False)
        else:
            seen.add(h)
            new.add(h)
            keep.append(True)
    df = df[np.asarray(keep, dtype=bool)] if len(df) else df
    stats["drop:exact_duplicate_text"] += n0 - len(df)
    if df.empty:
        return pd.DataFrame(columns=[f.name for f in LAKE_SCHEMA])
    out = df.assign(source=source, n_chars=df["text_clean"].str.len().astype("int32"),
                    n_tokens_est=df["text_clean"].map(n_tokens_estimate).astype("int32"))
    out = out[[f.name for f in LAKE_SCHEMA]].reset_index(drop=True)
    assert_privacy_safe(out, cfg)
    return out


# ================================================================ lake writing
def _part_dir(root: Path, source: str, year: int, month: int) -> Path:
    return root / f"source={source}" / f"year={year:04d}" / f"month={month:02d}"


def _write_part(path: Path, df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    ing = ingestion_settings(cfg)
    tbl = pa.Table.from_pandas(df[[f.name for f in LAKE_SCHEMA]], schema=LAKE_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp, compression=ing.get("compression", "zstd"), row_group_size=int(ing.get("row_group_rows", 100_000)))
    os.replace(tmp, path)


def _flush(buffers: dict[tuple[str, int, int], list[pd.DataFrame]], root: Path, prefix: str, counter: list[int], cfg: dict[str, Any]) -> list[str]:
    written = []
    for (src, y, m), frames in sorted(buffers.items()):
        p = _part_dir(root, src, y, m) / f"part-{prefix}-{counter[0]:05d}.parquet"
        counter[0] += 1
        _write_part(p, pd.concat(frames, ignore_index=True), cfg)
        written.append(p.relative_to(root).as_posix())
    buffers.clear()
    return written


def _input_key(p: Path) -> str:
    return hashlib.sha1(p.name.encode("utf-8")).hexdigest()[:8]


def _sig(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {"name": p.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _state_paths(root: Path, p: Path) -> tuple[Path, Path]:
    d = root / "_state"
    return d / f"{_input_key(p)}.json", d / f"{_input_key(p)}.npy"


def _ingest_one(adapter: SourceAdapter, path: Path, cfg: dict[str, Any], root: Path, window, lid, seen: set[int]) -> dict[str, Any]:
    ing = ingestion_settings(cfg)
    prefix = _input_key(path)
    for old in root.glob(f"source=*/year=*/month=*/part-{prefix}-*.parquet*"):      # an interrupted earlier attempt
        old.unlink()
    stats: Counter = Counter()
    new_hashes: set[int] = set()
    buffers: dict[tuple[str, int, int], list[pd.DataFrame]] = {}
    buffered, counter, parts = 0, [0], []
    for raw in adapter.iter_chunks(path, int(ing["chunk_rows"]), stats):
        out = process_chunk(raw, cfg, adapter.source, window, lid, seen, new_hashes, stats)
        if len(out):
            ts = out["created_at"]
            for (y, m), g in out.groupby([ts.dt.year, ts.dt.month]):
                buffers.setdefault((adapter.source, int(y), int(m)), []).append(g)
            buffered += len(out)
            stats["rows_out"] += len(out)
        if buffered >= int(ing["flush_rows"]):
            parts += _flush(buffers, root, prefix, counter, cfg)
            buffered = 0
    parts += _flush(buffers, root, prefix, counter, cfg)
    jp, npy = _state_paths(root, path)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, np.fromiter(new_hashes, dtype=np.int64, count=len(new_hashes)))
    state = {"input": _sig(path), "stats": dict(stats), "parts": parts, "done": True}
    write_json(jp, state)          # written LAST: its presence means the input is complete
    return state


def _balanced(stats: dict[str, int]) -> bool:
    return stats.get("rows_read", 0) - sum(v for k, v in stats.items() if k.startswith("drop:")) == stats.get("rows_out", 0)


def compact_lake(cfg: dict[str, Any], corpus: str | None = None) -> dict[str, int]:
    """Merge the small per-input part files of each partition into files of <= ingestion.target_rows_per_file rows."""
    root, target = lake_root(cfg, corpus), int(ingestion_settings(cfg)["target_rows_per_file"])
    merged = {"partitions_compacted": 0, "files_before": 0, "files_after": 0}
    for d in sorted({p.parent for p in root.glob("source=*/year=*/month=*/*.parquet")}):
        files = sorted(d.glob("*.parquet"))
        if len(files) < 2:
            continue
        tbl = pa.concat_tables([pq.read_table(f, schema=LAKE_SCHEMA) for f in files])
        tbl = tbl.sort_by([("created_at", "ascending"), ("record_id", "ascending")])
        new = []
        for i, off in enumerate(range(0, tbl.num_rows, target)):
            tmp = d / f"compact-{i:05d}.parquet.tmp"
            pq.write_table(tbl.slice(off, target), tmp, compression=ingestion_settings(cfg).get("compression", "zstd"),
                           row_group_size=int(ingestion_settings(cfg).get("row_group_rows", 100_000)))
            new.append(tmp)
        for f in files:
            f.unlink()
        for tmp in new:
            os.replace(tmp, tmp.with_suffix(""))
        merged["partitions_compacted"] += 1
        merged["files_before"] += len(files)
        merged["files_after"] += len(new)
    return merged


# ================================================================ driver + manifest
def manifest_path(cfg: dict[str, Any], corpus: str | None = None) -> Path:
    name, _ = corpus_spec(cfg, corpus)
    return Path(cfg["_paths"]["manifests"]) / f"step8_ingestion_{name}.json"


def ingest(cfg: dict[str, Any], corpus: str | None = None, force: bool = False, lid: LanguageIdentifier | None = None,
           adapter: SourceAdapter | None = None) -> dict[str, Any]:
    """Run (or resume) the ingestion of one corpus and write its manifest. Inputs already finished (same name, size and mtime)
    are skipped; a changed or unfinished input is redone. `force=True` wipes the corpus' lake first."""
    name, spec = corpus_spec(cfg, corpus)
    ing = ingestion_settings(cfg)
    root = lake_root(cfg, name)
    adapter = adapter or build_adapter(cfg, name)
    files = adapter.inputs()
    if not files:
        raise FileNotFoundError(
            f"adapter {adapter.name!r} found no input files for corpus {name!r}. Download them yourself (no scraping here) and check "
            f"the paths in config.yaml sources.*. Remember Reddit's terms, the dataset licence and your ethics requirements.")
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    window = date_range(cfg, name)
    if lid is None and ing.get("language_filter", True):
        lid = LanguageIdentifier(cfg)
    seen: set[int] = set()
    states: dict[str, dict[str, Any]] = {}
    redo = False
    for p in files:
        jp, npy = _state_paths(root, p)
        st = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else None
        if st and st.get("done") and st["input"] == _sig(p) and not redo:
            seen.update(np.load(npy).tolist())
            states[p.name] = st
            LOG.info("ingest %s: SKIP %s (already done)", name, p.name)
            continue
        redo = True      # a later input's duplicates depend on this one: keep order, redo from here on
        LOG.info("ingest %s: %s", name, p.name)
        states[p.name] = _ingest_one(adapter, p, cfg, root, window, lid, seen)
        LOG.info("   %s", {k: v for k, v in states[p.name]["stats"].items() if not k.startswith("edits:")})
    compact = compact_lake(cfg, name) if ing.get("compact", True) else {}
    return write_manifest_for(cfg, name, adapter, states, compact)


def write_manifest_for(cfg: dict[str, Any], corpus: str, adapter: SourceAdapter, states: dict[str, dict[str, Any]],
                       compact: dict[str, int]) -> dict[str, Any]:
    root = lake_root(cfg, corpus)
    total: Counter = Counter()
    per_input = {}
    for fname, st in states.items():
        total.update(st["stats"])
        s = st["stats"]
        per_input[fname] = {"rows_in": s.get("rows_read", 0), "rows_out": s.get("rows_out", 0),
                            "drops": {k[5:]: v for k, v in s.items() if k.startswith("drop:") and v}, "balanced": _balanced(s)}
    partitions = {}
    for d in sorted({p.parent for p in root.glob("source=*/year=*/month=*/*.parquet")}):
        fl = sorted(d.glob("*.parquet"))
        partitions[d.relative_to(root).as_posix()] = {
            "rows": int(sum(pq.ParquetFile(f).metadata.num_rows for f in fl)), "files": len(fl),
            "bytes": int(sum(f.stat().st_size for f in fl)), "sha256": {f.name: sha256_file(f) for f in fl}}
    rows_out = sum(v["rows"] for v in partitions.values())
    manifest = {
        "corpus": corpus, "adapter": adapter.name, "source": adapter.source, "lake": root.as_posix(),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "date_range": None if date_range(cfg, corpus) is None else [str(x) for x in date_range(cfg, corpus)],
        "rows_in": int(total.get("rows_read", 0)), "rows_out": int(rows_out),
        "drops_by_reason": {k[5:]: int(v) for k, v in sorted(total.items()) if k.startswith("drop:") and v},
        "clean_edits": {k[6:]: int(v) for k, v in sorted(total.items()) if k.startswith("edits:")},
        "balanced": bool(total.get("rows_read", 0) - sum(v for k, v in total.items() if k.startswith("drop:")) == rows_out),
        "per_input": per_input, "partitions": partitions, "compaction": compact,
        "schema": {f.name: str(f.type) for f in LAKE_SCHEMA}, "partitioning": "source / year / month (hive style)",
        "language_filter": bool(ingestion_settings(cfg).get("language_filter", True)),
        "privacy": {"author_fields": "never kept (reddit adapter reads `author` only to match bot accounts)", "k_min": cfg["privacy"]["k_min"]},
        "library_versions": library_versions(), "seed": cfg["primary_seed"],
        "licence_reminder": "Check Reddit's terms of use, the dataset licence and your institution's ethics requirements before use."}
    write_json(manifest_path(cfg, corpus), manifest)
    if not manifest["balanced"]:
        raise AssertionError(f"ingestion row accounting does not balance: in {manifest['rows_in']} - drops != out {rows_out}")
    return manifest


def read_lake(cfg: dict[str, Any], corpus: str | None = None) -> pds.Dataset:
    return pds.dataset(lake_root(cfg, corpus), format="parquet", partitioning="hive", ignore_prefixes=["_", "."])


# ================================================================ data quality, sampling frame, 1-day sample
def _suppress(df: pd.DataFrame, col: str, k_min: int) -> pd.DataFrame:
    df = df.copy()
    df["suppressed"] = df[col] < k_min
    df[col] = df[col].where(~df["suppressed"], pd.NA).astype("Int64")
    return df


def data_quality(cfg: dict[str, Any], corpus: str | None = None) -> dict[str, Any]:
    """Volume per day and community, duplicate rate, missing timestamps, text-length distribution. Aggregates only; any cell with
    n < privacy.k_min is suppressed (blank + suppressed=True) in the saved CSVs. Writes data_quality_<corpus>.md next to the reports."""
    name, _ = corpus_spec(cfg, corpus)
    k = int(cfg["privacy"]["k_min"])
    man = json.loads(manifest_path(cfg, name).read_text(encoding="utf-8"))
    ds = read_lake(cfg, name)
    daily: Counter = Counter()
    lens: list[np.ndarray] = []
    for b in ds.to_batches(columns=["community", "created_at", "n_chars"], batch_size=250_000):
        t = b.to_pandas()
        day = t["created_at"].dt.strftime("%Y-%m-%d")
        daily.update(zip(day, t["community"]))
        lens.append(t["n_chars"].to_numpy())
    L = np.concatenate(lens) if lens else np.array([], dtype=int)
    vol = pd.DataFrame([(d, c, n) for (d, c), n in daily.items()], columns=["day", "community", "n"]).sort_values(["community", "day"])
    mon = vol.assign(month=vol["day"].str[:7]).groupby(["month", "community"], as_index=False)["n"].sum()
    com = vol.groupby("community", as_index=False)["n"].sum()
    out_dir = Path(cfg["_paths"]["reports"]) / f"data_quality_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _suppress(vol, "n", k).to_csv(out_dir / "volume_by_day_community.csv", index=False)
    _suppress(mon, "n", k).to_csv(out_dir / "volume_by_month_community.csv", index=False)
    pct = {f"p{q}": float(np.percentile(L, q)) for q in (1, 5, 25, 50, 75, 95, 99)} if len(L) else {}
    hist_edges = [0, 100, 250, 500, 1000, 2000, 5000, 10000, 20001, np.inf]
    hist = pd.cut(pd.Series(L), bins=hist_edges, right=False).value_counts(sort=False)
    d = man["drops_by_reason"]
    kept_before_dedupe = man["rows_out"] + d.get("exact_duplicate_text", 0)
    dup_rate = d.get("exact_duplicate_text", 0) / kept_before_dedupe if kept_before_dedupe else None
    days = pd.to_datetime(vol["day"].unique()) if len(vol) else pd.DatetimeIndex([])
    span = (days.max() - days.min()).days + 1 if len(days) else 0
    res = {"corpus": name, "rows": int(man["rows_out"]), "rows_in": man["rows_in"], "missing_timestamp": d.get("missing_timestamp", 0),
           "duplicate_rate_exact": dup_rate, "length_chars": {"mean": float(L.mean()) if len(L) else None, "min": int(L.min()) if len(L) else None,
                                                             "max": int(L.max()) if len(L) else None, **pct},
           "days_with_data": int(len(days)), "calendar_span_days": int(span), "communities": int(len(com)),
           "cells_suppressed_day_community": int((vol["n"] < k).sum()), "cells_total_day_community": int(len(vol)), "k_min": k}
    L2 = [f"# Data quality report: corpus `{name}`", "",
          "Aggregate counts only. No text, no identifiers. Cells with n < k_min are suppressed in the CSVs "
          f"(k_min = {k}); the tables below show only communities and months at or above k_min.", "",
          f"- rows in (raw): **{man['rows_in']:,}** | rows out (lake): **{man['rows_out']:,}** | row accounting balanced: **{man['balanced']}**",
          f"- missing / unparseable timestamps dropped: **{res['missing_timestamp']:,}**",
          f"- exact duplicate rate (duplicates dropped / rows reaching the dedupe step): "
          f"**{'n/a' if dup_rate is None else f'{dup_rate:.2%}'}** (near-duplicates are NOT removed)",
          f"- days with data: {res['days_with_data']} of a {res['calendar_span_days']}-day span | communities: {res['communities']}",
          f"- day x community cells suppressed (n < {k}): {res['cells_suppressed_day_community']} of {res['cells_total_day_community']}", "",
          "## Rows dropped, by reason", "", "| reason | rows |", "|---|---|", *[f"| {r} | {n:,} |" for r, n in d.items()], "",
          "## Volume per community", "", "| community | rows |", "|---|---|",
          *[f"| {r.community} | {int(r.n):,} |" for r in com.itertuples() if r.n >= k], "",
          "## Text length (characters of the cleaned text)", "",
          "| " + " | ".join(["mean", "min", *pct.keys(), "max"]) + " |", "|" + "---|" * (len(pct) + 3),
          "| " + " | ".join(f"{v:,.0f}" for v in [res["length_chars"]["mean"] or 0, res["length_chars"]["min"] or 0, *pct.values(),
                                                  res["length_chars"]["max"] or 0]) + " |", "",
          "| length bin (chars) | rows |", "|---|---|", *[f"| {str(i)} | {int(n):,} |" for i, n in hist.items()], "",
          "## Notes", "",
          "- Per-day and per-month volumes by community are in `" + out_dir.name + "/*.csv` (suppressed cells blank).",
          "- The pilot `date` column has DAY resolution only; time of day is not available for it."]
    (Path(cfg["_paths"]["reports"]) / f"data_quality_{name}.md").write_text("\n".join(L2) + "\n", encoding="utf-8")
    write_json(out_dir / "summary.json", res)
    return res


def write_sampling_frame(cfg: dict[str, Any]) -> Path:
    """sampling_frame.md: which communities, why, what is excluded (incl. why suicide-related communities), and the selection bias.
    Counts are read from the ingestion manifests that exist; nothing is estimated."""
    zp, mc = cfg["sources"]["zenodo_pilot"], cfg["sources"]["main_corpus"]
    rows = []
    for corpus in cfg["ingestion"]["corpora"]:
        mp = manifest_path(cfg, corpus)
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            rows.append(f"- `{corpus}`: {m['rows_in']:,} rows read, {m['rows_out']:,} kept (manifest {mp.name}, {m['created_utc']})")
        else:
            rows.append(f"- `{corpus}`: not ingested yet (no manifest)")
    excl = ", ".join(f"r/{c}" for c in mc.get("excluded_subreddits", []))
    txt = f"""# Sampling frame and selection bias

*Generated from config.yaml by `stress_signals.ingestion.write_sampling_frame`. Counts come from the ingestion manifests only.*

**Aggregate, population-level language signals only. No diagnosis. No individual-level analysis.**

## What is in the frame

| corpus | source | communities | period |
|---|---|---|---|
| pilot | Zenodo Reddit Mental Health Dataset, record {zp['record_id']} (files you downloaded) | {', '.join('r/' + c for c in zp['subreddits'])} | `pre` {zp['window_dates']['pre'][0]} to {zp['window_dates']['pre'][1]}; `post` {zp['window_dates']['post'][0]} to {zp['window_dates']['post'][1]} (observed on disk, not the record's description) |
| main | Reddit SUBMISSIONS (posts) from per-subreddit dumps / Arctic Shift (files you downloaded) | {', '.join('r/' + c for c in mc['subreddits'])} | {mc['start']} to {mc['end']} |

Ingestion status:

{chr(10).join(rows)}

Why these communities (user decisions, Step 1): the pilot communities are kept in the main corpus so the pilot is comparable, plus r/legaladvice
and r/unemployment as communities where stressors such as legal and job problems are discussed. Other everyday-stress communities could
have been chosen; this is a convenience-and-comparability sample, not a probability sample of Reddit or of any population.

## What is excluded, and why

- **Suicide-related communities ({excl or 'none configured'})**: excluded by design. They hold posts from people who may be in acute crisis.
  Even aggregate monitoring of them invites misuse (targeting, outreach to individuals, re-identification of small groups), the
  project must not produce individual-level output, and small communities make low-count cells hard to protect. The project is a
  language-signal study of everyday pressure, not a crisis-detection system. **Consequence:** the most severe end of distress is
  under-represented; results must not be read as covering it.
- Posts removed or deleted (`[removed]`, `[deleted]`, moderator-removed), empty selftext (link-only posts), bot / moderator text.
- Non-English text (language filter on, `ingestion.language_filter`); texts under {cfg['preprocess']['min_tokens']} tokens.
- Exact duplicate texts after cleaning (the first seen is kept). Near-duplicates are not removed.
- Comments (main corpus is submissions only), images, links, and every author / user field (dropped at ingestion).
- Outside DATE_RANGE (main) and communities not listed above.

## Selection bias and what it means for the results

- **Who posts:** only people who choose to post publicly in these communities. Reddit users are not a random sample of any population
  (demographics not measured here; UNVERIFIED), and each community has its own norms and rules.
- **Survivorship and moderation:** removed or deleted posts are missing, so what was removed (often the most acute or the rule-breaking)
  is not seen. Dump timing decides which posts had been removed when the archive was taken (UNVERIFIED for the files you hold).
- **Community mix drives totals:** pooled numbers are dominated by the largest communities; compare within a community over time.
- **Pilot windows are not contiguous** ({zp['window_dates']['pre'][1]} to {zp['window_dates']['post'][0]} is missing) and the post window
  starts at the onset of the pandemic, so a pre/post difference mixes season, pandemic and community change. The pilot `date` has day resolution only.
- **Text-only signal:** language is not a measurement of stress or mental health of anyone. Findings are stated as "stress-related language in the
  sampled communities shifted", never as the population or any person becoming more stressed.
- **Machine labelling:** models were validated partly against machine-made labels (see the model cards); that limits what the numbers mean.

## Before using real data

Check Reddit's terms of use and API/data policy, the dataset licence (Zenodo record {zp['record_id']}; the Arctic Shift terms), and your
institution's ethics requirements. This code reads local files only and never scrapes.
"""
    p = Path(cfg["_paths"]["reports"]) / "sampling_frame.md"
    p.write_text(txt, encoding="utf-8")
    return p


def make_day_sample(cfg: dict[str, Any], corpus: str | None = None, day: str | None = None) -> Path:
    """One calendar day of the lake as a single Parquet file for the Step 9 parity test (Spark vs plain pandas). `day` defaults to
    ingestion.sample.day or the busiest day. Keeps all that day's rows up to ingestion.sample.max_rows (lowest record_id first, deterministic).
    Contains cleaned text: LOCAL ONLY, never commit or export it."""
    name, _ = corpus_spec(cfg, corpus)
    s = ingestion_settings(cfg).get("sample", {})
    ds = read_lake(cfg, name)
    if day is None:
        day = s.get("day")
    if day is None:
        cnt: Counter = Counter()
        for b in ds.to_batches(columns=["created_at"], batch_size=500_000):
            cnt.update(b.to_pandas()["created_at"].dt.strftime("%Y-%m-%d"))
        if not cnt:
            raise ValueError("the lake is empty")
        day = max(sorted(cnt), key=lambda d: cnt[d])
    lo = pd.Timestamp(day, tz="UTC")
    tbl = ds.to_table(filter=(pds.field("created_at") >= lo.to_pydatetime()) & (pds.field("created_at") < (lo + pd.Timedelta(days=1)).to_pydatetime()),
                      columns=[f.name for f in LAKE_SCHEMA])
    df = tbl.to_pandas().sort_values("record_id").head(int(s.get("max_rows", 20000))).reset_index(drop=True)
    out_dir = Path(cfg["_paths"]["processed"]) / "lake_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{day}.parquet"
    _write_part(path, df, cfg)
    write_json(path.with_suffix(".json"), {"corpus": name, "day": day, "rows": int(len(df)), "sha256": sha256_file(path),
                                          "note": "contains cleaned text: local only"})
    return path


def inspect_file(path: str | Path, n: int = 3) -> dict[str, Any]:
    """Field names (no values) of the first records of a Reddit dump file: use it to verify the UNVERIFIED field names."""
    keys: Counter = Counter()
    seen = 0
    for line in RedditSubmissionAdapter.open_lines(Path(path)):
        if line.strip():
            keys.update(json.loads(line).keys())
            seen += 1
            if seen >= n:
                break
    return {"records_read": seen, "fields": sorted(keys),
            "needed": ["id", "subreddit", "created_utc", "title", "selftext"], "missing_needed": [k for k in ("id", "subreddit", "created_utc", "title", "selftext") if k not in keys]}


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Step 8 ingestion (local files only)")
    ap.add_argument("cmd", choices=["ingest", "quality", "frame", "sample", "inspect"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--corpus")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        print(json.dumps(inspect_file(a.arg), indent=2))
        return
    cfg = load_config()
    if a.cmd == "ingest":
        m = ingest(cfg, a.corpus, force=a.force)
        print(json.dumps({k: m[k] for k in ("corpus", "rows_in", "rows_out", "balanced", "drops_by_reason")}, indent=2))
    elif a.cmd == "quality":
        print(json.dumps(data_quality(cfg, a.corpus), indent=2, default=str))
    elif a.cmd == "frame":
        print(write_sampling_frame(cfg))
    else:
        print(make_day_sample(cfg, a.corpus, a.arg))


if __name__ == "__main__":
    main()
