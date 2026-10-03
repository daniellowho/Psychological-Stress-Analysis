"""Step 2: text cleaning, unified record schema, language ID, deduplication, chunking and fixed splits.

Unified schema (every processed parquet starts with these columns, then dataset-specific label columns):
    record_id, source, community, created_at (UTC), text_clean, n_chars, n_tokens_est, lang
No author/username/URL fields are ever produced. `text_clean` is PII-scrubbed (URLs -> <URL>, mentions -> <USER>,
emails/phone numbers removed). Processed parquets are local working data; they never go to gold/aggregate tables.

Run:  python -m stress_signals.preprocess [--only dreaddit goemotions sad zenodo senticnet] [--force] [--force-splits]
(from the project root with PYTHONPATH=code, or from code/ itself; all paths resolve against the project root)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import load_config
from .privacy import EMAIL_RE, MENTION_RE, PHONE_RE, URL_RE, author_columns
from .utils import get_logger, library_versions, sha256_file, write_json, write_manifest

LOG = get_logger("stress_signals.preprocess")

UNIFIED_COLUMNS = ["record_id", "source", "community", "created_at", "text_clean", "n_chars", "n_tokens_est", "lang"]
FORBIDDEN_COLUMN_PATTERNS = re.compile(r"(?i)^(author.*|user.*|username|screen_name|.*_url|url|permalink)$")
DATASETS = ("dreaddit", "goemotions", "sad", "zenodo", "senticnet")

# ================================================================ cleaning (transformer path)
ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿­]")
URL_PLACEHOLDER_RE = re.compile(r"<\s*url\s*>", re.I)                 # Dreaddit's own <url> placeholder (and ours)
MD_LINK_RE = re.compile(r"!?\[([^\]\n]*)\]\(\s*(?:[^()\s]|\([^()\s]*\))*\s*(?:\"[^\"]*\")?\)")   # [text](url "title")
MD_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
MD_QUOTE_RE = re.compile(r"(?m)^[ \t]*(?:>[ \t]?)+")
MD_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*+][ \t]+")
MD_RULE_RE = re.compile(r"(?m)^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")
MD_STRONG_RE = re.compile(r"(\*\*|__|~~)(?=\S)(.+?)(?<=\S)\1")
MD_EM_STAR_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])")
MD_EM_UNDERSCORE_RE = re.compile(r"(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])")
MD_CODE_RE = re.compile(r"`{1,3}([^`]*)`{1,3}")
MD_SUPERSCRIPT_RE = re.compile(r"\^\(([^)]*)\)|\^(?=\S)")
MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|^])")
WS_RE = re.compile(r"\s+")
PLACEHOLDER_RE = re.compile(r"<(?:URL|USER|EMAIL|PHONE)>")
TOKEN_EST_RE = re.compile(r"\w+|[^\w\s]")


def _privacy_tokens(cfg: dict[str, Any] | None) -> tuple[str, str]:
    p = (cfg or {}).get("privacy", {})
    return p.get("url_token", "<URL>"), p.get("mention_token", "<USER>")


def _removed_markers_re(cfg: dict[str, Any] | None) -> re.Pattern[str]:
    markers = ((cfg or {}).get("preprocess", {}) or {}).get("removed_markers") or ["[removed]", "[deleted]"]
    return _markers_re(tuple(markers))


@lru_cache(maxsize=8)
def _markers_re(markers: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(m) for m in markers), re.I)


def _clean_once(text: str, cfg: dict[str, Any] | None, stats: Counter | None) -> str:
    url_tok, user_tok = _privacy_tokens(cfg)

    def sub(regex: re.Pattern[str], repl: str | Callable[[re.Match[str]], str], s: str, key: str,
            trigger: str | None = None) -> str:
        if trigger is not None and not any(ch in s for ch in trigger):   # cheap skip: the regex cannot match
            return s
        out, n = regex.subn(repl, s)
        if stats is not None and n:
            stats[key] += n
        return out

    t = unicodedata.normalize("NFC", text)
    for _ in range(5):                                    # nested escapes such as &amp;#x200B; -> &#x200B; -> ZWSP
        un = html.unescape(t)
        if un == t:
            break
        if stats is not None:
            stats["html_unescape"] += 1
        t = un
    t = sub(ZERO_WIDTH_RE, "", t, "zero_width")
    t = sub(URL_PLACEHOLDER_RE, url_tok, t, "url_placeholder", "<")
    t = sub(_removed_markers_re(cfg), " ", t, "removed_marker", "[")
    # markdown (line-anchored rules first, while newlines still exist)
    t = sub(MD_LINK_RE, lambda m: f"{m.group(1)} {url_tok}", t, "md_link", "]")
    t = sub(MD_RULE_RE, " ", t, "md_rule", "-*_")
    t = sub(MD_HEADING_RE, "", t, "md_heading", "#")
    t = sub(MD_QUOTE_RE, "", t, "md_quote", ">")
    t = sub(MD_BULLET_RE, "", t, "md_bullet", "-*+")
    t = sub(MD_STRONG_RE, r"\2", t, "md_strong", "*_~")
    t = sub(MD_EM_STAR_RE, r"\1", t, "md_em", "*")
    t = sub(MD_EM_UNDERSCORE_RE, r"\1", t, "md_em", "_")
    t = sub(MD_CODE_RE, r"\1", t, "md_code", "`")
    t = sub(MD_SUPERSCRIPT_RE, lambda m: m.group(1) or "", t, "md_superscript", "^")
    t = sub(MD_ESCAPE_RE, r"\1", t, "md_escape", "\\")
    # PII: emails before URLs (domains look like URLs) and before mentions (@domain looks like a mention)
    t = sub(EMAIL_RE, " ", t, "email_removed", "@")
    t = sub(URL_RE, url_tok, t, "url", ".:")
    t = sub(MENTION_RE, user_tok, t, "mention", "@/")
    t = sub(PHONE_RE, " ", t, "phone_removed", "0123456789")
    return WS_RE.sub(" ", t).strip()


def clean_text(text: Any, cfg: dict[str, Any] | None = None, stats: Counter | None = None) -> str:
    """Clean text for transformer models: unescape HTML, strip markdown/zero-width chars, scrub PII, collapse whitespace.

    Keeps casing, punctuation and emojis. URLs (incl. Dreaddit's `<url>`) -> <URL>, @x / u/x -> <USER>, emails and
    phone numbers removed, "[removed]"/"[deleted]" markers removed. All whitespace (incl. newlines) collapses to one
    space, which matches the Dreaddit training text (it has no newlines). Applied to a fixed point, so it is
    idempotent: clean_text(clean_text(x)) == clean_text(x). Non-strings return "". `stats` (optional) counts edits.
    """
    if not isinstance(text, str):
        return ""
    prev, cur = None, text
    for _ in range(4):
        if cur == prev:
            break
        prev, cur = cur, _clean_once(cur, cfg, stats)
    return cur


def is_bot_text(text: str, cfg: dict[str, Any] | None = None) -> bool:
    """True if a (cleaned) text matches any configured bot/moderator pattern (case-insensitive regex search)."""
    pats = ((cfg or {}).get("preprocess", {}) or {}).get("bot_patterns") or ["i am a bot", "this action was performed automatically"]
    return any(_compiled(p).search(text) for p in pats)


@lru_cache(maxsize=64)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


def n_tokens_estimate(text: str) -> int:
    """Tokenizer-independent length estimate: word runs + punctuation marks (NOT model tokens)."""
    return len(TOKEN_EST_RE.findall(text))


# ================================================================ cleaning (lexical path)
LEX_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
NEGATIONS = frozenset({"no", "not", "never", "nor", "nothing", "nobody", "none", "neither", "nowhere", "cannot"})


@lru_cache(maxsize=4)
def _stopwords(keep_negations: bool, extra: tuple[str, ...]) -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    sw = set(ENGLISH_STOP_WORDS) | set(extra)
    sw |= {w.replace("'", "") for w in sw}
    return frozenset(sw - NEGATIONS if keep_negations else sw)


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    import simplemma
    return simplemma.lemmatize(token, lang="en").lower()


def clean_text_lexical(text: Any, cfg: dict[str, Any] | None = None, already_clean: bool = False) -> str:
    """Lexical normalisation for TF-IDF / topic labelling / keyword features: clean_text, drop placeholders,
    lowercase, strip apostrophes, remove stop words (negations kept by default), lemmatise (simplemma, English).
    Returns space-joined tokens. NOT for transformer input."""
    lex = ((cfg or {}).get("preprocess", {}) or {}).get("lexical", {}) or {}
    keep_neg = bool(lex.get("keep_negations", True))
    sw = _stopwords(keep_neg, tuple(lex.get("extra_stopwords", ())))
    min_len = int(lex.get("min_token_len", 2))
    t = text if already_clean and isinstance(text, str) else clean_text(text, cfg)
    t = PLACEHOLDER_RE.sub(" ", t).lower().replace("’", "'")
    out = []
    for tok in LEX_TOKEN_RE.findall(t):
        tok = tok.replace("'", "")
        if len(tok) < min_len or tok in sw:
            continue
        lem = _lemma(tok)
        if lem in sw and lem not in NEGATIONS:
            continue
        out.append(lem)
    return " ".join(out)


# ================================================================ language ID
class LanguageIdentifier:
    """English filter backed by lingua-language-detector.

    Why lingua: deterministic (langdetect is random unless seeded), fully offline with models inside the pip wheel
    (fastText lid.176 needs a separate model download + native build, awkward on Windows and Spark workers), and
    its authors report higher accuracy than langdetect/langid/fastText on short texts (author benchmark: UNVERIFIED
    by us). Cost: ~1 GB RAM with all language models loaded, so build ONE detector per process.
    A text is English if P(en) >= preprocess.lang_id.min_confidence (P is lingua's relative confidence over all
    supported languages). For non-English texts the top language's ISO 639-1 code is stored.
    """

    def __init__(self, cfg: dict[str, Any]):
        from lingua import Language, LanguageDetectorBuilder
        lc = cfg["preprocess"]["lang_id"]
        b = LanguageDetectorBuilder.from_all_languages()
        if lc.get("low_accuracy_mode"):
            b = b.with_low_accuracy_mode()
        self._detector = b.build()
        self._en = Language.ENGLISH
        self.min_conf = float(lc.get("min_confidence", 0.5))
        self.max_chars = int(lc.get("max_chars", 1000))

    def _prep(self, t: str) -> str:
        return PLACEHOLDER_RE.sub(" ", t)[: self.max_chars]

    def detect(self, texts: Sequence[str]) -> tuple[list[str | None], list[float]]:
        """Return (lang codes, P(en)) for each text; code is 'en', another ISO 639-1 code, or None if undetectable."""
        prepped = [self._prep(t) for t in texts]
        p_en = list(self._detector.compute_language_confidence_in_parallel(prepped, self._en))
        codes: list[str | None] = ["en" if p >= self.min_conf else None for p in p_en]
        other = [i for i, c in enumerate(codes) if c is None]
        if other:
            found = self._detector.detect_languages_in_parallel_of([prepped[i] for i in other])
            for i, lang in zip(other, found):
                codes[i] = lang.iso_code_639_1.name.lower() if lang is not None else None
        return codes, [round(float(p), 4) for p in p_en]


# ================================================================ ids, hashing, dedupe
def stable_record_id(source: str, native_id: Any) -> str:
    """Deterministic 20-hex-char id from source + native id (same input -> same id on every machine/run)."""
    return hashlib.sha256(f"{source}:{native_id}".encode("utf-8")).hexdigest()[:20]


def dedupe_key(text: str) -> str:
    """sha1 of the lowercased, whitespace-collapsed text (exact-duplicate key)."""
    return hashlib.sha1(WS_RE.sub(" ", text).strip().lower().encode("utf-8")).hexdigest()


def exact_dedupe(df: pd.DataFrame, text_col: str = "text_clean", label_cols: Sequence[str] = (),
                 sort_cols: Sequence[str] = ()) -> tuple[pd.DataFrame, dict[str, int]]:
    """Exact dedupe on dedupe_key(text). With `label_cols`, duplicate groups whose labels disagree are dropped
    ENTIRELY (the label is unreliable); otherwise the first row (after sorting by `sort_cols`) is kept."""
    d = df.assign(_k=df[text_col].map(dedupe_key))
    if sort_cols:
        d = d.sort_values(list(sort_cols), kind="stable")
    stats = {"duplicate_rows_dropped": 0, "conflicting_label_groups": 0, "conflicting_label_rows_dropped": 0}
    if label_cols:
        nun = d.groupby("_k")[list(label_cols)].nunique(dropna=False).max(axis=1)
        bad = set(nun[nun > 1].index)
        stats["conflicting_label_groups"] = len(bad)
        mask = d["_k"].isin(bad)
        stats["conflicting_label_rows_dropped"] = int(mask.sum())
        d = d[~mask]
    dup = d.duplicated("_k", keep="first")
    stats["duplicate_rows_dropped"] = int(dup.sum())
    d = d[~dup]
    return d.drop(columns="_k").sort_index(), stats


def word_shingles(text: str, n: int) -> set[str]:
    w = WS_RE.sub(" ", text).strip().lower().split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def minhash_near_duplicates(texts: Sequence[str], num_perm: int = 128, threshold: float = 0.9, shingle_n: int = 5,
                            seed: int = 42) -> set[int]:
    """OPTIONAL near-duplicate detection (datasketch MinHash-LSH, Jaccard on word shingles).
    Returns positions to DROP: every text whose LSH bucket contains an earlier text. Off by default
    (preprocess.dedupe.near_duplicate_minhash) because it is O(n) memory in signatures and approximate."""
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    drop: set[int] = set()
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm, seed=seed)
        for sh in word_shingles(t, shingle_n):
            m.update(sh.encode("utf-8"))
        if lsh.query(m):
            drop.add(i)
        else:
            lsh.insert(str(i), m)
    return drop


# ================================================================ chunking
SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=\S)")


@dataclass(frozen=True)
class Chunk:
    """One window of a record. `n_tokens` includes special tokens and is <= max_tokens."""
    index: int
    text: str
    start_char: int
    end_char: int
    n_tokens: int
    n_chunks_total: int


def sentence_starts(text: str) -> list[int]:
    """Character offsets where sentences start (regex on . ! ? … followed by whitespace)."""
    return [0] + [m.end() for m in SENT_BOUNDARY_RE.finditer(text)]


def chunk_text(text: str, tokenizer: Any, max_tokens: int, stride: int = 64,
               max_chunks: int | None = None) -> list[Chunk]:
    """Split `text` into sentence-aligned windows of at most `max_tokens` model tokens (special tokens included).

    Windows end on a sentence boundary when one fits; a sentence longer than the window is cut at token level.
    Consecutive windows overlap by up to `stride` tokens (whole sentences when aligned, exactly `stride` tokens after
    a token-level cut). Each window is re-tokenised and shrunk if needed, so the max_tokens bound is guaranteed.
    With `max_chunks`, an evenly spaced subset (first and last always kept) is returned; `n_chunks_total` records
    the full count. Requires a *fast* HF tokenizer (offset mapping).
    """
    if not text:
        return []
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_tokens - n_special
    if budget <= 0 or not (0 <= stride < budget):
        raise ValueError(f"need 0 <= stride < max_tokens - special ({budget}); got stride={stride}")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = enc["offset_mapping"]
    n = len(offs)
    if n == 0:
        return []

    def n_model_tokens(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=True, truncation=False)["input_ids"])

    # token index at which each sentence starts (first token ending after the sentence's start char)
    ends = np.fromiter((e for _, e in offs), dtype=np.int64, count=n)
    bset = {int(np.searchsorted(ends, s, side="right")) for s in sentence_starts(text)}
    bounds = sorted(b for b in bset | {0, n} if 0 <= b <= n)
    barr = np.asarray(bounds)

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        cand = barr[(barr > start) & (barr <= start + budget)]
        forced = cand.size == 0
        end = min(start + budget, n) if forced else int(cand[-1])
        # verify against re-tokenisation of the substring (BPE can merge differently at the edges)
        while end > start + 1 and n_model_tokens(text[offs[start][0]:offs[end - 1][1]]) > max_tokens:
            end -= 1
        spans.append((start, end))
        if end >= n:
            break
        if forced or end not in bset:
            nxt = max(end - stride, start + 1)
        else:
            ov = barr[(barr >= end - stride) & (barr > start) & (barr < end)]
            nxt = int(ov[0]) if ov.size else end
        start = nxt

    if max_chunks is not None and len(spans) > max_chunks:
        keep = sorted(set(np.linspace(0, len(spans) - 1, max_chunks).round().astype(int).tolist()))
        sel = [spans[i] for i in keep]
    else:
        sel = spans
    out = []
    for i, (a, b) in enumerate(sel):
        s, e = offs[a][0], offs[b - 1][1]
        piece = text[s:e]
        out.append(Chunk(index=i, text=piece, start_char=s, end_char=e, n_tokens=n_model_tokens(piece),
                         n_chunks_total=len(spans)))
    return out


def aggregate_chunk_scores(scores: Sequence[float] | np.ndarray, method: str = "mean",
                           weights: Sequence[float] | None = None) -> np.ndarray | float:
    """Combine per-chunk CALIBRATED probabilities into one record score. `scores` is (n_chunks,) or (n_chunks, n_labels).

    Default `mean`: the record score is the expected chunk-level probability. Why not `max`: with noisy chunk
    scores the max grows with the number of chunks, so long posts (e.g. r/relationships, median ~400 tokens) would
    look systematically more "stressed" than short ones (r/teaching, ~140), which would bias exactly the cross-community
    and over-time aggregates this project reports. The mean is length-neutral and stays a probability, so calibration
    is preserved on average. Trade-off: a short stressful passage inside a long calm post is diluted (documented
    limitation; `max` is available for sensitivity analysis). `weighted_mean` weights chunks by `weights`
    (e.g. token counts, so a short tail chunk counts less).
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("no chunk scores to aggregate")
    if method == "mean":
        res = arr.mean(axis=0)
    elif method == "max":
        res = arr.max(axis=0)
    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean needs weights")
        w = np.asarray(weights, dtype=float)
        res = np.tensordot(w / w.sum(), arr, axes=(0, 0))
    else:
        raise ValueError(f"unknown aggregation {method!r}")
    return float(res) if np.ndim(res) == 0 else res


# ================================================================ splits (fixed, saved, never silently regenerated)
class SplitMismatchError(RuntimeError):
    """A saved split file does not match the current data. Regenerate only deliberately (force + version bump)."""


def _fingerprint(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def split_path(cfg: dict[str, Any], name: str) -> Path:
    return Path(cfg["_paths"]["splits"]) / f"{name}_split_v{cfg['splits']['version']}.json"


def make_dreaddit_splits(df: pd.DataFrame, val_fraction: float, seed: int, group_col: str = "post_id",
                         label_col: str = "label") -> dict[str, list[str]]:
    """Official test stays FINAL TEST. Official train -> train/validation with StratifiedGroupKFold
    (n_splits = round(1/val_fraction), fold 0): grouped by post_id (no post spans two splits), label-stratified as far
    as grouping allows. StratifiedGroupKFold is used instead of GroupShuffleSplit because the latter cannot stratify."""
    from sklearn.model_selection import StratifiedGroupKFold
    tr = df[df["official_split"] == "train"].sort_values("record_id").reset_index(drop=True)
    n_splits = max(2, round(1 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(tr, tr[label_col], groups=tr[group_col]))
    return {"train": sorted(tr.loc[tr_idx, "record_id"]), "validation": sorted(tr.loc[va_idx, "record_id"]),
            "test": sorted(df.loc[df["official_split"] == "test", "record_id"])}


def load_or_create_split(cfg: dict[str, Any], name: str, df: pd.DataFrame, create: Callable[[], dict[str, list[str]]],
                         meta: dict[str, Any], force: bool = False) -> dict[str, list[str]]:
    """Load a saved split if it matches `df` (same record_id set); create + save it if absent.
    A mismatch raises SplitMismatchError: splits are NEVER regenerated silently. `force=True` regenerates and keeps
    the old file as <name>.bak-<UTC timestamp>.json."""
    p = split_path(cfg, name)
    fp = _fingerprint(df["record_id"])
    if p.exists() and not force:
        saved = json.loads(p.read_text(encoding="utf-8"))
        if saved["data_fingerprint"] != fp:
            raise SplitMismatchError(
                f"{p.name}: saved split was built on different data (fingerprint {saved['data_fingerprint'][:12]} vs "
                f"{fp[:12]}). Investigate why the data changed; to regenerate deliberately, bump splits.version or "
                f"pass force=True (--force-splits).")
        LOG.info("loaded fixed split %s", p.name)
        return saved["splits"]
    if p.exists():
        bak = p.with_name(f"{p.stem}.bak-{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%SZ}.json")
        os.replace(p, bak)
        LOG.warning("force: regenerating split %s (old file kept as %s)", p.name, bak.name)
    splits = create()
    all_ids = [i for v in splits.values() for i in v]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(df["record_id"]):
        raise SplitMismatchError(f"{name}: generated split does not partition the data")
    write_json(p, {"name": name, "version": cfg["splits"]["version"], "created_utc": _dt.datetime.now(_dt.timezone.utc),
                   "data_fingerprint": fp, "n_records": len(df), "counts": {k: len(v) for k, v in splits.items()},
                   **meta, "splits": splits})
    LOG.info("saved fixed split %s %s", p.name, {k: len(v) for k, v in splits.items()})
    return splits


def group_overlap(df: pd.DataFrame, split_col: str = "split", group_col: str = "post_id") -> dict[str, int]:
    """Number of groups shared by each pair of splits (must be 0 for grouped splits)."""
    sets = {s: set(g[group_col]) for s, g in df.groupby(split_col)}
    names = sorted(sets)
    return {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}


# ================================================================ dataset builders
class StepCounter:
    """Row counts per processing step, with drop reasons (goes into the manifest)."""

    def __init__(self, name: str, n: int):
        self.name = name
        self.steps: list[dict[str, Any]] = [{"step": "loaded", "rows": int(n)}]

    def drop(self, reason: str, before: int, after: int, **extra: Any) -> None:
        self.steps.append({"step": f"drop:{reason}", "dropped": int(before - after), "rows": int(after), **extra})

    def note(self, step: str, **info: Any) -> None:
        self.steps.append({"step": step, **info})


def _finalise(df: pd.DataFrame, cfg: dict[str, Any], extra_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["n_chars"] = df["text_clean"].str.len().astype("int32")
    df["n_tokens_est"] = df["text_clean"].map(n_tokens_estimate).astype("int32")
    out = df[UNIFIED_COLUMNS + [c for c in extra_cols if c not in UNIFIED_COLUMNS]].reset_index(drop=True)
    assert_privacy_safe(out, cfg)
    return out


def assert_privacy_safe(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Fail loudly if an author/user/url column slipped through, or if emails/URLs/mentions survive in text_clean."""
    bad = author_columns(df.columns, cfg) + [c for c in df.columns if FORBIDDEN_COLUMN_PATTERNS.match(str(c))]
    if bad:
        raise AssertionError(f"forbidden columns in processed output: {bad}")
    t = df["text_clean"].astype(str)
    for name, rx in (("email", EMAIL_RE), ("url", URL_RE), ("mention", MENTION_RE)):
        hits = int(t.str.contains(rx).sum())
        if hits:
            raise AssertionError(f"{hits} rows still contain a {name} after cleaning")


def _clean_column(texts: pd.Series, cfg: dict[str, Any], stats: Counter) -> pd.Series:
    return texts.map(lambda x: clean_text(x, cfg, stats))


def _filter_unlabelled(df: pd.DataFrame, cfg: dict[str, Any], sc: StepCounter, lid: LanguageIdentifier | None) -> pd.DataFrame:
    """Drop empty/removed, bot text, too-short and non-English rows (unlabelled corpora only)."""
    n0 = len(df)
    df = df[df["text_clean"].str.len() > 0]
    sc.drop("empty_or_removed_after_cleaning", n0, len(df))
    n0 = len(df)
    df = df[~df["text_clean"].map(lambda t: is_bot_text(t, cfg))]
    sc.drop("bot_text", n0, len(df))
    n0 = len(df)
    df = df[df["text_clean"].map(n_tokens_estimate) >= int(cfg["preprocess"]["min_tokens"])]
    sc.drop(f"shorter_than_{cfg['preprocess']['min_tokens']}_tokens", n0, len(df))
    df = _add_lang(df, lid)
    if lid is None:
        sc.note("non_english_filter_skipped", reason="no language identifier")
        return df
    n0 = len(df)
    lang_counts = df["lang"].fillna("und").value_counts().head(10).to_dict()
    df = df[df["lang"] == "en"]
    sc.drop("non_english", n0, len(df), top_detected=lang_counts)
    return df


def _add_lang(df: pd.DataFrame, lid: LanguageIdentifier | None) -> pd.DataFrame:
    df = df.copy()
    if lid is None:
        df["lang"] = None
        return df
    codes, _ = lid.detect(df["text_clean"].tolist())
    df["lang"] = codes
    return df


def build_dreaddit(cfg: dict[str, Any], lid: LanguageIdentifier | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dreaddit -> unified schema. Text only (precomputed feature columns are discarded). Approved mitigations:
    drop exact-duplicate texts in train (conflicting-label groups dropped entirely), drop train texts also in test."""
    from datasets import load_dataset
    src = cfg["sources"]["dreaddit"]
    ds = load_dataset(src["hf_id"])
    raw = pd.concat([ds[s].to_pandas()[src["core_columns"]].assign(official_split=s) for s in ds.keys()], ignore_index=True)
    sc = StepCounter("dreaddit", len(raw))
    stats: Counter = Counter()
    df = pd.DataFrame({
        "record_id": raw["id"].map(lambda i: stable_record_id("dreaddit", i)),
        "source": "dreaddit", "community": raw["subreddit"].astype(str),
        "created_at": pd.to_datetime(raw["social_timestamp"], unit="s", utc=True),
        "text_clean": _clean_column(raw[src["text_column"]], cfg, stats),
        "label": raw[src["label_column"]].astype("int8"), "confidence": raw["confidence"].astype("float32"),
        "post_id": raw["post_id"].astype(str), "sentence_range": raw["sentence_range"].astype(str),
        "official_split": raw["official_split"],
    })
    n0 = len(df)
    df = df[df["text_clean"].str.len() > 0]
    sc.drop("empty_after_cleaning", n0, len(df))
    tr, te = df[df.official_split == "train"], df[df.official_split == "test"]
    n0 = len(tr) + len(te)
    tr, dd = exact_dedupe(tr, label_cols=["label"])
    sc.drop("train_exact_duplicate_text", n0, len(tr) + len(te), **dd)
    test_keys = set(te["text_clean"].map(dedupe_key))
    n0 = len(tr) + len(te)
    tr = tr[~tr["text_clean"].map(dedupe_key).isin(test_keys)]
    sc.drop("train_text_also_in_test", n0, len(tr) + len(te))
    n_dup_test = int(te["text_clean"].map(dedupe_key).duplicated().sum())
    sc.note("official_test_kept_unchanged", test_duplicate_rows=n_dup_test)
    df = _add_lang(pd.concat([tr, te]), lid)
    sc.note("lang_computed_not_filtered", counts=df["lang"].fillna("und").value_counts().to_dict())
    splits = load_or_create_split(
        cfg, "dreaddit", df,
        lambda: make_dreaddit_splits(df, cfg["splits"]["dreaddit"]["val_fraction"], cfg["primary_seed"],
                                     cfg["splits"]["dreaddit"]["group_column"]),
        {"method": "official test = FINAL TEST; train/validation = StratifiedGroupKFold(post_id) fold 0 of official train",
         "seed": cfg["primary_seed"], "val_fraction": cfg["splits"]["dreaddit"]["val_fraction"]},
        force=cfg.get("_force_splits", False))
    sid = {rid: s for s, ids in splits.items() for rid in ids}
    df["split"] = df["record_id"].map(sid)
    out = _finalise(df, cfg, ["label", "confidence", "post_id", "sentence_range", "official_split", "split"])
    info = {"steps": sc.steps, "clean_edits": dict(stats), "final_rows": int(len(out)),
            "split_counts": out["split"].value_counts().to_dict(),
            "label_by_split": out.groupby("split")["label"].value_counts().unstack().to_dict(orient="index"),
            "post_id_overlap_between_splits": group_overlap(out), "split_file": split_path(cfg, "dreaddit")}
    return out, info


def build_goemotions(cfg: dict[str, Any], lid: LanguageIdentifier | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """GoEmotions official TSVs (no header; text, comma-separated label ids, id) -> unified schema, official splits."""
    from .audit import _read_goemotions_tsv  # shared reader (handles the missing header row)
    src = cfg["sources"]["goemotions"]
    odir = Path(cfg["_root"]) / src["official_dir"]
    emotions = [l.strip() for l in (odir / "emotions.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(emotions) != src["num_labels"]:
        raise ValueError(f"emotions.txt has {len(emotions)} labels, config expects {src['num_labels']}")
    parts = []
    for fname, split in (("train.tsv", "train"), ("dev.tsv", "validation"), ("test.tsv", "test")):
        d, _ = _read_goemotions_tsv(odir / fname)
        parts.append(d.assign(official_split=split))
    raw = pd.concat(parts, ignore_index=True)
    sc = StepCounter("goemotions", len(raw))
    stats: Counter = Counter()
    df = pd.DataFrame({
        "record_id": raw["id"].map(lambda i: stable_record_id("goemotions", i)),
        "source": "goemotions", "community": None, "created_at": pd.NaT,
        "text_clean": _clean_column(raw["text"], cfg, stats),
        "labels": raw["labels"].map(lambda s: [int(x) for x in str(s).split(",") if x != ""]),
        "official_split": raw["official_split"],
    })
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    n0 = len(df)
    df = df[df["text_clean"].str.len() > 0]
    sc.drop("empty_after_cleaning", n0, len(df))
    if df["record_id"].duplicated().any():
        raise ValueError("GoEmotions ids are not unique: cannot build stable record ids")
    df = _add_lang(df, lid)
    sc.note("lang_computed_not_filtered", counts=df["lang"].fillna("und").value_counts().head(10).to_dict())
    keys = df["text_clean"].map(dedupe_key)
    by = {s: set(keys[df.official_split == s]) for s in ("train", "validation", "test")}
    sc.note("official_splits_kept_unchanged",
            cross_split_shared_texts={"train&validation": len(by["train"] & by["validation"]),
                                      "train&test": len(by["train"] & by["test"]),
                                      "validation&test": len(by["validation"] & by["test"])},
            within_train_duplicate_rows=int(keys[df.official_split == "train"].duplicated().sum()))
    splits = load_or_create_split(
        cfg, "goemotions", df,
        lambda: {s: sorted(df.loc[df.official_split == s, "record_id"]) for s in ("train", "validation", "test")},
        {"method": "official GoEmotions train/dev/test (dev renamed validation)", "label_names": emotions},
        force=cfg.get("_force_splits", False))
    sid = {rid: s for s, ids in splits.items() for rid in ids}
    df["split"] = df["record_id"].map(sid)
    out = _finalise(df, cfg, ["labels", "official_split", "split"])
    info = {"steps": sc.steps, "clean_edits": dict(stats), "final_rows": int(len(out)),
            "split_counts": out["split"].value_counts().to_dict(), "label_names": emotions,
            "split_file": split_path(cfg, "goemotions")}
    return out, info


SAD_KEEP = ["is_stressor", "is_stressor_conf", "is_covid", "top_label", "second_label", "avg_severity", "Source"]


def build_sad(cfg: dict[str, Any], lid: LanguageIdentifier | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """SAD -> unified schema (no split here: Step 5 owns stressor splits and the Reddit gold set)."""
    src = cfg["sources"]["sad"]
    rdir = Path(cfg["_root"]) / src["raw_dir"]
    path = next((p for p in rdir.rglob("SAD_v1.xlsx") if not p.name.startswith("~$")), Path(cfg["_root"]) / src["local_path"])
    raw = pd.read_excel(path, sheet_name=0)
    sc = StepCounter("sad", len(raw))
    stats: Counter = Counter()
    df = raw[["sID"] + [c for c in SAD_KEEP if c in raw.columns]].copy()
    df.insert(0, "record_id", raw["sID"].map(lambda i: stable_record_id("sad", i)))
    df["source"], df["community"], df["created_at"] = "sad", None, pd.NaT
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["text_clean"] = _clean_column(raw["sentence"], cfg, stats)
    df = df.rename(columns={"Source": "sad_source"})
    n0 = len(df)
    df = df[df["text_clean"].str.len() > 0]
    sc.drop("empty_after_cleaning", n0, len(df))
    n0 = len(df)
    df, dd = exact_dedupe(df, label_cols=["top_label", "is_stressor"])
    sc.drop("exact_duplicate_text", n0, len(df), **dd)
    df = _add_lang(df, lid)
    sc.note("lang_computed_not_filtered", counts=df["lang"].fillna("und").value_counts().head(10).to_dict())
    extra = ["sID"] + [c if c != "Source" else "sad_source" for c in SAD_KEEP if c in raw.columns]
    out = _finalise(df, cfg, extra)
    return out, {"steps": sc.steps, "clean_edits": dict(stats), "final_rows": int(len(out)), "xlsx": path.name}


def _zenodo_files(cfg: dict[str, Any]) -> list[tuple[str, str, Path]]:
    src = cfg["sources"]["zenodo_pilot"]
    zdir = Path(cfg["_root"]) / src["local_dir"]
    files = []
    for sub in src["subreddits"]:
        for win in src["windows"]:
            p = zdir / src["file_pattern"].format(subreddit=sub, window=win)
            if p.exists():
                files.append((sub, win, p))
    return files


def build_zenodo(cfg: dict[str, Any], lid: LanguageIdentifier | None, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Zenodo pilot -> unified schema. Reads ONLY the whitelisted columns (never `author` or feature columns).
    Per-file results are cached in data/processed/_cache/zenodo_clean/ (resumable). Then global exact dedupe on text."""
    src = cfg["sources"]["zenodo_pilot"]
    cols = list(src["load_columns"])
    if author_columns(cols, cfg):
        raise AssertionError("zenodo load_columns must not include author fields")
    cache = Path(cfg["_paths"]["interim"]) / "zenodo_clean"
    cache.mkdir(parents=True, exist_ok=True)
    per_file: dict[str, Any] = {}
    stats: Counter = Counter()
    frames = []
    for order, (sub, win, p) in enumerate(_zenodo_files(cfg)):
        cp, ip = cache / f"{p.stem}.parquet", cache / f"{p.stem}.json"
        if cp.exists() and ip.exists() and not force:
            frames.append(pd.read_parquet(cp))
            per_file[p.stem] = json.loads(ip.read_text(encoding="utf-8"))
            continue
        raw = pd.read_csv(p, usecols=cols, dtype=str, keep_default_na=False, encoding="utf-8")
        sc = StepCounter(p.stem, len(raw))
        fstats: Counter = Counter()
        mism = int((raw["subreddit"].str.lower() != sub.lower()).sum())
        df = pd.DataFrame({
            "record_id": [stable_record_id("zenodo", f"{sub}|{d}|{hashlib.sha1(t.encode('utf-8')).hexdigest()}")
                          for d, t in zip(raw[src["date_column"]], raw[src["text_column"]])],
            "source": "zenodo_pilot", "community": sub.lower(),
            "created_at": pd.to_datetime(raw[src["date_column"]], format="%Y/%m/%d", utc=True, errors="coerce"),
            "text_clean": _clean_column(raw[src["text_column"]], cfg, fstats),
            "window": win, "_file_order": order,
        })
        n0 = len(df)
        df = df[df["created_at"].notna()]
        sc.drop("unparseable_date", n0, len(df))
        df = _filter_unlabelled(df, cfg, sc, lid)
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in src["window_dates"][win])
        sc.note("outside_observed_window_kept", n=int(((df.created_at < lo) | (df.created_at > hi)).sum()))
        info = {"steps": sc.steps, "subreddit_value_mismatch": mism, "clean_edits": dict(fstats)}
        _write_parquet(df, cp)
        write_json(ip, info)
        frames.append(df)
        per_file[p.stem] = info
        LOG.info("zenodo %s: %d -> %d rows", p.stem, sc.steps[0]["rows"], len(df))
    for v in per_file.values():
        stats.update(v.get("clean_edits", {}))
    df = pd.concat(frames, ignore_index=True)
    sc = StepCounter("zenodo_all", len(df))
    n0 = len(df)
    df, dd = exact_dedupe(df, sort_cols=["created_at", "_file_order"])
    sc.drop("exact_duplicate_text_global", n0, len(df), **dd)
    dcfg = cfg["preprocess"]["dedupe"]
    if dcfg.get("near_duplicate_minhash"):
        n0 = len(df)
        df = df.sort_values(["created_at", "_file_order"], kind="stable").reset_index(drop=True)
        drop = minhash_near_duplicates(df["text_clean"].tolist(), dcfg["minhash_num_perm"], dcfg["minhash_threshold"],
                                       dcfg["minhash_shingle_n"], cfg["primary_seed"])
        df = df.drop(index=sorted(drop))
        sc.drop("near_duplicate_minhash", n0, len(df))
    if df["record_id"].duplicated().any():
        raise AssertionError("zenodo record_id collision after dedupe")
    out = _finalise(df.sort_values(["community", "created_at", "record_id"]), cfg, ["window"])
    info = {"per_file": per_file, "global_steps": sc.steps, "clean_edits": dict(stats), "final_rows": int(len(out)),
            "rows_by_community_window": out.groupby(["community", "window"]).size().rename("n").reset_index().to_dict(orient="records"),
            "date_range": {"min": out.created_at.min(), "max": out.created_at.max()},
            "note": "created_at has DAY resolution (source `date` column); time of day is not available"}
    return out, info


def _reddit_containment_rows(queries: Sequence[str], docs: Sequence[str], n: int, threshold: float) -> set[int]:
    """Positions in `docs` that contain >= threshold of some query's word n-gram shingles (inverted index)."""
    index: dict[str, list[int]] = {}
    for j, d in enumerate(docs):
        for sh in word_shingles(d, n):
            index.setdefault(sh, []).append(j)
    hit: set[int] = set()
    for q in queries:
        qs = word_shingles(q, n)
        if not qs:
            continue
        c: Counter = Counter()
        for sh in qs:
            c.update(index.get(sh, ()))
        hit |= {j for j, k in c.items() if k / len(qs) >= threshold}
    return hit


def build_senticnet(cfg: dict[str, Any], lid: LanguageIdentifier | None,
                    dreaddit_texts: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """SenticNet -> validation-only parquet (AUTOMATED labels; never ground truth). Approved mitigations:
    within-file dedupe with conflicting-label groups dropped; Twitter_Non-Advert texts present in Twitter_Full dropped;
    rows containing Dreaddit segments (shingle containment) or exact Dreaddit texts dropped."""
    from .audit import load_senticnet, senticnet_text_label_cols
    sv = cfg["preprocess"]["senticnet_validation"]
    frames = load_senticnet(cfg)
    info: dict[str, Any] = {"label_provenance": "automated DNN-based labels (SenticNet README); NOT human ground truth"}
    stats: Counter = Counter()
    out_frames = {}
    dre_keys = {dedupe_key(t) for t in dreaddit_texts}
    n_sh, thr = int(cfg["audit"]["shingle_n"]), float(cfg["audit"]["overlap_threshold"])
    for name, raw in frames.items():
        tcol, lcol = senticnet_text_label_cols(raw)
        sc = StepCounter(name, len(raw))
        df = pd.DataFrame({
            "record_id": [stable_record_id("senticnet", f"{name}|{i}") for i in raw.index],
            "source": "senticnet", "community": name,
            "created_at": pd.to_datetime(pd.Series(pd.NaT, index=raw.index), utc=True),
            "text_clean": _clean_column(raw[tcol], cfg, stats),
            "label_auto": pd.to_numeric(raw[lcol], errors="coerce"),
            "platform": "reddit" if name.lower().startswith("reddit") else "twitter",
        })
        n0 = len(df)
        df = df[(df.text_clean.str.len() > 0) & df.label_auto.isin([0, 1])]
        sc.drop("empty_text_or_invalid_label", n0, len(df))
        n0 = len(df)
        df, dd = exact_dedupe(df, label_cols=["label_auto"] if sv["drop_conflicting_duplicate_groups"] else ())
        sc.drop("within_file_duplicates", n0, len(df), **dd)
        n0 = len(df)
        df = df[~df.text_clean.map(dedupe_key).isin(dre_keys)]
        sc.drop("exact_dreaddit_text", n0, len(df))
        if sv["drop_dreaddit_contained"] and df.platform.iloc[0] == "reddit" and dreaddit_texts:
            n0 = len(df)
            hit = _reddit_containment_rows(dreaddit_texts, df.text_clean.tolist(), n_sh, thr)
            df = df.drop(index=df.index[sorted(hit)])
            sc.drop(f"contains_dreaddit_segment_ge_{thr:.0%}", n0, len(df))
        out_frames[name] = (df, sc)
    flag_info: dict[str, Any] = {}
    if sv.get("non_advert_as_flag_on_twitter_full") and {"Twitter_Non-Advert", "Twitter_Full"} <= set(out_frames):
        # user decision (Step 2 final): no separate Non-Advert set; flag matching Twitter_Full rows by text hash
        na, na_sc = out_frames.pop("Twitter_Non-Advert")
        full, full_sc = out_frames["Twitter_Full"]
        na_keys = set(na.text_clean.map(dedupe_key))
        full = full.assign(is_non_advert=full.text_clean.map(dedupe_key).isin(na_keys))
        na_sc.drop("not_kept_as_separate_set", len(na), 0)
        full_sc.note("is_non_advert_flag_added", rows_flagged=int(full["is_non_advert"].sum()),
                     non_advert_texts=len(na_keys),
                     non_advert_texts_without_match_in_full=int(len(na_keys - set(full.text_clean.map(dedupe_key)))))
        out_frames["Twitter_Full"] = (full, full_sc)
        flag_info = {"Twitter_Non-Advert": na_sc.steps}
    df = pd.concat([d for d, _ in out_frames.values()], ignore_index=True)
    df = _add_lang(df, lid)
    df["label_auto"] = df["label_auto"].astype("int8")
    extra = ["label_auto", "platform"]
    if "is_non_advert" in df.columns:
        df["is_non_advert"] = df["is_non_advert"].astype("boolean")      # null for rows not from Twitter_Full
        extra.append("is_non_advert")
    out = _finalise(df, cfg, extra)
    info.update({"files": {**{k: s.steps for k, (_, s) in out_frames.items()}, **flag_info}, "clean_edits": dict(stats),
                 "final_rows": int(len(out)),
                 "label_auto_by_file": out.groupby("community")["label_auto"].value_counts().unstack().fillna(0).astype(int).to_dict(orient="index"),
                 "lang_counts": out["lang"].fillna("und").value_counts().head(10).to_dict()})
    return out, info


# ================================================================ GoEmotions overlap flags (sidecar)
def goemotions_overlap_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_paths"]["processed"]) / "goemotions_overlap_flags.parquet"


def build_goemotions_overlap_flags(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """User decision (Step 2 final): keep the published GoEmotions splits, but flag every row whose cleaned text
    (dedupe_key) also occurs in ANOTHER split. Written as a sidecar so goemotions.parquet and its split file stay
    byte-identical. Columns: record_id, split, text_in_train, text_in_validation, text_in_test, exclude_for_test_sensitivity."""
    out = goemotions_overlap_path(cfg)
    src = processed_path(cfg, "goemotions")
    df = pd.read_parquet(src, columns=["record_id", "split", "text_clean"])
    keys = df["text_clean"].map(dedupe_key)
    by = {s: set(keys[df["split"] == s]) for s in ("train", "validation", "test")}
    flags = pd.DataFrame({"record_id": df["record_id"], "split": df["split"],
                          **{f"text_in_{s}": keys.isin(by[s]) & (df["split"] != s) for s in ("train", "validation", "test")}})
    flags["exclude_for_test_sensitivity"] = (flags["split"] == "test") & (flags["text_in_train"] | flags["text_in_validation"])
    t = flags[flags["split"] == "test"]
    summary = {"shared_texts": {"train&validation": len(by["train"] & by["validation"]), "train&test": len(by["train"] & by["test"]),
                                "validation&test": len(by["validation"] & by["test"])},
               "test_rows_text_in_train": int(t["text_in_train"].sum()),
               "test_rows_text_in_validation": int(t["text_in_validation"].sum()),
               "test_rows_excluded_in_sensitivity_check": int(t["exclude_for_test_sensitivity"].sum()),
               "test_rows_total": int(len(t)),
               "validation_rows_text_in_train": int(flags.loc[flags["split"] == "validation", "text_in_train"].sum()),
               "source_sha256": sha256_file(src)}
    if force or not out.exists():
        _write_parquet(flags, out)
    return summary


def goemotions_test_sensitivity_frames(pred: pd.DataFrame, flags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """For Step 4: test predictions (must contain record_id) with and without the overlapping test rows."""
    excl = set(flags.loc[flags["exclude_for_test_sensitivity"], "record_id"])
    return {"all_test": pred, "test_without_cross_split_overlap": pred[~pred["record_id"].isin(excl)]}


# ================================================================ orchestration
def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    os.replace(tmp, path)
    return path


def processed_path(cfg: dict[str, Any], name: str) -> Path:
    fname = {"zenodo": "zenodo_pilot", "senticnet": "senticnet_validation"}.get(name, name)
    return Path(cfg["_paths"]["processed"]) / f"{fname}.parquet"


def _dataset_versions(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("dreaddit", "goemotions", "sad", "zenodo_pilot", "senticnet"):
        a = Path(cfg["_paths"]["interim"]) / f"audit_{name}.json"
        if a.exists():
            j = json.loads(a.read_text(encoding="utf-8"))
            out[name] = j.get("version") or j.get("record") or j.get("repo") or {"audit_sha256": sha256_file(a)}
    return out


def run(cfg: dict[str, Any], only: Iterable[str] | None = None, force: bool = False,
        force_splits: bool = False) -> dict[str, Any]:
    """Build every processed parquet (skips ones that exist unless force), then write manifest + report."""
    cfg = {**cfg, "_force_splits": force_splits}
    names = [n for n in DATASETS if not only or n in set(only)]
    man_path = Path(cfg["_paths"]["manifests"]) / "step2_preprocess_manifest.json"
    prev = json.loads(man_path.read_text(encoding="utf-8")).get("extra", {}).get("datasets", {}) if man_path.exists() else {}
    results: dict[str, Any] = dict(prev)
    lid: LanguageIdentifier | None = None
    for name in names:
        out_p = processed_path(cfg, name)
        if out_p.exists() and not force and name in prev:
            LOG.info("%s: %s exists, skipping (use --force to rebuild)", name, out_p.name)
            continue
        if lid is None:
            LOG.info("building lingua language detector (all languages; one-off ~1 GB)")
            lid = LanguageIdentifier(cfg)
        LOG.info("building %s", name)
        if name == "dreaddit":
            df, info = build_dreaddit(cfg, lid)
        elif name == "goemotions":
            df, info = build_goemotions(cfg, lid)
        elif name == "sad":
            df, info = build_sad(cfg, lid)
        elif name == "zenodo":
            df, info = build_zenodo(cfg, lid, force=force)
        else:
            dre = processed_path(cfg, "dreaddit")
            dre_texts = pd.read_parquet(dre, columns=["text_clean"])["text_clean"].tolist() if dre.exists() else []
            if not dre_texts:
                LOG.warning("senticnet: processed dreaddit missing, leakage filter uses no Dreaddit texts")
            df, info = build_senticnet(cfg, lid, dre_texts)
        _write_parquet(df, out_p)
        info["output"] = out_p
        info["columns"] = list(df.columns)
        results[name] = info
        LOG.info("%s: wrote %d rows to %s", name, len(df), out_p)
    if processed_path(cfg, "goemotions").exists():
        results.setdefault("goemotions", {})["overlap_flags"] = build_goemotions_overlap_flags(cfg, force=force)
    write_manifest(
        man_path,
        artifacts=[processed_path(cfg, n) for n in DATASETS if processed_path(cfg, n).exists()]
        + ([goemotions_overlap_path(cfg)] if goemotions_overlap_path(cfg).exists() else [])
        + sorted(Path(cfg["_paths"]["splits"]).glob("*.json")),
        seeds=[cfg["primary_seed"]], dataset_versions=_dataset_versions(cfg),
        extra={"step": 2, "preprocess_config": cfg["preprocess"], "splits_config": cfg["splits"],
               "chunking_config": cfg["chunking"], "lang_id": {"library": "lingua-language-detector",
                                                              "version": _pkg_version("lingua-language-detector")},
               "lemmatizer": {"library": "simplemma", "version": _pkg_version("simplemma")},
               "datasets": results},
        root=cfg["_root"])
    write_report(cfg, results)
    return results


def _pkg_version(dist: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:  # noqa: BLE001
        return None


def write_report(cfg: dict[str, Any], results: dict[str, Any]) -> Path:
    """data/outputs/reports/preprocess_report.md: row counts per step with drop reasons (counts only, no text)."""
    lines = ["# Step 2: preprocessing report", "", "Counts only; every number comes from "
             "`data/outputs/manifests/step2_preprocess_manifest.json`. No text is included.", ""]

    def steps_table(steps: list[dict[str, Any]]) -> list[str]:
        rows = ["| step | dropped | rows after | details |", "|---|---|---|---|"]
        for s in steps:
            det = {k: v for k, v in s.items() if k not in ("step", "dropped", "rows")}
            rows.append(f"| {s['step']} | {s.get('dropped', '')} | {s.get('rows', '')} | {json.dumps(det, default=str) if det else ''} |")
        return rows

    for name in DATASETS:
        r = results.get(name)
        if not r:
            continue
        lines += [f"## {name}", "", f"Final rows: **{r['final_rows']}**", ""]
        if name == "zenodo":
            for f, v in r["per_file"].items():
                lines += [f"### {f}", ""] + steps_table(v["steps"]) + [""]
            lines += ["### all files", ""] + steps_table(r["global_steps"]) + [""]
            lines += ["Rows by community × window:", "", "| community | window | n |", "|---|---|---|"]
            lines += [f"| {x['community']} | {x['window']} | {x['n']} |" for x in r["rows_by_community_window"]] + [""]
        elif name == "senticnet":
            for f, st in r["files"].items():
                lines += [f"### {f}", ""] + steps_table(st) + [""]
            lines += [f"Automated label counts by file: `{json.dumps(r['label_auto_by_file'])}`", ""]
        else:
            lines += steps_table(r["steps"]) + [""]
        if "split_counts" in r:
            lines += [f"Split counts: `{json.dumps(r['split_counts'])}`", ""]
        if "label_by_split" in r:
            lines += [f"Label by split: `{json.dumps(r['label_by_split'], default=str)}`", ""]
        if "post_id_overlap_between_splits" in r:
            lines += [f"post_id overlap between splits (must be 0): `{json.dumps(r['post_id_overlap_between_splits'])}`", ""]
        lines += [f"Cleaning edits (counts of substitutions): `{json.dumps(r['clean_edits'], sort_keys=True)}`", ""]
    p = Path(cfg["_paths"]["reports"]) / "preprocess_report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Step 2 preprocessing (no model training)")
    ap.add_argument("--only", nargs="*", choices=DATASETS)
    ap.add_argument("--force", action="store_true", help="rebuild outputs even if they exist")
    ap.add_argument("--force-splits", action="store_true", help="regenerate saved splits (old files kept as .bak)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    get_logger("stress_signals", cfg["logging"]["level"], Path(cfg["_paths"]["logs"]) / "pipeline.log")
    res = run(cfg, a.only, a.force, a.force_splits)
    for k, v in res.items():
        print(f"{k:<11} final_rows={v.get('final_rows')}")


if __name__ == "__main__":
    main()
