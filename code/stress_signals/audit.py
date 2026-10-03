"""Step 1: dataset acquisition and audit.

Every audit returns a JSON-serialisable dict (saved to data/processed/_cache/audit_<name>.json) and a Markdown
fragment (data/processed/_cache/audit_<name>.md). `build_report` concatenates fragments into
data/outputs/reports/data_audit.md. No raw user text is written to any JSON/Markdown output; author fields are
never loaded. Run everything with (from the project root):  PYTHONPATH=code python -m stress_signals.audit  [--only dreaddit goemotions ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import subprocess
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import load_config
from .privacy import author_columns
from .utils import get_logger, java_version, sha256_file, write_json, write_manifest

LOG = get_logger("stress_signals.audit")
UNVERIFIED = "UNVERIFIED"

# Dreaddit subreddit -> domain, from Turcan & McKeown (2019) "Dreaddit", Table 1. Written from memory of the
# paper: UNVERIFIED against the paper text; any subreddit in the data that is not listed here is reported.
DREADDIT_DOMAINS: dict[str, str] = {
    "domesticviolence": "abuse", "survivorsofabuse": "abuse",
    "anxiety": "anxiety", "stress": "anxiety",
    "almosthomeless": "financial", "assistance": "financial", "food_pantry": "financial", "homeless": "financial",
    "ptsd": "PTSD",
    "relationships": "social",
}

# Mendeley column groups, copied from the authors' own "Key Features" section (public API description).
MENDELEY_GROUPS: dict[str, list[str]] = {
    "demographics": ["gender", "age"],
    "symptoms_emotional_responses": ["stress_experience", "heartbeat_palpitations", "anxiety_tension", "sleep_problems",
                                     "restlessness", "irritability", "sadness_low_mood", "loneliness_isolation",
                                     "concentration_problems"],
    "health_indicators": ["headaches", "health_issues", "weight_changes"],
    "stressors_academic_environment": ["academic_overload", "peer_competition", "low_academic_confidence",
                                       "subject_confidence", "academic_conflicts", "class_attendance",
                                       "professor_difficulties", "work_environment", "home_environment"],
    "stressors_social_relationship": ["relationship_stress", "lack_relaxation_time"],
    "target": ["stress_type"],
}

PERCENTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


# ================================================================ generic helpers
def project_path(cfg: dict[str, Any], rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def interim_dir(cfg: dict[str, Any]) -> Path:
    d = Path(cfg["_paths"]["interim"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path, force: bool = False, timeout: int = 120) -> dict[str, Any]:
    """Idempotent download: skips if `dest` exists. Returns {url, path, sha256, bytes, downloaded}."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetched = False
    if force or not dest.exists():
        LOG.info("downloading %s", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
        fetched = True
    return {"url": url, "path": dest.as_posix(), "sha256": sha256_file(dest), "bytes": dest.stat().st_size,
            "downloaded_now": fetched}


def fetch_json(url: str, cache: Path | None = None, timeout: int = 60) -> dict[str, Any] | list[Any]:
    """GET JSON (cached to disk when `cache` is given, so reruns are offline and reproducible)."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "stress-signals-audit", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    if cache is not None:
        write_json(cache, data)
    return data


def github_repo_meta(cfg: dict[str, Any], repo: str) -> dict[str, Any]:
    """Licence + default branch + latest commit of a GitHub repo (cached)."""
    mdir = project_path(cfg, cfg["audit"]["metadata_dir"])
    safe = repo.replace("/", "__")
    try:
        info = fetch_json(f"https://api.github.com/repos/{repo}", mdir / f"github_{safe}.json")
        branch = info.get("default_branch", "main")
        commit = fetch_json(f"https://api.github.com/repos/{repo}/commits/{branch}", mdir / f"github_{safe}_commit.json")
        return {"repo": repo, "licence_spdx": (info.get("license") or {}).get("spdx_id"),
                "default_branch": branch, "commit_sha": commit.get("sha"),
                "commit_date": commit.get("commit", {}).get("committer", {}).get("date")}
    except Exception as exc:  # noqa: BLE001 - offline or rate-limited; report, don't crash
        return {"repo": repo, "error": repr(exc)}


def describe_numeric(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(pd.Series(list(values) if not isinstance(values, (pd.Series, np.ndarray)) else values).dropna(), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    out = {"n": int(arr.size), "mean": round(float(arr.mean()), 3), "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
           "min": float(arr.min()), "max": float(arr.max())}
    for q in PERCENTILES:
        out[f"p{int(q * 100):02d}"] = round(float(np.quantile(arr, q)), 3)
    return out


def counts(series: pd.Series, normalize: bool = True) -> dict[str, Any]:
    vc = series.astype("object").where(series.notna(), "<NA>").astype(str).value_counts()
    out = {"counts": {k: int(v) for k, v in vc.items()}}
    if normalize:
        out["proportions"] = {k: round(v / max(int(vc.sum()), 1), 4) for k, v in vc.items()}
    return out


def frame_overview(df: pd.DataFrame) -> dict[str, Any]:
    return {"n_rows": int(len(df)), "n_columns": int(df.shape[1]),
            "columns": [{"name": str(c), "dtype": str(df[c].dtype), "n_missing": int(df[c].isna().sum())} for c in df.columns]}


def normalise_text(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def text_hash(t: Any) -> str:
    return hashlib.sha1(normalise_text(t).encode("utf-8")).hexdigest()


def duplicate_stats(texts: pd.Series) -> dict[str, Any]:
    """Exact (raw string) and normalised (lowercase, whitespace-collapsed) duplicate counts."""
    t = texts.dropna().astype(str)
    raw_vc = t.value_counts()
    norm_vc = t.map(text_hash).value_counts()
    return {"n_texts": int(len(t)),
            "exact_duplicate_rows": int((raw_vc - 1)[raw_vc > 1].sum()),
            "exact_duplicated_distinct_texts": int((raw_vc > 1).sum()),
            "normalised_duplicate_rows": int((norm_vc - 1)[norm_vc > 1].sum())}


def missing_text_stats(texts: pd.Series) -> dict[str, int]:
    s = texts.astype("object")
    return {"n_missing": int(s.isna().sum()),
            "n_empty_or_whitespace": int(s.dropna().astype(str).str.strip().eq("").sum())}


def load_tokenizer(cfg: dict[str, Any], component: str):
    """Tokenizer of the base model configured for `component` (the one we will train with). None if unavailable."""
    name = cfg["models"][component]["base_model"]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        LOG.info("tokenizer for %s: %s (fast=%s)", component, name, tok.is_fast)
        return tok
    except Exception as exc:  # noqa: BLE001
        LOG.warning("tokenizer %s unavailable (%r): token stats skipped", name, exc)
        return None


def length_stats(texts: pd.Series, tokenizer, cfg: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    """Character lengths on all rows; token lengths (incl. special tokens, no truncation) on a seeded sample."""
    t = texts.dropna().astype(str)
    out: dict[str, Any] = {"chars": describe_numeric(t.str.len())}
    if tokenizer is None or len(t) == 0:
        out["tokens"] = {"skipped": "no tokenizer"}
        return out
    n = int(cfg["audit"]["token_sample_size"])
    sample = t.sample(n=min(n, len(t)), random_state=seed if seed is not None else cfg["primary_seed"])
    enc = tokenizer(sample.tolist(), add_special_tokens=True, truncation=False)["input_ids"]
    lens = pd.Series([len(x) for x in enc])
    out["tokens"] = describe_numeric(lens)
    out["tokens"]["tokenizer"] = getattr(tokenizer, "name_or_path", "?")
    out["tokens"]["sampled_rows"] = int(len(sample))
    out["tokens"]["share_over"] = {str(k): round(float((lens > k).mean()), 4) for k in cfg["audit"]["token_thresholds"]}
    return out


def shingles(text: str, n: int) -> set[str]:
    w = normalise_text(text).split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def containment_overlap(queries: Sequence[str], corpus: Iterable[str], n: int, threshold: float) -> dict[str, Any]:
    """How many `queries` have >= threshold of their word n-gram shingles inside `corpus` (near-duplicate containment)."""
    pool: set[str] = set()
    for doc in corpus:
        pool |= shingles(doc, n)
    scores = []
    for q in queries:
        sh = shingles(q, n)
        scores.append(len(sh & pool) / len(sh) if sh else 0.0)
    s = np.asarray(scores)
    return {"n_queries": int(len(s)), "shingle_n": n, "threshold": threshold,
            "n_contained": int((s >= threshold).sum()), "share_contained": round(float((s >= threshold).mean()) if len(s) else 0.0, 4),
            "n_any_overlap": int((s > 0).sum())}


# ---------------------------------------------------------------- markdown helpers
def md_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    def cell(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(map(cell, r)) + " |" for r in rows]
    return "\n".join(lines)


def md_counts(c: dict[str, Any], title: str = "value") -> str:
    props = c.get("proportions", {})
    return md_table([(k, v, props.get(k, "")) for k, v in c["counts"].items()], [title, "n", "share"])


def md_desc(d: dict[str, Any]) -> str:
    keys = [k for k in ("n", "mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "p99", "max") if k in d]
    return md_table([[d[k] for k in keys]], keys) if keys else str(d)


def md_lengths(ls: dict[str, Any]) -> str:
    out = ["Characters:", "", md_desc(ls["chars"]), ""]
    tk = ls.get("tokens", {})
    if "n" in tk:
        out += [f"Tokens (`{tk['tokenizer']}`, special tokens included, sample of {tk['sampled_rows']} rows):", "",
                md_desc(tk), "", "Share of texts longer than: " + ", ".join(f"{k} tokens = {v:.2%}" for k, v in tk["share_over"].items())]
    else:
        out.append(f"Tokens: {tk.get('skipped', 'n/a')}")
    return "\n".join(out)


def md_overview(ov: dict[str, Any], max_cols: int = 60) -> str:
    rows = [(c["name"], c["dtype"], c["n_missing"]) for c in ov["columns"][:max_cols]]
    more = "" if ov["n_columns"] <= max_cols else f"\n\n_{ov['n_columns'] - max_cols} further columns listed in the JSON._"
    return f"Rows: **{ov['n_rows']}**, columns: **{ov['n_columns']}**\n\n" + md_table(rows, ["column", "dtype", "missing"]) + more


def save_audit(cfg: dict[str, Any], name: str, audit: dict[str, Any], md: str) -> None:
    d = interim_dir(cfg)
    write_json(d / f"audit_{name}.json", audit)
    (d / f"audit_{name}.md").write_text(md, encoding="utf-8")
    LOG.info("saved audit_%s.json / .md", name)


# ================================================================ Dreaddit
def audit_dreaddit(cfg: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    src = cfg["sources"]["dreaddit"]
    hf_id, text_col = src["hf_id"], src["text_column"]
    ds = load_dataset(hf_id)
    frames = {split: ds[split].to_pandas() for split in ds.keys()}
    try:
        info = HfApi().dataset_info(hf_id)
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
        licence = (card.get("license") if hasattr(card, "get") else getattr(card, "license", None))
        version = {"hf_id": hf_id, "revision_sha": info.sha, "last_modified": str(getattr(info, "last_modified", ""))}
    except Exception as exc:  # noqa: BLE001
        licence, version = None, {"hf_id": hf_id, "error": repr(exc)}

    core = src["core_columns"]
    all_cols = list(frames[next(iter(frames))].columns)
    feature_cols = [c for c in all_cols if c not in core]
    groups: dict[str, list[str]] = {}
    for c in feature_cols:
        key = c.split("_")[0] if "_" in c else c
        key = "_".join(c.split("_")[:2]) if key == "lex" else key
        groups.setdefault(key, []).append(c)

    full = pd.concat([f.assign(_split=s) for s, f in frames.items()], ignore_index=True)
    tok = load_tokenizer(cfg, "stress")
    flags: list[str] = []

    split_sizes = {s: int(len(f)) for s, f in frames.items()}
    expected = {"train": 2838, "test": 715}
    if split_sizes != expected:
        flags.append(f"split sizes {split_sizes} differ from expected {expected}")
    missing_core = [c for c in core if c not in all_cols]
    if missing_core:
        flags.append(f"missing core columns: {missing_core}")

    seg_per_post = full.groupby("post_id").size()
    train_posts, test_posts = set(frames.get("train", pd.DataFrame()).get("post_id", [])), set(frames.get("test", pd.DataFrame()).get("post_id", []))
    leak_posts = sorted(train_posts & test_posts)
    if leak_posts:
        flags.append(f"{len(leak_posts)} post_id values appear in BOTH train and test (segment-level leakage risk)")
    train_h = set(frames["train"][text_col].map(text_hash)) if "train" in frames else set()
    test_h = set(frames["test"][text_col].map(text_hash)) if "test" in frames else set()

    subs = sorted(full["subreddit"].unique())
    unmapped = [s for s in subs if s not in DREADDIT_DOMAINS]
    if unmapped:
        flags.append(f"subreddits without a domain mapping: {unmapped}")
    full["_domain"] = full["subreddit"].map(DREADDIT_DOMAINS).fillna("UNMAPPED")

    ts = pd.to_datetime(full["social_timestamp"], unit="s", utc=True, errors="coerce")
    artefacts = {pat: int(full[text_col].astype(str).str.contains(pat, regex=False).sum()) for pat in ("&#x200B;", "<url>")}
    artefacts["&#x200b; (lowercase)"] = int(full[text_col].astype(str).str.contains("&#x200b;", regex=False).sum())
    artefacts["any HTML entity &...;"] = int(full[text_col].astype(str).str.contains(r"&[#a-zA-Z0-9]+;", regex=True).sum())

    audit = {
        "name": "dreaddit", "version": version,
        "licence": {"value": licence or UNVERIFIED, "source": f"Hugging Face dataset card of {hf_id}"},
        "split_sizes": split_sizes, "total_rows": int(len(full)),
        "overview": {s: frame_overview(f[[c for c in f.columns if c in core]]) for s, f in frames.items()},
        "core_columns_present": [c for c in core if c in all_cols], "missing_core_columns": missing_core,
        "precomputed_feature_columns": {"n": len(feature_cols), "groups": groups,
                                        "decision": "NOT USED: train on `text` only (cannot be reproduced on new data)"},
        "labels": {s: counts(f["label"]) for s, f in frames.items()},
        "missing_text": {s: missing_text_stats(f[text_col]) for s, f in frames.items()},
        "missing_values_core": {s: {c: int(f[c].isna().sum()) for c in core if c in f} for s, f in frames.items()},
        "duplicates": {s: duplicate_stats(f[text_col]) for s, f in frames.items()},
        "text_overlap_train_test_normalised": len(train_h & test_h),
        "segments_per_post": {"n_posts": int(seg_per_post.size), "distribution": {str(k): int(v) for k, v in seg_per_post.value_counts().sort_index().items()}},
        "post_id_in_train_and_test": {"n": len(leak_posts)},
        "subreddit_x_label": {s: pd.crosstab(f["subreddit"], f["label"]).to_dict(orient="index") for s, f in frames.items()},
        "subreddit_x_label_all": pd.crosstab(full["subreddit"], full["label"]).to_dict(orient="index"),
        "domain_mapping": {"mapping": DREADDIT_DOMAINS, "source": "Turcan & McKeown 2019, Table 1 (" + UNVERIFIED + ")", "unmapped": unmapped},
        "domain_x_label": pd.crosstab(full["_domain"], full["label"]).to_dict(orient="index"),
        "confidence": {"describe": describe_numeric(full["confidence"]), "values": counts(full["confidence"].round(3))},
        "timestamp_range_utc": {"min": str(ts.min()), "max": str(ts.max()), "n_unparseable": int(ts.isna().sum())},
        "year_counts": {str(k): int(v) for k, v in ts.dt.year.value_counts().sort_index().items()},
        "text_artefacts": artefacts,
        "lengths": {s: length_stats(f[text_col], tok, cfg) for s, f in frames.items()},
        "flags": flags,
    }

    def xtab_md(d: dict[str, Any]) -> str:
        labels = sorted({k for v in d.values() for k in v})
        return md_table([[sub] + [v.get(l, 0) for l in labels] for sub, v in d.items()], ["group"] + [f"label={l}" for l in labels])

    md = [f"## Dreaddit (`{hf_id}`)", "",
          f"- Version: revision `{version.get('revision_sha', UNVERIFIED)}`",
          f"- Licence (from HF card): **{audit['licence']['value']}**",
          f"- Split sizes: {split_sizes} (expected train 2,838 / test 715) → {'MATCH' if split_sizes == expected else 'MISMATCH'}",
          f"- Core columns present: {audit['core_columns_present']}; missing: {missing_core or 'none'}",
          f"- Precomputed feature columns: **{len(feature_cols)}** (groups: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()) + "). **Decision: not used.**",
          f"- post_id values in both train and test: **{len(leak_posts)}**; normalised texts shared by train and test: **{audit['text_overlap_train_test_normalised']}**",
          f"- Segments per post_id ({seg_per_post.size} posts): {audit['segments_per_post']['distribution']}",
          f"- social_timestamp range (UTC): {audit['timestamp_range_utc']['min']} → {audit['timestamp_range_utc']['max']}; by year: {audit['year_counts']}",
          f"- Text artefacts (rows containing): {artefacts}", "",
          "### Label distribution", ""]
    for s in frames:
        md += [f"**{s}**", "", md_counts(audit["labels"][s], "label"), ""]
    md += ["### Subreddit × label (all splits)", "", xtab_md(audit["subreddit_x_label_all"]), "",
           f"### Domain × label (mapping {audit['domain_mapping']['source']})", "", xtab_md(audit["domain_x_label"]), "",
           "### Annotator confidence", "", md_desc(audit["confidence"]["describe"]), "",
           "### Missing values and duplicates", ""]
    md += [md_table([(s, audit["missing_text"][s]["n_missing"], audit["missing_text"][s]["n_empty_or_whitespace"],
                      audit["duplicates"][s]["exact_duplicate_rows"], audit["duplicates"][s]["normalised_duplicate_rows"]) for s in frames],
                    ["split", "missing text", "empty text", "exact dup rows", "normalised dup rows"]), ""]
    for s in frames:
        md += [f"### Text length: {s}", "", md_lengths(audit["lengths"][s]), ""]
    md += ["### Flags", ""] + ([f"- ⚠ {f}" for f in flags] or ["- none"]) + [""]
    save_audit(cfg, "dreaddit", audit, "\n".join(md))
    return audit


# ================================================================ GoEmotions
def _read_goemotions_tsv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = path.read_bytes()
    n_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    first = raw.split(b"\n", 1)[0].decode("utf-8").split("\t")
    has_header = not (len(first) >= 2 and re.fullmatch(r"\d+(,\d+)*", first[1] or "") is not None)
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=0 if has_header else None, names=None if has_header else ["text", "labels", "id"],
                     dtype=str, keep_default_na=False, encoding="utf-8")  # standard CSV quoting: texts with quotes are "..."-wrapped
    return df, {"n_lines": n_lines, "has_header_row": has_header, "n_parsed_rows": int(len(df)), "n_fields_first_line": len(first)}


def try_kaggle_download(dataset_id: str, dest: Path) -> dict[str, Any]:
    """Download a Kaggle dataset if credentials exist. Never raises; returns a status dict."""
    if dest.exists() and any(dest.iterdir()):
        return {"status": "present", "path": dest.as_posix()}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # importing `kaggle` authenticates immediately
        api = KaggleApi()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=str(dest), unzip=True, quiet=True)
        return {"status": "downloaded", "path": dest.as_posix()}
    except BaseException as exc:  # noqa: BLE001 - kaggle raises OSError/SystemExit without credentials
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"[:400]}


def _label_matrix(label_lists: Sequence[Sequence[int]], n_labels: int) -> np.ndarray:
    m = np.zeros((len(label_lists), n_labels), dtype=np.int8)
    for i, ls in enumerate(label_lists):
        m[i, list(ls)] = 1
    return m


def audit_goemotions(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg["sources"]["goemotions"]
    odir = project_path(cfg, src["official_dir"])
    files = {f: download(f"{src['official_base_url']}/{f}", odir / f) for f in src["official_files"]}
    emotions = [l.strip() for l in (odir / "emotions.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    ekman = json.loads((odir / "ekman_mapping.json").read_text(encoding="utf-8"))
    sentiment = json.loads((odir / "sentiment_mapping.json").read_text(encoding="utf-8"))
    repo_meta = github_repo_meta(cfg, "google-research/google-research")

    official: dict[str, pd.DataFrame] = {}
    parse: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        official[split], parse[split] = _read_goemotions_tsv(odir / f"{split}.tsv")
        official[split]["label_ids"] = official[split]["labels"].map(lambda s: [int(x) for x in s.split(",") if x != ""])
    sizes = {s: int(len(d)) for s, d in official.items()}
    expected = {k: int(v) for k, v in src["official_split_sizes"].items()}
    flags: list[str] = []
    if sizes != expected:
        flags.append(f"official split sizes {sizes} != expected {expected}")
    for s, p in parse.items():
        if p["n_lines"] != p["n_parsed_rows"] + (1 if p["has_header_row"] else 0):
            flags.append(f"{s}.tsv: {p['n_lines']} lines but {p['n_parsed_rows']} parsed rows (quoting/embedded newline issue)")

    # --- Hugging Face fallback
    hf: dict[str, Any] = {"id": src["hf_fallback_id"], "config": src["hf_fallback_config"]}
    try:
        from datasets import load_dataset
        hds = load_dataset(src["hf_fallback_id"], src["hf_fallback_config"])
        hf_names = hds["train"].features["labels"].feature.names
        hf["split_sizes"] = {s: int(len(hds[s])) for s in hds.keys()}
        hf["label_names"] = hf_names
        hf["label_names_identical_to_official"] = hf_names == emotions
        cmp = {}
        for s_hf, s_off in (("train", "train"), ("validation", "dev"), ("test", "test")):
            if s_hf not in hds:
                continue
            h = hds[s_hf].to_pandas()
            o = official[s_off]
            merged = o.merge(h, on="id", how="outer", suffixes=("_off", "_hf"), indicator=True)
            both = merged[merged["_merge"] == "both"]
            cmp[s_hf] = {"ids_only_official": int((merged["_merge"] == "left_only").sum()),
                         "ids_only_hf": int((merged["_merge"] == "right_only").sum()),
                         "text_identical": int((both["text_off"] == both["text_hf"]).sum()),
                         "labels_identical": int(sum(sorted(a) == sorted(list(b)) for a, b in zip(both["label_ids"], both["labels_hf"]))),
                         "n_joined": int(len(both))}
        hf["comparison_with_official"] = cmp
        try:
            from huggingface_hub import HfApi
            info = HfApi().dataset_info(src["hf_fallback_id"])
            card = getattr(info, "card_data", None) or {}
            hf["licence"] = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
            hf["revision_sha"] = info.sha
        except Exception as exc:  # noqa: BLE001
            hf["licence_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001
        hf["error"] = repr(exc)

    # --- Kaggle copy (UNVERIFIED contents)
    kdir = project_path(cfg, src["kaggle_dir"])
    kaggle = try_kaggle_download(src["kaggle_id"], kdir)
    kaggle["kaggle_id"] = src["kaggle_id"]
    if kaggle["status"] in ("present", "downloaded"):
        off_hashes = {s: set(d["text"].map(text_hash)) for s, d in official.items()}
        kfiles = []
        for p in sorted(kdir.rglob("*")):
            if not p.is_file():
                continue
            entry: dict[str, Any] = {"file": p.relative_to(kdir).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            if p.suffix.lower() in (".csv", ".tsv", ".txt"):
                sep = "\t" if p.suffix.lower() == ".tsv" else ","
                try:
                    head = p.open(encoding="utf-8", errors="replace").readline().rstrip("\n").split(sep)
                    entry["first_line_fields"] = len(head)
                    entry["first_line_looks_like_header"] = any(h.strip().lower() in ("text", "id", "labels", "comment_text") or h.strip() in emotions for h in head)
                    kdf = pd.read_csv(p, sep=sep, header=0 if entry["first_line_looks_like_header"] else None, dtype=str,
                                      keep_default_na=False, quoting=csv.QUOTE_NONE if sep == "\t" else csv.QUOTE_MINIMAL)
                    entry["n_rows"] = int(len(kdf))
                    entry["columns"] = [str(c) for c in kdf.columns][:60]
                    entry["n_columns"] = int(kdf.shape[1])
                    entry["emotion_columns_present"] = [e for e in emotions if e in kdf.columns]
                    tcol = "text" if "text" in kdf.columns else (0 if 0 in kdf.columns else None)
                    if tcol is not None:
                        kh = set(kdf[tcol].map(text_hash))
                        entry["text_overlap_with_official"] = {s: len(kh & h) for s, h in off_hashes.items()}
                    entry["row_count_matches_official_split"] = [s for s, n in sizes.items() if n == entry["n_rows"]]
                except Exception as exc:  # noqa: BLE001
                    entry["parse_error"] = repr(exc)[:300]
            kfiles.append(entry)
        kaggle["files"] = kfiles

    # --- label statistics on the official files
    lab = {}
    for s, d in official.items():
        m = _label_matrix(d["label_ids"].tolist(), len(emotions))
        card = m.sum(1)
        lab[s] = {"per_label": {emotions[i]: int(m[:, i].sum()) for i in range(len(emotions))},
                  "labels_per_example": {str(k): int(v) for k, v in pd.Series(card).value_counts().sort_index().items()},
                  "multi_label_rate": round(float((card > 1).mean()), 4)}
    m_tr = _label_matrix(official["train"]["label_ids"].tolist(), len(emotions)).astype(np.int32)
    co = m_tr.T @ m_tr
    pairs = sorted(((emotions[i], emotions[j], int(co[i, j])) for i, j in itertools.combinations(range(len(emotions)), 2)), key=lambda x: -x[2])
    tok = load_tokenizer(cfg, "emotion")
    ekman_inv = {e: k for k, v in ekman.items() for e in v}
    unmapped_ekman = [e for e in emotions if e not in ekman_inv and e != "neutral"]

    audit = {
        "name": "goemotions",
        "official": {"files": files, "repo": repo_meta, "parse": parse, "split_sizes": sizes, "expected": expected,
                     "labels_emotions_txt": emotions, "n_labels": len(emotions)},
        "licence": {"official_repo_spdx": repo_meta.get("licence_spdx", UNVERIFIED), "hf_card": hf.get("licence") or UNVERIFIED,
                    "kaggle": UNVERIFIED, "note": "repo licence covers the code repository; dataset terms per HF card"},
        "hf_fallback": hf, "kaggle": kaggle,
        "ekman_mapping": {"source": f"{src['official_base_url']}/ekman_mapping.json (official google-research repo)",
                          "mapping": ekman, "emotions_not_mapped_excluding_neutral": unmapped_ekman},
        "sentiment_mapping": sentiment,
        "labels": lab,
        "cooccurrence_train": {"labels": emotions, "matrix": co.tolist(), "top_pairs": pairs[:20]},
        "missing_text": {s: missing_text_stats(d["text"]) for s, d in official.items()},
        "duplicates": {s: duplicate_stats(d["text"]) for s, d in official.items()},
        "lengths": {s: length_stats(d["text"], tok, cfg) for s, d in official.items()},
        "flags": flags,
    }
    md = ["## GoEmotions", "",
          f"- Official files: `{src['official_base_url']}` (google-research repo, licence {repo_meta.get('licence_spdx', UNVERIFIED)}, latest repo commit {repo_meta.get('commit_sha', '?')})",
          f"- Header row in official TSVs: " + ", ".join(f"{s}={p['has_header_row']}" for s, p in parse.items()),
          f"- Split sizes (official files): {sizes} vs expected {expected} → {'MATCH' if sizes == expected else 'MISMATCH'}",
          f"- Label set ({len(emotions)}): {', '.join(emotions)}",
          f"- Ekman mapping: from `ekman_mapping.json` in the official repo; groups: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in ekman.items())
          + f"; not mapped (besides neutral): {unmapped_ekman or 'none'}",
          f"- HF fallback `{hf['id']}` / `{hf['config']}`: sizes {hf.get('split_sizes', hf.get('error'))}; label names identical to official: {hf.get('label_names_identical_to_official')}; licence (card): {hf.get('licence', UNVERIFIED)}",
          ]
    if "comparison_with_official" in hf:
        md += ["", md_table([(s, c["n_joined"], c["ids_only_official"], c["ids_only_hf"], c["text_identical"], c["labels_identical"])
                             for s, c in hf["comparison_with_official"].items()],
                            ["HF split", "ids joined", "only official", "only HF", "text identical", "labels identical"])]
    md += ["", f"- Kaggle `{src['kaggle_id']}`: **{kaggle['status']}**" + (f" ({kaggle.get('reason')})" if kaggle.get("reason") else "")]
    for f in kaggle.get("files", []):
        md.append(f"  - `{f['file']}`: rows={f.get('n_rows')}, cols={f.get('n_columns')}, header-like first line={f.get('first_line_looks_like_header')}, "
                  f"matches split size={f.get('row_count_matches_official_split')}, text overlap with official={f.get('text_overlap_with_official')}")
    md += ["", "### Multi-label structure", "",
           md_table([(s, l["multi_label_rate"], l["labels_per_example"]) for s, l in lab.items()], ["split", "multi-label rate", "labels per example"]), "",
           "### Per-label counts", "",
           md_table([(e, *(lab[s]["per_label"][e] for s in lab)) for e in emotions], ["label", *lab.keys()]), "",
           "### Top 15 co-occurring label pairs (train)", "",
           md_table(pairs[:15], ["label A", "label B", "n"]), "",
           "### Missing values and duplicates", "",
           md_table([(s, audit["missing_text"][s]["n_missing"], audit["missing_text"][s]["n_empty_or_whitespace"],
                      audit["duplicates"][s]["exact_duplicate_rows"], audit["duplicates"][s]["normalised_duplicate_rows"]) for s in official],
                    ["split", "missing", "empty", "exact dup rows", "normalised dup rows"]), ""]
    for s in official:
        md += [f"### Text length: {s}", "", md_lengths(audit["lengths"][s]), ""]
    md += ["### Flags", ""] + ([f"- ⚠ {f}" for f in flags] or ["- none"]) + [""]
    save_audit(cfg, "goemotions", audit, "\n".join(md))
    return audit


# ================================================================ SAD
def audit_sad(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg["sources"]["sad"]
    rdir = project_path(cfg, src["raw_dir"])
    zinfo = download(src["zip_url"], rdir / "SAD_v1.zip")
    lic = download(src["zip_url"].rsplit("/", 1)[0] + "/LICENSE", rdir / "LICENSE")
    with zipfile.ZipFile(rdir / "SAD_v1.zip") as z:
        members = [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()]
        z.extractall(rdir / "unzipped")
    xlsx = next((p for p in (rdir / "unzipped").rglob("*.xlsx") if not p.name.startswith("~$")), None)
    schema = next((p for p in (rdir / "unzipped").rglob("*.txt")), None)
    local_xlsx = project_path(cfg, src["local_path"])
    flags: list[str] = []
    same_as_local = None
    if xlsx is not None and local_xlsx.exists():
        same_as_local = sha256_file(xlsx) == sha256_file(local_xlsx)
        if not same_as_local:
            flags.append("SAD_v1.xlsx in the GitHub zip differs from the local copy in data/raw/SAD_v1 (using the zip)")
    path = xlsx or local_xlsx
    sheets = pd.read_excel(path, sheet_name=None)
    sheet_name, df = max(sheets.items(), key=lambda kv: len(kv[1]))
    repo = github_repo_meta(cfg, src["repo"])
    lic_text = Path(lic["path"]).read_text(encoding="utf-8", errors="replace")
    lic_first = lic_text.strip().splitlines()[0] if lic_text.strip() else ""

    text_col = "sentence" if "sentence" in df.columns else None
    if text_col is None:
        flags.append(f"no `sentence` column; columns are {list(df.columns)}")
    cat_cols = [c for c in df.columns if c in ("Financial Problem", "Other", "Everyday Decision Making", "Emotional Turmoil", "School",
                                                 "Family Issues", "Social Relationships", "Work", "Health, Fatigue, or Physical Pain")]
    schema_text = schema.read_text(encoding="utf-8", errors="replace") if schema else ""
    if "Source" in df.columns and schema_text:
        undocumented = sorted(v for v in df["Source"].astype(str).unique() if v not in schema_text)
        if undocumented:
            flags.append(f"Source values not named in Schema.txt: {undocumented}")
    if "is_stressor" in df.columns:
        share = float((df["is_stressor"] == 1).mean())
        if share > 0.9:
            flags.append(f"is_stressor=1 for {share:.1%} of rows: almost no non-stressor negatives")
    tok = load_tokenizer(cfg, "stressor")
    sev_col = next((c for c in ("avg_severity", "median_severity") if c in df.columns), None)
    audit = {
        "name": "sad", "zip": zinfo, "zip_members": members, "xlsx_used": path.as_posix(), "sheets": {k: int(len(v)) for k, v in sheets.items()},
        "sheet_used": sheet_name, "xlsx_identical_to_local_copy": same_as_local,
        "licence": {"github_spdx": repo.get("licence_spdx", UNVERIFIED), "license_file_first_line": lic_first, "license_sha256": lic["sha256"]},
        "repo": repo,
        "schema_txt": schema.read_text(encoding="utf-8", errors="replace") if schema else UNVERIFIED,
        "overview": frame_overview(df),
        "label_columns": {c: counts(df[c]) for c in ("top_label", "second_label", "original_label", "is_stressor", "is_covid", "Source", "is_seed") if c in df.columns},
        "category_weight_columns": {c: describe_numeric(df[c]) for c in cat_cols},
        "category_weight_nonzero": {c: int((df[c].fillna(0) > 0).sum()) for c in cat_cols},
        "severity": {c: describe_numeric(df[c]) for c in ("avg_severity", "median_severity", "SD_severity") if c in df.columns},
        "agreement": {c: describe_numeric(df[c]) for c in ("is_stressor_conf", "is_covid_conf") if c in df.columns},
        "missing_text": missing_text_stats(df[text_col]) if text_col else None,
        "duplicates": duplicate_stats(df[text_col]) if text_col else None,
        "lengths": length_stats(df[text_col], tok, cfg) if text_col else None,
        "source_x_top_label": pd.crosstab(df["Source"], df["top_label"]).to_dict(orient="index") if {"Source", "top_label"} <= set(df.columns) else None,
        "flags": flags,
    }
    md = ["## SAD (Stress-Annotated Dataset)", "",
          f"- Zip: `{src['zip_url']}` sha256 `{zinfo['sha256'][:16]}…`; members: " + ", ".join(f"{m['name']} ({m['bytes']} B)" for m in members),
          f"- XLSX used: `{Path(path).name}`, sheets {audit['sheets']}; identical to local `data/raw/SAD_v1` copy: {same_as_local}",
          f"- Licence: GitHub reports **{audit['licence']['github_spdx']}**; LICENSE file first line: \"{lic_first}\"",
          f"- Repo commit: {repo.get('commit_sha')} ({repo.get('commit_date')})", "",
          "### Schema.txt (verbatim)", "", "```", audit["schema_txt"].strip(), "```", "",
          "### Columns", "", md_overview(audit["overview"]), ""]
    for c, v in audit["label_columns"].items():
        md += [f"### `{c}` (the file's own counts)", "", md_counts(v, c), ""]
    if cat_cols:
        md += ["### Category weight columns (rows with weight > 0)", "", md_table(list(audit["category_weight_nonzero"].items()), ["category", "n rows > 0"]), ""]
    for c, v in audit["severity"].items():
        md += [f"### `{c}`", "", md_desc(v), ""]
    if text_col:
        md += ["### Missing / duplicates", "", f"Missing: {audit['missing_text']}; duplicates: {audit['duplicates']}", "",
               "### Text length", "", md_lengths(audit["lengths"]), ""]
    md += ["### Flags", ""] + ([f"- ⚠ {f}" for f in flags] or ["- none"]) + [""]
    save_audit(cfg, "sad", audit, "\n".join(md))
    return audit


# ================================================================ Zenodo pilot
def zenodo_header(cfg: dict[str, Any], subreddit: str = "teaching", window: str = "post") -> dict[str, Any]:
    """Read ONLY the header of one Zenodo CSV and classify its columns."""
    src = cfg["sources"]["zenodo_pilot"]
    p = project_path(cfg, src["local_dir"]) / src["file_pattern"].format(subreddit=subreddit, window=window)
    with p.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    feat_prefixes = ("liwc_", "tfidf_", "sent_", "n_")
    text_candidates = [c for c in header if c.lower() in ("post", "text", "body", "selftext", "title")]
    return {"file": p.name, "n_columns": len(header), "columns": header,
            "non_feature_columns": [c for c in header if not c.startswith(feat_prefixes)],
            "text_column_candidates": text_candidates,
            "author_columns": author_columns(header, cfg),
            "date_columns": [c for c in header if c.lower() in ("date", "created_utc", "timestamp", "time")],
            "id_columns": [c for c in header if c.lower() in ("id", "post_id", "postid", "name", "link_id")]}


def audit_zenodo(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg["sources"]["zenodo_pilot"]
    zdir = project_path(cfg, src["local_dir"])
    mdir = project_path(cfg, cfg["audit"]["metadata_dir"])
    try:
        rec = fetch_json(src["api_url"], mdir / f"zenodo_{src['record_id']}.json")
        meta = {"title": rec["metadata"].get("title"), "licence": (rec["metadata"].get("license") or {}).get("id"),
                "version": rec["metadata"].get("version"), "doi": rec.get("doi"), "publication_date": rec["metadata"].get("publication_date"),
                "files_in_record": len(rec.get("files", []))}
        desc = re.sub(r"<[^>]+>", " ", rec["metadata"].get("description", "")).replace("&nbsp;", " ")
        i = desc.find("License")
        meta["licence_quote"] = re.sub(r"\s+", " ", desc[i:i + 330]).strip() if i >= 0 else UNVERIFIED
        rec_sizes = {f["key"]: f["size"] for f in rec.get("files", [])}
    except Exception as exc:  # noqa: BLE001
        meta, rec_sizes = {"error": repr(exc)}, {}

    head = zenodo_header(cfg)
    text_col = head["text_column_candidates"][0] if head["text_column_candidates"] else None
    present = sorted(p.name for p in zdir.glob("*.csv"))
    flags: list[str] = []
    audit: dict[str, Any] = {"name": "zenodo_pilot", "record": meta, "teaching_post_header": head, "files_present": present,
                             "suicidewatch_file_present": any("suicidewatch" in f.lower() for f in present),
                             "windows_present": sorted({m.group(2) for f in present if (m := re.match(r"(.+)_(pre|post|2018|2019)_features", f))}),
                             "flags": flags}
    if audit["suicidewatch_file_present"]:
        flags.append("a suicidewatch file is present in the pilot folder")
    if not head["id_columns"]:
        flags.append("no post id column: deduplication must use text hashes")
    if head["author_columns"]:
        flags.append(f"author column(s) {head['author_columns']} present: never loaded; dropped at ingestion")
    if text_col is None:
        flags.append("NO usable text column: recommend piloting on one small subreddit from the main corpus instead")
        audit["usable_text"] = False
    else:
        audit["usable_text"] = True
        tok = load_tokenizer(cfg, "stress")
        usecols = [c for c in ("subreddit", "date", text_col, "n_chars", "n_words") if c in head["columns"]]  # author never read
        per_file, hashes = {}, {}
        for sub in src["subreddits"]:
            for win in src["windows"]:
                p = zdir / src["file_pattern"].format(subreddit=sub, window=win)
                if not p.exists():
                    continue
                df = pd.read_csv(p, usecols=usecols, dtype={text_col: "string"}, keep_default_na=False, na_values=[""])
                dt = pd.to_datetime(df["date"], errors="coerce", utc=True) if "date" in df else pd.Series(dtype="datetime64[ns, UTC]")
                txt = df[text_col].dropna().astype(str)
                hashes[(sub, win)] = set(txt.map(text_hash))
                char_len = txt.str.len()
                entry = {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                         "bytes_match_zenodo_record": rec_sizes.get(p.name) == p.stat().st_size if rec_sizes else None,
                         "n_rows": int(len(df)), "subreddit_values": counts(df["subreddit"], normalize=False)["counts"] if "subreddit" in df else None,
                         "date_min": str(dt.min()), "date_max": str(dt.max()), "n_date_unparseable": int(dt.isna().sum()),
                         "date_format_example": str(df["date"].iloc[0]) if len(df) else None,
                         "missing_text": missing_text_stats(df[text_col]), "duplicates": duplicate_stats(df[text_col]),
                         "share_with_uppercase": round(float(txt.str.contains(r"[A-Z]").mean()), 4),
                         "share_with_sentence_punct": round(float(txt.str.contains(r"[.!?]").mean()), 4),
                         "share_with_newline": round(float(txt.str.contains("\n").mean()), 4),
                         "share_with_url_like": round(float(txt.str.contains(r"https?://|www\.", regex=True).mean()), 4),
                         "lengths": length_stats(df[text_col], tok, cfg)}
                if "n_chars" in df:
                    nc = pd.to_numeric(df["n_chars"], errors="coerce")
                    entry["n_chars_feature_vs_len_text"] = {"corr": round(float(np.corrcoef(nc.fillna(0), df[text_col].fillna("").str.len())[0, 1]), 4),
                                                            "median_ratio": round(float((df[text_col].fillna("").str.len() / nc.replace(0, np.nan)).median()), 4)}
                per_file[f"{sub}_{win}"] = entry
        cross = {}
        for sub in src["subreddits"]:
            wins = [w for w in src["windows"] if (sub, w) in hashes]
            for a, b in itertools.combinations(wins, 2):
                cross[f"{sub}: {a}∩{b}"] = len(hashes[(sub, a)] & hashes[(sub, b)])
        subs = list(src["subreddits"])
        cross_sub = {f"{a}∩{b} ({w})": len(hashes[(a, w)] & hashes[(b, w)])
                     for w in src["windows"] for a, b in itertools.combinations(subs, 2) if (a, w) in hashes and (b, w) in hashes}
        audit.update({"text_column": text_col, "per_file": per_file, "cross_window_duplicate_texts": cross,
                      "cross_subreddit_duplicate_texts_nonzero": {k: v for k, v in cross_sub.items() if v}})
        missing_files = [f"{s}_{w}" for s in src["subreddits"] for w in src["windows"] if f"{s}_{w}" not in per_file]
        audit["configured_but_missing"] = missing_files
        for win in src["windows"]:
            mins = [v["date_min"][:10] for k, v in per_file.items() if k.endswith(f"_{win}")]
            maxs = [v["date_max"][:10] for k, v in per_file.items() if k.endswith(f"_{win}")]
            if mins:
                audit.setdefault("window_observed_ranges", {})[win] = {"min": min(mins), "max": max(maxs)}
        desc_txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", rec["metadata"].get("description", "")).replace("&nbsp;", " ")) if meta.get("title") else ""
        audit["window_definitions_from_record"] = {w: (m.group(0) if (m := re.search(rf"{w}:\s*([^.]*)", desc_txt)) else UNVERIFIED) for w in src["windows"]}
        obs_pre = audit.get("window_observed_ranges", {}).get("pre")
        if obs_pre and "Dec 2018" in audit["window_definitions_from_record"].get("pre", "") and obs_pre["min"] < "2018-12-01":
            flags.append(f"pre window observed {obs_pre['min']}..{obs_pre['max']} but the record describes it as 'Dec 2018 to Dec 2019'")
        uc = [v["share_with_uppercase"] for v in per_file.values()]
        if uc and max(uc) < 0.05:
            flags.append("text appears lower-cased/pre-processed (almost no uppercase): may not be the original post text")

    md = ["## Zenodo Reddit Mental Health Dataset (pilot)", "",
          f"- Record {src['record_id']}: {meta.get('title')} v{meta.get('version')} ({meta.get('publication_date')}), licence id **{meta.get('licence', UNVERIFIED)}**",
          f"- Licence quote: \"{meta.get('licence_quote', UNVERIFIED)}\"",
          f"- Header of `{head['file']}` ({head['n_columns']} columns). Non-feature columns: {head['non_feature_columns']}",
          f"- Text column: **{text_col or 'NONE'}**; author column(s): **{head['author_columns'] or 'none'}** (never loaded); "
          f"date column(s): **{head['date_columns'] or 'none'}**; post id column: **{head['id_columns'] or 'NONE'}**",
          f"- Files present: {len(present)}; windows present: {audit['windows_present']}; suicidewatch file present: **{audit['suicidewatch_file_present']}**"]
    if audit.get("usable_text"):
        md += ["", "### Rows, dates and text form per file", "",
               md_table([(k, v["n_rows"], v["date_min"][:10], v["date_max"][:10], v["n_date_unparseable"], v["missing_text"]["n_missing"],
                          v["duplicates"]["exact_duplicate_rows"], v["share_with_uppercase"], v["share_with_sentence_punct"], v["share_with_url_like"],
                          v["lengths"]["chars"].get("p50"), v["lengths"]["tokens"].get("p50"), v["lengths"]["tokens"].get("share_over", {}).get("512"),
                          v["bytes_match_zenodo_record"])
                         for k, v in audit["per_file"].items()],
                        ["file", "rows", "date min", "date max", "bad dates", "missing text", "exact dup rows", "has uppercase", "has . ! ?",
                         "url-like", "median chars", "median tokens", ">512 tok", "size = record"]), "",
               "### Duplicate texts between windows (normalised hash)", "",
               md_table(list(audit["cross_window_duplicate_texts"].items()), ["pair", "shared texts"]), "",
               f"Non-zero cross-subreddit duplicates: {audit['cross_subreddit_duplicate_texts_nonzero'] or 'none'}", ""]
        md += ["Window definitions (record) vs observed date ranges:", "",
               md_table([(w, audit["window_definitions_from_record"].get(w), audit.get("window_observed_ranges", {}).get(w)) for w in src["windows"]],
                        ["window", "record description", "observed"]), ""]
        nchar = {k: v.get("n_chars_feature_vs_len_text") for k, v in audit["per_file"].items()}
        md += [f"`n_chars` feature vs length of text column (corr, median ratio): {nchar}", ""]
    md += ["### Flags", ""] + ([f"- ⚠ {f}" for f in flags] or ["- none"]) + [""]
    save_audit(cfg, "zenodo_pilot", audit, "\n".join(md))
    return audit


# ================================================================ Mendeley
def audit_mendeley(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg["sources"]["mendeley_stress_indicators"]
    path = project_path(cfg, src["local_path"])
    mdir = project_path(cfg, cfg["audit"]["metadata_dir"])
    flags: list[str] = []
    try:
        meta = fetch_json(src["api_url"], mdir / "mendeley_2gsjv8m7ch.json")
    except Exception as exc:  # noqa: BLE001
        meta = {"error": repr(exc)}
    df = pd.read_csv(path)
    file_sha = sha256_file(path)
    api_sha = next((f.get("content_details", {}).get("sha256_hash") for f in meta.get("files", []) if f.get("filename") == path.name), None)
    desc = meta.get("description", "")
    m = re.search(r"stress_type.*?:\s*(.*?)(?:\n\s*\n\s*📂|\Z)", desc, flags=re.S)
    code_lines = re.findall(r"(\d)\s*=\s*([^\n]+)", m.group(1)) if m else []
    codes = {c: lab.strip() for c, lab in code_lines}
    numeric = df.select_dtypes(include="number").columns.tolist()
    grouped = {c for v in MENDELEY_GROUPS.values() for c in v}
    ungrouped = [c for c in df.columns if c not in grouped]
    if ungrouped:
        flags.append(f"columns not in the authors' grouping: {ungrouped}")
    expansion_quote = next((c.get("contribution") for c in meta.get("contributors", []) if c.get("contribution")), None)
    n_dup = int(df.duplicated().sum())
    if n_dup and re.search(r"Duplicates:\s*None", desc):
        flags.append(f"{n_dup} of {len(df)} rows are exact duplicates, but the authors' description states 'Duplicates: None'")
    tgt = df["stress_type"].map(lambda v: f"{v} = {codes.get(str(v), UNVERIFIED)}") if "stress_type" in df else None
    audit = {
        "name": "mendeley", "version": {"doi": (meta.get("doi") or {}).get("id"), "version": meta.get("version"), "publish_date": meta.get("publish_date")},
        "licence": {"value": (meta.get("data_licence") or {}).get("short_name", UNVERIFIED), "url": (meta.get("data_licence") or {}).get("url")},
        "sha256_file": file_sha, "sha256_repository": api_sha, "sha256_match": api_sha == file_sha if api_sha else None,
        "overview": frame_overview(df), "is_tabular": True, "all_numeric": len(numeric) == df.shape[1],
        "free_text_columns": [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 30],
        "value_ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max()), "n_unique": int(df[c].nunique())} for c in numeric},
        "groups_from_authors": MENDELEY_GROUPS, "ungrouped": ungrouped,
        "stress_type_codes_from_authors": codes or UNVERIFIED,
        "stress_type_balance": counts(tgt) if tgt is not None else None,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "expansion_note": {"quote_description": "The original dataset was collected via Google Forms survey and later expanded to ensure sufficient sample size for machine learning research." if "later expanded" in desc else UNVERIFIED,
                           "quote_contributor": expansion_quote or UNVERIFIED,
                           "implication": "Rows are partly synthetic/expanded: NOT suitable for prevalence estimation. Used only to derive the stressor taxonomy."},
        "flags": flags,
    }
    md = ["## Mendeley: Stress Indicators Dataset for Mental Health Classification", "",
          f"- DOI {audit['version']['doi']}, version {audit['version']['version']} ({audit['version']['publish_date']}); licence **{audit['licence']['value']}**",
          f"- File SHA-256 matches the repository's published hash: **{audit['sha256_match']}**",
          f"- **TABULAR** survey data: {df.shape[0]} rows × {df.shape[1]} columns; all numeric: {audit['all_numeric']}; free-text columns: {audit['free_text_columns'] or 'none'}",
          f"- Missing values: {audit['missing_total']}; duplicate rows: {audit['duplicate_rows']}",
          "", "> **Note:** the authors describe the file as follows: \"" + str(audit["expansion_note"]["quote_description"]) + "\" and list their contribution as \""
          + str(audit["expansion_note"]["quote_contributor"]) + "\". The rows are therefore not a sample of a real population and **must not be used for prevalence estimation**. Role: stressor taxonomy only.",
          "", "### Column groups (taken from the authors' own \"Key Features\" list)", "",
          md_table([(g, ", ".join(c)) for g, c in MENDELEY_GROUPS.items()], ["group", "columns"]), "",
          "### Value ranges", "", md_table([(c, v["min"], v["max"], v["n_unique"]) for c, v in audit["value_ranges"].items()], ["column", "min", "max", "unique"]), "",
          "### stress_type balance (codes from the authors' description)", "", md_counts(audit["stress_type_balance"], "stress_type") if tgt is not None else "n/a", "",
          "### Flags", ""] + ([f"- ⚠ {f}" for f in flags] or ["- none"]) + [""]
    save_audit(cfg, "mendeley", audit, "\n".join(md))
    return audit


# ================================================================ TensiStrength
def run_tensistrength(cfg: dict[str, Any], text: str, explain: bool = False) -> str | None:
    """Run the TensiStrength jar on one SYNTHETIC/scrubbed text. Returns raw stdout or None if Java/jar missing."""
    src = cfg["sources"]["tensistrength"]
    d = project_path(cfg, src["local_dir"])
    jar, data = d / src["jar_name"], d / "TensiStrength_Data"
    if java_version() is None or not jar.exists() or not data.exists():
        return None
    args = ["java", "-jar", str(jar), "sentidata", data.as_posix() + "/", "text", text.replace(" ", "+")] + (["explain"] if explain else [])
    res = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return (res.stdout or res.stderr).strip()


def audit_tensistrength(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg["sources"]["tensistrength"]
    d = project_path(cfg, src["local_dir"])
    files = {f: download(f"{src['base_url']}/{f}", d / f) for f in src["files"]}
    with zipfile.ZipFile(d / src["resource_zip"]) as z:
        z.extractall(d / "TensiStrength_Data")
    res_files = []
    for p in sorted((d / "TensiStrength_Data").rglob("*")):
        if p.is_file():
            lines = p.read_text(encoding="latin-1").splitlines()
            ncols = Counter(len(l.split("\t")) for l in lines if l.strip())
            res_files.append({"file": p.name, "bytes": p.stat().st_size, "n_lines": len(lines), "tab_fields": dict(ncols.most_common(3)),
                              "first_line": lines[0][:120] if lines else ""})
    lut = d / "TensiStrength_Data" / "SentimentLookupTable.txt"
    cats: dict[str, int] = {}
    if lut.exists():
        rows = [l.split("\t") for l in lut.read_text(encoding="latin-1").splitlines() if l.strip()]
        cats = dict(Counter(r[3].strip().lower() if len(r) > 3 else "<none>" for r in rows).most_common())
    readme = (d / "README.md").read_text(encoding="utf-8", errors="replace")
    lic_lines = [l.strip() for l in readme.splitlines() if "TensiStrength" in l or "£1000" in l or "commercial" in l.lower()]
    manual_txt = ""
    raw = (d / "SentiStrengthJavaManual.doc").read_bytes()
    manual_txt = " ".join(m.decode("cp1252") for m in re.findall(rb"[\x20-\x7e\r\n\t]{25,}", raw)).replace("\r", "\n")
    (d / "manual_extracted.txt").write_text(manual_txt, encoding="utf-8")
    probes = ["i am so stressed and worried about exams", "feeling calm and relaxed today", "the bus leaves at noon"]
    smoke = {t: run_tensistrength(cfg, t) for t in probes}
    jv = java_version()
    audit = {
        "name": "tensistrength", "files": files, "repo": github_repo_meta(cfg, src["repo"]), "resource_files": res_files,
        "lookup_table_category_counts": cats,
        "contains_labelled_text": False,
        "assessment": "Resource/lexicon data only (term lookup tables, word lists, dictionary). No texts with labels, so no validation set.",
        "licence": {"readme_quotes": lic_lines, "summary": "Free for academic use; GBP 1000 for commercial use (README)."},
        "cli": {"single_text": "java -jar TensiStrengthMain.jar sentidata <TensiStrength_Data>/ text word+word+word [explain]",
                "file": "java -jar TensiStrengthMain.jar sentidata <dir>/ input <file.txt>   (one text per line; writes <file>_out.txt)",
                "column": "java -jar TensiStrengthMain.jar sentidata <dir>/ annotateCol <col#> inputFolder <dir> fileSubstring <txt>",
                "stdin": "java -jar TensiStrengthMain.jar stdin sentidata <dir>/",
                "manual_scale": "Manual (SentiStrength): result '+ve space -ve'; positive 1..5, negative -1..-5; trinary/binary/scale options add an overall column.",
                "observed_scale": "Observed on synthetic probes: first number = relaxation (1..5), second = stress (-1..-5); 1/-1 = none. Mapping of SentiStrength's +/- to relaxation/stress is inferred from outputs, not documented in the manual."},
        "java_version": jv, "smoke_test_synthetic": smoke,
        "flags": [] if jv else ["Java not found: TensiStrength cannot run"],
    }
    md = ["## TensiStrength (auxiliary lexicon signal)", "",
          f"- Downloaded from `{src['base_url']}`: " + ", ".join(f"{k} ({v['bytes']} B, sha256 {v['sha256'][:12]}…)" for k, v in files.items()),
          f"- Repo commit {audit['repo'].get('commit_sha')} ({audit['repo'].get('commit_date')})",
          f"- **Contains labelled text: NO.** {audit['assessment']}",
          f"- Licence (README): " + " / ".join(f"\"{q}\"" for q in lic_lines),
          f"- Java: {jv or 'NOT FOUND'}", "",
          "### Resource files", "", md_table([(r["file"], r["bytes"], r["n_lines"], r["tab_fields"]) for r in res_files], ["file", "bytes", "lines", "tab-field counts"]), "",
          "### SentimentLookupTable category column (4th field)", "", md_table(list(cats.items()), ["category", "n terms"]), "",
          "### Command-line usage (from SentiStrengthJavaManual.doc)", "", *[f"- {k}: `{v}`" if k != "manual_scale" and k != "observed_scale" else f"- {k}: {v}" for k, v in audit["cli"].items()], "",
          "### Smoke test (synthetic sentences, not user data)", "", md_table([(t, o) for t, o in smoke.items()], ["probe", "output"]), "",
          "### Flags", ""] + ([f"- ⚠ {f}" for f in audit["flags"]] or ["- none"]) + [""]
    save_audit(cfg, "tensistrength", audit, "\n".join(md))
    return audit


# ================================================================ SenticNet
def load_senticnet(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    src = cfg["sources"]["senticnet"]
    d = project_path(cfg, src["local_dir"])
    return {Path(f).stem: pd.read_excel(d / f) for f in src["files"]}


def senticnet_text_label_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Guess text/label columns from the real header: label = binary 0/1 column, text = longest-string column."""
    label = next((c for c in df.columns if set(pd.Series(df[c]).dropna().unique()) <= {0, 1} and df[c].nunique() == 2), None)
    obj = [c for c in df.columns if df[c].dtype == object]
    text = max(obj, key=lambda c: df[c].astype(str).str.len().mean()) if obj else None
    return text, label


def audit_senticnet(cfg: dict[str, Any], dreaddit_texts: Sequence[str] | None = None) -> dict[str, Any]:
    src = cfg["sources"]["senticnet"]
    d = project_path(cfg, src["local_dir"])
    frames = load_senticnet(cfg)
    lic_text = (d / "LICENSE").read_text(encoding="utf-8", errors="replace") if (d / "LICENSE").exists() else ""
    readme = (d / "README.md").read_text(encoding="utf-8", errors="replace") if (d / "README.md").exists() else ""
    ann_quote = next((s.strip() for s in re.split(r"(?<=\.)\s", readme) if "automated" in s.lower()), UNVERIFIED)
    if dreaddit_texts is None:
        try:
            from datasets import load_dataset
            dd = load_dataset(cfg["sources"]["dreaddit"]["hf_id"])
            dreaddit_texts = [t for s in dd for t in dd[s][cfg["sources"]["dreaddit"]["text_column"]]]
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Dreaddit unavailable for overlap check: %r", exc)
            dreaddit_texts = []
    dre_hash = {text_hash(t) for t in dreaddit_texts}
    tok = load_tokenizer(cfg, "stress")
    n, thr = int(cfg["audit"]["shingle_n"]), float(cfg["audit"]["overlap_threshold"])
    per: dict[str, Any] = {}
    flags: list[str] = []
    hashes = {}
    for name, df in frames.items():
        tcol, lcol = senticnet_text_label_cols(df)
        if tcol is None or lcol is None:
            flags.append(f"{name}: could not identify text/label columns from {list(df.columns)}")
        t = df[tcol].dropna().astype(str) if tcol else pd.Series(dtype=str)
        hashes[name] = set(t.map(text_hash))
        per[name] = {"overview": frame_overview(df), "text_column": tcol, "label_column": lcol,
                     "labels": counts(df[lcol]) if lcol else None, "missing_text": missing_text_stats(df[tcol]) if tcol else None,
                     "duplicates": duplicate_stats(df[tcol]) if tcol else None,
                     "lengths": length_stats(df[tcol], tok, cfg) if tcol else None,
                     "exact_overlap_with_dreaddit": len(hashes[name] & dre_hash),
                     "dreaddit_segments_contained": containment_overlap(dreaddit_texts, t.tolist(), n, thr) if (dreaddit_texts and name.startswith("Reddit")) else None}
        if per[name]["duplicates"] and per[name]["duplicates"]["normalised_duplicate_rows"]:
            # conflicting labels among duplicates are a direct noise signal
            g = df.assign(_h=df[tcol].map(text_hash)).groupby("_h")[lcol].nunique()
            per[name]["duplicate_groups_with_conflicting_labels"] = int((g > 1).sum())
    cross = {f"{a}∩{b}": len(hashes[a] & hashes[b]) for a, b in itertools.combinations(hashes, 2)}
    for k, v in per.items():
        if v.get("duplicate_groups_with_conflicting_labels"):
            flags.append(f"{k}: {v['duplicate_groups_with_conflicting_labels']} duplicate-text groups carry conflicting labels (direct label noise)")
        if v["labels"]:
            top = max(v["labels"]["proportions"].values())
            if top > 0.75:
                flags.append(f"{k}: majority class share {top:.1%}")
        dc = v.get("dreaddit_segments_contained") or {}
        if dc.get("n_contained"):
            flags.append(f"{k}: {dc['n_contained']} Dreaddit segments are ≥{thr:.0%} contained in this file (exclude from any Dreaddit-trained evaluation)")
    for pair, nshared in cross.items():
        a_, b_ = pair.split("∩")
        if nshared and nshared >= 0.9 * min(len(hashes[a_]), len(hashes[b_])):
            flags.append(f"{pair}: {nshared} shared texts, so one file is (nearly) a subset of the other; do not count them as independent")
    audit = {"name": "senticnet", "repo": github_repo_meta(cfg, src["repo"]),
             "licence": {"repo_licence_file_first_line": lic_text.strip().splitlines()[0] if lic_text.strip() else UNVERIFIED,
                         "note": "Repository licence; underlying Reddit/Twitter content remains subject to platform terms (UNVERIFIED for redistribution)."},
             "annotation_quote": ann_quote, "files": per, "cross_file_shared_texts": cross,
             "handcheck": "see data/processed/_cache/senticnet_handcheck.csv (filled in by the user; results in audit_senticnet_handcheck.json)",
             "flags": flags}
    md = ["## SenticNet stress-detection (cross-platform agreement check only)", "",
          f"- Licence file: \"{audit['licence']['repo_licence_file_first_line']}\" ({audit['repo'].get('licence_spdx')}); {audit['licence']['note']}",
          f"- README on labels: \"{ann_quote}\" → **automated labels, not ground truth**", "",
          md_table([(k, v["overview"]["n_rows"], v["text_column"], v["label_column"],
                     (v["labels"] or {}).get("counts"), v["duplicates"]["exact_duplicate_rows"] if v["duplicates"] else None,
                     v.get("duplicate_groups_with_conflicting_labels", 0),
                     v["lengths"]["chars"].get("p50") if v["lengths"] else None,
                     v["lengths"]["tokens"].get("p50") if v["lengths"] else None,
                     v["exact_overlap_with_dreaddit"],
                     (v["dreaddit_segments_contained"] or {}).get("n_contained"))
                    for k, v in per.items()],
                   ["file", "rows", "text col", "label col", "label counts", "exact dup rows", "dup groups w/ conflicting labels",
                    "median chars", "median tokens", "exact overlap w/ Dreaddit", f"Dreaddit segments ≥{thr:.0%} contained"]), "",
          "Columns per file: " + "; ".join(f"{k}: {[c['name'] for c in v['overview']['columns']]}" for k, v in per.items()), "",
          f"Shared texts between files (normalised): {cross}", ""]
    hc = interim_dir(cfg) / "audit_senticnet_handcheck.json"
    if hc.exists():
        h = json.loads(hc.read_text(encoding="utf-8"))
        md += ["### Hand-check of automated labels (user-annotated)", "", md_table([(k, v) for k, v in h.items()], ["metric", "value"]), ""]
    else:
        md += ["### Hand-check of automated labels", "", "_Pending: user fills data/processed/_cache/senticnet_handcheck.csv (see notebook Step 1)._", ""]
    for k, v in per.items():
        if v["lengths"]:
            md += [f"### Text length: {k}", "", md_lengths(v["lengths"]), ""]
    md += ["### Flags", ""] + ([f"- ⚠ {f}" for f in flags] or ["- none"]) + [""]
    save_audit(cfg, "senticnet", audit, "\n".join(md))
    return audit


def make_senticnet_handcheck(cfg: dict[str, Any], overwrite: bool = False) -> Path:
    """Write a stratified sample (file × label) for MANUAL checking. Local file only (contains raw text; gitignored)."""
    out = interim_dir(cfg) / "senticnet_handcheck.csv"
    if out.exists() and not overwrite:
        LOG.info("hand-check sheet exists, not overwriting: %s", out)
        return out
    from .privacy import scrub_text
    frames = load_senticnet(cfg)
    total = int(cfg["sources"]["senticnet"]["handcheck_n"])
    cells = [(k, lab) for k, df in frames.items() for lab in (0, 1)]
    per_cell = max(1, total // len(cells))
    rows = []
    for name, df in frames.items():
        tcol, lcol = senticnet_text_label_cols(df)
        for lab in (0, 1):
            sub = df[df[lcol] == lab]
            for idx, r in sub.sample(n=min(per_cell, len(sub)), random_state=cfg["primary_seed"]).iterrows():
                rows.append({"file": name, "row_index": idx, "auto_label": int(lab),
                             "text": scrub_text(str(r[tcol]), cfg), "human_label": "", "notes": ""})
    pd.DataFrame(rows).sample(frac=1.0, random_state=cfg["primary_seed"]).to_csv(out, index=False, encoding="utf-8-sig")
    LOG.info("wrote %d rows to %s (fill `human_label` with 0/1, leave blank if unsure)", len(rows), out)
    return out


def score_senticnet_handcheck(cfg: dict[str, Any]) -> dict[str, Any]:
    """Agreement between automated labels and the user's manual labels. Saves audit_senticnet_handcheck.json."""
    p = interim_dir(cfg) / "senticnet_handcheck.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    df = df[pd.to_numeric(df["human_label"], errors="coerce").isin([0, 1])].copy()
    df["human_label"] = df["human_label"].astype(int)
    res: dict[str, Any] = {"n_checked": int(len(df))}
    if len(df):
        agree = df["auto_label"] == df["human_label"]
        res["agreement"] = round(float(agree.mean()), 4)
        res["estimated_label_noise"] = round(1 - float(agree.mean()), 4)
        res["auto1_human0"] = int(((df.auto_label == 1) & (df.human_label == 0)).sum())
        res["auto0_human1"] = int(((df.auto_label == 0) & (df.human_label == 1)).sum())
        for f, g in df.groupby("file"):
            res[f"agreement_{f}"] = f"{round(float((g.auto_label == g.human_label).mean()), 3)} (n={len(g)})"
        try:
            from sklearn.metrics import cohen_kappa_score
            res["cohen_kappa"] = round(float(cohen_kappa_score(df.auto_label, df.human_label)), 4)
        except Exception:  # noqa: BLE001
            pass
    write_json(interim_dir(cfg) / "audit_senticnet_handcheck.json", res)
    return res


# ================================================================ report + manifest
ORDER = ("dreaddit", "goemotions", "sad", "zenodo_pilot", "mendeley", "tensistrength", "senticnet")


def build_report(cfg: dict[str, Any]) -> Path:
    """Concatenate per-dataset fragments (+ usage decision file if present) into data/outputs/reports/data_audit.md."""
    d = interim_dir(cfg)
    out = Path(cfg["_paths"]["reports"]) / "data_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Data audit (Step 1)", "", f"> {cfg['project']['ethics_banner']}", "",
             "All numbers below are generated by `stress_signals.audit` from the files listed. No raw user text is reproduced. "
             "Anything not confirmed from a source file or the user is marked UNVERIFIED.", ""]
    for name in ORDER:
        f = d / f"audit_{name}.md"
        parts += [f.read_text(encoding="utf-8") if f.exists() else f"## {name}\n\n_Not audited yet._\n", ""]
    decision = Path(cfg["_paths"]["reports"]) / "dataset_usage_decision.md"
    if decision.exists():
        parts += [decision.read_text(encoding="utf-8")]
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def write_audit_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    d = interim_dir(cfg)
    arts = sorted(d.glob("audit_*.json")) + [Path(cfg["_paths"]["reports"]) / "data_audit.md"]
    versions = {}
    for name in ORDER:
        f = d / f"audit_{name}.json"
        if f.exists():
            a = json.loads(f.read_text(encoding="utf-8"))
            versions[name] = a.get("version") or a.get("record") or a.get("repo") or a.get("official", {}).get("repo")
    return write_manifest(Path(cfg["_paths"]["manifests"]) / "step1_audit_manifest.json", artifacts=arts, seeds=cfg["seeds"],
                          dataset_versions=versions, root=cfg["_root"], extra={"step": 1})


AUDITS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "dreaddit": audit_dreaddit, "goemotions": audit_goemotions, "sad": audit_sad, "zenodo_pilot": audit_zenodo,
    "mendeley": audit_mendeley, "tensistrength": audit_tensistrength, "senticnet": audit_senticnet,
}


def run(cfg: dict[str, Any], only: Iterable[str] | None = None) -> dict[str, Any]:
    """Run selected audits (default: all). Failures are recorded, not fatal, so one bad source doesn't block the rest."""
    results: dict[str, Any] = {}
    for name in (list(only) if only else ORDER):
        try:
            results[name] = AUDITS[name](cfg)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("audit %s failed", name)
            results[name] = {"error": repr(exc)}
            save_audit(cfg, name, {"name": name, "error": repr(exc)}, f"## {name}\n\n**AUDIT FAILED:** `{exc!r}`\n")
    build_report(cfg)
    write_audit_manifest(cfg)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Step 1 dataset audit (no training).")
    ap.add_argument("--only", nargs="*", choices=list(AUDITS), help="subset of audits to run")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    res = run(cfg, a.only)
    for k, v in res.items():
        print(f"{k:<14} {'ERROR ' + v['error'] if 'error' in v else 'ok'}  flags={v.get('flags', [])}")


if __name__ == "__main__":
    main()
