"""Step 5: stressor taxonomy + stressor classifiers (anything that fits weights is RUN BY THE USER; see the notebook
banner cells). Stressor labels are a RESEARCH taxonomy of stressor LANGUAGE, not a clinical assessment.

Pieces:
    5A taxonomy      code/taxonomy/stressors_v1.yaml (validated here; SAD -> classifier classes -> taxonomy categories)
    5B gold set      ~400 Dreaddit stress-positive segments (train + validation splits only), annotation sheets,
                     dev/test split (ids only, in git), gold-set report. SINGLE annotator (user decision): no
                     inter-annotator agreement is measured or simulated
    5C methods       (1) keyword baseline, (2a) frozen sentence embeddings + logistic regression on SAD,
                     (2b) fine-tuned distilroberta-base on SAD (softmax over 9 SAD classes), (3) zero-shot similarity
                     to the taxonomy descriptions (the only option for categories SAD lacks)
    inference unit   posts -> sentence windows -> per-window category scores -> post score = aggregation over windows
                     (default max) -> per-category thresholds tuned on GOLD-DEV -> label set; empty set -> none_unclear
    5D bundle        <paths.models>/stressor/v<YYYYMMDD>/

Run layout: trained_models/stressor/runs/{embed_lr/<emb>/, finetune/<model>/seed<k>/, gold/<method key>/}
Nothing here stores raw text except the LOCAL annotation sheets under data/outputs/gold/ (gitignored).
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from . import stress_model as SM
from .metrics import bootstrap_multilabel, ece, fit_temperature, multilabel_metrics, nll, softmax, tune_label_thresholds
from .utils import (checkpoint_root, device_label, get_logger, model_version_dir, models_root, select_device, set_seed,
                    sha256_file, write_json, write_manifest)

LOG = get_logger("stress_signals.stressor_model")
TRAINING_BANNER = SM.TRAINING_BANNER
COMPONENT = "stressor"
NONE_COL = "none_unclear"
METHODS = ("keyword", "zero_shot", "embed_lr", "finetune")
SAD_LABELS = ("Work", "School", "Financial Problem", "Family Issues", "Social Relationships",
              "Health, Fatigue, or Physical Pain", "Everyday Decision Making", "Emotional Turmoil", "Other")
MENDELEY_COLUMNS = (
    "gender", "age", "stress_experience", "heartbeat_palpitations", "anxiety_tension", "sleep_problems", "restlessness",
    "headaches", "irritability", "concentration_problems", "sadness_low_mood", "health_issues", "loneliness_isolation",
    "academic_overload", "peer_competition", "relationship_stress", "professor_difficulties", "work_environment",
    "lack_relaxation_time", "home_environment", "low_academic_confidence", "subject_confidence", "academic_conflicts",
    "class_attendance", "weight_changes", "stress_type")
MENDELEY_ROLES = ("stressor", "symptom_response", "appraisal_not_stressor", "behaviour_not_stressor", "not_used")
CATEGORY_KEYS = ("id", "name", "coverage", "definition", "include", "exclude", "keywords", "paraphrases", "sad_labels",
                 "mendeley_columns", "dreaddit_subreddits")
SCOPE_NOTE = (
    "Stressor labels are a RESEARCH taxonomy of what a text names as a cause of pressure. They are not a clinical "
    "assessment, say nothing about any person, and feed aggregate signals only.")
LEAKAGE_NOTE = (
    "Gold texts come ONLY from Dreaddit's train and validation splits (stress label 1). Dreaddit's official test split, "
    "on which the Step 3 stress model was evaluated once (3D.2, written before this gold set existed), is never "
    "sampled, so that evaluation is untouched. Gold labels never train or tune the stress model. The stressor models "
    "never train on Dreaddit; gold-dev only tunes stressor thresholds and picks the method; gold-test is scored once. "
    "Caveat: the Step 3 model was TRAINED on these segments, so its stress probabilities on gold texts are in-sample "
    "and must not be evaluated on the gold set (dreaddit_split is recorded per gold row).")
ANNOTATION_NOTE = "single annotator, no inter-annotator agreement"
MACHINE_ANNOTATION_NOTE = "MACHINE-LABELLED by a zero-shot NLI model (not a human annotation); no inter-annotator agreement"
MACHINE_ANNOTATION_LIMITATION = (
    "The 'gold' labels were produced by a zero-shot NLI model (machine_labeler), NOT by a person. Every gold metric is "
    "AGREEMENT WITH THAT LABELLER, not correctness, and inherits its errors; it cannot show that the stressor classifier "
    "is right on Reddit. Treat all stressor results as provisional until a human annotates the sheet.")
ANNOTATION_LIMITATION = (
    "The gold set was labelled by a SINGLE annotator; no inter-annotator agreement (e.g. Cohen's kappa) was measured, "
    "so label reliability is unknown and every gold metric includes annotator error of unknown size.")


# ================================================================ settings / paths
def settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """stressor_model config (deep copy) with resolved embedding and fine-tune model ids."""
    s = copy.deepcopy(cfg["stressor_model"])
    e = s["embedding"]
    if e["model_choice"] not in e["model_options"]:
        raise ValueError(f"embedding.model_choice {e['model_choice']!r} not in {list(e['model_options'])}")
    e["hf_id"] = e["model_options"][e["model_choice"]]["hf_id"]
    f = s["finetune"]
    if f["model_choice"] not in f["model_options"]:
        raise ValueError(f"finetune.model_choice {f['model_choice']!r} not in {list(f['model_options'])}")
    f["hf_id"] = f["model_options"][f["model_choice"]]["hf_id"]
    return s


def runs_root(cfg: dict[str, Any]) -> Path:
    return models_root(cfg) / COMPONENT / "runs"


def lr_run_dir(cfg: dict[str, Any], emb_choice: str | None = None) -> Path:
    return runs_root(cfg) / "embed_lr" / (emb_choice or settings(cfg)["embedding"]["model_choice"])


def ft_run_dir(cfg: dict[str, Any], seed: int, model_choice: str | None = None) -> Path:
    return runs_root(cfg) / "finetune" / (model_choice or settings(cfg)["finetune"]["model_choice"]) / f"seed{seed}"


def ft_checkpoint_dir(cfg: dict[str, Any], run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    return checkpoint_root(cfg) / COMPONENT / run_dir.parent.name / run_dir.name


def method_key(cfg: dict[str, Any], method: str) -> str:
    """File-safe name of a method + its model variant, e.g. 'embed_lr__all-MiniLM-L6-v2'."""
    s = settings(cfg)
    variant = {"keyword": f"taxonomy-v{load_taxonomy(cfg)['version']}", "zero_shot": s["embedding"]["model_choice"],
               "embed_lr": s["embedding"]["model_choice"], "finetune": s["finetune"]["model_choice"]}[method]
    return f"{method}__{variant}"


def gold_run_dir(cfg: dict[str, Any], method: str) -> Path:
    return runs_root(cfg) / "gold" / method_key(cfg, method)


def embedding_cache_dir(cfg: dict[str, Any]) -> Path:
    """Local cache of sentence embeddings (derived from text, so it stays under data/, which is gitignored)."""
    return Path(cfg["_paths"]["interim"]) / "stressor_embeddings"


# ================================================================ 5A taxonomy
def taxonomy_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_root"]) / cfg["stressor_model"]["taxonomy_file"]


def load_taxonomy(cfg: dict[str, Any] | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate the taxonomy YAML (raises ValueError listing every problem)."""
    p = Path(path) if path else taxonomy_path(cfg)
    tax = yaml.safe_load(p.read_text(encoding="utf-8"))
    validate_taxonomy(tax)
    return tax


def validate_taxonomy(tax: dict[str, Any]) -> None:
    """Every category complete; every SAD label mapped exactly once; every Mendeley column given exactly one role,
    consistent with the categories' mendeley_columns lists."""
    errors: list[str] = []
    cats = tax.get("categories") or []
    ids = [c.get("id") for c in cats]
    if len(set(ids)) != len(ids):
        errors.append(f"duplicate category ids: {ids}")
    if NONE_COL in ids:
        errors.append(f"{NONE_COL!r} is a gold column, not a category")
    seen_sad: list[str] = []
    for c in cats:
        miss = [k for k in CATEGORY_KEYS if k not in c]
        if miss:
            errors.append(f"category {c.get('id')}: missing {miss}")
            continue
        if c["coverage"] not in ("sad", "zero_shot"):
            errors.append(f"category {c['id']}: coverage must be sad|zero_shot")
        if c["coverage"] == "sad" and not c["sad_labels"]:
            errors.append(f"category {c['id']}: coverage sad needs sad_labels")
        if c["coverage"] == "zero_shot" and c["sad_labels"]:
            errors.append(f"category {c['id']}: zero_shot coverage cannot have sad_labels")
        if len(c["paraphrases"]) != 3:
            errors.append(f"category {c['id']}: needs exactly 3 paraphrase descriptions")
        if not c["keywords"]:
            errors.append(f"category {c['id']}: no seed keywords")
        sub = c.get("sad_label_subcategory")
        if sub is not None and set(sub) != set(c["sad_labels"]):
            errors.append(f"category {c['id']}: sad_label_subcategory keys must equal sad_labels")
        if sub is None and len(c["sad_labels"]) > 1:
            errors.append(f"category {c['id']}: several SAD labels need sad_label_subcategory")
        seen_sad += list(c["sad_labels"])
    for n in tax.get("non_stressor_classes") or []:
        if n.get("id") in ids:
            errors.append(f"non-stressor class {n.get('id')} clashes with a category id")
        seen_sad += list(n.get("sad_labels") or [])
    dup = sorted({x for x in seen_sad if seen_sad.count(x) > 1})
    if dup or set(seen_sad) != set(SAD_LABELS):
        errors.append(f"SAD labels must map exactly once: duplicated {dup}, missing {sorted(set(SAD_LABELS) - set(seen_sad))}, "
                      f"unknown {sorted(set(seen_sad) - set(SAD_LABELS))}")
    rows = tax.get("mendeley") or []
    cols = [r.get("column") for r in rows]
    if sorted(cols) != sorted(MENDELEY_COLUMNS):
        errors.append(f"Mendeley columns must each appear once: missing {sorted(set(MENDELEY_COLUMNS) - set(cols))}, "
                      f"extra/duplicate {sorted({c for c in cols if cols.count(c) > 1 or c not in MENDELEY_COLUMNS})}")
    by_cat: dict[str, set[str]] = {i: set() for i in ids}
    for r in rows:
        if r.get("role") not in MENDELEY_ROLES:
            errors.append(f"Mendeley {r.get('column')}: unknown role {r.get('role')!r}")
        if not r.get("reason"):
            errors.append(f"Mendeley {r.get('column')}: every decision needs a reason")
        if r.get("role") == "stressor":
            if r.get("category") not in by_cat:
                errors.append(f"Mendeley {r.get('column')}: stressor role needs a valid category")
            else:
                by_cat[r["category"]].add(r["column"])
        elif r.get("category"):
            errors.append(f"Mendeley {r.get('column')}: only stressor rows map to a category")
    for c in cats:
        if "mendeley_columns" in c and set(c["mendeley_columns"]) != by_cat.get(c["id"], set()):
            errors.append(f"category {c['id']}: mendeley_columns {sorted(c['mendeley_columns'])} disagree with the "
                          f"mendeley section {sorted(by_cat.get(c['id'], set()))}")
    if errors:
        raise ValueError("taxonomy invalid:\n  - " + "\n  - ".join(errors))


def category_ids(tax: dict[str, Any]) -> list[str]:
    """Stressor categories in output order."""
    return [c["id"] for c in tax["categories"]]


def coverage(tax: dict[str, Any]) -> dict[str, str]:
    return {c["id"]: c["coverage"] for c in tax["categories"]}


def sad_label_to_class(tax: dict[str, Any]) -> dict[str, str]:
    """SAD top_label -> classifier class ('relationship_issues/family' for sub-categories)."""
    m: dict[str, str] = {}
    for c in tax["categories"]:
        sub = c.get("sad_label_subcategory")
        for lab in c["sad_labels"]:
            m[lab] = f"{c['id']}/{sub[lab]}" if sub else c["id"]
    for n in tax.get("non_stressor_classes") or []:
        for lab in n["sad_labels"]:
            m[lab] = n["id"]
    return m


def classifier_classes(tax: dict[str, Any]) -> list[str]:
    """Softmax classes of the SAD-trained classifiers, in fixed order (taxonomy order, then non-stressor classes)."""
    m = sad_label_to_class(tax)
    order = [m[lab] for c in tax["categories"] for lab in c["sad_labels"]]
    order += [n["id"] for n in tax.get("non_stressor_classes") or []]
    return list(dict.fromkeys(order))


def class_category_matrix(tax: dict[str, Any]) -> np.ndarray:
    """(n_classes, n_categories) 0/1: which category each class's probability is added to. Non-stressor classes map to
    no category (their mass is discarded at post level). Sub-classes of one category are summed (they are exclusive)."""
    classes, cats = classifier_classes(tax), category_ids(tax)
    M = np.zeros((len(classes), len(cats)))
    for i, cl in enumerate(classes):
        cat = cl.split("/")[0]
        if cat in cats:
            M[i, cats.index(cat)] = 1.0
    return M


def merge_class_probs(P: np.ndarray, tax: dict[str, Any]) -> np.ndarray:
    """Class probabilities (n, n_classes) -> category probabilities (n, n_categories). Zero-shot-only categories get
    NaN (no SAD class); a hybrid scorer fills them."""
    out = np.asarray(P, dtype=float) @ class_category_matrix(tax)
    zs = [j for j, c in enumerate(category_ids(tax)) if coverage(tax)[c] == "zero_shot"]
    out[:, zs] = np.nan
    return out


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r.tolist()) + " |")
    return "\n".join(lines)


def sad_mapping_table(tax: dict[str, Any], sad_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """SAD label -> classifier class -> taxonomy category (+ SAD top_label counts when given)."""
    m = sad_label_to_class(tax)
    rows = []
    for lab in SAD_LABELS:
        cl = m[lab]
        cat = cl.split("/")[0]
        is_cat = cat in category_ids(tax)
        rows.append({"SAD label": lab, "n (top_label)": (sad_counts or {}).get(lab, ""), "classifier class": cl,
                     "taxonomy category": cat if is_cat else "(none: not a stressor)",
                     "sub-category": cl.split("/")[1] if "/" in cl else "",
                     "stressor output": "yes" if is_cat else "no (discarded at post level)"})
    return pd.DataFrame(rows)


def mendeley_table(tax: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"column": r["column"], "authors' group": r.get("author_group", ""), "role": r["role"],
                          "category": r.get("category", ""), "reason": r["reason"]} for r in tax["mendeley"]])


def category_table(tax: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": c["id"], "name": c["name"], "coverage": c["coverage"],
        "how it is scored": ("SAD-supervised (or keyword/zero-shot baselines)" if c["coverage"] == "sad"
                             else "zero-shot embedding similarity ONLY (lower confidence)"),
        "SAD labels": ", ".join(c["sad_labels"]) or "-", "Mendeley columns": ", ".join(c["mendeley_columns"]) or "-",
        "Dreaddit subreddits (weak check)": ", ".join(c["dreaddit_subreddits"]) or "-",
        "n keywords": len(c["keywords"])} for c in tax["categories"]])


def write_taxonomy_report(cfg: dict[str, Any], tax: dict[str, Any], sad_counts: dict[str, int] | None = None) -> Path:
    """data/outputs/reports/stressor_taxonomy_v<k>.md: categories, SAD mapping, Mendeley decisions, uncovered categories.
    Aggregate counts only, no text."""
    out = Path(cfg["_paths"]["reports"]) / f"stressor_taxonomy_v{tax['version']}.md"
    zs = [c["id"] for c in tax["categories"] if c["coverage"] == "zero_shot"]
    parts = [f"# Stressor taxonomy v{tax['version']}", "", f"_{SCOPE_NOTE}_", "",
             f"Source of truth: `{cfg['stressor_model']['taxonomy_file']}` (definitions, include/exclude rules, seed "
             "keywords, 3 paraphrase descriptions per category). Keywords and paraphrases are author-chosen and unvalidated.",
             "", "## Categories", "", _md_table(category_table(tax)), "",
             "## SAD categories -> taxonomy", "", _md_table(sad_mapping_table(tax, sad_counts)), "",
             "## Decisions", ""]
    parts += [f"- **{k}**: {v}" for k, v in tax["decisions"].items()]
    parts += ["", "## Categories SAD does not cover", "",
              f"{', '.join(zs)}: no SAD label exists, so only the keyword baseline and zero-shot embedding similarity "
              "(window embedding vs the mean embedding of the category definition + 3 paraphrases) can score them. "
              "In the SAD-trained methods these columns come from the zero-shot scorer (hybrid). They are flagged "
              "lower-confidence everywhere, and their thresholds are tuned on the small gold-dev set only.", "",
              "## Mendeley columns: cause vs response (structure only, no statistics)", "",
              _md_table(mendeley_table(tax)), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


# ================================================================ SAD data + fixed split
def load_sad(cfg: dict[str, Any], tax: dict[str, Any]) -> pd.DataFrame:
    """Processed SAD (Step 2) with the classifier class. Severity columns are not loaded (never labels)."""
    from .preprocess import processed_path
    df = pd.read_parquet(processed_path(cfg, "sad"),
                         columns=["record_id", "text_clean", "top_label", "is_stressor", "is_covid", "sad_source"])
    m = sad_label_to_class(tax)
    unknown = sorted(set(df["top_label"]) - set(m))
    if unknown:
        raise ValueError(f"SAD top_label values not in the taxonomy: {unknown}")
    df["cls"] = df["top_label"].map(m)
    return df


def make_sad_split(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Stratified (by classifier class) train/validation/test split of SAD (no official split exists)."""
    from sklearn.model_selection import train_test_split
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"SAD split fractions must sum to 1: {fractions}")
    d = df.sort_values("record_id").reset_index(drop=True)
    rest, te = train_test_split(d, test_size=fractions["test"], stratify=d["cls"], random_state=seed)
    tr, va = train_test_split(rest, test_size=fractions["validation"] / (1 - fractions["test"]), stratify=rest["cls"],
                              random_state=seed)
    return {"train": sorted(tr["record_id"]), "validation": sorted(va["record_id"]), "test": sorted(te["record_id"])}


def load_sad_splits(cfg: dict[str, Any], tax: dict[str, Any], force: bool = False) -> dict[str, pd.DataFrame]:
    """train / validation / test frames; the split file data/processed/splits/sad_split_v1.json is created once."""
    from .preprocess import load_or_create_split
    df = load_sad(cfg, tax)
    fr = cfg["stressor_model"]["sad"]["split"]
    splits = load_or_create_split(cfg, "sad", df, lambda: make_sad_split(df, fr, cfg["primary_seed"]),
                                  {"stratified_by": "stressor classifier class (SAD top_label)", "fractions": fr,
                                   "seed": cfg["primary_seed"]}, force=force)
    where = {rid: s for s, ids in splits.items() for rid in ids}
    df["split"] = df["record_id"].map(where)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")}


def class_counts(data: dict[str, pd.DataFrame], tax: dict[str, Any]) -> pd.DataFrame:
    classes = classifier_classes(tax)
    return pd.DataFrame({s: d["cls"].value_counts().reindex(classes, fill_value=0) for s, d in data.items()})


# ================================================================ inference unit: sentence windows
_WORD_RE = re.compile(r"\w+")


def split_sentences(text: str) -> list[str]:
    """Sentences: paragraphs (line breaks) first, then the Step 2 sentence-boundary regex inside each paragraph."""
    from .preprocess import sentence_starts
    out: list[str] = []
    for para in re.split(r"\n+", text or ""):
        para = para.strip()
        if not para:
            continue
        st = sentence_starts(para)
        out += [s for s in (para[a:b].strip() for a, b in zip(st, st[1:] + [len(para)])) if s]
    return out


def sentence_windows(text: str, sentences_per_window: int = 1, min_words: int = 4,
                     max_windows: int | None = None) -> list[str]:
    """Split a post into SAD-like units. Sentences with fewer than `min_words` words are merged into the next one (a
    trailing short remainder joins the previous unit); units are grouped `sentences_per_window` at a time without
    overlap. With `max_windows`, an evenly spaced subset (first and last kept) is returned."""
    merged: list[str] = []
    buf = ""
    for s in split_sentences(text):
        buf = f"{buf} {s}" if buf else s
        if len(_WORD_RE.findall(buf)) >= min_words:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]} {buf}"
        else:
            merged.append(buf)
    k = max(1, int(sentences_per_window))
    wins = [" ".join(merged[i:i + k]) for i in range(0, len(merged), k)]
    if max_windows is not None and len(wins) > max_windows:
        keep = sorted(set(np.linspace(0, len(wins) - 1, max_windows).round().astype(int).tolist()))
        wins = [wins[i] for i in keep]
    return wins


def posts_to_windows(texts: Sequence[str], wcfg: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Flat window list + owner index (post position) for every window. Every post must yield >= 1 window."""
    flat: list[str] = []
    owner: list[int] = []
    for i, t in enumerate(texts):
        w = sentence_windows(t, wcfg["sentences_per_window"], wcfg["min_words"], wcfg["max_windows"])
        if not w:
            raise ValueError(f"post {i} produced no sentence window (empty text)")
        flat += w
        owner += [i] * len(w)
    return flat, np.asarray(owner)


AGGREGATIONS = ("max", "mean", "top2_mean", "sum")


def aggregate_windows(S: np.ndarray, owner: np.ndarray, n_posts: int, method: str = "max") -> np.ndarray:
    """Post scores (n_posts, L) from window scores (n_windows, L).
    max (default): a stressor named in ONE sentence is named in the post; a mean would dilute it by post length
      (most sentences of a post about work are not about work). Cost: max grows with the number of windows, so long
      posts get more labels; the window cap (max_windows) and thresholds tuned on post-level gold-dev limit this, and
      the dev table reports mean / top2_mean as sensitivity.
    sum: used by the keyword baseline (number of windows with a keyword hit)."""
    S = np.asarray(S, dtype=float)
    owner = np.asarray(owner)
    out = np.full((n_posts, S.shape[1]), np.nan)
    for i in range(n_posts):
        rows = S[owner == i]
        if rows.size == 0:
            raise ValueError(f"post {i} has no windows")
        if method == "max":
            out[i] = rows.max(0)
        elif method == "mean":
            out[i] = rows.mean(0)
        elif method == "top2_mean":
            out[i] = np.sort(rows, 0)[-2:].mean(0)
        elif method == "sum":
            out[i] = rows.sum(0)
        else:
            raise ValueError(f"unknown aggregation {method!r}")
    return out


# ================================================================ sentence embeddings (frozen; inference only)
@dataclass(frozen=True)
class EmbeddingSpec:
    """How a sentence-transformers checkpoint turns token states into a sentence vector, read from the checkpoint's
    own modules.json / 1_Pooling/config.json / sentence_bert_config.json (not assumed)."""
    hf_id: str
    pooling: str            # mean | cls
    normalize: bool
    max_length: int
    lowercase: bool
    revision: str | None


def embedding_spec(hf_id: str) -> EmbeddingSpec:
    from huggingface_hub import hf_hub_download
    mods = json.loads(Path(hf_hub_download(hf_id, "modules.json")).read_text(encoding="utf-8"))
    pool = json.loads(Path(hf_hub_download(hf_id, "1_Pooling/config.json")).read_text(encoding="utf-8"))
    stc = json.loads(Path(hf_hub_download(hf_id, "sentence_bert_config.json")).read_text(encoding="utf-8"))
    types = [m["type"] for m in mods]
    if pool.get("pooling_mode_cls_token"):
        mode = "cls"
    elif pool.get("pooling_mode_mean_tokens"):
        mode = "mean"
    else:
        raise ValueError(f"{hf_id}: unsupported pooling config {pool}")
    if any(t.endswith("Dense") for t in types):
        raise ValueError(f"{hf_id}: a Dense module follows pooling; not supported by this plain-transformers embedder")
    revision = Path(hf_hub_download(hf_id, "config.json")).parent.name     # snapshots/<commit sha>/config.json
    return EmbeddingSpec(hf_id, mode, any(t.endswith("Normalize") for t in types), int(stc.get("max_seq_length", 256)),
                         bool(stc.get("do_lower_case", False)), revision)


def pool_hidden(hidden, attention_mask, mode: str):
    """(batch, tokens, dim) -> (batch, dim). mean: average over non-padding tokens; cls: first token."""
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-9)
    raise ValueError(f"unknown pooling {mode!r}")


class Embedder:
    """Frozen sentence encoder (fp32, no gradients). `model`/`tokenizer` may be injected (tests use a tiny random
    model); otherwise they are loaded at the spec's revision."""

    def __init__(self, spec: EmbeddingSpec, device: str = "cpu", batch_size: int = 64, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
            model = AutoModel.from_pretrained(spec.hf_id, revision=spec.revision)
        self.spec, self.tok, self.device, self.batch_size = spec, tokenizer, device, int(batch_size)
        self.model = model.to(device).eval()

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], model_choice: str | None = None) -> "Embedder":
        e = settings(cfg)["embedding"]
        hf_id = e["model_options"][model_choice or e["model_choice"]]["hf_id"]
        return cls(embedding_spec(hf_id), select_device(cfg), e["batch_size"])

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        texts = [t.lower() if self.spec.lowercase else t for t in texts]
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                idx = order[i:i + self.batch_size]
                enc = self.tok([texts[j] for j in idx], truncation=True, max_length=self.spec.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                v = pool_hidden(self.model(**enc).last_hidden_state.float(), enc["attention_mask"], self.spec.pooling)
                if self.spec.normalize:
                    v = torch.nn.functional.normalize(v, p=2, dim=1)
                out[idx] = v.cpu().numpy()
        return out


def embed_cached(cfg: dict[str, Any], embedder: Embedder, texts: Sequence[str], name: str) -> np.ndarray:
    """Embeddings for `texts`, cached under data/ (keyed by model, revision, pooling and a hash of the texts)."""
    h = hashlib.sha256()
    for part in (embedder.spec.hf_id, str(embedder.spec.revision), embedder.spec.pooling, str(embedder.spec.max_length),
                 *texts):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    key = h.hexdigest()[:16]
    p = embedding_cache_dir(cfg) / f"{name}__{embedder.spec.hf_id.replace('/', '--')}__{key}.npy"
    if p.exists():
        return np.load(p)
    E = embedder.encode(texts)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npy")
    np.save(tmp, E)
    os.replace(tmp, p)
    return E


VERIFY_SENTENCES = ("I have three exams next week and I have not started studying.",
                    "My final exams are soon and I am far behind on revision.",
                    "The weather was sunny at the beach today.")


def verify_embedding_models(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load every configured embedding model and encode 3 SYNTHETIC sentences (inference only). Passes when vectors
    are unit length and the paraphrase pair is more similar than the unrelated pair."""
    rows = []
    for choice, o in settings(cfg)["embedding"]["model_options"].items():
        try:
            emb = Embedder(embedding_spec(o["hf_id"]), select_device(cfg), 8)
            E = emb.encode(list(VERIFY_SENTENCES))
            sim_para, sim_unrel = float(E[0] @ E[1]), float(E[0] @ E[2])
            norms = np.linalg.norm(E, axis=1)
            rows.append({"model_choice": choice, "hf_id": o["hf_id"], "revision": emb.spec.revision,
                         "pooling": emb.spec.pooling, "normalize": emb.spec.normalize, "max_length": emb.spec.max_length,
                         "lowercase": emb.spec.lowercase, "dim": emb.dim,
                         "n_params_M": round(sum(p.numel() for p in emb.model.parameters()) / 1e6, 1),
                         "sim_paraphrase": round(sim_para, 3), "sim_unrelated": round(sim_unrel, 3),
                         "loads_ok": bool(np.allclose(norms, 1, atol=1e-3) and sim_para > sim_unrel), "error": ""})
            del emb
        except Exception as exc:  # noqa: BLE001 - report, do not stop Run All
            rows.append({"model_choice": choice, "hf_id": o["hf_id"], "loads_ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
    return pd.DataFrame(rows)


# ================================================================ scorers (window level)
def keyword_regex(keywords: Sequence[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive alternation; a trailing * matches any word ending, spaces match any whitespace."""
    parts = []
    for kw in keywords:
        kw = kw.strip().lower()
        star = kw.endswith("*")
        body = r"\s+".join(re.escape(w) for w in kw.rstrip("*").split())
        parts.append(body + (r"\w*" if star else ""))
    return re.compile(r"(?<!\w)(?:" + "|".join(sorted(parts, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)


class KeywordScorer:
    """(1) Lexicon baseline: a window scores 1 for a category if it contains any of the category's seed keywords.
    Post score = number of matching windows (aggregation 'sum'); threshold = minimum count, tuned on gold-dev."""
    method = "keyword"
    aggregation = "sum"

    def __init__(self, tax: dict[str, Any]):
        self.cats = category_ids(tax)
        self.patterns = [keyword_regex(c["keywords"]) for c in tax["categories"]]
        self.kinds = ["count"] * len(self.cats)

    def window_scores(self, windows: Sequence[str], E: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros((len(windows), len(self.patterns)))
        for i, w in enumerate(windows):
            out[i] = [1.0 if p.search(w) else 0.0 for p in self.patterns]
        return out


def category_description_texts(cat: dict[str, Any]) -> list[str]:
    return [" ".join(str(cat["definition"]).split())] + list(cat["paraphrases"])


class ZeroShotScorer:
    """(3) Label-free: cosine similarity between the window embedding and a category prototype (normalised mean of the
    embeddings of the category definition and its 3 paraphrases). Scores are cosines, not probabilities."""
    method = "zero_shot"
    needs_embeddings = True

    def __init__(self, tax: dict[str, Any], embedder: Embedder, aggregation: str = "max"):
        self.cats, self.aggregation = category_ids(tax), aggregation
        protos = []
        for c in tax["categories"]:
            v = embedder.encode(category_description_texts(c)).mean(0)
            protos.append(v / np.linalg.norm(v))
        self.prototypes = np.stack(protos).astype(np.float32)
        self.kinds = ["cosine"] * len(self.cats)

    def window_scores(self, windows: Sequence[str], E: np.ndarray | None = None) -> np.ndarray:
        if E is None:
            raise ValueError("zero-shot scoring needs the window embeddings")
        return np.asarray(E, dtype=float) @ self.prototypes.T.astype(float)


class SupervisedScorer:
    """(2a)/(2b): SAD-trained softmax over the classifier classes, temperature-calibrated on SAD-validation, merged into
    taxonomy categories (sub-classes summed, non-stressor classes discarded). Categories SAD lacks are filled with the
    zero-shot cosine (hybrid); their `kind` is 'cosine', so they get their own threshold grid."""

    def __init__(self, tax: dict[str, Any], method: str, class_logits_fn, temperature: float,
                 zero_shot: ZeroShotScorer | None, aggregation: str = "max"):
        self.tax, self.method, self.fn, self.T = tax, method, class_logits_fn, float(temperature)
        self.zs, self.aggregation = zero_shot, aggregation
        self.cats = category_ids(tax)
        cov = coverage(tax)
        self.kinds = ["probability" if cov[c] == "sad" else "cosine" for c in self.cats]
        self.needs_embeddings = method == "embed_lr" or zero_shot is not None

    def class_probs(self, windows: Sequence[str], E: np.ndarray | None = None) -> np.ndarray:
        return softmax(self.fn(windows, E), self.T)

    def window_scores(self, windows: Sequence[str], E: np.ndarray | None = None) -> np.ndarray:
        S = merge_class_probs(self.class_probs(windows, E), self.tax)
        miss = np.isnan(S).any(0)
        if miss.any():
            if self.zs is None:
                raise ValueError("categories without SAD labels need a zero-shot scorer")
            S[:, miss] = self.zs.window_scores(windows, E)[:, miss]
        return S


def lr_logits_fn(params: dict[str, np.ndarray]):
    coef, intercept = np.asarray(params["coef"], dtype=float), np.asarray(params["intercept"], dtype=float)

    def fn(windows: Sequence[str], E: np.ndarray | None) -> np.ndarray:
        if E is None:
            raise ValueError("embed_lr scoring needs the window embeddings")
        return np.asarray(E, dtype=float) @ coef.T + intercept
    return fn


def finetune_logits_fn(model, tokenizer, max_length: int, batch_size: int, device: str):
    def fn(windows: Sequence[str], E: np.ndarray | None = None) -> np.ndarray:
        return SM.predict_logits(model, tokenizer, list(windows), max_length, batch_size, device)
    return fn


def score_posts(cfg: dict[str, Any], scorer, texts: Sequence[str], embedder: Embedder | None = None,
                cache_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Window-level scores (n_windows, L) and the owner (post index) of each window. Inference only."""
    wcfg = settings(cfg)["windows"]
    windows, owner = posts_to_windows(texts, wcfg)
    E = None
    if getattr(scorer, "needs_embeddings", False):
        if embedder is None:
            raise ValueError(f"{scorer.method} needs an embedder")
        E = embed_cached(cfg, embedder, windows, cache_name) if cache_name else embedder.encode(windows)
    return scorer.window_scores(windows, E), owner


# ================================================================ (2a) embedding + logistic regression (USER RUNS)
def fit_embedding_lr(cfg: dict[str, Any], tax: dict[str, Any], data: dict[str, pd.DataFrame], embedder: Embedder) -> Path:
    """FITS WEIGHTS (user-run cell). Multinomial logistic regression on frozen SAD-train sentence embeddings; C chosen
    on SAD-validation macro-F1 over the classifier classes (ties -> smaller C). Saves lr.npz (coef, intercept, classes;
    no pickle), lr_summary.json, SAD validation/test logits, DONE.json. Skips if DONE.json exists."""
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    s = settings(cfg)
    d = lr_run_dir(cfg)
    if (d / "DONE.json").exists():
        LOG.info("embed_lr: DONE.json present in %s, skipping", d)
        return d
    d.mkdir(parents=True, exist_ok=True)
    classes = classifier_classes(tax)
    idx = {c: i for i, c in enumerate(classes)}
    E = {k: embed_cached(cfg, embedder, v["text_clean"].tolist(), f"sad_{k}") for k, v in data.items()}
    y = {k: v["cls"].map(idx).to_numpy() for k, v in data.items()}
    grid, best = [], None
    for C in s["embed_lr"]["C_grid"]:
        clf = LogisticRegression(C=C, max_iter=s["embed_lr"]["max_iter"], random_state=cfg["primary_seed"])
        clf.fit(E["train"], y["train"])
        f1 = float(f1_score(y["validation"], clf.predict(E["validation"]), average="macro",
                            labels=list(range(len(classes))), zero_division=0))
        grid.append({"C": C, "validation_macro_f1": f1, "n_iter": int(np.max(clf.n_iter_))})
        if best is None or f1 > best[1] + 1e-12:
            best = (C, f1, clf)
    C, f1, clf = best
    if list(clf.classes_) != list(range(len(classes))):
        raise ValueError("a SAD class is missing from SAD-train")
    np.savez(d / "lr.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array(classes))
    for split in ("validation", "test"):
        Z = E[split] @ clf.coef_.T + clf.intercept_
        _save_class_logits(data[split], y[split], Z, classes, d / f"predictions_{split}.parquet")
    write_json(d / "lr_summary.json", {
        "method": "embed_lr", "embedding": embedder.spec.__dict__, "classes": classes, "C_grid": grid, "best_C": C,
        "best_validation_macro_f1": f1, "selection": "max SAD-validation macro-F1 over classes; ties -> smaller C",
        "n_train": int(len(y["train"])), "n_validation": int(len(y["validation"])), "sklearn": sklearn.__version__,
        "seed": cfg["primary_seed"], "device": device_label()})
    write_json(d / "DONE.json", {"finished_utc": _dt.datetime.now(_dt.timezone.utc)})
    return d


def load_lr(d: Path) -> dict[str, np.ndarray]:
    z = np.load(Path(d) / "lr.npz", allow_pickle=False)
    return {"coef": z["coef"], "intercept": z["intercept"], "classes": z["classes"].tolist()}


def _save_class_logits(df: pd.DataFrame, y: np.ndarray, Z: np.ndarray, classes: Sequence[str], path: Path) -> None:
    out = pd.concat([pd.DataFrame({"record_id": df["record_id"].to_numpy(), "y": np.asarray(y, dtype=int)}),
                     pd.DataFrame(np.asarray(Z, dtype=np.float32), columns=[f"logit__{c}" for c in classes])], axis=1)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_class_logits(d: Path, split: str, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(Path(d) / f"predictions_{split}.parquet")
    return df["y"].to_numpy(), df[[f"logit__{c}" for c in classes]].to_numpy(dtype=float)


# ================================================================ (2b) fine-tuned distilroberta (USER RUNS)
def ft_compute_metrics(eval_pred) -> dict[str, float]:
    """Validation metrics during training (uncalibrated argmax). Trainer adds the eval_ prefix."""
    from sklearn.metrics import f1_score
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    y, yhat = np.asarray(eval_pred.label_ids), np.asarray(logits).argmax(1)
    return {"macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)), "accuracy": float((y == yhat).mean())}


def train_finetune_seed(cfg: dict[str, Any], tax: dict[str, Any], seed: int, data: dict[str, pd.DataFrame]) -> Path:
    """FITS WEIGHTS (user-run cell). Fine-tune distilroberta-base on SAD-train, softmax over the classifier classes,
    plain cross-entropy (Step 3's WeightedTrainer with all weights 1), checkpoint every epoch, early stopping on
    SAD-validation macro-F1, best epoch kept. Resume-safe like Step 3: DONE.json -> skip; best_model/ -> only missing
    predictions; else resume from the last epoch checkpoint."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint
    s = settings(cfg)["finetune"]
    if s["gradient_accumulation_steps"] != 1:
        raise ValueError("stressor fine-tuning needs gradient_accumulation_steps = 1: the installed Trainer does not divide "
                         "a custom loss by GA (Step 4 measurement), and SAD sentences are short enough for batch 32 on 4 GB")
    run_dir = ft_run_dir(cfg, seed)
    if (run_dir / "DONE.json").exists():
        LOG.info("stressor seed %s: DONE.json present, skipping", seed)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    classes = classifier_classes(tax)
    idx = {c: i for i, c in enumerate(classes)}
    id2label = dict(enumerate(classes))
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    best_dir = run_dir / "best_model"
    tok = AutoTokenizer.from_pretrained(s["hf_id"], use_fast=True)
    cuda = torch.cuda.is_available()
    if not (best_dir / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            s["hf_id"], num_labels=len(classes), id2label=id2label, label2id={v: k for k, v in id2label.items()})
        tr, va = data["train"], data["validation"]
        ds_tr = SM.EncodedDataset(SM.encode(tok, tr["text_clean"], s["max_length"]), tr["cls"].map(idx))
        ds_va = SM.EncodedDataset(SM.encode(tok, va["text_clean"], s["max_length"]), va["cls"].map(idx))
        ck = ft_checkpoint_dir(cfg, run_dir)
        args = SM.build_training_args(s, ck, seed, cuda, bool(cuda and torch.cuda.is_bf16_supported()))
        WeightedTrainer, CSVLogger = SM._trainer_classes()
        compat = SM.hf_compat()
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, data_collator=SM.WeightedCollator(tok),
            compute_metrics=ft_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=s["early_stopping_patience"]),
                       CSVLogger(run_dir / "train_log.csv")],
            **{compat["tokenizer_key"]: tok})
        last = get_last_checkpoint(str(ck)) if ck.is_dir() else None
        LOG.info("stressor seed %s: training %s on %s, resume_from=%s", seed, s["hf_id"], device_label(), last)
        result = trainer.train(resume_from_checkpoint=last)
        trainer.save_model(str(best_dir))
        tok.save_pretrained(str(best_dir))
        write_json(run_dir / "train_summary.json", {
            "seed": seed, "model_choice": s["model_choice"], "hf_id": s["hf_id"], "settings": s, "classes": classes,
            "best_metric_validation_macro_f1": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint, "epochs_completed": trainer.state.epoch,
            "global_step": trainer.state.global_step, "train_runtime_s": result.metrics.get("train_runtime"),
            "checkpoint_dir": ck, "log_history": trainer.state.log_history, "transformers": compat["transformers"],
            "n_train": len(tr), "n_validation": len(va), "device": device_label()})
        del trainer, model
        if cuda:
            torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    device = select_device(cfg)
    for split in ("validation", "test"):
        out = run_dir / f"predictions_{split}.parquet"
        if out.exists():
            continue
        df = data[split]
        Z = SM.predict_logits(model, tok, df["text_clean"].tolist(), s["max_length"], s["eval_batch_size"], device)
        _save_class_logits(df, df["cls"].map(idx).to_numpy(), Z, classes, out)
    write_json(run_dir / "DONE.json", {"seed": seed, "finished_utc": _dt.datetime.now(_dt.timezone.utc)})
    return run_dir


def ft_readiness(cfg: dict[str, Any]) -> dict[str, Any]:
    s = settings(cfg)["finetune"]
    done = [k for k in s["seeds"] if (ft_run_dir(cfg, k) / "DONE.json").exists()]
    missing = [k for k in s["seeds"] if k not in done]
    return {"seeds_done": done, "seeds_missing": missing, "all_seeds": not missing,
            "why": f"needs every stressor fine-tune seed {s['seeds']}; missing {missing} (trained in 5C.4)"}


def ft_select_seed(cfg: dict[str, Any], tax: dict[str, Any]) -> dict[str, Any]:
    """Per-seed SAD validation/test macro-F1 (uncalibrated argmax), mean +/- std; final seed = best VALIDATION."""
    from sklearn.metrics import f1_score
    classes = classifier_classes(tax)
    rows = []
    for seed in settings(cfg)["finetune"]["seeds"]:
        d = ft_run_dir(cfg, seed)
        r = {"seed": seed}
        for split in ("validation", "test"):
            y, Z = load_class_logits(d, split, classes)
            r[f"{split}_macro_f1"] = float(f1_score(y, Z.argmax(1), average="macro", zero_division=0))
            r[f"{split}_accuracy"] = float((y == Z.argmax(1)).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    final = int(t.loc[t["validation_macro_f1"].idxmax(), "seed"])
    out = {"per_seed": rows, "final_seed": final, "rule": "best SAD-validation macro-F1 (test never used)",
           "mean_std": {c: {"mean": float(t[c].mean()), "std": float(t[c].std(ddof=1)) if len(t) > 1 else 0.0}
                        for c in t.columns if c != "seed"}}
    write_json(runs_root(cfg) / "finetune" / settings(cfg)["finetune"]["model_choice"] / "seed_selection.json", out)
    return out


# ================================================================ SAD calibration + SAD-test report (no training)
def calibrate_and_report_sad(cfg: dict[str, Any], tax: dict[str, Any], run_dir: Path, method: str,
                             force: bool = False) -> dict[str, Any]:
    """Temperature on SAD-VALIDATION logits, then the SAD-test report ONCE (SAD_TEST_REPORT.json): class-level
    accuracy / macro-F1 / per-class F1, and the CATEGORY-level macro-F1 over the SAD-covered stressor categories
    (argmax after merging sub-classes; non-stressor classes compete in the argmax). That last number is the reference
    for the domain-shift cost on Reddit gold-test."""
    from sklearn.metrics import confusion_matrix, f1_score
    run_dir = Path(run_dir)
    path = run_dir / "SAD_TEST_REPORT.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    classes = classifier_classes(tax)
    yv, Zv = load_class_logits(run_dir, "validation", classes)
    yt, Zt = load_class_logits(run_dir, "test", classes)
    T = fit_temperature(Zv, yv)
    bins = cfg["stress_model"]["calibration"]["ece_bins"]
    conf_v0, conf_v1 = softmax(Zv).max(1), softmax(Zv, T).max(1)
    Pt = softmax(Zt, T)
    yhat = Pt.argmax(1)
    cats = category_ids(tax)
    sad_cats = [c for c in cats if coverage(tax)[c] == "sad"]
    groups = sad_cats + [n["id"] for n in tax.get("non_stressor_classes") or []]
    G = np.zeros((len(classes), len(groups)))
    for i, cl in enumerate(classes):
        G[i, groups.index(cl.split("/")[0])] = 1
    true_g, pred_g = G[yt].argmax(1), (Pt @ G).argmax(1)
    per_cat = f1_score(true_g, pred_g, labels=list(range(len(sad_cats))), average=None, zero_division=0)
    rep = {
        "method": method, "run_dir": run_dir, "calibration": {
            "method": "temperature (one scalar, SAD-validation)", "temperature": T,
            "validation_nll_before": nll(Zv, yv), "validation_nll_after": nll(Zv, yv, T),
            "validation_top_label_ece_before": ece((Zv.argmax(1) == yv).astype(int), conf_v0, bins),
            "validation_top_label_ece_after": ece((Zv.argmax(1) == yv).astype(int), conf_v1, bins)},
        "n_test": int(len(yt)), "classes": classes,
        "test_accuracy": float((yhat == yt).mean()),
        "test_macro_f1_classes": float(f1_score(yt, yhat, average="macro", labels=list(range(len(classes))), zero_division=0)),
        "test_per_class_f1": dict(zip(classes, map(float, f1_score(yt, yhat, labels=list(range(len(classes))), average=None,
                                                                    zero_division=0)))),
        "test_confusion": confusion_matrix(yt, yhat, labels=list(range(len(classes)))).tolist(),
        "test_top_label_ece": ece((yhat == yt).astype(int), Pt.max(1), bins),
        "test_category_macro_f1_sad_covered": float(per_cat.mean()),
        "test_category_f1": dict(zip(sad_cats, map(float, per_cat))),
        "category_rule": "argmax over merged groups (stressor categories + non-stressor classes); one-vs-rest F1 per "
                         "SAD-covered stressor category, macro over them",
        "created_utc": _dt.datetime.now(_dt.timezone.utc)}
    write_json(path, rep)
    return json.loads(path.read_text(encoding="utf-8"))


def keyword_sad_report(cfg: dict[str, Any], tax: dict[str, Any], data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Keyword baseline on SAD-test at sentence level (no training): a sentence's predicted category = the SAD-covered
    category with a keyword hit, ties/no hit -> no prediction. Category macro-F1 over SAD-covered categories."""
    from sklearn.metrics import f1_score
    ks = KeywordScorer(tax)
    cats = category_ids(tax)
    sad_cats = [c for c in cats if coverage(tax)[c] == "sad"]
    te = data["test"]
    H = ks.window_scores(te["text_clean"].tolist())[:, [cats.index(c) for c in sad_cats]]
    pred = np.where(H.sum(1) == 1, H.argmax(1), -1)
    true = te["cls"].map(lambda c: sad_cats.index(c.split("/")[0]) if c.split("/")[0] in sad_cats else -2).to_numpy()
    per = f1_score(true, pred, labels=list(range(len(sad_cats))), average=None, zero_division=0)
    return {"method": "keyword", "n_test": int(len(te)), "test_category_macro_f1_sad_covered": float(per.mean()),
            "test_category_f1": dict(zip(sad_cats, map(float, per))),
            "share_no_or_multiple_hits": float((pred == -1).mean()),
            "rule": "exactly one SAD-covered category with a keyword hit -> that category; else no prediction"}


# ================================================================ 5B gold set
def label_source(cfg: dict[str, Any]) -> str:
    """'human' (the annotator sheet) or 'machine' (machine_labeler's sheet). Config stressor_model.gold.label_source."""
    src = cfg["stressor_model"]["gold"].get("label_source", "human")
    if src not in ("human", "machine"):
        raise ValueError(f"stressor_model.gold.label_source must be human or machine, got {src!r}")
    return src


def annotation_note(cfg: dict[str, Any]) -> str:
    return MACHINE_ANNOTATION_NOTE if label_source(cfg) == "machine" else ANNOTATION_NOTE


def annotation_limitation(cfg: dict[str, Any]) -> str:
    return MACHINE_ANNOTATION_LIMITATION if label_source(cfg) == "machine" else ANNOTATION_LIMITATION


def gold_files(cfg: dict[str, Any]) -> dict[str, Path]:
    from .preprocess import split_path
    v = cfg["stressor_model"]["gold"]["version"]
    g = Path(cfg["_paths"]["gold"])
    human, machine = g / f"stressor_gold_v{v}_annotator1.csv", g / f"stressor_gold_v{v}_machine.csv"
    return {"sheet1": machine if label_source(cfg) == "machine" else human, "human_sheet": human, "machine_sheet": machine,
            "split": split_path(cfg, "stressor_gold"),
            "report": Path(cfg["_paths"]["reports"]) / f"stressor_gold_set_v{v}.md"}


def allocate_stratified(available: dict[str, int], n_total: int, floor: int) -> dict[str, int]:
    """Per-stratum sample sizes: min(available, floor) each, the rest proportional to remaining availability
    (largest-remainder rounding, never above availability). Raises if n_total exceeds what is available."""
    if n_total > sum(available.values()):
        raise ValueError(f"asked for {n_total}, only {sum(available.values())} available")
    take = {k: min(v, floor) for k, v in available.items()}
    if sum(take.values()) > n_total:
        raise ValueError(f"floor {floor} x {len(available)} strata exceeds n_total {n_total}")
    while (rem := n_total - sum(take.values())) > 0:
        cap = {k: available[k] - take[k] for k in available if available[k] > take[k]}
        tot = sum(cap.values())
        quota = {k: rem * c / tot for k, c in cap.items()}
        add = {k: min(cap[k], int(np.floor(q))) for k, q in quota.items()}
        left = rem - sum(add.values())
        for k in sorted(cap, key=lambda k: (-(quota[k] - np.floor(quota[k])), k)):
            if left <= 0:
                break
            if add[k] < cap[k]:
                add[k] += 1
                left -= 1
        for k, a in add.items():
            take[k] += a
    return take


def sample_gold(cfg: dict[str, Any], dreaddit: pd.DataFrame) -> pd.DataFrame:
    """Gold key (no text): Dreaddit label 1, confidence >= min_confidence, allowed splits only, one segment per post,
    stratified by subreddit; dev/test 50/50 stratified by subreddit; seeded shuffle gives the annotation order."""
    from sklearn.model_selection import train_test_split
    g, seed = cfg["stressor_model"]["gold"], cfg["primary_seed"]
    pool = dreaddit[(dreaddit["label"] == 1) & (dreaddit["confidence"] >= g["min_confidence"])
                    & dreaddit["split"].isin(g["dreaddit_splits"])].sort_values("record_id")
    if g["one_segment_per_post"]:
        pool = pool.sample(frac=1.0, random_state=seed).drop_duplicates("post_id").sort_values("record_id")
    alloc = allocate_stratified(pool["community"].value_counts().to_dict(), g["n_total"], g["floor_per_subreddit"])
    pick = pd.concat([pool[pool["community"] == c].sample(n, random_state=seed) for c, n in sorted(alloc.items()) if n > 0])
    dev, test = train_test_split(pick, train_size=g["dev_fraction"], stratify=pick["community"], random_state=seed)
    pick = pd.concat([dev.assign(gold_split="dev"), test.assign(gold_split="test")])
    pick = pick.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pick["gold_id"] = [f"g{i:04d}" for i in range(1, len(pick) + 1)]
    pick["source"] = "dreaddit"
    return pick[["gold_id", "record_id", "community", "split", "confidence", "gold_split", "source"]] \
        .rename(columns={"split": "dreaddit_split"})


def gold_columns(tax: dict[str, Any]) -> list[str]:
    return category_ids(tax) + [NONE_COL]


def build_gold_set(cfg: dict[str, Any], tax: dict[str, Any]) -> dict[str, Any]:
    """Create the gold key (data/processed/splits/stressor_gold_split_v1.json: ids, subreddit, splits; NO text and NO
    Reddit post ids) and the LOCAL annotation sheets (CSV, scrubbed text, blank label columns). Never overwrites: if the
    key exists it is loaded and only missing sheets are re-created from it."""
    from .privacy import scrub_text
    f = gold_files(cfg)
    if f["split"].exists():
        meta = json.loads(f["split"].read_text(encoding="utf-8"))
        key = pd.DataFrame(meta["rows"])
    else:
        dd = pd.concat(SM.load_dreaddit(cfg).values(), ignore_index=True)
        key = sample_gold(cfg, dd)
        g = cfg["stressor_model"]["gold"]
        meta = {"name": "stressor_gold", "version": g["version"], "created_utc": _dt.datetime.now(_dt.timezone.utc),
                "seed": cfg["primary_seed"], "sampling": {k: g[k] for k in ("n_total", "min_confidence", "dreaddit_splits",
                                                                            "one_segment_per_post", "floor_per_subreddit",
                                                                            "dev_fraction", "annotators")},
                "rule": "Dreaddit HUMAN label 1 (stress), confidence >= min_confidence; one segment per post; stratified "
                        "by subreddit (floor, rest proportional); dev/test stratified by subreddit; test never used for tuning",
                "leakage": LEAKAGE_NOTE,
                "counts": {"total": int(len(key)), "by_gold_split": key["gold_split"].value_counts().to_dict(),
                           "by_community": key["community"].value_counts().to_dict(),
                           "annotators": int(g["annotators"])},
                "annotation": annotation_note(cfg),
                "rows": key.to_dict(orient="records")}
        write_json(f["split"], meta)
    texts = pd.concat(SM.load_dreaddit(cfg).values(), ignore_index=True).set_index("record_id")["text_clean"]
    cols = gold_columns(tax)
    for name, rows in (("human_sheet", key),):
        if f[name].exists():
            continue
        sheet = pd.DataFrame({"gold_id": rows["gold_id"].to_numpy(),
                              "text": rows["record_id"].map(texts).map(lambda t: scrub_text(t, cfg)).to_numpy(),
                              **{c: "" for c in cols}, "notes": ""})
        f[name].parent.mkdir(parents=True, exist_ok=True)
        tmp = f[name].with_suffix(".tmp.csv")
        sheet.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, f[name])
    write_gold_report(cfg, tax)
    return {"files": f, "counts": meta["counts"], "leakage": meta["leakage"], "annotation": annotation_note(cfg)}


def write_gold_report(cfg: dict[str, Any], tax: dict[str, Any]) -> Path:
    """data/outputs/reports/stressor_gold_set_v<k>.md: sampling rule, counts, leakage, annotation status and per-category
    support per split (counts only, no text). Re-written whenever called, so it follows the annotation progress."""
    f = gold_files(cfg)
    meta = json.loads(f["split"].read_text(encoding="utf-8"))
    key = pd.DataFrame(meta["rows"])
    comm = pd.crosstab(key["community"], key["gold_split"]).reset_index()
    parts = [f"# Stressor gold set v{meta['version']}", "", f"_{SCOPE_NOTE}_", "",
             f"**Annotation: {annotation_note(cfg)}.** {annotation_limitation(cfg)}", "",
             f"**Sampling.** {meta['rule']}. Settings: `{json.dumps(meta['sampling'])}`, seed {meta['seed']}.", "",
             f"**Leakage.** {meta['leakage']}", "", f"**Size.** {meta['counts']['total']} segments: "
             f"{meta['counts']['by_gold_split']}.", "", "## Segments per subreddit and split", "", _md_table(comm), ""]
    if f["sheet1"].exists():
        lab, probs = read_sheet(f["sheet1"], gold_columns(tax))
        m = lab.merge(key[["gold_id", "gold_split"]], on="gold_id")
        done = m[m["annotated"]]
        sup = pd.DataFrame({"column": gold_columns(tax),
                            **{f"positives_{s}": [int(done.loc[done["gold_split"] == s, c].sum()) for c in gold_columns(tax)]
                               for s in ("dev", "test")}})
        sup["coverage"] = sup["column"].map(lambda c: coverage(tax).get(c, "gold-only column"))
        parts += [f"## Annotation progress and support", "",
                  f"Annotated {int(m['annotated'].sum())} of {len(m)} rows ({len(probs)} problems).", "", _md_table(sup), ""]
    out = f["report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def gold_key(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(json.loads(gold_files(cfg)["split"].read_text(encoding="utf-8"))["rows"])


def read_sheet(path: Path, cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Label columns as 0/1 (blank = 0). Returns (labels incl. `annotated`, problems). Problems: values other than
    blank/0/1, or none_unclear = 1 together with a category."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = [c for c in ["gold_id", *cols] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    problems: list[str] = []
    out = pd.DataFrame({"gold_id": raw["gold_id"]})
    for c in cols:
        v = raw[c].str.strip().str.lower().replace({"1.0": "1", "0.0": "0", "x": "1", "y": "1", "yes": "1"})
        bad = ~v.isin(["", "0", "1"])
        problems += [f"{gid}: {c}={raw.loc[i, c]!r}" for i, gid in raw.loc[bad, "gold_id"].items()]
        out[c] = (v == "1").astype(int)
    cats = [c for c in cols if c != NONE_COL]
    clash = (out[NONE_COL] == 1) & (out[cats].sum(1) > 0)
    problems += [f"{gid}: none_unclear together with a category" for gid in out.loc[clash, "gold_id"]]
    out["annotated"] = out[list(cols)].sum(1) > 0
    return out, problems


def gold_status(cfg: dict[str, Any], tax: dict[str, Any]) -> dict[str, Any]:
    """Annotation progress (counts only, no text) for both sheets."""
    key, cols, f = gold_key(cfg), gold_columns(tax), gold_files(cfg)
    out: dict[str, Any] = {}
    for name in ("sheet1",):
        if not f[name].exists():
            out[name] = {"exists": False}
            continue
        lab, probs = read_sheet(f[name], cols)
        m = lab.merge(key[["gold_id", "gold_split"]], on="gold_id", how="left")
        out[name] = {"exists": True, "rows": int(len(m)), "annotated": int(m["annotated"].sum()),
                     "annotated_by_split": m.groupby("gold_split")["annotated"].sum().astype(int).to_dict(),
                     "rows_by_split": m["gold_split"].value_counts().to_dict(),
                     "n_problems": len(probs), "problems_first_10": probs[:10],
                     "positives": {c: int(m.loc[m["annotated"], c].sum()) for c in cols}}
    s1 = out.get("sheet1", {})
    out["complete"] = bool(s1.get("exists") and s1["annotated"] == s1["rows"] and s1["n_problems"] == 0)
    out["annotation"] = annotation_note(cfg)
    out["label_source"] = label_source(cfg)
    return out


def load_gold_labels(cfg: dict[str, Any], tax: dict[str, Any]) -> pd.DataFrame:
    """Key + 0/1 labels of the COMPLETE, problem-free annotation sheet (raises otherwise)."""
    lab, probs = read_sheet(gold_files(cfg)["sheet1"], gold_columns(tax))
    if probs or not lab["annotated"].all():
        raise ValueError(f"gold sheet: {int((~lab['annotated']).sum())} rows unannotated, "
                         f"{len(probs)} problems (first: {probs[:3]})")
    return gold_key(cfg).merge(lab.drop(columns="annotated"), on="gold_id", how="inner")


def gold_texts(cfg: dict[str, Any]) -> pd.DataFrame:
    """Gold key + Dreaddit text_clean (in memory only; never written by this function)."""
    dd = pd.concat(SM.load_dreaddit(cfg).values(), ignore_index=True)[["record_id", "text_clean"]]
    return gold_key(cfg).merge(dd, on="record_id", how="left")


# ================================================================ scoring the gold set (inference only)
def build_scorer(cfg: dict[str, Any], tax: dict[str, Any], method: str, embedder: Embedder | None = None):
    """Scorer for a method from its saved artefacts (no fitting here)."""
    s = settings(cfg)
    agg = s["aggregation"]
    if method == "keyword":
        return KeywordScorer(tax)
    zs = ZeroShotScorer(tax, embedder, agg) if embedder is not None else None
    if method == "zero_shot":
        return zs
    if method == "embed_lr":
        d = lr_run_dir(cfg)
        T = calibrate_and_report_sad(cfg, tax, d, "embed_lr")["calibration"]["temperature"]
        return SupervisedScorer(tax, "embed_lr", lr_logits_fn(load_lr(d)), T, zs, agg)
    if method == "finetune":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        f = s["finetune"]
        d = ft_run_dir(cfg, ft_select_seed(cfg, tax)["final_seed"])
        T = calibrate_and_report_sad(cfg, tax, d, "finetune")["calibration"]["temperature"]
        model = AutoModelForSequenceClassification.from_pretrained(str(d / "best_model"))
        tok = AutoTokenizer.from_pretrained(str(d / "best_model"))
        device = select_device(cfg)
        bs = f["eval_batch_size"] if device == "cuda" else cfg["inference"]["batch_size"]
        return SupervisedScorer(tax, "finetune", finetune_logits_fn(model, tok, f["max_length"], bs, device), T, zs, agg)
    raise ValueError(f"unknown method {method!r}")


def score_gold(cfg: dict[str, Any], tax: dict[str, Any], method: str, scorer, embedder: Embedder | None = None) -> Path:
    """Window-level scores for every gold post (dev AND test; thresholds come later from dev only). Writes
    window_scores.parquet (gold_id, gold_split, window index, s__<category>; no text) + kinds.json. Inference only."""
    d = gold_run_dir(cfg, method)
    out = d / "window_scores.parquet"
    if out.exists():
        return out
    d.mkdir(parents=True, exist_ok=True)
    g = gold_texts(cfg)
    S, owner = score_posts(cfg, scorer, g["text_clean"].tolist(), embedder, cache_name="gold_windows")
    cats = category_ids(tax)
    w = pd.DataFrame(S.astype(np.float32), columns=[f"s__{c}" for c in cats])
    w.insert(0, "window", pd.Series(owner).groupby(owner).cumcount().to_numpy())
    w.insert(0, "gold_split", g["gold_split"].to_numpy()[owner])
    w.insert(0, "gold_id", g["gold_id"].to_numpy()[owner])
    tmp = out.with_suffix(".tmp.parquet")
    w.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    write_json(d / "kinds.json", {"method": method, "method_key": method_key(cfg, method), "kinds": dict(zip(cats, scorer.kinds)),
                                  "aggregation": getattr(scorer, "aggregation", settings(cfg)["aggregation"]),
                                  "windows": settings(cfg)["windows"]})
    return out


def post_scores(cfg: dict[str, Any], tax: dict[str, Any], method: str, aggregation: str | None = None) -> pd.DataFrame:
    """Post-level scores (gold_id, gold_split, n_windows, s__<category>) from the saved window scores."""
    d = gold_run_dir(cfg, method)
    w = pd.read_parquet(d / "window_scores.parquet")
    agg = aggregation or json.loads((d / "kinds.json").read_text(encoding="utf-8"))["aggregation"]
    ids = pd.unique(w["gold_id"])
    owner = pd.Categorical(w["gold_id"], categories=ids).codes
    cols = [f"s__{c}" for c in category_ids(tax)]
    S = aggregate_windows(w[cols].to_numpy(), owner, len(ids), agg)
    first = w.drop_duplicates("gold_id").set_index("gold_id")
    return pd.DataFrame({"gold_id": ids, "gold_split": first.loc[ids, "gold_split"].to_numpy(),
                         "n_windows": np.bincount(owner, minlength=len(ids)), **{c: S[:, j] for j, c in enumerate(cols)}})


NEVER_COSINE = 1.01   # above the largest possible cosine (unit vectors): a threshold that never flags


def _grid(kind: str, tcfg: dict[str, Any]) -> list[float]:
    return {"probability": tcfg["probability_grid"], "cosine": tcfg["cosine_grid"], "count": tcfg["count_grid"]}[kind]


def global_cosine_threshold(Y: np.ndarray, S: np.ndarray, kinds: Sequence[str], tcfg: dict[str, Any]) -> float | None:
    """ONE threshold for all cosine-scored categories of a method: max pooled (micro) F1 over every (post, cosine
    category) cell of GOLD-DEV. None if the method has no cosine categories."""
    cols = [j for j, k in enumerate(kinds) if k == "cosine"]
    if not cols:
        return None
    tab, _ = tune_label_thresholds(Y[:, cols].reshape(-1, 1), S[:, cols].reshape(-1, 1), _grid("cosine", tcfg), ["pooled"])
    return float(tab.iloc[0]["threshold"])


def tune_thresholds(Y: np.ndarray, S: np.ndarray, kinds: Sequence[str], cats: Sequence[str], tcfg: dict[str, Any]) -> pd.DataFrame:
    """Per-category threshold maximising that category's F1 on GOLD-DEV (shared helper: ties -> closest to 0.5; count
    thresholds are rounded up to whole windows). Categories with fewer than low_support_dev_positives dev positives are
    flagged; with low_support_rule keep_default (user decision) they get: probability -> 0.5, count -> 1 window,
    cosine -> the method's ONE global cosine threshold (global_cosine_threshold), since a cosine has no natural default."""
    g = global_cosine_threshold(Y, S, kinds, tcfg)
    keep = tcfg.get("low_support_rule", "tune") == "keep_default"
    rows = []
    for j, (c, k) in enumerate(zip(cats, kinds)):
        tab, _ = tune_label_thresholds(Y[:, [j]], S[:, [j]], _grid(k, tcfg), [c])
        r = tab.iloc[0].to_dict()
        t = float(np.ceil(r["threshold"])) if k == "count" else float(r["threshold"])
        low = r["n_pos"] < tcfg["low_support_dev_positives"]
        default = {"probability": 0.5, "count": 1.0, "cosine": g}[k]
        rule = ({"probability": "default_0.5", "count": "default_1_window", "cosine": "global_cosine"}[k]
                if low and keep else "tuned")
        # Gold labels that are positive for almost every post cannot say where a threshold belongs: the F1 optimum is "flag
        # everything" (thresholds at the grid floor). Probability -> the SAD-calibrated 0.5; cosine has no natural default and
        # no human-labelled evidence -> NEVER_COSINE (the raw score is still reported, the category is just never flagged).
        if k != "count" and r["n_pos"] / max(len(Y), 1) > tcfg.get("degenerate_prevalence", 1.1):
            rule, default = "degenerate_prevalence", (0.5 if k == "probability" else NEVER_COSINE)
        rows.append({"category": c, "kind": k, "tuned": t, "applied": default if rule != "tuned" else t, "rule": rule,
                     "f1_dev_at_tuned": r["f1_at_threshold"], "n_pos_dev": int(r["n_pos"]), "low_support": bool(low)})
    out = pd.DataFrame(rows)
    out.attrs["global_cosine_threshold"] = g
    return out


def _supported_macro(m: dict[str, Any], names: Sequence[str] | None = None) -> float:
    f = [r["f1"] for r in m["per_label"] if r["support"] > 0 and (names is None or r["label"] in names)]
    return float(np.mean(f)) if f else float("nan")


def evaluate_split(Y: np.ndarray, Ynone: np.ndarray, S: np.ndarray, thr: Sequence[float], tax: dict[str, Any]) -> dict[str, Any]:
    """Multi-label metrics at the given thresholds + macro-F1 over supported categories (all / SAD-covered / zero-shot
    only) + the none_unclear rule (no category above threshold) against the gold none_unclear column."""
    cats, cov = category_ids(tax), coverage(tax)
    m = multilabel_metrics(Y, S, np.asarray(thr, dtype=float), cats)
    H = S >= np.asarray(thr, dtype=float)
    none_pred = ~H.any(1)
    tp = int((none_pred & (Ynone == 1)).sum()); fp = int((none_pred & (Ynone == 0)).sum())
    fn = int((~none_pred & (Ynone == 1)).sum())
    m.update({
        "f1_macro_supported": _supported_macro(m),
        "f1_macro_sad_covered": _supported_macro(m, [c for c in cats if cov[c] == "sad"]),
        "f1_macro_zero_shot_only": _supported_macro(m, [c for c in cats if cov[c] == "zero_shot"]),
        "none_unclear": {"gold_rate": float(Ynone.mean()), "predicted_rate": float(none_pred.mean()),
                         "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                         "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0},
        "mean_labels_per_post": float(H.sum(1).mean()),
    })
    return m


def _xy(post: pd.DataFrame, gold: pd.DataFrame, tax: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cats = category_ids(tax)
    m = post[post["gold_split"] == split].merge(gold[["gold_id", *cats, NONE_COL]], on="gold_id", how="inner")
    return m[cats].to_numpy(), m[NONE_COL].to_numpy(), m[[f"s__{c}" for c in cats]].to_numpy(dtype=float)


def tune_on_dev(cfg: dict[str, Any], tax: dict[str, Any], method: str, gold: pd.DataFrame) -> dict[str, Any]:
    """Thresholds from GOLD-DEV only; dev metrics (optimistic) + aggregation sensitivity on dev. Writes thresholds.json."""
    s = settings(cfg)
    d = gold_run_dir(cfg, method)
    kinds_meta = json.loads((d / "kinds.json").read_text(encoding="utf-8"))
    cats = category_ids(tax)
    kinds = [kinds_meta["kinds"][c] for c in cats]
    post = post_scores(cfg, tax, method)
    Y, Yn, S = _xy(post, gold, tax, "dev")
    tab = tune_thresholds(Y, S, kinds, cats, s["thresholds"])
    thr = tab["applied"].to_numpy()
    sens = {}
    if method != "keyword":
        for agg in ("max", "mean", "top2_mean"):
            Ya, Yna, Sa = _xy(post_scores(cfg, tax, method, agg), gold, tax, "dev")
            ta = tune_thresholds(Ya, Sa, kinds, cats, s["thresholds"])["applied"].to_numpy()
            sens[agg] = _summary(evaluate_split(Ya, Yna, Sa, ta, tax))
    out = {"method": method, "method_key": method_key(cfg, method), "fitted_on": "gold-dev", "n_dev": int(len(Y)),
           "aggregation": kinds_meta["aggregation"], "thresholds": dict(zip(cats, map(float, thr))),
           "global_cosine_threshold": tab.attrs.get("global_cosine_threshold"),
           "low_support_rule": s["thresholds"]["low_support_rule"], "support_dev": dict(zip(cats, map(int, Y.sum(0)))),
           "per_category": tab.to_dict(orient="records"), "dev": evaluate_split(Y, Yn, S, thr, tax),
           "dev_aggregation_sensitivity": sens,
           "caveat": "dev metrics are optimistic: thresholds were tuned on the same rows"}
    write_json(d / "thresholds.json", out)
    return out


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k != "per_label"}


def _select_method(s: dict[str, Any], methods: Sequence[str], res: dict[str, Any]) -> None:
    """Fill selected_method / ranking / selection_rule in `res`: config force_method wins, else the best dev (or test) metric."""
    sel = s["method_selection"]
    split, metric = sel["split"], sel["metric"]
    key = "dev" if split == "dev" else "test"
    ranking = sorted(methods, key=lambda m: -np.nan_to_num(res["methods"][m][key][metric], nan=-1))
    forced = sel.get("force_method")
    if forced and forced not in methods:
        raise ValueError(f"method_selection.force_method {forced!r} is not one of {list(methods)}")
    rule = f"max {split} {metric}" + ("" if split == "dev" else " (TEST used for selection: the selected test number is optimistic)")
    if forced:
        rule = (f"forced to {forced} by config (method_selection.force_method); highest {split} {metric} was {ranking[0]}. "
                f"Reason: the gold labels are machine-made (zero-shot labeller, nearly every category positive), so gold scores cannot separate the methods; SAD is the only human-labelled evidence")
    res.update({"selected_method": forced or ranking[0], "ranking": ranking, "selection_rule": rule,
                "test_ranking_for_reference": sorted(methods, key=lambda m: -np.nan_to_num(res["methods"][m]["test"][metric], nan=-1)),
                "created_utc": res.get("created_utc") or _dt.datetime.now(_dt.timezone.utc)})


def final_gold_report(cfg: dict[str, Any], tax: dict[str, Any], methods: Sequence[str], gold: pd.DataFrame,
                      sad_reports: dict[str, dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """Gold-TEST evaluation of every method at its dev-tuned thresholds, computed ONCE (GOLD_TEST_REPORT.json).
    Row bootstrap CIs over categories with >= 1 test positive. Method chosen by the config selection split/metric.
    Domain-shift cost = SAD-test category macro-F1 (SAD-covered) - gold-test macro-F1 over the same categories."""
    path = runs_root(cfg) / "gold" / "GOLD_TEST_REPORT.json"
    if path.exists() and not force:
        res = json.loads(path.read_text(encoding="utf-8"))
        before = res.get("selected_method"), res.get("selection_rule")
        _select_method(settings(cfg), methods, res)         # config may have changed; test numbers are untouched
        if (res["selected_method"], res["selection_rule"]) != before:
            write_json(path, res)
            res = json.loads(path.read_text(encoding="utf-8"))
        return res
    s = settings(cfg)
    cats, cov = category_ids(tax), coverage(tax)
    ev = s["eval"]
    res: dict[str, Any] = {"methods": {}, "selection": s["method_selection"], "leakage": LEAKAGE_NOTE,
                           "annotation": annotation_note(cfg), "annotation_limitation": annotation_limitation(cfg),
                           "label_source": label_source(cfg)}
    for meth in methods:
        d = gold_run_dir(cfg, meth)
        thr_meta = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        thr = np.array([thr_meta["thresholds"][c] for c in cats])
        post = post_scores(cfg, tax, meth)
        Y, Yn, S = _xy(post, gold, tax, "test")
        test = evaluate_split(Y, Yn, S, thr, tax)
        sup = [j for j in range(len(cats)) if Y[:, j].sum() > 0]
        ci = bootstrap_multilabel(Y[:, sup], S[:, sup], thr[sup], ev["bootstrap_resamples"], cfg["primary_seed"],
                                  ev["ci_level"], [cats[j] for j in sup])
        sad_cov = [c for c in cats if cov[c] == "sad"]
        sad = sad_reports.get(meth, {})
        sad_f1 = sad.get("test_category_macro_f1_sad_covered")
        gold_sad = test["f1_macro_sad_covered"]
        res["methods"][meth] = {
            "method_key": method_key(cfg, meth), "thresholds": thr_meta["thresholds"],
            "low_support_categories": [r["category"] for r in thr_meta["per_category"] if r["low_support"]],
            "threshold_rule": {r["category"]: r["rule"] for r in thr_meta["per_category"]},
            "support": {c: {"dev": int(thr_meta["support_dev"][c]), "test": int(Y[:, j].sum())} for j, c in enumerate(cats)},
            "dev": _summary(thr_meta["dev"]), "test": test, "test_ci_supported_categories": ci,
            "sad_test_category_macro_f1": sad_f1,
            "domain_shift_cost": None if sad_f1 is None else {
                "sad_test": sad_f1, "gold_test_sad_covered": gold_sad, "drop": sad_f1 - gold_sad,
                "categories": sad_cov,
                "caveat": "SAD-test is single-label argmax on sentences; gold-test is multi-label thresholded on posts, "
                          "so the drop is an approximate measure of the domain shift, not a like-for-like difference"}}
    _select_method(s, methods, res)
    write_json(path, res)
    return json.loads(path.read_text(encoding="utf-8"))


# ================================================================ weak Dreaddit-subreddit cross-check (sanity only)
def dreaddit_crosscheck(cfg: dict[str, Any], tax: dict[str, Any], method: str, scorer, thresholds: dict[str, float],
                        embedder: Embedder | None = None) -> dict[str, Any]:
    """Predicted stressor rates per Dreaddit subreddit on stress-positive segments NOT in the gold set. For each
    category with mapped subreddits: rate in mapped vs other subreddits. A SANITY CHECK, never accuracy (a subreddit is
    not a stressor label). Cells with n < k_min are suppressed. No text, no ids in the output."""
    d = gold_run_dir(cfg, method)
    path = d / "dreaddit_crosscheck.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    c = settings(cfg)["crosscheck"]
    dd = pd.concat(SM.load_dreaddit(cfg).values(), ignore_index=True)
    dd = dd[(dd["label"] == c["label"]) & dd["split"].isin(c["dreaddit_splits"]) & ~dd["record_id"].isin(gold_key(cfg)["record_id"])]
    dd = dd.reset_index(drop=True)
    S, owner = score_posts(cfg, scorer, dd["text_clean"].tolist(), embedder, cache_name="dreaddit_crosscheck")
    cats = category_ids(tax)
    P = aggregate_windows(S, owner, len(dd), getattr(scorer, "aggregation", "max"))
    H = P >= np.array([thresholds[x] for x in cats])
    k_min = cfg["privacy"]["k_min"]
    by_sub = []
    for sub, idx in dd.groupby("community").indices.items():
        row: dict[str, Any] = {"community": sub, "n": int(len(idx)), "suppressed": len(idx) < k_min}
        if not row["suppressed"]:
            row.update({f"rate__{x}": float(H[idx, j].mean()) for j, x in enumerate(cats)})
        by_sub.append(row)
    checks = []
    for j, cat in enumerate(tax["categories"]):
        mapped = set(cat["dreaddit_subreddits"])
        if not mapped:
            continue
        inm = dd["community"].isin(mapped).to_numpy()
        n_in, n_out = int(inm.sum()), int((~inm).sum())
        r_in = float(H[inm, j].mean()) if n_in >= k_min else None
        r_out = float(H[~inm, j].mean()) if n_out >= k_min else None
        checks.append({"category": cat["id"], "mapped_subreddits": sorted(mapped), "n_mapped": n_in, "n_other": n_out,
                       "rate_mapped": r_in, "rate_other": r_out,
                       "ratio": (r_in / r_out) if (r_in is not None and r_out) else None,
                       "expected": "rate_mapped > rate_other"})
    out = {"method": method, "n_segments": int(len(dd)), "k_min": k_min, "by_subreddit": by_sub, "checks": checks,
           "note": "weak sanity check: subreddits are not stressor labels; NOT an accuracy estimate"}
    write_json(path, out)
    return out


# ================================================================ 5D bundle
def save_bundle(cfg: dict[str, Any], tax: dict[str, Any], method: str, date: _dt.date | None = None,
                overwrite: bool = False) -> Path:
    """<paths.models>/stressor/v<YYYYMMDD>/: taxonomy copy, stressor_config.json (method, categories + coverage,
    classes -> categories, embedding model + revision, window rule, aggregation, calibration, thresholds, none rule),
    thresholds.json, weights (lr.npz or the fine-tuned model + tokenizer; zero-shot prototypes.npy), metrics.json,
    model_card.md, manifest.json. Every number comes from saved run files."""
    s = settings(cfg)
    bundle = model_version_dir(cfg, COMPONENT, date)
    if (bundle / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"{bundle} already holds a bundle; pass another `date` or overwrite=True")
    if bundle.exists() and any(bundle.iterdir()):
        # overwrite: keep the old bundle aside (a different method writes different weight files; never mix them)
        old = bundle.with_name(f"{bundle.name}.bak-{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%SZ}")
        os.replace(bundle, old)
        LOG.warning("existing bundle moved to %s", old)
    bundle.mkdir(parents=True, exist_ok=True)
    gold_rep =json.loads((runs_root(cfg) / "gold" / "GOLD_TEST_REPORT.json").read_text(encoding="utf-8"))
    d = gold_run_dir(cfg, method)
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
    kinds = json.loads((d / "kinds.json").read_text(encoding="utf-8"))
    shutil.copyfile(taxonomy_path(cfg), bundle / taxonomy_path(cfg).name)
    cal, weights = None, {}
    if method == "embed_lr":
        shutil.copyfile(lr_run_dir(cfg) / "lr.npz", bundle / "lr.npz")
        cal = _read_json(lr_run_dir(cfg) / "SAD_TEST_REPORT.json")
        weights = {"classifier": "lr.npz (coef, intercept, classes)"}
    elif method == "finetune":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        rd = ft_run_dir(cfg, ft_select_seed(cfg, tax)["final_seed"])
        AutoModelForSequenceClassification.from_pretrained(str(rd / "best_model")).save_pretrained(str(bundle))
        AutoTokenizer.from_pretrained(str(rd / "best_model")).save_pretrained(str(bundle))
        cal = _read_json(rd / "SAD_TEST_REPORT.json")
        weights = {"classifier": "model.safetensors + config.json + tokenizer files (fine-tuned)", "run_dir": rd}
    emb = None
    if method != "keyword":
        embedder = Embedder.from_cfg(cfg)
        np.save(bundle / "prototypes.npy", ZeroShotScorer(tax, embedder).prototypes)
        emb = embedder.spec.__dict__
        weights["zero_shot"] = "prototypes.npy (one unit vector per category, taxonomy order)"
    cats = category_ids(tax)
    scfg = {
        "component": COMPONENT, "method": method, "method_key": method_key(cfg, method), "taxonomy_file": taxonomy_path(cfg).name,
        "taxonomy_version": tax["version"], "categories": [{"id": c, "coverage": coverage(tax)[c], "score_kind": kinds["kinds"][c],
                                                             "lower_confidence": coverage(tax)[c] == "zero_shot"} for c in cats],
        "classifier_classes": classifier_classes(tax), "class_to_category": {cl: (cl.split("/")[0] if cl.split("/")[0] in cats else None)
                                                                            for cl in classifier_classes(tax)},
        "embedding_model": emb, "weights": weights,
        "finetune_max_length": s["finetune"]["max_length"] if method == "finetune" else None,
        "calibration": None if cal is None else cal["calibration"],
        "inference_unit": {"windows": s["windows"], "rule": "sentence windows (stressor_model.sentence_windows)"},
        "aggregation": kinds["aggregation"], "thresholds": thr["thresholds"], "thresholds_fitted_on": "gold-dev",
        "none_rule": "none_unclear when no category score reaches its threshold",
        "scope": SCOPE_NOTE, "created_utc": _dt.datetime.now(_dt.timezone.utc)}
    write_json(bundle / "stressor_config.json", scfg)
    write_json(bundle / "thresholds.json", thr)
    metrics = {"gold_test_report": gold_rep, "selected": gold_rep["methods"].get(method),
               "sad_test_report": cal, "annotation": annotation_note(cfg), "annotation_limitation": annotation_limitation(cfg),
               "label_source": label_source(cfg),
               "dreaddit_crosscheck": _read_json(d / "dreaddit_crosscheck.json")}
    write_json(bundle / "metrics.json", metrics)
    (bundle / "model_card.md").write_text(model_card(scfg, metrics), encoding="utf-8")
    from .preprocess import processed_path, split_path
    files = [p for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "manifest.json"]
    write_manifest(bundle / "manifest.json", artifacts=files, seeds=s["finetune"]["seeds"] if method == "finetune" else [cfg["primary_seed"]],
                   dataset_versions={"sad_parquet_sha256": sha256_file(processed_path(cfg, "sad")),
                                     "sad_split": split_path(cfg, "sad").name, "gold_split": gold_files(cfg)["split"].name,
                                     "gold_split_sha256": sha256_file(gold_files(cfg)["split"])},
                   extra={"component": COMPONENT, "method": method}, root=bundle)
    return bundle


def _read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def _f(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def model_card(sc: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Short model card; every number is read from `metrics` (saved run files)."""
    sel = metrics["selected"] or {}
    t, ci = sel.get("test", {}), sel.get("test_ci_supported_categories", {})
    ds = sel.get("domain_shift_cost") or {}
    sup = sel.get("support", {})
    per = ", ".join(f"{r['label']} {r['f1']:.2f} (test n={r['support']}, dev n={sup.get(r['label'], {}).get('dev', 'n/a')})"
                    for r in t.get("per_label", []))
    zs = [c["id"] for c in sc["categories"] if c["lower_confidence"]]
    cc = metrics.get("dreaddit_crosscheck") or {}
    cross = "; ".join(f"{r['category']}: mapped {_f(r['rate_mapped'])} vs other {_f(r['rate_other'])}" for r in cc.get("checks", []))
    return f"""# Model card: stressor-language classifier ({sc['method_key']})

**What it is.** A multi-label tagger of the stressor categories a text *names as a cause of pressure*
(taxonomy v{sc['taxonomy_version']}, {len(sc['categories'])} categories). Outputs feed **aggregate, population-level**
signals only.

**Stressor labels are a research taxonomy, not a clinical assessment.** They describe language in sampled text. They do
not diagnose, assess or describe any person, and must not be used for screening, profiling, ranking or moderation.

**Method.** {sc['method']}. Posts are split into sentence windows; each window is scored; post score = {sc['aggregation']}
over windows; per-category thresholds tuned on the Reddit gold-dev set; no category above threshold -> none_unclear.
Categories without SAD training labels ({', '.join(zs)}) are scored by zero-shot embedding similarity and are
**lower confidence**. Embedding model: {(sc['embedding_model'] or {}).get('hf_id', 'n/a')}.

**Data.** Training: SAD v1 (MIT; crowd-written and LiveJournal sentences, no Reddit). Evaluation: hand-labelled gold set
of Dreaddit stress-positive segments (train/validation splits only), split 50/50 into dev (tuning) and test (scored once).

**Gold-test (n = {t.get('n', 'n/a')}).** macro-F1 over categories with test positives {_f(t.get('f1_macro_supported'))}
[95% CI {_f(ci.get('f1_macro', {}).get('lo'))}–{_f(ci.get('f1_macro', {}).get('hi'))}]; micro-F1 {_f(t.get('f1_micro'))};
SAD-covered categories {_f(t.get('f1_macro_sad_covered'))}; zero-shot-only categories {_f(t.get('f1_macro_zero_shot_only'))};
none_unclear F1 {_f((t.get('none_unclear') or {}).get('f1'))}. Per category: {per or 'n/a'}.
Low-support categories (< 10 dev positives; default threshold, or one global threshold for cosine scores):
{', '.join(sel.get('low_support_categories', [])) or 'none'}.

**Domain shift.** SAD-test category macro-F1 {_f(ds.get('sad_test'))} vs Reddit gold-test {_f(ds.get('gold_test_sad_covered'))}
(drop {_f(ds.get('drop'))}; approximate: different units and decision rules).

**Weak sanity check (not accuracy).** Predicted rates in mapped vs other Dreaddit subreddits: {cross or 'not run'}.

**Gold labels: {metrics.get('annotation', ANNOTATION_NOTE)}.**

**Limitations.**
- {metrics.get('annotation_limitation', ANNOTATION_LIMITATION)}
- SAD is short, mostly non-Reddit, 94.5% stressor-positive text; the gold set is small (~200 test posts) and comes from
  10 Dreaddit subreddits (2017-2018), so per-category F1 has wide CIs and may not transfer to other communities or years.
- Max aggregation favours long posts (more windows, more chances to pass a threshold).
- 'Everyday Decision Making' was mapped to overload (user decision), so overload also absorbs some uncertainty language.
- Seed keywords and paraphrase descriptions are author-chosen and unvalidated.
- English only; no demographic fairness audit is possible (no such attributes, by design).
"""
