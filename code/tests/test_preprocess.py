"""Step 2 preprocessing tests: synthetic strings only (no user text), no model weights touched."""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from stress_signals.config import load_config
from stress_signals.preprocess import (
    UNIFIED_COLUMNS, SplitMismatchError, aggregate_chunk_scores, assert_privacy_safe, chunk_text, clean_text,
    clean_text_lexical, dedupe_key, exact_dedupe, group_overlap, is_bot_text, load_or_create_split,
    make_dreaddit_splits, minhash_near_duplicates, n_tokens_estimate, split_path, stable_record_id,
)
from stress_signals.privacy import EMAIL_RE, MENTION_RE, PHONE_RE, URL_RE


@pytest.fixture(scope="module")
def cfg():
    return load_config()


SYNTHETIC = [
    "Hello &amp;#x200B; **bold** and *italic* [my link](https://example.com/a_(b)) <url> end.",
    "&#x200B;\n\n> quoted line\n\n# Heading\n* bullet one\n- bullet two\n---\nlast line [removed]",
    "Ping u/some_user, /u/other-user and @handle!! Email jane.doe@example.org or call (555) 123-4567 😭😭",
    "Visit www.example.net/path?q=1 or http://foo.io now… I'm SO stressed!!! 2019 2020 2021",
    "Nested &amp;amp;gt; entity, `code`, ^(tiny) super^script, \\*escaped\\* and ~~struck~~ text",
    "",
    "   \t\n  ",
    "[deleted]",
    "Plain sentence with snake_case_name and 3*4=12 maths.",
]


# ---------------------------------------------------------------- cleaning
@pytest.mark.parametrize("s", SYNTHETIC, ids=range(len(SYNTHETIC)))
def test_clean_is_idempotent(s, cfg):
    once = clean_text(s, cfg)
    assert clean_text(once, cfg) == once


@pytest.mark.parametrize("s", SYNTHETIC, ids=range(len(SYNTHETIC)))
def test_no_pii_remains(s, cfg):
    out = clean_text(s, cfg)
    assert not EMAIL_RE.search(out)
    assert not URL_RE.search(out)
    assert not MENTION_RE.search(out)
    assert not PHONE_RE.search(out)
    for leaked in ("jane.doe", "example.org", "example.com", "some_user", "other-user", "@handle", "555", "foo.io"):
        assert leaked not in out


def test_placeholders_markdown_and_entities(cfg):
    out = clean_text(SYNTHETIC[0], cfg)
    assert "​" not in out and "&" not in out and "**" not in out
    assert out.count("<URL>") == 2 and "my link" in out and "bold" in out and "italic" in out
    out2 = clean_text(SYNTHETIC[1], cfg)
    assert out2 == "quoted line Heading bullet one bullet two last line"
    assert "<USER>" in clean_text(SYNTHETIC[2], cfg)


def test_keeps_case_punctuation_emoji_and_numbers(cfg):
    out = clean_text(SYNTHETIC[3], cfg)
    assert "SO stressed!!!" in out and "I'm" in out and "…" in out
    assert "2019 2020 2021" in out                     # year runs are not mistaken for phone numbers
    assert "😭😭" in clean_text(SYNTHETIC[2], cfg)
    assert "snake_case_name" in clean_text(SYNTHETIC[8], cfg) and "3*4=12" in clean_text(SYNTHETIC[8], cfg)


def test_removed_and_empty_become_empty(cfg):
    for s in ("", "   \t\n  ", "[deleted]", "[removed]", None, 3.5):
        assert clean_text(s, cfg) == ""


def test_whitespace_collapsed(cfg):
    out = clean_text("a\n\n b\t\tc   d", cfg)
    assert out == "a b c d"


def test_clean_stats_counted(cfg):
    st = Counter()
    clean_text(SYNTHETIC[2], cfg, st)
    assert st["mention"] >= 3 and st["email_removed"] == 1 and st["phone_removed"] == 1


def test_bot_text(cfg):
    assert is_bot_text("I am a bot, and this action was performed automatically.", cfg)
    assert is_bot_text("Your post has been removed for breaking rule 2", cfg)
    assert not is_bot_text("My robot vacuum stresses me out", cfg)


def test_lexical_path(cfg):
    out = clean_text_lexical("I'm NOT sleeping and my exams are worrying me https://x.com", cfg)
    toks = out.split()
    assert out == out.lower()
    assert "not" in toks                                   # negation kept
    assert "and" not in toks and "my" not in toks and "im" not in toks
    assert "exam" in toks and "url" not in toks


def test_token_estimate():
    assert n_tokens_estimate("Hello, world!") == 4
    assert n_tokens_estimate("") == 0


# ---------------------------------------------------------------- ids / dedupe
def test_stable_record_id():
    a = stable_record_id("dreaddit", 123)
    assert a == stable_record_id("dreaddit", "123") and len(a) == 20
    assert a != stable_record_id("goemotions", 123)


def test_exact_dedupe_drops_conflicting_groups():
    df = pd.DataFrame({"text_clean": ["A b", "a  B", "c", "c", "d"], "label": [1, 1, 0, 1, 0]})
    out, st = exact_dedupe(df, label_cols=["label"])
    assert list(out["text_clean"]) == ["A b", "d"]
    assert st == {"duplicate_rows_dropped": 1, "conflicting_label_groups": 1, "conflicting_label_rows_dropped": 2}
    assert dedupe_key("A  b ") == dedupe_key("a b")


def test_minhash_near_duplicates_optional():
    pytest.importorskip("datasketch")
    base = "the rent went up again this month and i do not know how we will pay for everything now"
    texts = [base, base + " honestly", "completely different text about teaching a class of kids today"]
    drop = minhash_near_duplicates(texts, num_perm=128, threshold=0.7, shingle_n=3)
    assert drop == {1}


# ---------------------------------------------------------------- chunking
@pytest.fixture(scope="module")
def tokenizer():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer not cached locally: {exc!r}")


LONG_TEXTS = [
    " ".join(f"Sentence number {i} talks about work deadlines and money worries." for i in range(120)),
    ("word " * 1500).strip(),                                            # no punctuation at all
    "Short. " + "x" * 3000 + " end.",                                    # one giant token-split "word"
    " ".join(f"I can't sleep 😭 ({i})!" for i in range(300)),           # emoji + punctuation heavy
    "Tiny text.",
]
LONG_IDS = ["sentences", "no_punct", "giant_word", "emoji", "tiny"]


@pytest.mark.parametrize("max_tokens,stride", [(64, 16), (128, 32), (256, 64), (512, 128)])
@pytest.mark.parametrize("text", LONG_TEXTS, ids=LONG_IDS)
def test_chunks_never_exceed_max_tokens(tokenizer, text, max_tokens, stride):
    chunks = chunk_text(text, tokenizer, max_tokens, stride)
    assert chunks
    for c in chunks:
        assert c.n_tokens <= max_tokens
        assert len(tokenizer(c.text, add_special_tokens=True)["input_ids"]) <= max_tokens
        assert text[c.start_char:c.end_char] == c.text


@pytest.mark.parametrize("text", LONG_TEXTS[:4], ids=LONG_IDS[:4])
def test_chunks_cover_text_and_overlap(tokenizer, text):
    chunks = chunk_text(text, tokenizer, 128, 32)
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(text)   # text_clean is always stripped
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char <= a.end_char + 1            # no gap (at most the separating space)
        assert b.start_char > a.start_char              # progress


def test_max_chunks_subset_keeps_first_and_last(tokenizer):
    full = chunk_text(LONG_TEXTS[0], tokenizer, 64, 16)
    sub = chunk_text(LONG_TEXTS[0], tokenizer, 64, 16, max_chunks=4)
    assert len(full) > 4 and len(sub) == 4
    assert sub[0].text == full[0].text and sub[-1].text == full[-1].text
    assert all(c.n_chunks_total == len(full) for c in sub)


def test_chunk_bad_stride(tokenizer):
    with pytest.raises(ValueError):
        chunk_text("hello world.", tokenizer, 64, 64)


def test_aggregation():
    assert aggregate_chunk_scores([0.2, 0.4, 0.9]) == pytest.approx(0.5)
    assert aggregate_chunk_scores([0.2, 0.4, 0.9], "max") == pytest.approx(0.9)
    assert aggregate_chunk_scores([0.2, 0.8], "weighted_mean", weights=[3, 1]) == pytest.approx(0.35)
    multi = aggregate_chunk_scores(np.array([[0.1, 0.9], [0.3, 0.5]]))
    assert np.allclose(multi, [0.2, 0.7])
    with pytest.raises(ValueError):
        aggregate_chunk_scores([])


# ---------------------------------------------------------------- splits
def _toy_dreaddit(n_posts=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_posts):
        for s in range(int(rng.integers(1, 4))):
            rows.append({"record_id": stable_record_id("toy", f"{p}-{s}"), "post_id": f"p{p}",
                         "label": int(rng.integers(0, 2)), "official_split": "test" if p >= 160 else "train"})
    return pd.DataFrame(rows)


def test_dreaddit_split_grouped_and_stratified():
    df = _toy_dreaddit()
    sp = make_dreaddit_splits(df, 0.15, seed=42)
    sid = {r: k for k, v in sp.items() for r in v}
    df["split"] = df["record_id"].map(sid)
    assert df["split"].notna().all()
    assert set(df.loc[df.official_split == "test", "split"]) == {"test"}
    assert all(v == 0 for v in group_overlap(df).values())
    tr_rate = df.loc[df.split == "train", "label"].mean()
    va_rate = df.loc[df.split == "validation", "label"].mean()
    assert abs(tr_rate - va_rate) < 0.15
    assert sp == make_dreaddit_splits(df, 0.15, seed=42)   # deterministic


def test_split_file_never_regenerated_silently(cfg, tmp_path):
    c = {**cfg, "_paths": {**cfg["_paths"], "splits": tmp_path}}
    df = _toy_dreaddit()
    make = lambda: make_dreaddit_splits(df, 0.15, 42)  # noqa: E731
    first = load_or_create_split(c, "toy", df, make, {"seed": 42})
    again = load_or_create_split(c, "toy", df, lambda: pytest.fail("must load, not regenerate"), {})
    assert first == again
    with pytest.raises(SplitMismatchError):
        load_or_create_split(c, "toy", df.iloc[:-1], make, {})
    load_or_create_split(c, "toy", df.iloc[:-5], lambda: make_dreaddit_splits(df.iloc[:-5], 0.15, 42), {}, force=True)
    assert len(list(tmp_path.glob("toy_split_v*.bak-*.json"))) == 1


def test_saved_dreaddit_split_has_no_post_overlap(cfg):
    """Runs against the real processed data once Step 2 has been run (skipped before)."""
    p = cfg["_paths"]["processed"] / "dreaddit.parquet"
    if not p.exists():
        pytest.skip("data/processed/dreaddit.parquet not built yet")
    df = pd.read_parquet(p, columns=["record_id", "post_id", "split", "official_split"])
    assert all(v == 0 for v in group_overlap(df).values())
    assert set(df.loc[df.official_split == "test", "split"]) == {"test"}
    saved = json.loads(split_path(cfg, "dreaddit").read_text(encoding="utf-8"))
    assert sum(saved["counts"].values()) == len(df)


def test_processed_outputs_schema_and_privacy(cfg):
    from stress_signals.preprocess import DATASETS, processed_path
    found = [processed_path(cfg, n) for n in DATASETS if processed_path(cfg, n).exists()]   # sidecars excluded
    if not found:
        pytest.skip("no processed parquet yet")
    for p in found:
        df = pd.read_parquet(p)
        assert list(df.columns[:len(UNIFIED_COLUMNS)]) == UNIFIED_COLUMNS
        assert df["record_id"].is_unique
        assert_privacy_safe(df, cfg)


def test_assert_privacy_safe_rejects_author_column(cfg):
    df = pd.DataFrame({c: ["x"] for c in UNIFIED_COLUMNS} | {"author": ["someone"]})
    with pytest.raises(AssertionError):
        assert_privacy_safe(df, cfg)


# ---------------------------------------------------------------- language id
def test_language_id(cfg):
    pytest.importorskip("lingua")
    from stress_signals.preprocess import LanguageIdentifier
    lid = LanguageIdentifier(cfg)
    codes, p_en = lid.detect(["I am really stressed about my exams next week and cannot sleep.",
                              "Estoy muy estresado por los exámenes de la próxima semana."])
    assert codes[0] == "en" and codes[1] == "es" and p_en[0] > p_en[1]
