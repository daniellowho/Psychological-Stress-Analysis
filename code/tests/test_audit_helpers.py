"""Audit helper tests on synthetic data (no downloads)."""

import pandas as pd

from stress_signals.audit import containment_overlap, duplicate_stats, md_table, text_hash


def test_duplicates_raw_and_normalised():
    s = pd.Series(["Hello world", "Hello world", "hello   WORLD", "other", None])
    d = duplicate_stats(s)
    assert d["n_texts"] == 4
    assert d["exact_duplicate_rows"] == 1
    assert d["normalised_duplicate_rows"] == 2


def test_text_hash_normalises():
    assert text_hash(" A  b ") == text_hash("a b")


def test_containment_overlap():
    corpus = ["one two three four five six seven eight nine ten eleven"]
    r = containment_overlap(["two three four five six seven eight nine", "completely different words here and there ok fine"], corpus, 8, 0.8)
    assert r["n_contained"] == 1 and r["n_queries"] == 2


def test_md_table_escapes_pipes():
    assert "a\|b" in md_table([("a|b", 1)], ["x", "y"])
