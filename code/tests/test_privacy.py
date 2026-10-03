"""PII scrubbing tests (synthetic strings only)."""

from stress_signals.config import load_config
from stress_signals.privacy import author_columns, scrub_text


def test_scrub_replaces_pii():
    s = "mail me at jane.doe@example.com or see https://x.com/a?b=1, ping u/some_user and @handle, call 555-123-4567"
    out = scrub_text(s)
    for token in ("<EMAIL>", "<URL>", "<USER>", "<PHONE>"):
        assert token in out
    for leaked in ("jane.doe", "x.com/a", "some_user", "@handle", "555-123-4567"):
        assert leaked not in out


def test_scrub_keeps_plain_text_and_non_strings():
    assert scrub_text("I have 3 exams in 2 weeks") == "I have 3 exams in 2 weeks"
    assert scrub_text(None) is None


def test_author_columns_detected():
    cfg = load_config()
    assert author_columns(["subreddit", "Author", "post", "date"], cfg) == ["Author"]


def test_phone_regex_ignores_years_and_amounts():
    for s in ("2019 2020 2021", "10 000 000", "I owe $12,500 at 4.5%", "401k since 2019-11-01", "in 1999 I was 25"):
        assert scrub_text(s) == s
    for s in ("(555) 123-4567", "555.123.4567", "+44 20 7946 0958", "5551234567"):
        assert "<PHONE>" in scrub_text(s)
