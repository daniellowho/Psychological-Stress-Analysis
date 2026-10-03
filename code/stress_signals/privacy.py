"""Privacy helpers: PII scrubbing and author-field removal. Applied before any text is shown or stored."""

from __future__ import annotations

import re
from typing import Any, Iterable

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|io|co|uk|ly|me)(?:/\S*)?\b")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
# Phone numbers: NANP 3-3-4 (optional +1 / parentheses), "+"-prefixed international, or a bare 10-11 digit run.
# Deliberately does NOT match runs of years or amounts such as "2019 2020 2021" or "10 000 000" (Step 2 fix).
PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}"
    r"|\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}"
    r"|\d{10,11}"
    r")(?!\w)")
MENTION_RE = re.compile(r"(?i)(?<![\w/])(?:/?u/[A-Za-z0-9_-]{2,}|@[A-Za-z0-9_]{2,})")


def scrub_text(text: Any, cfg: dict[str, Any] | None = None) -> Any:
    """Replace URLs, emails, phone numbers and @/u/ mentions with placeholder tokens. Non-strings pass through."""
    if not isinstance(text, str):
        return text
    p = (cfg or {}).get("privacy", {})
    flags = p.get("scrub", {})
    if flags.get("emails", True):
        text = EMAIL_RE.sub(p.get("email_token", "<EMAIL>"), text)
    if flags.get("urls", True):
        text = URL_RE.sub(p.get("url_token", "<URL>"), text)
    if flags.get("user_mentions", True):
        text = MENTION_RE.sub(p.get("mention_token", "<USER>"), text)
    if flags.get("phone_numbers", True):
        text = PHONE_RE.sub(p.get("phone_token", "<PHONE>"), text)
    return text


def author_columns(columns: Iterable[str], cfg: dict[str, Any]) -> list[str]:
    """Columns that must be dropped at ingestion (case-insensitive match against privacy.drop_fields)."""
    drop = {f.lower() for f in cfg["privacy"]["drop_fields"]}
    return [c for c in columns if str(c).lower() in drop]
