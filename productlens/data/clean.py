"""
Text cleaning pipeline for reviews.

Implements Unicode normalization, HTML cleanup, whitespace normalization,
minimum length filtering, and optional language detection. Always preserves
raw_text alongside clean_text.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import List, Optional

from productlens.schemas import ReviewRecord

logger = logging.getLogger("productlens.data.clean")

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")
_RE_MULTI_WHITESPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_URL = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def remove_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    # First decode entities
    text = html.unescape(text)
    # Remove any residual encoded entities
    text = _RE_HTML_ENTITIES.sub(" ", text)
    # Remove tags
    text = _RE_HTML_TAGS.sub(" ", text)
    return text


def remove_urls(text: str) -> str:
    """Remove URLs (never visit them)."""
    return _RE_URL.sub("", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return _RE_EMAIL.sub("", text)


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters."""
    return _RE_CONTROL_CHARS.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs and excessive newlines."""
    text = _RE_MULTI_WHITESPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a text string.

    Pipeline order:
    1. Unicode normalization (NFC)
    2. HTML entity decode + tag removal
    3. Control character removal
    4. URL removal
    5. Email removal
    6. Whitespace normalization
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = remove_html(text)
    text = remove_control_chars(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Review-level cleaning
# ---------------------------------------------------------------------------

def clean_review(review: ReviewRecord) -> ReviewRecord:
    """
    Clean a single review, preserving raw_text.

    Parameters
    ----------
    review : ReviewRecord
        Input review. ``raw_text`` must be populated.

    Returns
    -------
    ReviewRecord
        Review with ``clean_text`` set and ``text`` updated.
    """
    raw = review.raw_text or review.text
    cleaned = clean_text(raw)

    # Also clean title
    clean_title = clean_text(review.title) if review.title else ""

    return ReviewRecord(
        review_id=review.review_id,
        product_id=review.product_id,
        parent_product_id=review.parent_product_id,
        category=review.category,
        title=clean_title,
        text=cleaned,
        raw_text=raw,
        clean_text=cleaned,
        rating=review.rating,
        verified_purchase=review.verified_purchase,
        helpful_vote=review.helpful_vote,
        timestamp=review.timestamp,
        source=review.source,
    )


def clean_reviews(
    reviews: List[ReviewRecord],
    min_chars: int = 20,
    language: Optional[str] = None,
) -> List[ReviewRecord]:
    """
    Clean a list of reviews, filtering by length and optionally language.

    Parameters
    ----------
    reviews : list of ReviewRecord
        Input reviews.
    min_chars : int
        Minimum character length for ``clean_text``. Reviews shorter
        than this are discarded.
    language : str, optional
        If set, filter to this language (requires ``langdetect``).
        Not applied if the library is unavailable.

    Returns
    -------
    list of ReviewRecord
        Cleaned and filtered reviews.
    """
    cleaned: List[ReviewRecord] = []
    removed_short = 0
    removed_empty = 0
    removed_lang = 0

    detect_lang = None
    if language:
        try:
            from langdetect import detect as _detect
            from langdetect import LangDetectException
            detect_lang = _detect
        except ImportError:
            logger.warning(
                "langdetect not installed — skipping language filter. "
                "Install with: pip install langdetect"
            )

    for review in reviews:
        result = clean_review(review)

        # Filter empty
        if not result.clean_text:
            removed_empty += 1
            continue

        # Filter short
        if len(result.clean_text) < min_chars:
            removed_short += 1
            continue

        # Filter language
        if detect_lang and language:
            try:
                detected = detect_lang(result.clean_text)
                if detected != language:
                    removed_lang += 1
                    continue
            except Exception:
                pass  # Keep review if detection fails

        cleaned.append(result)

    logger.info(
        "Cleaning: %d → %d reviews (removed: %d empty, %d short, %d lang)",
        len(reviews),
        len(cleaned),
        removed_empty,
        removed_short,
        removed_lang,
    )
    return cleaned
