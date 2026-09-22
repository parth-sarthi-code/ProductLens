"""
Sentence splitting for reviews.

Rule-based splitter that handles abbreviations, punctuation, Unicode, and
edge cases. Produces SentenceRecord objects with character offsets into
the review's clean_text field.

Design choice: rule-based rather than spaCy to avoid a heavy dependency
in smoke mode. SpaCy can be used as an optional enhancement.
"""

from __future__ import annotations

import logging
import re
from typing import List

from productlens.schemas import ReviewRecord, SentenceRecord
from productlens.utils import stable_id

logger = logging.getLogger("productlens.data.sentence_split")

# ---------------------------------------------------------------------------
# Abbreviations that should NOT cause sentence breaks
# ---------------------------------------------------------------------------

# Title abbreviations: NEVER cause a break because they always precede a name
_TITLE_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "gen", "sgt",
    "cpl", "pvt", "capt", "lt", "col", "maj", "cmdr", "adm",
})

# Regular abbreviations: do NOT cause breaks UNLESS followed by an uppercase
# multi-character word (indicating a new sentence)
_ABBREVIATIONS = frozenset({
    "st", "ave", "blvd",
    "vs", "etc", "inc", "ltd", "corp", "co", "dept", "univ",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "fig", "eq", "vol", "no", "approx", "est", "min", "max",
    "e.g", "i.e", "viz", "cf",
    "u.s", "u.s.a", "u.k",
})

# ---------------------------------------------------------------------------
# Sentence boundary regex
# ---------------------------------------------------------------------------

# Matches sentence-ending punctuation followed by whitespace or end of string
_RE_SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<=[\.\!\?])           # lookbehind: sentence-ending punctuation
    (?:\s+)                 # whitespace after punctuation
    (?=[A-Z\"\'\(\[\u201c]) # lookahead: uppercase letter, quote, or bracket
    """,
    re.VERBOSE | re.UNICODE,
)

# Simpler fallback: split on .!? followed by space
_RE_SIMPLE_BOUNDARY = re.compile(
    r"""
    (?<=[\.\!\?])   # lookbehind: sentence-ending punctuation
    \s+             # whitespace
    """,
    re.VERBOSE,
)


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    """
    Check if a period at position dot_pos is part of an abbreviation
    that should NOT cause a sentence break.

    Key heuristic: even known abbreviations like "etc." or "U.S." should
    cause a sentence break when followed by whitespace + a multi-character
    capitalized word (indicating a new sentence starts).

    Parameters
    ----------
    text : str
        Full text.
    dot_pos : int
        Position of the period character.

    Returns
    -------
    bool
    """
    # Find the word before the dot
    start = dot_pos
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    word = text[start:dot_pos].lower()

    is_known_abbrev = False

    # Title abbreviations (Mr., Dr., etc.) NEVER cause sentence breaks
    if word in _TITLE_ABBREVIATIONS:
        return True

    if word in _ABBREVIATIONS:
        is_known_abbrev = True
    elif len(word) == 1 and word.isalpha():
        # Single letter like "A.", "U.", "S." — could be abbreviation
        is_known_abbrev = True
    elif "." in word:
        # Dotted abbreviation like "U.S" found before dot
        is_known_abbrev = True

    if not is_known_abbrev:
        return False

    # If this abbreviation is followed by whitespace + a multi-character
    # uppercase word, it's likely a sentence boundary, not an abbreviation.
    # E.g., "etc. The" → sentence boundary, "U.S. Army" → also a boundary,
    # but "U.S.A." → not a boundary (next char is another period pattern).
    after = dot_pos + 1
    # Skip whitespace
    while after < len(text) and text[after] in " \t":
        after += 1
    if after < len(text) and text[after].isupper():
        # Find the next word
        word_end = after
        while word_end < len(text) and text[word_end].isalpha():
            word_end += 1
        next_word = text[after:word_end]
        # If it's a multi-character word (not single letter abbreviation
        # continuation), treat as sentence boundary
        if len(next_word) > 1:
            return False  # Not an abbreviation context → will split

    return True


def _is_decimal(text: str, dot_pos: int) -> bool:
    """Check if a period is part of a decimal number (e.g. 3.5)."""
    if dot_pos > 0 and dot_pos < len(text) - 1:
        return text[dot_pos - 1].isdigit() and text[dot_pos + 1].isdigit()
    return False


def _is_ellipsis(text: str, dot_pos: int) -> bool:
    """Check if a period is part of an ellipsis (...)."""
    if dot_pos >= 2:
        return text[dot_pos - 2: dot_pos + 1] == "..."
    return False


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using rule-based heuristics.

    Handles:
    - Standard sentence boundaries (.!?)
    - Abbreviations (Mr., Dr., U.S., etc.)
    - Decimal numbers (3.5, 4.0)
    - Ellipsis (...)
    - Multiple consecutive punctuation (!!, ?!, ...)
    - Unicode punctuation

    Parameters
    ----------
    text : str
        Input text to split.

    Returns
    -------
    list of str
        Non-empty sentences.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Replace Unicode sentence-ending punctuation with ASCII equivalents
    # for consistent processing, but keep the original in output
    sentences: List[str] = []
    current_start = 0
    i = 0

    while i < len(text):
        char = text[i]

        if char in ".!?\u2026":  # \u2026 = …
            # Handle ellipsis character
            if char == "\u2026":
                # Check if next char starts a new sentence
                rest = text[i + 1:].lstrip()
                if rest and rest[0].isupper():
                    end = i + 1
                    # Skip whitespace
                    while end < len(text) and text[end] in " \t\n\r":
                        end += 1
                    sentence = text[current_start:end].strip()
                    if sentence:
                        sentences.append(sentence)
                    current_start = end
                    i = end
                    continue

            # Handle regular periods
            if char == ".":
                # Skip if abbreviation
                if _is_abbreviation(text, i):
                    i += 1
                    continue

                # Skip if decimal number
                if _is_decimal(text, i):
                    i += 1
                    continue

                # Skip if ellipsis
                if _is_ellipsis(text, i):
                    i += 1
                    continue

            # Skip consecutive punctuation (!!!, ?!, etc.)
            end_punct = i + 1
            while end_punct < len(text) and text[end_punct] in ".!?":
                end_punct += 1

            # Check what follows
            rest_start = end_punct
            while rest_start < len(text) and text[rest_start] in " \t\n\r":
                rest_start += 1

            if rest_start >= len(text):
                # End of text — this is the last sentence
                i = end_punct
                continue

            # If next character could start a new sentence
            next_char = text[rest_start] if rest_start < len(text) else ""
            if next_char and (next_char.isupper() or next_char in "\"'([{" or
                             next_char == "\u201c"):
                sentence = text[current_start:end_punct].strip()
                if sentence:
                    sentences.append(sentence)
                current_start = rest_start
                i = rest_start
                continue

            i = end_punct
            continue

        # Handle newline as potential sentence boundary
        if char == "\n":
            # Double newline is a paragraph break = sentence boundary
            if i + 1 < len(text) and text[i + 1] == "\n":
                sentence = text[current_start:i].strip()
                if sentence:
                    sentences.append(sentence)
                # Skip whitespace
                rest_start = i + 2
                while rest_start < len(text) and text[rest_start] in " \t\n\r":
                    rest_start += 1
                current_start = rest_start
                i = rest_start
                continue

        i += 1

    # Remaining text
    remaining = text[current_start:].strip()
    if remaining:
        sentences.append(remaining)

    return sentences


def split_review_sentences(review: ReviewRecord) -> List[SentenceRecord]:
    """
    Split a review into sentence records with character offsets.

    Offsets refer to the review's ``clean_text`` field.

    Parameters
    ----------
    review : ReviewRecord
        Must have ``clean_text`` populated.

    Returns
    -------
    list of SentenceRecord
    """
    source_text = review.clean_text or review.text
    if not source_text:
        return []

    raw_sentences = split_into_sentences(source_text)
    records: List[SentenceRecord] = []

    search_start = 0
    for sent_text in raw_sentences:
        if not sent_text:
            continue

        # Find the sentence in the source text
        idx = source_text.find(sent_text, search_start)
        if idx == -1:
            # Fallback: try stripping and searching
            stripped = sent_text.strip()
            idx = source_text.find(stripped, search_start)
            if idx == -1:
                # Last resort: use current search position
                logger.debug(
                    "Could not find sentence offset for review %s: '%s'",
                    review.review_id[:12],
                    sent_text[:50],
                )
                idx = search_start
                sent_text = stripped

        start_char = idx
        end_char = start_char + len(sent_text)

        record = SentenceRecord(
            sentence_id=stable_id(review.review_id, start_char, end_char),
            review_id=review.review_id,
            text=sent_text,
            start_char=start_char,
            end_char=end_char,
        )
        records.append(record)

        # Advance search position past this sentence
        search_start = end_char

    return records


def split_reviews_to_sentences(
    reviews: List[ReviewRecord],
) -> List[SentenceRecord]:
    """
    Split all reviews into sentence records.

    Parameters
    ----------
    reviews : list of ReviewRecord

    Returns
    -------
    list of SentenceRecord
    """
    all_sentences: List[SentenceRecord] = []
    for review in reviews:
        sentences = split_review_sentences(review)
        all_sentences.extend(sentences)

    logger.info(
        "Sentence splitting: %d reviews → %d sentences",
        len(reviews),
        len(all_sentences),
    )
    return all_sentences
