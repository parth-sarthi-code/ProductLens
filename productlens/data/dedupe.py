"""
Review deduplication.

Provides exact and near-duplicate detection. Must run before train/test
splitting to prevent data leakage (Spec §28).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Set, Tuple

from productlens.schemas import ReviewRecord
from productlens.utils import content_hash

logger = logging.getLogger("productlens.data.dedupe")


# ---------------------------------------------------------------------------
# Exact deduplication
# ---------------------------------------------------------------------------

def exact_deduplicate(reviews: List[ReviewRecord]) -> Tuple[List[ReviewRecord], List[str]]:
    """
    Remove exact text duplicates, keeping the first occurrence.

    Parameters
    ----------
    reviews : list of ReviewRecord

    Returns
    -------
    deduplicated : list of ReviewRecord
    duplicate_ids : list of str
        Review IDs that were removed as duplicates.
    """
    seen_hashes: Set[str] = set()
    deduplicated: List[ReviewRecord] = []
    duplicate_ids: List[str] = []

    for review in reviews:
        text = review.clean_text or review.text or review.raw_text
        if not text:
            deduplicated.append(review)
            continue

        h = content_hash(text.lower().strip())
        if h in seen_hashes:
            duplicate_ids.append(review.review_id)
        else:
            seen_hashes.add(h)
            deduplicated.append(review)

    logger.info(
        "Exact dedup: %d → %d reviews (%d duplicates removed)",
        len(reviews),
        len(deduplicated),
        len(duplicate_ids),
    )
    return deduplicated, duplicate_ids


# ---------------------------------------------------------------------------
# Near-duplicate detection (Jaccard on character n-grams)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-gram set from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i: i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def near_deduplicate(
    reviews: List[ReviewRecord],
    threshold: float = 0.9,
    ngram_size: int = 3,
) -> Tuple[List[ReviewRecord], List[str]]:
    """
    Remove near-duplicate reviews using character n-gram Jaccard similarity.

    This is O(n²) and suitable for moderate datasets. For very large datasets,
    MinHash/LSH should be used instead.

    Parameters
    ----------
    reviews : list of ReviewRecord
    threshold : float
        Jaccard similarity threshold above which reviews are considered
        near-duplicates.
    ngram_size : int
        Character n-gram size.

    Returns
    -------
    deduplicated : list of ReviewRecord
    duplicate_ids : list of str
    """
    if threshold >= 1.0:
        # No near-dedup at threshold 1.0
        return reviews, []

    deduplicated: List[ReviewRecord] = []
    duplicate_ids: List[str] = []
    kept_ngrams: List[Set[str]] = []

    for review in reviews:
        text = review.clean_text or review.text or review.raw_text
        if not text:
            deduplicated.append(review)
            continue

        current_ngrams = _char_ngrams(text, ngram_size)
        is_dup = False

        for existing in kept_ngrams:
            if _jaccard_similarity(current_ngrams, existing) >= threshold:
                is_dup = True
                break

        if is_dup:
            duplicate_ids.append(review.review_id)
        else:
            kept_ngrams.append(current_ngrams)
            deduplicated.append(review)

    logger.info(
        "Near dedup (threshold=%.2f): %d → %d reviews (%d near-duplicates removed)",
        threshold,
        len(reviews),
        len(deduplicated),
        len(duplicate_ids),
    )
    return deduplicated, duplicate_ids


# ---------------------------------------------------------------------------
# Combined deduplication
# ---------------------------------------------------------------------------

def deduplicate(
    reviews: List[ReviewRecord],
    near_dedupe: bool = False,
    near_threshold: float = 0.9,
) -> Tuple[List[ReviewRecord], Dict[str, List[str]]]:
    """
    Run full deduplication pipeline: exact first, then optionally near.

    Parameters
    ----------
    reviews : list of ReviewRecord
    near_dedupe : bool
        Whether to also run near-duplicate detection.
    near_threshold : float
        Jaccard threshold for near-dedup.

    Returns
    -------
    deduplicated : list of ReviewRecord
    log : dict
        ``{"exact_duplicates": [...], "near_duplicates": [...]}``.
    """
    result, exact_dups = exact_deduplicate(reviews)
    near_dups: List[str] = []

    if near_dedupe:
        result, near_dups = near_deduplicate(result, threshold=near_threshold)

    log = {
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
    }
    return result, log
