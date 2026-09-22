"""
Train / validation / test splitting with leakage prevention.

Supports product-aware splitting, category-aware splitting, and
cross-product / cross-category test sets. Deduplication must run before
this module (Spec §28).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from productlens.schemas import ReviewRecord

logger = logging.getLogger("productlens.data.split")


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------

def split_reviews(
    reviews: List[ReviewRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    by_product: bool = True,
) -> Dict[str, List[ReviewRecord]]:
    """
    Split reviews into train / val / test sets.

    When ``by_product=True`` (default), splitting is done at the product
    level to prevent product leakage — all reviews for a product go into
    the same split.

    Parameters
    ----------
    reviews : list of ReviewRecord
    train_ratio : float
    val_ratio : float
    test_ratio : float
    seed : int
    by_product : bool
        If True, split by product ID to prevent product leakage.

    Returns
    -------
    dict
        Keys: ``train``, ``val``, ``test``.
    """
    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    rng = random.Random(seed)

    if by_product:
        return _split_by_product(reviews, train_ratio, val_ratio, rng)
    else:
        return _split_random(reviews, train_ratio, val_ratio, rng)


def _split_random(
    reviews: List[ReviewRecord],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> Dict[str, List[ReviewRecord]]:
    """Random splitting (no leakage protection)."""
    indices = list(range(len(reviews)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train: n_train + n_val])

    splits: Dict[str, List[ReviewRecord]] = {"train": [], "val": [], "test": []}
    for i, review in enumerate(reviews):
        if i in train_idx:
            splits["train"].append(review)
        elif i in val_idx:
            splits["val"].append(review)
        else:
            splits["test"].append(review)

    _log_splits(splits)
    return splits


def _split_by_product(
    reviews: List[ReviewRecord],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> Dict[str, List[ReviewRecord]]:
    """
    Split by product ID to prevent product leakage.

    All reviews for a given product go into the same split.
    """
    # Group by product
    product_reviews: Dict[str, List[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        pid = review.product_id or "unknown"
        product_reviews[pid].append(review)

    product_ids = sorted(product_reviews.keys())
    rng.shuffle(product_ids)

    n = len(product_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pids = set(product_ids[:n_train])
    val_pids = set(product_ids[n_train: n_train + n_val])

    splits: Dict[str, List[ReviewRecord]] = {"train": [], "val": [], "test": []}
    for pid in product_ids:
        if pid in train_pids:
            splits["train"].extend(product_reviews[pid])
        elif pid in val_pids:
            splits["val"].extend(product_reviews[pid])
        else:
            splits["test"].extend(product_reviews[pid])

    _log_splits(splits)
    return splits


def _log_splits(splits: Dict[str, List[ReviewRecord]]) -> None:
    """Log split sizes."""
    for name, data in splits.items():
        logger.info("Split '%s': %d reviews", name, len(data))


# ---------------------------------------------------------------------------
# Cross-product and cross-category test sets
# ---------------------------------------------------------------------------

def create_cross_product_test(
    reviews: List[ReviewRecord],
    known_product_ids: Set[str],
    max_reviews: int = 500,
    seed: int = 42,
) -> List[ReviewRecord]:
    """
    Create a test set of reviews from products NOT in the known set.

    Parameters
    ----------
    reviews : list of ReviewRecord
    known_product_ids : set of str
        Product IDs already in train/val.
    max_reviews : int
    seed : int

    Returns
    -------
    list of ReviewRecord
    """
    candidates = [r for r in reviews if r.product_id not in known_product_ids]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    result = candidates[:max_reviews]
    logger.info("Cross-product test set: %d reviews", len(result))
    return result


def create_cross_category_test(
    reviews: List[ReviewRecord],
    known_categories: Set[str],
    max_reviews: int = 500,
    seed: int = 42,
) -> List[ReviewRecord]:
    """
    Create a test set of reviews from categories NOT in the known set.

    Parameters
    ----------
    reviews : list of ReviewRecord
    known_categories : set of str
        Categories already in train/val.
    max_reviews : int
    seed : int

    Returns
    -------
    list of ReviewRecord
    """
    candidates = [r for r in reviews if r.category not in known_categories]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    result = candidates[:max_reviews]
    logger.info("Cross-category test set: %d reviews", len(result))
    return result


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_no_leakage(
    splits: Dict[str, List[ReviewRecord]],
    check_product: bool = True,
    check_review: bool = True,
    check_text: bool = True,
) -> Dict[str, List[str]]:
    """
    Verify no data leakage across splits.

    Parameters
    ----------
    splits : dict
        Keys are split names, values are lists of ReviewRecord.
    check_product : bool
        Check for product ID overlap.
    check_review : bool
        Check for review ID overlap.
    check_text : bool
        Check for identical text overlap.

    Returns
    -------
    dict
        Keys: ``review_leaks``, ``product_leaks``, ``text_leaks``.
        Each is a list of overlapping identifiers.
    """
    issues: Dict[str, List[str]] = {
        "review_leaks": [],
        "product_leaks": [],
        "text_leaks": [],
    }

    split_names = list(splits.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name_a = split_names[i]
            name_b = split_names[j]
            data_a = splits[name_a]
            data_b = splits[name_b]

            if check_review:
                ids_a = {r.review_id for r in data_a}
                ids_b = {r.review_id for r in data_b}
                overlap = ids_a & ids_b
                for rid in overlap:
                    issues["review_leaks"].append(
                        f"{name_a}↔{name_b}: {rid}"
                    )

            if check_product:
                pids_a = {r.product_id for r in data_a}
                pids_b = {r.product_id for r in data_b}
                overlap = pids_a & pids_b
                for pid in overlap:
                    issues["product_leaks"].append(
                        f"{name_a}↔{name_b}: {pid}"
                    )

            if check_text:
                texts_a = {(r.clean_text or r.text).lower().strip() for r in data_a}
                texts_b = {(r.clean_text or r.text).lower().strip() for r in data_b}
                overlap = texts_a & texts_b - {""}
                for t in list(overlap)[:20]:  # Cap reporting
                    issues["text_leaks"].append(
                        f"{name_a}↔{name_b}: {t[:80]}..."
                    )

    # Log results
    for key, items in issues.items():
        if items:
            logger.warning("Leakage detected — %s: %d issues", key, len(items))
        else:
            logger.info("No %s detected ✓", key)

    return issues
