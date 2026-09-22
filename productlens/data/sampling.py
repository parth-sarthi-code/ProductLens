"""
Data sampling utilities.

Provides category-based, product-based, and stratified sampling with
reproducible seed control.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from productlens.schemas import ReviewRecord

logger = logging.getLogger("productlens.data.sampling")


def sample_by_category(
    reviews: List[ReviewRecord],
    categories: List[str],
    max_per_category: int = 1000,
    seed: int = 42,
) -> List[ReviewRecord]:
    """
    Sample reviews from specified categories.

    Parameters
    ----------
    reviews : list of ReviewRecord
    categories : list of str
        Category names to include. If empty, all categories are included.
    max_per_category : int
        Maximum reviews per category.
    seed : int

    Returns
    -------
    list of ReviewRecord
    """
    rng = random.Random(seed)

    # Group by category
    by_cat: Dict[str, List[ReviewRecord]] = defaultdict(list)
    for r in reviews:
        by_cat[r.category].append(r)

    # Filter to requested categories
    if categories:
        target_cats = set(categories)
    else:
        target_cats = set(by_cat.keys())

    result: List[ReviewRecord] = []
    for cat in sorted(target_cats):
        pool = by_cat.get(cat, [])
        rng.shuffle(pool)
        sampled = pool[:max_per_category]
        result.extend(sampled)
        logger.info("Category '%s': sampled %d / %d", cat, len(sampled), len(pool))

    logger.info("Total sampled: %d reviews from %d categories", len(result), len(target_cats))
    return result


def sample_by_product(
    reviews: List[ReviewRecord],
    product_ids: Optional[List[str]] = None,
    max_products: int = 100,
    max_reviews_per_product: int = 50,
    seed: int = 42,
) -> List[ReviewRecord]:
    """
    Sample reviews by product.

    Parameters
    ----------
    reviews : list of ReviewRecord
    product_ids : list of str, optional
        Specific product IDs to include. If None, randomly sample products.
    max_products : int
        Maximum number of products to sample.
    max_reviews_per_product : int
        Maximum reviews per product.
    seed : int

    Returns
    -------
    list of ReviewRecord
    """
    rng = random.Random(seed)

    # Group by product
    by_product: Dict[str, List[ReviewRecord]] = defaultdict(list)
    for r in reviews:
        by_product[r.product_id].append(r)

    # Select products
    if product_ids:
        selected = [pid for pid in product_ids if pid in by_product]
    else:
        all_pids = sorted(by_product.keys())
        rng.shuffle(all_pids)
        selected = all_pids[:max_products]

    result: List[ReviewRecord] = []
    for pid in selected:
        pool = by_product[pid]
        rng.shuffle(pool)
        sampled = pool[:max_reviews_per_product]
        result.extend(sampled)

    logger.info(
        "Product sampling: %d products, %d reviews",
        len(selected),
        len(result),
    )
    return result


def sample_stratified_by_rating(
    reviews: List[ReviewRecord],
    max_per_rating: int = 200,
    seed: int = 42,
) -> List[ReviewRecord]:
    """
    Stratified sampling by star rating.

    Parameters
    ----------
    reviews : list of ReviewRecord
    max_per_rating : int
        Maximum reviews per rating bucket.
    seed : int

    Returns
    -------
    list of ReviewRecord
    """
    rng = random.Random(seed)

    by_rating: Dict[int, List[ReviewRecord]] = defaultdict(list)
    for r in reviews:
        bucket = max(1, min(5, round(r.rating))) if r.rating > 0 else 0
        by_rating[bucket].append(r)

    result: List[ReviewRecord] = []
    for rating in sorted(by_rating.keys()):
        pool = by_rating[rating]
        rng.shuffle(pool)
        sampled = pool[:max_per_rating]
        result.extend(sampled)
        logger.info("Rating %d: sampled %d / %d", rating, len(sampled), len(pool))

    logger.info("Stratified total: %d reviews", len(result))
    return result


def random_sample(
    reviews: List[ReviewRecord],
    n: int = 1000,
    seed: int = 42,
) -> List[ReviewRecord]:
    """
    Simple random sample of reviews.

    Parameters
    ----------
    reviews : list of ReviewRecord
    n : int
        Number of reviews to sample.
    seed : int

    Returns
    -------
    list of ReviewRecord
    """
    rng = random.Random(seed)
    pool = list(reviews)
    rng.shuffle(pool)
    result = pool[:n]
    logger.info("Random sample: %d / %d reviews", len(result), len(reviews))
    return result
