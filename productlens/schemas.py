"""
ProductLens data schemas.

Defines the canonical data structures used throughout the pipeline.
Internal schemas use dataclasses for speed; API schemas will use Pydantic
(added in Stage 5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Review records
# ---------------------------------------------------------------------------

@dataclass
class ReviewRecord:
    """
    Canonical review record produced by ingestion / cleaning.

    Attributes
    ----------
    review_id : str
        Stable SHA-256 ID derived from (source, product_id, text, timestamp).
    product_id : str
        Product identifier (ASIN or equivalent).
    parent_product_id : str
        Parent product identifier (parent_asin), may equal product_id.
    category : str
        Product category.
    title : str
        Review title.
    text : str
        Current working text (initially raw, then clean).
    raw_text : str
        Original unmodified review text — never overwritten.
    clean_text : str
        Text after cleaning pipeline — empty until cleaning runs.
    rating : float
        Star rating (1–5). Use -1 for missing.
    verified_purchase : bool
        Whether the purchase was verified.
    helpful_vote : int
        Count of helpful votes.
    timestamp : str
        ISO-8601 timestamp string. Empty string if unavailable.
    source : str
        Data source identifier (e.g. dataset name, file path).
    """

    review_id: str = ""
    product_id: str = ""
    parent_product_id: str = ""
    category: str = ""
    title: str = ""
    text: str = ""
    raw_text: str = ""
    clean_text: str = ""
    rating: float = -1.0
    verified_purchase: bool = False
    helpful_vote: int = 0
    timestamp: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRecord":
        """Construct from dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Sentence records
# ---------------------------------------------------------------------------

@dataclass
class SentenceRecord:
    """
    A single sentence extracted from a review.

    Offsets refer to the review's ``clean_text`` field.

    Attributes
    ----------
    sentence_id : str
        Stable ID derived from (review_id, start_char, end_char).
    review_id : str
        Parent review ID.
    text : str
        Sentence text.
    start_char : int
        Start character offset in clean_text.
    end_char : int
        End character offset in clean_text (exclusive).
    """

    sentence_id: str = ""
    review_id: str = ""
    text: str = ""
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Product records
# ---------------------------------------------------------------------------

@dataclass
class ProductRecord:
    """
    Product-level record.

    Attributes
    ----------
    product_id : str
        Product identifier (ASIN).
    parent_product_id : str
        Parent product identifier.
    asin : str
        Amazon Standard Identification Number.
    category : str
        Product category.
    title : str
        Product title.
    review_count : int
        Total number of reviews loaded.
    average_rating : float
        Mean star rating across reviews.
    """

    product_id: str = ""
    parent_product_id: str = ""
    asin: str = ""
    category: str = ""
    title: str = ""
    review_count: int = 0
    average_rating: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Product-aspect aggregation records
# ---------------------------------------------------------------------------

@dataclass
class ProductAspectRecord:
    """
    Aggregated aspect-level statistics for a product.

    Attributes
    ----------
    product_id : str
    aspect_id : str
        Cluster / normalized aspect identifier.
    canonical_aspect : str
        Human-readable canonical aspect name.
    mention_count : int
    unique_review_count : int
    positive_ratio : float
    neutral_ratio : float
    negative_ratio : float
    confidence : float
        Mean model confidence.
    evidence_quality : float
        Mean evidence quality score.
    contradiction_ratio : float
        Ratio of disagreement among mentions.
    evidence_ids : list of str
        IDs of selected evidence records.
    """

    product_id: str = ""
    aspect_id: str = ""
    canonical_aspect: str = ""
    mention_count: int = 0
    unique_review_count: int = 0
    positive_ratio: float = 0.0
    neutral_ratio: float = 0.0
    negative_ratio: float = 0.0
    confidence: float = 0.0
    evidence_quality: float = 0.0
    contradiction_ratio: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductAspectRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Aspect-sentiment prediction record (§18)
# ---------------------------------------------------------------------------

@dataclass
class AspectSentimentRecord:
    """
    Full prediction record for a single aspect–sentiment extraction.

    Every field required by Spec §18.
    """

    review_id: str = ""
    sentence_id: str = ""
    product_id: str = ""
    category: str = ""
    aspect_surface: str = ""
    aspect_normalized: str = ""
    aspect_cluster_id: str = ""
    sentiment: str = ""  # positive | neutral | negative
    sentiment_probabilities: Dict[str, float] = field(
        default_factory=lambda: {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    )
    sentiment_confidence: float = 0.0
    extraction_confidence: float = 0.0
    evidence_text: str = ""
    span_start: int = 0
    span_end: int = 0
    model_version: str = ""
    aspect_type: str = "product"  # product | service | delivery | seller | packaging | unknown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AspectSentimentRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Evidence record (§21)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """
    Verified evidence record linking a prediction to its source review.

    Attributes
    ----------
    review_id : str
    sentence_id : str
    source_text : str
        Full source text of the review or sentence.
    span_start : int
    span_end : int
    evidence_text : str
        Exact extracted span.
    source_text_hash : str
        SHA-256 of source_text for integrity verification.
    relevance : str
        ``relevant``, ``irrelevant``, or ``uncertain``.
    aspect_confidence : float
    sentiment_confidence : float
    helpful_vote : int
    verified_purchase : bool
    specificity : float
        How specific / informative the evidence is (0–1).
    """

    review_id: str = ""
    sentence_id: str = ""
    source_text: str = ""
    span_start: int = 0
    span_end: int = 0
    evidence_text: str = ""
    source_text_hash: str = ""
    relevance: str = "relevant"
    aspect_confidence: float = 0.0
    sentiment_confidence: float = 0.0
    helpful_vote: int = 0
    verified_purchase: bool = False
    specificity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Cluster / normalization records (§14)
# ---------------------------------------------------------------------------

@dataclass
class AspectClusterRecord:
    """
    Metadata for a single normalized aspect cluster.

    Attributes
    ----------
    cluster_id : str
    canonical_aspect : str
        Derived from actual cluster members.
    cluster_confidence : float
    representative_terms : list of str
    member_count : int
    """

    cluster_id: str = ""
    canonical_aspect: str = ""
    cluster_confidence: float = 0.0
    representative_terms: List[str] = field(default_factory=list)
    member_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AspectClusterRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def records_to_dicts(records: List[Any]) -> List[Dict[str, Any]]:
    """Convert a list of dataclass records to list of dicts."""
    return [r.to_dict() for r in records]


def records_to_json(records: List[Any], path: str) -> None:
    """Write records as a JSON array to a file."""
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(records_to_dicts(records), fh, indent=2, default=str)


def records_from_json(path: str, cls: type) -> List[Any]:
    """Read records from a JSON array file."""
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [cls.from_dict(d) for d in data]
