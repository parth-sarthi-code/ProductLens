"""
Tests for data loading, cleaning, deduplication, splitting, and schemas.

Tests schema discovery, cleaning preserves raw_text, duplicate removal,
stable ID determinism, missing field handling, and edge cases.
"""

from __future__ import annotations

import hashlib
from typing import List

import pytest

from productlens.config import load_config, config_hash, ProductLensConfig
from productlens.data.clean import clean_text, clean_review, clean_reviews
from productlens.data.dedupe import exact_deduplicate, near_deduplicate, deduplicate
from productlens.data.load_amazon import discover_schema, normalize_record
from productlens.data.sampling import (
    sample_by_category,
    sample_by_product,
    sample_stratified_by_rating,
    random_sample,
)
from productlens.data.split import split_reviews, verify_no_leakage
from productlens.data.synthetic import generate_synthetic_reviews
from productlens.schemas import ReviewRecord
from productlens.utils import stable_id, content_hash


# ===================================================================
# Configuration tests
# ===================================================================

class TestConfig:
    """Test configuration loading and profiles."""

    def test_default_config_loads(self):
        cfg = load_config()
        assert cfg.project.seed == 42
        assert cfg.data.max_reviews == 10_000
        assert cfg.models.aspect_extractor == "microsoft/deberta-v3-base"

    def test_smoke_profile(self):
        cfg = load_config(profile="smoke")
        assert cfg.data.max_reviews == 300
        assert cfg.training.epochs == 1
        assert cfg.training.fp16 is False

    def test_dev_profile(self):
        cfg = load_config(profile="dev")
        assert cfg.data.max_reviews == 5_000

    def test_full_profile(self):
        cfg = load_config(profile="full")
        assert cfg.data.max_reviews == 100_000

    def test_override_single(self):
        cfg = load_config(overrides=["data.max_reviews=999"])
        assert cfg.data.max_reviews == 999

    def test_override_multiple(self):
        cfg = load_config(overrides=[
            "data.max_reviews=50",
            "training.epochs=5",
            "training.fp16=false",
        ])
        assert cfg.data.max_reviews == 50
        assert cfg.training.epochs == 5
        assert cfg.training.fp16 is False

    def test_config_hash_deterministic(self):
        cfg1 = load_config()
        cfg2 = load_config()
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_config_hash_changes_with_values(self):
        cfg1 = load_config()
        cfg2 = load_config(overrides=["data.max_reviews=999"])
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_all_yaml_keys_match_dataclass(self):
        """Every key in default.yaml must map to a dataclass field."""
        import yaml
        from dataclasses import fields
        from pathlib import Path

        yaml_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if not yaml_path.exists():
            pytest.skip("default.yaml not found")

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        cfg = load_config()
        section_map = {
            "project": cfg.project,
            "data": cfg.data,
            "models": cfg.models,
            "training": cfg.training,
            "normalization": cfg.normalization,
            "evidence": cfg.evidence,
            "aggregation": cfg.aggregation,
            "pipeline": cfg.pipeline,
        }

        for section_name, section_data in raw.items():
            assert section_name in section_map, f"Unknown section: {section_name}"
            dc = section_map[section_name]
            dc_fields = {f.name for f in fields(dc.__class__)}
            for key in section_data.keys():
                assert key in dc_fields, (
                    f"YAML key '{section_name}.{key}' has no matching dataclass field"
                )


# ===================================================================
# Schema discovery tests
# ===================================================================

class TestSchemaDiscovery:
    """Test schema discovery and field normalization."""

    def test_discover_standard_fields(self):
        sample = {
            "rating": 5,
            "title": "Great",
            "text": "Review text",
            "asin": "B001",
            "verified_purchase": True,
        }
        mapping = discover_schema(sample)
        assert mapping["rating"] == "rating"
        assert mapping["text"] == "text"
        assert mapping["asin"] == "product_id"

    def test_discover_alternative_fields(self):
        sample = {
            "overall": 4.0,
            "reviewText": "Some text",
            "reviewerID": "user123",
        }
        mapping = discover_schema(sample)
        assert mapping["overall"] == "rating"
        assert mapping["reviewText"] == "text"

    def test_normalize_record(self):
        sample = {
            "rating": 5,
            "text": "Great product",
            "asin": "B001",
            "parent_asin": "B000",
            "verified_purchase": True,
            "helpful_vote": 10,
            "timestamp": "2024-01-01",
        }
        mapping = discover_schema(sample)
        record = normalize_record(sample, mapping, source="test")
        assert record.product_id == "B001"
        assert record.parent_product_id == "B000"
        assert record.text == "Great product"
        assert record.rating == 5.0
        assert record.verified_purchase is True
        assert record.helpful_vote == 10

    def test_missing_fields_safe_defaults(self):
        sample = {"text": "Just text"}
        mapping = discover_schema(sample)
        record = normalize_record(sample, mapping, source="test")
        assert record.text == "Just text"
        assert record.product_id == ""
        assert record.parent_product_id == ""
        assert record.rating == -1.0
        assert record.verified_purchase is False
        assert record.helpful_vote == 0

    def test_parent_defaults_to_product_id(self):
        sample = {"text": "Text", "asin": "B123"}
        mapping = discover_schema(sample)
        record = normalize_record(sample, mapping, source="test")
        assert record.parent_product_id == "B123"


# ===================================================================
# Stable ID tests
# ===================================================================

class TestStableIds:
    """Test deterministic ID generation."""

    def test_stable_id_deterministic(self):
        id1 = stable_id("source", "product", "text", "time")
        id2 = stable_id("source", "product", "text", "time")
        assert id1 == id2

    def test_stable_id_changes_with_input(self):
        id1 = stable_id("source", "product", "text1", "time")
        id2 = stable_id("source", "product", "text2", "time")
        assert id1 != id2

    def test_stable_id_is_hex(self):
        sid = stable_id("a", "b")
        assert len(sid) == 64
        assert all(c in "0123456789abcdef" for c in sid)

    def test_content_hash_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_different_for_different_text(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2


# ===================================================================
# Cleaning tests
# ===================================================================

class TestCleaning:
    """Test text cleaning pipeline."""

    def test_clean_unicode(self):
        result = clean_text("café résumé naïve")
        assert "café" in result

    def test_clean_html_tags(self):
        result = clean_text("<b>Bold</b> text <br/> here")
        assert "<b>" not in result
        assert "Bold" in result

    def test_clean_html_entities(self):
        result = clean_text("Tom &amp; Jerry &lt;love&gt;")
        assert "Tom & Jerry" in result

    def test_clean_urls(self):
        result = clean_text("Visit https://example.com for details. Good product.")
        assert "https" not in result
        assert "Good product" in result

    def test_clean_whitespace(self):
        result = clean_text("Too    many     spaces   here")
        assert "  " not in result

    def test_clean_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_preserves_raw_text(self):
        review = ReviewRecord(
            review_id="test",
            text="<b>Raw</b> text",
            raw_text="<b>Raw</b> text",
        )
        cleaned = clean_review(review)
        assert cleaned.raw_text == "<b>Raw</b> text"
        assert "<b>" not in cleaned.clean_text
        assert "Raw" in cleaned.clean_text

    def test_clean_reviews_filters_short(self):
        reviews = [
            ReviewRecord(review_id="1", text="OK", raw_text="OK"),
            ReviewRecord(review_id="2", text="This is a long enough review.", raw_text="This is a long enough review."),
        ]
        cleaned = clean_reviews(reviews, min_chars=10)
        assert len(cleaned) == 1
        assert cleaned[0].review_id == "2"

    def test_clean_reviews_filters_empty(self):
        reviews = [
            ReviewRecord(review_id="1", text="", raw_text=""),
            ReviewRecord(review_id="2", text="Valid review text here", raw_text="Valid review text here"),
        ]
        cleaned = clean_reviews(reviews, min_chars=5)
        assert len(cleaned) == 1

    def test_control_chars_removed(self):
        result = clean_text("Hello\x00World\x07Test")
        assert "\x00" not in result
        assert "\x07" not in result

    def test_newlines_normalized(self):
        result = clean_text("Line1\n\n\n\n\nLine2")
        assert "\n\n\n" not in result


# ===================================================================
# Synthetic data tests
# ===================================================================

class TestSyntheticData:
    """Test synthetic review generator."""

    def test_generates_reviews(self, synthetic_reviews):
        assert len(synthetic_reviews) > 200

    def test_deterministic(self):
        r1 = generate_synthetic_reviews(seed=42)
        r2 = generate_synthetic_reviews(seed=42)
        assert len(r1) == len(r2)
        # Same content (shuffled identically)
        texts1 = [r.text for r in r1]
        texts2 = [r.text for r in r2]
        assert texts1 == texts2

    def test_covers_all_categories(self, synthetic_reviews):
        categories = {r.category for r in synthetic_reviews}
        expected = {"Electronics", "Beauty", "Home", "Sports", "Books", "Automotive"}
        assert expected.issubset(categories)

    def test_has_multi_aspect_reviews(self, synthetic_reviews):
        """At least some reviews contain multiple aspects."""
        multi_aspect_indicators = [
            "but", "however", "although", "though",
        ]
        count = sum(
            1 for r in synthetic_reviews
            if any(word in r.text.lower() for word in multi_aspect_indicators)
        )
        assert count > 5

    def test_has_empty_reviews(self, synthetic_reviews):
        """Edge case: empty reviews exist for testing cleaning."""
        empty = [r for r in synthetic_reviews if not r.text.strip()]
        assert len(empty) > 0

    def test_has_html_reviews(self, synthetic_reviews):
        """Edge case: reviews with HTML tags."""
        html_reviews = [r for r in synthetic_reviews if "<" in r.text]
        assert len(html_reviews) > 0

    def test_has_unicode_reviews(self, synthetic_reviews):
        """Edge case: reviews with non-ASCII characters."""
        unicode_reviews = [
            r for r in synthetic_reviews
            if any(ord(c) > 127 for c in r.text)
        ]
        assert len(unicode_reviews) > 0

    def test_has_url_reviews(self, synthetic_reviews):
        """Edge case: reviews containing URLs."""
        url_reviews = [r for r in synthetic_reviews if "http" in r.text]
        assert len(url_reviews) > 0

    def test_has_duplicate_text(self, synthetic_reviews):
        """Duplicates exist for testing dedup."""
        texts = [r.text for r in synthetic_reviews if r.text.strip()]
        unique = set(texts)
        assert len(texts) > len(unique)

    def test_has_irrelevant_sentences(self, synthetic_reviews):
        """Reviews with delivery/service mentions."""
        irrelevant_indicators = ["delivered", "package arrived", "brother bought"]
        count = sum(
            1 for r in synthetic_reviews
            if any(ind in r.text.lower() for ind in irrelevant_indicators)
        )
        assert count > 0

    def test_all_have_review_ids(self, synthetic_reviews):
        for r in synthetic_reviews:
            assert r.review_id, "Review must have a non-empty ID"
            assert len(r.review_id) == 64, "ID must be SHA-256 hex"

    def test_all_have_source(self, synthetic_reviews):
        for r in synthetic_reviews:
            assert r.source == "synthetic"


# ===================================================================
# Deduplication tests
# ===================================================================

class TestDeduplication:
    """Test exact and near-duplicate detection."""

    def test_exact_dedup_removes_duplicates(self):
        reviews = [
            ReviewRecord(review_id="a", text="Same text", raw_text="Same text", clean_text="Same text"),
            ReviewRecord(review_id="b", text="Same text", raw_text="Same text", clean_text="Same text"),
            ReviewRecord(review_id="c", text="Different text", raw_text="Different text", clean_text="Different text"),
        ]
        deduped, dups = exact_deduplicate(reviews)
        assert len(deduped) == 2
        assert len(dups) == 1
        assert "b" in dups

    def test_exact_dedup_case_insensitive(self):
        reviews = [
            ReviewRecord(review_id="a", text="Same Text", raw_text="Same Text", clean_text="Same Text"),
            ReviewRecord(review_id="b", text="same text", raw_text="same text", clean_text="same text"),
        ]
        deduped, dups = exact_deduplicate(reviews)
        assert len(deduped) == 1

    def test_exact_dedup_empty_reviews(self):
        reviews = [
            ReviewRecord(review_id="a", text="", raw_text="", clean_text=""),
            ReviewRecord(review_id="b", text="Valid text", raw_text="Valid text", clean_text="Valid text"),
        ]
        deduped, dups = exact_deduplicate(reviews)
        assert len(deduped) == 2

    def test_near_dedup(self):
        reviews = [
            ReviewRecord(review_id="a", text="The sound quality is great", clean_text="The sound quality is great"),
            ReviewRecord(review_id="b", text="The sound quality is greatt", clean_text="The sound quality is greatt"),
            ReviewRecord(review_id="c", text="Completely different text here", clean_text="Completely different text here"),
        ]
        deduped, dups = near_deduplicate(reviews, threshold=0.8)
        assert len(deduped) == 2  # "a" and "c" kept, "b" removed
        assert "b" in dups

    def test_combined_dedup_with_synthetic(self, synthetic_reviews):
        cleaned = clean_reviews(synthetic_reviews, min_chars=10)
        deduped, log = deduplicate(cleaned)
        assert len(deduped) <= len(cleaned)
        assert len(log["exact_duplicates"]) > 0  # Synthetic data has duplicates


# ===================================================================
# Splitting tests
# ===================================================================

class TestSplitting:
    """Test train/val/test splitting."""

    def test_split_sizes(self, cleaned_reviews):
        splits = split_reviews(cleaned_reviews, seed=42)
        total = sum(len(s) for s in splits.values())
        assert total == len(cleaned_reviews)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_split_deterministic(self, cleaned_reviews):
        s1 = split_reviews(cleaned_reviews, seed=42)
        s2 = split_reviews(cleaned_reviews, seed=42)
        assert len(s1["train"]) == len(s2["train"])
        ids1 = {r.review_id for r in s1["train"]}
        ids2 = {r.review_id for r in s2["train"]}
        assert ids1 == ids2

    def test_split_different_seeds(self, cleaned_reviews):
        s1 = split_reviews(cleaned_reviews, seed=42)
        s2 = split_reviews(cleaned_reviews, seed=99)
        ids1 = {r.review_id for r in s1["train"]}
        ids2 = {r.review_id for r in s2["train"]}
        # Could overlap but shouldn't be identical
        assert ids1 != ids2

    def test_no_review_leakage_product_split(self, cleaned_reviews):
        splits = split_reviews(cleaned_reviews, seed=42, by_product=True)
        issues = verify_no_leakage(splits, check_product=True, check_review=True, check_text=True)
        assert len(issues["review_leaks"]) == 0
        assert len(issues["product_leaks"]) == 0

    def test_no_review_leakage_random_split(self, cleaned_reviews):
        splits = split_reviews(cleaned_reviews, seed=42, by_product=False)
        issues = verify_no_leakage(splits, check_review=True, check_text=True)
        assert len(issues["review_leaks"]) == 0


# ===================================================================
# Sampling tests
# ===================================================================

class TestSampling:
    """Test sampling utilities."""

    def test_sample_by_category(self, cleaned_reviews):
        sampled = sample_by_category(
            cleaned_reviews, categories=["Electronics"], max_per_category=10
        )
        assert len(sampled) <= 10
        assert all(r.category == "Electronics" for r in sampled)

    def test_sample_by_product(self, cleaned_reviews):
        sampled = sample_by_product(
            cleaned_reviews, max_products=2, max_reviews_per_product=5
        )
        product_ids = {r.product_id for r in sampled}
        assert len(product_ids) <= 2

    def test_random_sample(self, cleaned_reviews):
        sampled = random_sample(cleaned_reviews, n=10)
        assert len(sampled) == min(10, len(cleaned_reviews))

    def test_stratified_sample(self, cleaned_reviews):
        sampled = sample_stratified_by_rating(cleaned_reviews, max_per_rating=5)
        ratings = {round(r.rating) for r in sampled if r.rating > 0}
        # Should have multiple rating buckets
        assert len(ratings) >= 2

    def test_sampling_deterministic(self, cleaned_reviews):
        s1 = random_sample(cleaned_reviews, n=10, seed=42)
        s2 = random_sample(cleaned_reviews, n=10, seed=42)
        assert [r.review_id for r in s1] == [r.review_id for r in s2]
