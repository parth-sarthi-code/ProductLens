"""
Unit tests for aspect normalization, clustering, and alias handling.

Spec §42: Test deterministic normalization and clustering interfaces.
"""

import tempfile
import numpy as np
import pytest
import yaml

from productlens.aspects.aliases import AliasResolver, DEFAULT_ALIASES
from productlens.aspects.candidates import CandidateAspect
from productlens.aspects.clustering import AspectClusterer
from productlens.aspects.normalize import AspectEmbedder, HashingEmbedder, EmbeddingCache


# ---------------------------------------------------------------------------
# Hashing Embedder & Cache Tests
# ---------------------------------------------------------------------------

def test_hashing_embedder_determinism_and_shape():
    """Verify HashingEmbedder produces deterministic, normalized 384-d vectors."""
    embedder = HashingEmbedder(dimension=384)
    texts = ["battery life", "display screen", "customer service"]

    vecs1 = embedder.encode(texts)
    vecs2 = embedder.encode(texts)

    assert vecs1.shape == (3, 384)
    np.testing.assert_allclose(vecs1, vecs2, rtol=1e-5)

    # Check unit normalization
    norms = np.linalg.norm(vecs1, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1e-4)


def test_embedding_cache():
    """Test disk embedding cache get and put."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        texts = ["screen", "display"]
        model = "test-model"

        # Initially absent
        assert cache.get(texts, model) is None

        # Store
        embs = np.random.randn(2, 384).astype(np.float32)
        cache.put(texts, embs, model, dimension=384)

        # Retrieve
        loaded = cache.get(texts, model)
        assert loaded is not None
        assert loaded.shape == (2, 384)
        np.testing.assert_allclose(embs, loaded, rtol=1e-5)


# ---------------------------------------------------------------------------
# Alias Resolution & Merge Safety Tests (Spec §15)
# ---------------------------------------------------------------------------

def test_alias_resolution():
    """Test canonical alias mapping for common synonyms."""
    resolver = AliasResolver()

    assert resolver.resolve("display") == "screen"
    assert resolver.resolve("panel") == "screen"
    assert resolver.resolve("lcd") == "screen"
    assert resolver.resolve("battery life") == "battery"
    assert resolver.resolve("battery backup") == "battery"
    assert resolver.resolve("runtime") == "battery"
    assert resolver.resolve("customer support") == "customer service"


def test_alias_merge_safety_disallowed_pairs():
    """Verify that disallowed pairs like 'screen' and 'screen protector' are NEVER merged."""
    resolver = AliasResolver()

    # Screen vs Screen protector
    assert not resolver.can_merge("screen", "screen protector")
    assert not resolver.can_merge("screen protector", "screen")

    # Phone vs Phone case
    assert not resolver.can_merge("phone", "phone case")
    assert not resolver.can_merge("phone case", "phone")

    # Generic word only overlap
    assert not resolver.can_merge("sound quality", "build quality")
    assert not resolver.can_merge("build quality", "sound quality")
    assert not resolver.can_merge("battery performance", "gaming performance")


def test_alias_manual_overrides():
    """Test loading and applying manual YAML alias overrides."""
    resolver = AliasResolver()

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump({"custom_panel": "screen", "my_gizmo": "gizmo_canonical"}, f)
        yaml_path = f.name

    resolver.load_overrides(yaml_path)
    assert resolver.resolve("custom_panel") == "screen"
    assert resolver.resolve("my_gizmo") == "gizmo_canonical"


# ---------------------------------------------------------------------------
# Clustering & Outlier Preservation Tests (Spec §14)
# ---------------------------------------------------------------------------

def test_aspect_clustering_and_canonical_names():
    """Verify clustering groups mentions and derives canonical aspect from members."""
    candidates = [
        CandidateAspect(surface="display", start_char=0, end_char=7, category="Electronics", product_id="P1"),
        CandidateAspect(surface="screen", start_char=0, end_char=6, category="Electronics", product_id="P1"),
        CandidateAspect(surface="panel", start_char=0, end_char=5, category="Electronics", product_id="P2"),
        CandidateAspect(surface="battery", start_char=0, end_char=7, category="Electronics", product_id="P1"),
        CandidateAspect(surface="battery life", start_char=0, end_char=12, category="Electronics", product_id="P2"),
    ]

    clusterer = AspectClusterer(
        mode="category_aware",
        min_cluster_size=2,
        min_samples=1,
    )
    result = clusterer.cluster_candidates(candidates)

    assert result.total_candidates == 5
    assert len(result.cluster_records) > 0

    # Check surface mappings
    assert "screen" in result.surface_to_canonical
    assert "battery" in result.surface_to_canonical

    # "display" and "screen" should resolve to canonical "screen"
    assert result.surface_to_canonical["display"] == "screen"
    assert result.surface_to_canonical["screen"] == "screen"


def test_clustering_preserves_outliers():
    """Verify that outliers (HDBSCAN label -1) are NEVER dropped."""
    # A single unique candidate that won't cluster with anything
    candidates = [
        CandidateAspect(surface="unusual quirky feature xyz", start_char=0, end_char=26, category="Home", product_id="P3"),
    ]

    clusterer = AspectClusterer(
        mode="category_aware",
        min_cluster_size=3,
        min_samples=2,
    )
    result = clusterer.cluster_candidates(candidates)

    assert result.total_candidates == 1
    # Outlier must be represented in cluster records
    assert len(result.cluster_records) == 1
    rec = result.cluster_records[0]
    assert rec.member_count == 1
    assert "unusual quirky feature xyz" in result.surface_to_canonical


def test_category_aware_vs_global_clustering():
    """Verify category_aware partitions by category while global processes all."""
    candidates = [
        CandidateAspect(surface="screen", start_char=0, end_char=6, category="Electronics", product_id="P1"),
        CandidateAspect(surface="display", start_char=0, end_char=7, category="Electronics", product_id="P1"),
        CandidateAspect(surface="zipper", start_char=0, end_char=6, category="Sports", product_id="P2"),
        CandidateAspect(surface="fabric", start_char=0, end_char=6, category="Sports", product_id="P2"),
    ]

    cat_clusterer = AspectClusterer(mode="category_aware", min_cluster_size=2, min_samples=1)
    cat_result = cat_clusterer.cluster_candidates(candidates)

    global_clusterer = AspectClusterer(mode="global", min_cluster_size=2, min_samples=1)
    global_result = global_clusterer.cluster_candidates(candidates)

    assert cat_result.total_candidates == 4
    assert global_result.total_candidates == 4

    # Verify cluster_ids in category-aware include the category prefix
    cat_ids = [rec.cluster_id for rec in cat_result.cluster_records]
    assert any("Electronics" in cid for cid in cat_ids)
    assert any("Sports" in cid for cid in cat_ids)


# ---------------------------------------------------------------------------
# Determinism & Multi-Domain Tests
# ---------------------------------------------------------------------------

def test_cluster_determinism():
    """Verify that clustering is 100% deterministic across repeated runs."""
    candidates = [
        CandidateAspect(surface="screen", start_char=0, end_char=6, category="Electronics"),
        CandidateAspect(surface="display", start_char=0, end_char=7, category="Electronics"),
        CandidateAspect(surface="panel", start_char=0, end_char=5, category="Electronics"),
        CandidateAspect(surface="battery", start_char=0, end_char=7, category="Electronics"),
    ]

    clusterer1 = AspectClusterer(mode="category_aware", min_cluster_size=2, min_samples=1)
    res1 = clusterer1.cluster_candidates(candidates)

    clusterer2 = AspectClusterer(mode="category_aware", min_cluster_size=2, min_samples=1)
    res2 = clusterer2.cluster_candidates(candidates)

    assert len(res1.cluster_records) == len(res2.cluster_records)
    assert res1.surface_to_canonical == res2.surface_to_canonical
    assert res1.surface_to_cluster_id == res2.surface_to_cluster_id


def test_normalization_consistency():
    """Verify consistent canonical resolution across synonyms."""
    resolver = AliasResolver()
    terms = ["display", "panel", "screen", "lcd"]
    canonical_set = {resolver.resolve(t) for t in terms}
    # All display synonyms must resolve to the single canonical term 'screen'
    assert len(canonical_set) == 1
    assert "screen" in canonical_set


def test_multi_category_domain_generalization():
    """Verify system handles non-laptop categories (Beauty, Sports, Home, Auto, Books)."""
    multi_domain_candidates = [
        CandidateAspect(surface="gentle formula", start_char=0, end_char=14, category="Beauty"),
        CandidateAspect(surface="sensitive skin", start_char=0, end_char=14, category="Beauty"),
        CandidateAspect(surface="brake pads", start_char=0, end_char=10, category="Automotive"),
        CandidateAspect(surface="wiper blades", start_char=0, end_char=12, category="Automotive"),
        CandidateAspect(surface="cushioning", start_char=0, end_char=10, category="Sports"),
        CandidateAspect(surface="rubber grip", start_char=0, end_char=11, category="Sports"),
    ]

    clusterer = AspectClusterer(mode="category_aware", min_cluster_size=2, min_samples=1)
    result = clusterer.cluster_candidates(multi_domain_candidates)

    assert result.total_candidates == 6
    # All surface mentions must be represented in canonical mappings
    for c in multi_domain_candidates:
        assert c.surface in result.surface_to_canonical
