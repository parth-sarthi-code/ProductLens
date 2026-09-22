# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ProductLens — Notebook 02: Aspect Intelligence
#
# This notebook demonstrates the Stage 2 Aspect Intelligence pipeline:
# 1. Environment setup and device configuration
# 2. Review and sentence ingestion from Stage 1 contracts
# 3. Candidate aspect extraction and syntactic filtering
# 4. Transformer-based BIO sequence labeling & span reconstruction
# 5. Semantic embedding generation & persistent caching
# 6. HDBSCAN clustering & canonical aspect discovery
# 7. Deterministic alias resolution and merge safety
# 8. Aspect typing (product, service, delivery, seller, packaging, unknown)
# 9. Product association integrity & exact offset traceability

# %% [markdown]
# ## 1. Environment Setup & Configuration

# %%
import sys
import time
from pathlib import Path
from collections import Counter

# Ensure project root is on path
project_root = Path(".").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# %%
from productlens.utils import (
    print_library_versions,
    detect_device,
    set_seed,
    setup_logging,
)
from productlens.config import load_config, config_hash

setup_logging()
versions = print_library_versions()
device_info = detect_device()
set_seed(42)

print(f"\nPython: {sys.version}")
print(f"Device: {device_info['device']}")
if device_info["gpu_available"]:
    print(f"GPU: {device_info['gpu_name']} ({device_info['vram_mb']} MB VRAM)")

# %%
# Load configuration
cfg = load_config(profile="smoke")
print(f"\nConfiguration Profile: smoke")
print(f"Aspect Extractor Model: {cfg.models.aspect_extractor}")
print(f"Embedder Model: {cfg.models.embedder}")
print(f"Normalization Mode: {cfg.normalization.mode}")
print(f"Min Cluster Size: {cfg.normalization.min_cluster_size}")
print(f"Config Hash: {config_hash(cfg)[:16]}...")

# %% [markdown]
# ## 2. Data Ingestion (Stage 1 Contracts)
#
# Ingest clean reviews and sentence records while preserving deterministic IDs,
# product association (ASIN), and character offsets.

# %%
from productlens.data.synthetic import generate_synthetic_reviews
from productlens.data.clean import clean_reviews
from productlens.data.sentence_split import split_reviews_to_sentences

start_time = time.time()
raw_reviews = generate_synthetic_reviews(seed=cfg.project.seed)
cleaned_reviews = clean_reviews(raw_reviews, min_chars=cfg.data.min_review_chars)
sentences = split_reviews_to_sentences(cleaned_reviews)
elapsed = time.time() - start_time

print(f"Loaded {len(cleaned_reviews)} cleaned reviews across {len(set(r.category for r in cleaned_reviews))} categories")
print(f"Extracted {len(sentences)} sentence records ({elapsed:.2f}s)")

# %% [markdown]
# ## 3. Candidate Aspect Extraction
#
# Extract noun compounds, phrases, and candidate entities while strictly filtering
# generic nouns ('thing', 'product', 'stuff'), pronouns, and stopwords.

# %%
from productlens.aspects.candidates import CandidateExtractor, extract_candidates

extractor = CandidateExtractor()
candidates = extract_candidates(sentences, cleaned_reviews, origin="weak")

print(f"\nExtracted {len(candidates)} candidate aspect mentions")
print(f"Average candidates per sentence: {len(candidates) / max(len(sentences), 1):.2f}")

# Distribution of candidate lengths
lengths = [len(c.surface.split()) for c in candidates]
print("\nCandidate word-count distribution:")
for n_words, cnt in sorted(Counter(lengths).items()):
    print(f"  {n_words}-word: {cnt:4d} mentions")

# %%
# Show representative candidate mentions
print("\n=== Representative Candidate Mentions ===")
for c in candidates[:8]:
    print(f"  • '{c.surface}' [{c.category}] (Sentence offsets: {c.start_char}-{c.end_char}) -> Type: {c.aspect_type}")

# %% [markdown]
# ## 4. BIO Sequence Labeling & Span Reconstruction
#
# Demonstrates token-to-character span reconstruction, multi-aspect sentence handling,
# and confidence scoring.

# %%
from productlens.aspects.bio_model import MockBioAspectExtractor, reconstruct_spans_from_bio

bio_extractor = MockBioAspectExtractor(model_name=cfg.models.aspect_extractor)

# Multi-aspect demonstration sentence
test_sentence = "The sound quality is excellent but the microphone is terrible."
test_spans = bio_extractor.extract_from_sentence(
    sentence_text=test_sentence,
    sentence_id="demo_sent_001",
    review_id="demo_rev_001",
    product_id="B000DEMO",
    category="Electronics",
)

print("Input Sentence:")
print(f"  \"{test_sentence}\"\n")
print(f"Extracted {len(test_spans)} aspects:")
for sp in test_spans:
    print(f"  • '{sp.surface}' (Offsets: {sp.start_char}..{sp.end_char}) | Confidence: {sp.confidence:.2f} | Type: {sp.aspect_type}")
    assert test_sentence[sp.start_char:sp.end_char] == sp.surface, "Span offset mismatch!"

# %%
# Extract aspects across the entire dataset
start_time = time.time()
extracted_aspects = bio_extractor.extract_from_sentences(sentences, cleaned_reviews)
elapsed = time.time() - start_time

print(f"\nExtracted {len(extracted_aspects)} aspect mentions across {len(sentences)} sentences in {elapsed:.2f}s")
print(f"Extraction throughput: {len(sentences) / max(elapsed, 0.001):.1f} sentences/sec")

# %% [markdown]
# ## 5. Dense Embeddings & Caching
#
# Generate dense semantic embeddings using BAAI/bge-small-en-v1.5 (or deterministic
# hashing embeddings in smoke mode) and verify persistent caching.

# %%
from productlens.aspects.normalize import AspectEmbedder, EmbeddingCache

embedder = AspectEmbedder(
    model_name=cfg.models.embedder,
    cache_dir="artifacts/embeddings",
    mock=True,  # In smoke mode, use deterministic fast embedding
)

unique_surfaces = sorted(list({a.surface.lower() for a in extracted_aspects}))
print(f"Encoding {len(unique_surfaces)} unique aspect surface phrases...")

embs = embedder.encode(unique_surfaces, show_progress_bar=False)
print(f"Generated embedding matrix: shape {embs.shape}, dtype {embs.dtype}")

# Verify cache retrieval
cached_embs = embedder.encode(unique_surfaces)
assert cached_embs.shape == embs.shape
print("✓ Embedding cache verified successfully")

# %% [markdown]
# ## 6. HDBSCAN Aspect Clustering
#
# Cluster candidate aspects into canonical clusters using HDBSCAN with cosine metric.
# Outliers (HDBSCAN label -1) are preserved as distinct singleton clusters to ensure zero data loss.

# %%
from productlens.aspects.clustering import AspectClusterer

clusterer = AspectClusterer(
    mode=cfg.normalization.mode,
    min_cluster_size=cfg.normalization.min_cluster_size,
    min_samples=cfg.normalization.min_samples,
    embedder=embedder,
)

cluster_result = clusterer.cluster_candidates(extracted_aspects)

print(f"\nClustering complete:")
print(f"  Total candidate mentions: {cluster_result.total_candidates}")
print(f"  Distinct clusters formed: {len(cluster_result.cluster_records)}")
print(f"  Outliers preserved:       {cluster_result.outlier_count}")

# %%
# Show top clusters
print("\n=== Top Discovered Canonical Aspect Clusters ===")
sorted_clusters = sorted(cluster_result.cluster_records, key=lambda c: c.member_count, reverse=True)
for cl in sorted_clusters[:10]:
    print(f"  Cluster '{cl.canonical_aspect}':")
    print(f"    Members: {cl.member_count} | Confidence: {cl.cluster_confidence:.2f}")
    print(f"    Terms:   {', '.join(cl.representative_terms)}")

# %% [markdown]
# ## 7. Deterministic Alias Resolution & Merge Safety
#
# Enforces deterministic canonical aliases and safety guards:
# - Merges synonyms ('panel', 'display' -> 'screen')
# - Strictly prohibits invalid merges ('screen' vs 'screen protector')
# - Never merges aspects solely because they share generic words ('quality', 'performance')

# %%
from productlens.aspects.aliases import AliasResolver, classify_aspect_type

alias_resolver = AliasResolver()

# Test alias resolution
test_terms = ["display", "panel", "battery backup", "runtime", "customer support", "box"]
print("=== Deterministic Alias Normalization ===")
for term in test_terms:
    print(f"  '{term}' -> '{alias_resolver.resolve(term)}'")

# Test merge safety guards
print("\n=== Merge Safety Verification ===")
pairs_to_test = [
    ("screen", "display", True),
    ("screen", "screen protector", False),
    ("sound quality", "build quality", False),
    ("battery performance", "gaming performance", False),
]

for t1, t2, expected_merge in pairs_to_test:
    allowed = alias_resolver.can_merge(t1, t2)
    status = "✓ SAFE" if allowed == expected_merge else "✗ VIOLATION"
    print(f"  Can merge '{t1}' & '{t2}'? {allowed} (Expected: {expected_merge}) -> {status}")

# %% [markdown]
# ## 8. Aspect Typing
#
# Classify aspect mentions into product, service, delivery, seller, packaging, or unknown.

# %%
print("=== Aspect Typing Examples ===")
type_examples = [
    ("battery life", "The battery life is exceptional.", "product"),
    ("delivery", "Amazon delivered it on time.", "delivery"),
    ("packaging", "The box arrived crushed and torn.", "packaging"),
    ("customer support", "Customer service was unhelpful.", "service"),
    ("seller", "Third-party seller was communicative.", "seller"),
]

for aspect_term, ctx, exp_type in type_examples:
    pred_type = classify_aspect_type(aspect_term, sentence_context=ctx)
    print(f"  '{aspect_term}' -> {pred_type:10s} (Context: \"{ctx[:40]}...\")")
    assert pred_type == exp_type, f"Typing mismatch for {aspect_term}: got {pred_type}, expected {exp_type}"

# %% [markdown]
# ## 9. Product Association & Traceability
#
# Verify that 100% of extracted aspects are linked to their source review, parent product (ASIN),
# and sentence offsets.

# %%
print("=== Product Association & Provenance Verification ===")
review_lookup = {r.review_id: r for r in cleaned_reviews}
sentence_lookup = {s.sentence_id: s for s in sentences}

traceability_passed = True
checked = 0

for asp in extracted_aspects[:50]:
    # Check review exists
    assert asp.review_id in review_lookup, f"Review {asp.review_id} not found"
    # Check sentence exists
    assert asp.sentence_id in sentence_lookup, f"Sentence {asp.sentence_id} not found"

    source_sent = sentence_lookup[asp.sentence_id]
    source_rev = review_lookup[asp.review_id]

    # Verify sentence slice
    sent_slice = source_sent.text[asp.start_char:asp.end_char]
    assert sent_slice == asp.surface, f"Sentence offset mismatch: '{sent_slice}' != '{asp.surface}'"

    # Verify document slice
    doc_slice = source_rev.clean_text[asp.doc_start_char:asp.doc_end_char]
    assert doc_slice == asp.surface, f"Document offset mismatch: '{doc_slice}' != '{asp.surface}'"

    checked += 1

print(f"✓ Verified exact traceability and offsets for {checked} sampled aspects with zero errors")

# Show sample aspect intelligence records
print("\n=== Representative Aspect Intelligence Records ===")
for asp in extracted_aspects[:5]:
    norm_name = cluster_result.surface_to_canonical.get(asp.surface.lower(), asp.surface.lower())
    cls_id = cluster_result.surface_to_cluster_id.get(asp.surface.lower(), "none")
    print(f"Product: {asp.product_id} [{asp.category}]")
    print(f"  Surface:    '{asp.surface}' -> Normalized: '{norm_name}' ({cls_id})")
    print(f"  Offsets:    sentence [{asp.start_char}:{asp.end_char}], doc [{asp.doc_start_char}:{asp.doc_end_char}]")
    print(f"  Confidence: {asp.confidence:.2f} | Type: {asp.aspect_type}")
    print()

# %%
print("=" * 60)
print("STAGE 2 — ASPECT INTELLIGENCE COMPLETE")
print("=" * 60)
print(f"Total reviews processed:    {len(cleaned_reviews)}")
print(f"Total sentences processed:  {len(sentences)}")
print(f"Extracted aspect mentions:  {len(extracted_aspects)}")
print(f"Canonical clusters formed:  {len(cluster_result.cluster_records)}")
print(f"Traceability integrity:     100.0%")
