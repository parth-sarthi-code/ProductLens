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
# # ProductLens — Notebook 01: Data Foundation
#
# This notebook demonstrates the complete data pipeline:
# 1. Environment setup and version checking
# 2. Synthetic data generation (smoke mode)
# 3. Data cleaning
# 4. Deduplication
# 5. Train/val/test splitting with leakage verification
# 6. Sentence splitting with offset verification
# 7. Data statistics and validation

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
import os
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

# Setup
setup_logging()
versions = print_library_versions()
device_info = detect_device()
set_seed(42)

print(f"\nPython: {sys.version}")
print(f"Device: {device_info['device']}")
if device_info['gpu_available']:
    print(f"GPU: {device_info['gpu_name']} ({device_info['vram_mb']} MB VRAM)")

# %%
# Load configuration
cfg = load_config(profile="smoke")
print(f"\nProfile: smoke")
print(f"Max reviews: {cfg.data.max_reviews}")
print(f"Chunk size: {cfg.data.chunk_size}")
print(f"Min review chars: {cfg.data.min_review_chars}")
print(f"Config hash: {config_hash(cfg)[:16]}...")

# %% [markdown]
# ## 2. Synthetic Data Generation
#
# Generate ~300 deterministic reviews across 6 categories for smoke testing.
# No downloads required.

# %%
from productlens.data.synthetic import generate_synthetic_reviews

start = time.time()
reviews = generate_synthetic_reviews(seed=cfg.project.seed)
elapsed = time.time() - start

print(f"Generated {len(reviews)} synthetic reviews in {elapsed:.2f}s")
print(f"\nCategories: {sorted(set(r.category for r in reviews))}")
print(f"\nCategory distribution:")
cat_counts = Counter(r.category for r in reviews)
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")

# %%
# Show sample reviews
print("=== Sample Reviews ===\n")
for i, r in enumerate(reviews[:5]):
    print(f"[{i}] Category: {r.category}")
    print(f"    Product: {r.product_id}")
    print(f"    Rating: {r.rating}")
    print(f"    Text: {r.text[:100]}{'...' if len(r.text) > 100 else ''}")
    print()

# %%
# Show edge cases
print("=== Edge Case Reviews ===\n")
edge_cases = [r for r in reviews if r.title.startswith("Edge case:")]
for r in edge_cases[:8]:
    print(f"  [{r.title}]")
    print(f"    Text: {repr(r.text[:80])}")
    print()

# %% [markdown]
# ## 3. Data Cleaning
#
# Clean reviews: Unicode normalization, HTML removal, URL stripping,
# whitespace normalization. Filter short/empty reviews.

# %%
from productlens.data.clean import clean_reviews, clean_text

start = time.time()
cleaned = clean_reviews(reviews, min_chars=cfg.data.min_review_chars)
elapsed = time.time() - start

print(f"Cleaning: {len(reviews)} → {len(cleaned)} reviews ({elapsed:.2f}s)")
print(f"Removed: {len(reviews) - len(cleaned)} reviews")

# %%
# Verify raw_text is preserved
for r in cleaned[:5]:
    assert r.raw_text, "raw_text must be preserved"
    assert r.clean_text, "clean_text must be set"
    print(f"Review {r.review_id[:12]}:")
    if r.raw_text != r.clean_text:
        print(f"  RAW:   {r.raw_text[:80]}...")
        print(f"  CLEAN: {r.clean_text[:80]}...")
    else:
        print(f"  TEXT:   {r.clean_text[:80]}...")
    print()

print("✓ raw_text preserved for all cleaned reviews")

# %%
# Demonstrate cleaning on specific edge cases
print("=== Cleaning Examples ===\n")
test_cases = [
    "<b>Great</b> sound quality! <i>Love</i> the bass. <br/>Recommended.",
    "The &amp; cream is &lt;amazing&gt; for dry skin.",
    "Check out my full review at https://example.com/review/123 . The sound is great.",
    "Too    many     spaces   here",
]
for tc in test_cases:
    print(f"  Input:  {tc}")
    print(f"  Output: {clean_text(tc)}")
    print()

# %% [markdown]
# ## 4. Deduplication
#
# Remove exact and near-duplicate reviews before splitting.
# This prevents data leakage across train/test boundaries.

# %%
from productlens.data.dedupe import deduplicate

start = time.time()
deduped, dup_log = deduplicate(
    cleaned,
    near_dedupe=cfg.data.dedupe_near,
    near_threshold=cfg.data.near_dedupe_threshold,
)
elapsed = time.time() - start

print(f"Deduplication: {len(cleaned)} → {len(deduped)} reviews ({elapsed:.2f}s)")
print(f"Exact duplicates removed: {len(dup_log['exact_duplicates'])}")
print(f"Near duplicates removed: {len(dup_log['near_duplicates'])}")

# %% [markdown]
# ## 5. Train / Val / Test Splitting
#
# Product-aware splitting ensures no product appears in multiple splits,
# preventing product leakage.

# %%
from productlens.data.split import split_reviews, verify_no_leakage

start = time.time()
splits = split_reviews(
    deduped,
    train_ratio=cfg.data.train_ratio,
    val_ratio=cfg.data.val_ratio,
    test_ratio=cfg.data.test_ratio,
    seed=cfg.project.seed,
    by_product=True,
)
elapsed = time.time() - start

print(f"\nSplit results ({elapsed:.2f}s):")
for name, data in splits.items():
    products = len(set(r.product_id for r in data))
    categories = len(set(r.category for r in data))
    print(f"  {name:5s}: {len(data):4d} reviews, {products:3d} products, {categories:2d} categories")

# %%
# Verify no leakage
print("\n=== Leakage Verification ===")
issues = verify_no_leakage(splits)
total_issues = sum(len(v) for v in issues.values())
if total_issues == 0:
    print("✓ No data leakage detected")
else:
    print(f"✗ {total_issues} leakage issues detected!")
    for key, items in issues.items():
        if items:
            print(f"  {key}: {len(items)} issues")

# %% [markdown]
# ## 6. Sentence Splitting
#
# Split each review into individual sentences with character offsets.
# Offsets reference the review's `clean_text` field.

# %%
from productlens.data.sentence_split import split_reviews_to_sentences

start = time.time()
sentences = split_reviews_to_sentences(deduped)
elapsed = time.time() - start

print(f"Sentence splitting: {len(deduped)} reviews → {len(sentences)} sentences ({elapsed:.2f}s)")
print(f"Average sentences per review: {len(sentences) / max(len(deduped), 1):.1f}")

# %%
# Verify offsets
print("=== Offset Verification ===\n")
review_map = {r.review_id: r for r in deduped}
offset_errors = 0

for s in sentences:
    review = review_map.get(s.review_id)
    if review and review.clean_text:
        actual = review.clean_text[s.start_char:s.end_char]
        if actual != s.text:
            offset_errors += 1
            if offset_errors <= 3:
                print(f"  Mismatch in review {s.review_id[:12]}:")
                print(f"    Expected: {s.text[:60]}")
                print(f"    Got:      {actual[:60]}")
                print(f"    Offsets:  [{s.start_char}:{s.end_char}]")

if offset_errors == 0:
    print(f"✓ All {len(sentences)} sentence offsets verified correctly")
else:
    print(f"✗ {offset_errors} offset mismatches found")

# %%
# Show sample sentences
print("\n=== Sample Sentences ===\n")
for s in sentences[:8]:
    print(f"  [{s.sentence_id[:12]}] ({s.start_char}-{s.end_char}) {s.text[:80]}")

# %% [markdown]
# ## 7. Data Statistics

# %%
print("=" * 60)
print("PRODUCTLENS DATA FOUNDATION — SUMMARY")
print("=" * 60)
print(f"\nTotal synthetic reviews generated: {len(reviews)}")
print(f"After cleaning: {len(cleaned)}")
print(f"After deduplication: {len(deduped)}")
print(f"Total sentences: {len(sentences)}")
print()

print("Category breakdown:")
for cat in sorted(set(r.category for r in deduped)):
    cat_reviews = [r for r in deduped if r.category == cat]
    cat_sents = [s for s in sentences if review_map.get(s.review_id, ReviewRecord()).category == cat]
    print(f"  {cat:15s}: {len(cat_reviews):4d} reviews, {len(cat_sents):5d} sentences")

print()
print("Split sizes:")
for name, data in splits.items():
    print(f"  {name:5s}: {len(data):4d} reviews")

print()
review_lengths = [len(r.clean_text) for r in deduped]
print(f"Review length stats:")
print(f"  Min: {min(review_lengths)}")
print(f"  Max: {max(review_lengths)}")
print(f"  Mean: {sum(review_lengths) / len(review_lengths):.0f}")

sent_lengths = [len(s.text) for s in sentences]
print(f"\nSentence length stats:")
print(f"  Min: {min(sent_lengths)}")
print(f"  Max: {max(sent_lengths)}")
print(f"  Mean: {sum(sent_lengths) / len(sent_lengths):.0f}")

# %%
print("\n✅ Stage 1 — Data Foundation complete")
