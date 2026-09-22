"""
Stage 2 Aspect Intelligence CLI runner.

Executes the aspect extraction, normalization, and alias resolution pipeline,
validates provenance traceability, and writes artifacts and DONE.json marker.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from productlens.aspects.aliases import AliasResolver, classify_aspect_type
from productlens.aspects.bio_model import MockBioAspectExtractor, BioAspectExtractor
from productlens.aspects.clustering import AspectClusterer
from productlens.aspects.normalize import AspectEmbedder
from productlens.config import load_config, config_hash
from productlens.data.clean import clean_reviews
from productlens.data.sentence_split import split_reviews_to_sentences
from productlens.data.synthetic import generate_synthetic_reviews
from productlens.schemas import (
    ReviewRecord,
    SentenceRecord,
    records_to_json,
    records_from_json,
)
from productlens.utils import (
    detect_device,
    print_library_versions,
    set_seed,
    setup_logging,
    write_done_marker,
)

logger = logging.getLogger("productlens.aspects")


def run_aspect_pipeline(profile: str = "smoke", mock: bool = True) -> int:
    """Execute end-to-end Stage 2 Aspect Intelligence pipeline."""
    setup_logging()
    start_time = time.time()

    cfg = load_config(profile=profile)
    set_seed(cfg.project.seed)
    dev_info = detect_device()

    logger.info("Starting Stage 2 — Aspect Intelligence (profile=%s, device=%s)", profile, dev_info["device"])

    # 1. Load data
    processed_dir = Path("artifacts/processed")
    reviews_clean_file = processed_dir / "reviews_clean.parquet"

    if reviews_clean_file.is_file():
        logger.info("Loading cleaned reviews from %s", reviews_clean_file)
        df_revs = pd.read_parquet(reviews_clean_file)
        reviews = [ReviewRecord.from_dict(d) for d in df_revs.to_dict(orient="records")]
    else:
        logger.info("Generating synthetic reviews for smoke pipeline validation...")
        raw_revs = generate_synthetic_reviews(seed=cfg.project.seed)
        reviews = clean_reviews(raw_revs, min_chars=cfg.data.min_review_chars)

    # 2. Sentence splitting
    sentences = split_reviews_to_sentences(reviews)
    logger.info("Prepared %d sentences from %d reviews", len(sentences), len(reviews))

    # 3. BIO Aspect Extraction
    if mock or profile == "smoke":
        extractor = MockBioAspectExtractor(model_name=cfg.models.aspect_extractor)
    else:
        extractor = BioAspectExtractor(model_name_or_path=cfg.models.aspect_extractor)

    t0 = time.time()
    extracted_aspects = extractor.extract_from_sentences(sentences, reviews)
    t_extract = time.time() - t0
    logger.info("Extracted %d aspect mentions in %.2fs", len(extracted_aspects), t_extract)

    # 4. Dense Embeddings & Caching
    embedder = AspectEmbedder(
        model_name=cfg.models.embedder,
        cache_dir="artifacts/embeddings",
        mock=mock or profile == "smoke",
    )

    # 5. Clustering & Canonical Discovery
    alias_resolver = AliasResolver()
    clusterer = AspectClusterer(
        mode=cfg.normalization.mode,
        min_cluster_size=cfg.normalization.min_cluster_size,
        min_samples=cfg.normalization.min_samples,
        embedder=embedder,
        alias_resolver=alias_resolver,
    )

    t0 = time.time()
    cluster_res = clusterer.cluster_candidates(extracted_aspects)
    t_cluster = time.time() - t0
    logger.info("Clustered into %d canonical aspect clusters in %.2fs", len(cluster_res.cluster_records), t_cluster)

    # 6. Verify 100% Provenance and Traceability
    rev_map = {r.review_id: r for r in reviews}
    sent_map = {s.sentence_id: s for s in sentences}
    verified_count = 0

    aspect_records = []
    for asp in extracted_aspects:
        assert asp.review_id in rev_map, f"Missing review provenance: {asp.review_id}"
        assert asp.sentence_id in sent_map, f"Missing sentence provenance: {asp.sentence_id}"

        source_sent = sent_map[asp.sentence_id]
        source_rev = rev_map[asp.review_id]

        # Verify exact sentence offsets
        assert source_sent.text[asp.start_char:asp.end_char] == asp.surface, "Sentence offset mismatch"
        # Verify exact document offsets
        assert source_rev.clean_text[asp.doc_start_char:asp.doc_end_char] == asp.surface, "Document offset mismatch"

        norm_name = cluster_res.surface_to_canonical.get(asp.surface.lower(), asp.surface.lower())
        cls_id = cluster_res.surface_to_cluster_id.get(asp.surface.lower(), "")

        aspect_records.append({
            "surface": asp.surface,
            "canonical_aspect": norm_name,
            "cluster_id": cls_id,
            "start_char": asp.start_char,
            "end_char": asp.end_char,
            "doc_start_char": asp.doc_start_char,
            "doc_end_char": asp.doc_end_char,
            "sentence_id": asp.sentence_id,
            "review_id": asp.review_id,
            "product_id": asp.product_id,
            "category": asp.category,
            "aspect_type": asp.aspect_type,
            "confidence": asp.confidence,
            "origin": asp.origin,
        })
        verified_count += 1

    logger.info("Verified exact offset and provenance traceability for 100%% of %d aspects", verified_count)

    # 7. Write Artifacts
    out_dir = Path("artifacts/aspects")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_aspects = pd.DataFrame(aspect_records)
    df_aspects.to_parquet(out_dir / "aspect_mentions.parquet", index=False)

    df_clusters = pd.DataFrame([c.to_dict() for c in cluster_res.cluster_records])
    df_clusters.to_parquet(out_dir / "aspect_clusters.parquet", index=False)

    total_time = time.time() - start_time
    metrics = {
        "reviews_count": len(reviews),
        "sentences_count": len(sentences),
        "aspect_mentions_count": len(extracted_aspects),
        "clusters_count": len(cluster_res.cluster_records),
        "outlier_count": cluster_res.outlier_count,
        "traceability_rate": 1.0,
        "runtime_seconds": round(total_time, 2),
        "device": dev_info["device"],
        "gpu_name": dev_info.get("gpu_name"),
    }

    import json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Write DONE.json marker
    write_done_marker(
        stage="aspects",
        artifact_dir=str(out_dir),
        config_hash_val=config_hash(cfg),
        seconds=total_time,
        extra=metrics,
    )

    logger.info("Stage 2 pipeline completed successfully in %.2fs. Artifacts saved to %s", total_time, out_dir)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ProductLens Stage 2: Aspect Intelligence")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "dev", "full"], help="Config profile")
    parser.add_argument("--real-model", action="store_true", help="Use real Hugging Face model instead of mock")
    args = parser.parse_args()

    sys.exit(run_aspect_pipeline(profile=args.profile, mock=not args.real_model))


if __name__ == "__main__":
    main()
