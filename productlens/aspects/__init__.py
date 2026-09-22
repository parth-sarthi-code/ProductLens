"""
ProductLens Aspect Intelligence module.

Provides:
- Candidate aspect extraction (rule, POS, dependency, candidate generation)
- Transformer-based BIO sequence labeling (DeBERTa-v3-base / RoBERTa-base)
- Aspect normalization via dense embeddings (BGE-small) and HDBSCAN clustering
- Deterministic alias resolution and manual override handling
- Aspect typing (product, service, delivery, seller, packaging, unknown)
- Span evaluation metrics (exact and partial F1, token-level BIO metrics)
"""

from productlens.aspects.candidates import (
    CandidateAspect,
    CandidateExtractor,
    extract_candidates,
)
from productlens.aspects.bio_model import (
    BioAspectExtractor,
    MockBioAspectExtractor,
    BIO_LABELS,
    ID2LABEL,
    LABEL2ID,
)
from productlens.aspects.normalize import (
    AspectEmbedder,
    HashingEmbedder,
    EmbeddingCache,
)
from productlens.aspects.clustering import (
    AspectClusterer,
    ClusterResult,
)
from productlens.aspects.aliases import (
    AliasResolver,
    classify_aspect_type,
)

__all__ = [
    "CandidateAspect",
    "CandidateExtractor",
    "extract_candidates",
    "BioAspectExtractor",
    "MockBioAspectExtractor",
    "BIO_LABELS",
    "ID2LABEL",
    "LABEL2ID",
    "AspectEmbedder",
    "HashingEmbedder",
    "EmbeddingCache",
    "AspectClusterer",
    "ClusterResult",
    "AliasResolver",
    "classify_aspect_type",
]
