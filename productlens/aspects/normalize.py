"""
Aspect embedding generation and caching.

Supports dense sentence embeddings via BAAI/bge-small-en-v1.5,
hashing embeddings for smoke mode and unit tests, and persistent disk caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("productlens.aspects.normalize")


# ---------------------------------------------------------------------------
# Embedding Cache (Spec §30)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Persistent embedding cache under ``artifacts/embeddings/``.

    Stores embedding matrices as .npy files alongside a metadata.json
    specifying model_name, embedding_dimension, normalization, and input_hash.
    """

    def __init__(self, cache_dir: str = "artifacts/embeddings") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_inputs(self, texts: List[str], model_name: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(model_name.encode("utf-8"))
        for text in texts:
            hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def get(
        self,
        texts: List[str],
        model_name: str,
    ) -> Optional[np.ndarray]:
        """Retrieve cached embeddings if they exist for the exact inputs and model."""
        input_hash = self._hash_inputs(texts, model_name)
        npy_path = self.cache_dir / f"{input_hash}.npy"
        meta_path = self.cache_dir / f"{input_hash}.meta.json"

        if npy_path.is_file() and meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if meta.get("input_hash") == input_hash and meta.get("count") == len(texts):
                    data = np.load(npy_path)
                    logger.info("Loaded %d cached embeddings from %s", len(texts), npy_path)
                    return data
            except Exception as exc:
                logger.warning("Failed to load embedding cache %s: %s", npy_path, exc)
        return None

    def put(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        model_name: str,
        dimension: int,
        normalized: bool = True,
        model_revision: str = "main",
    ) -> None:
        """Save embeddings and metadata to cache."""
        input_hash = self._hash_inputs(texts, model_name)
        npy_path = self.cache_dir / f"{input_hash}.npy"
        meta_path = self.cache_dir / f"{input_hash}.meta.json"

        try:
            np.save(npy_path, embeddings)
            metadata = {
                "model_name": model_name,
                "model_revision": model_revision,
                "embedding_dimension": dimension,
                "normalization": normalized,
                "input_hash": input_hash,
                "count": len(texts),
            }
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            logger.info("Cached %d embeddings to %s", len(texts), npy_path)
        except Exception as exc:
            logger.warning("Failed to write embedding cache: %s", exc)


# ---------------------------------------------------------------------------
# Hashing Embedder (Spec §41 Smoke Mode / Offline / Mock)
# ---------------------------------------------------------------------------

class HashingEmbedder:
    """
    Deterministic feature hashing embedder for smoke mode and unit tests.

    Produces normalized 384-dimensional dense vectors from character n-grams
    and token hashes with zero GPU, network, or external model dependencies.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.model_name = "mock/hashing-embedder"

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts into deterministic dense embedding vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            clean = text.lower().strip()
            # Token-level hashing
            words = clean.split()
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vectors[i, idx] += sign * 1.5

            # Character 3-gram hashing for subword / surface similarity
            padded = f" {clean} "
            for j in range(max(1, len(padded) - 2)):
                tri = padded[j:j+3]
                h = int(hashlib.sha256(tri.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vectors[i, idx] += sign * 0.8

            if normalize_embeddings:
                norm = np.linalg.norm(vectors[i])
                if norm > 1e-9:
                    vectors[i] /= norm
                else:
                    vectors[i, 0] = 1.0

        return vectors


# ---------------------------------------------------------------------------
# Aspect Embedder (Spec §14, §30)
# ---------------------------------------------------------------------------

class AspectEmbedder:
    """
    Aspect embedding generator supporting BAAI/bge-small-en-v1.5 and caching.

    Automatically handles FP16 on CUDA, CPU fallback, and caching.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = "artifacts/embeddings",
        use_cache: bool = True,
        mock: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        self.mock = mock
        self.dimension = 384
        self._model = None
        self._device = device

        if not mock:
            self._init_model()

    def _init_model(self) -> None:
        """Lazily initialize sentence transformer model."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading embedding model %s on %s", self.model_name, self._device)
            self._model = SentenceTransformer(self.model_name, device=self._device)
            if self._device == "cuda":
                self._model.half()  # FP16
            self.dimension = self._model.get_sentence_embedding_dimension()
        except Exception as exc:
            logger.warning(
                "Could not load SentenceTransformer %s (%s). Falling back to HashingEmbedder.",
                self.model_name,
                exc,
            )
            self.mock = True

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode aspect phrases into normalized dense vectors.

        Checks disk cache before computing.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Check cache
        if self.cache is not None:
            cached = self.cache.get(texts, self.model_name)
            if cached is not None:
                return cached

        if self.mock or self._model is None:
            hashing = HashingEmbedder(dimension=self.dimension)
            embs = hashing.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            embs = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

        # Store in cache
        if self.cache is not None:
            self.cache.put(
                texts=texts,
                embeddings=embs,
                model_name=self.model_name,
                dimension=self.dimension,
                normalized=normalize_embeddings,
            )

        return embs
