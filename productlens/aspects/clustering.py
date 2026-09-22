"""
Aspect normalization and clustering.

Clusters candidate aspect embeddings using HDBSCAN (with scikit-learn DBSCAN fallback).
Supports category-aware and global clustering modes, represents outliers without data loss,
and derives canonical names from actual cluster members.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from productlens.aspects.aliases import AliasResolver, DEFAULT_ALIASES
from productlens.aspects.candidates import CandidateAspect
from productlens.aspects.normalize import AspectEmbedder
from productlens.schemas import AspectClusterRecord

logger = logging.getLogger("productlens.aspects.clustering")


# ---------------------------------------------------------------------------
# Clustering Result Container
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Result of aspect clustering."""
    cluster_records: List[AspectClusterRecord] = field(default_factory=list)
    # Maps candidate surface string (lower) -> canonical_aspect
    surface_to_canonical: Dict[str, str] = field(default_factory=dict)
    # Maps candidate surface string (lower) -> cluster_id
    surface_to_cluster_id: Dict[str, str] = field(default_factory=dict)
    # Total candidates processed
    total_candidates: int = 0
    # Outlier count
    outlier_count: int = 0


# ---------------------------------------------------------------------------
# Aspect Clusterer
# ---------------------------------------------------------------------------

class AspectClusterer:
    """
    Clusters candidate aspect mentions using HDBSCAN and semantic embeddings.

    Implements Spec §14:
    - Default mode: category_aware
    - metric: cosine
    - Derives canonical aspect name from actual cluster members
    - Never discards outliers
    """

    def __init__(
        self,
        mode: str = "category_aware",
        min_cluster_size: int = 5,
        min_samples: int = 2,
        metric: str = "cosine",
        embedder: Optional[AspectEmbedder] = None,
        alias_resolver: Optional[AliasResolver] = None,
    ) -> None:
        self.mode = mode
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.embedder = embedder or AspectEmbedder(mock=True)
        self.alias_resolver = alias_resolver or AliasResolver()

    def cluster_candidates(
        self,
        candidates: List[CandidateAspect],
    ) -> ClusterResult:
        """
        Cluster candidate aspects.

        Parameters
        ----------
        candidates : list of CandidateAspect
            Candidate aspect mentions extracted from review sentences.

        Returns
        -------
        ClusterResult containing cluster metadata and surface mappings.
        """
        if not candidates:
            return ClusterResult()

        if self.mode == "category_aware":
            return self._cluster_category_aware(candidates)
        else:
            return self._cluster_global(candidates)

    def _cluster_category_aware(
        self,
        candidates: List[CandidateAspect],
    ) -> ClusterResult:
        """Cluster candidates grouped by category to prevent cross-domain contamination."""
        cat_groups: Dict[str, List[CandidateAspect]] = {}
        for c in candidates:
            cat = c.category if c.category else "general"
            cat_groups.setdefault(cat, []).append(c)

        merged_result = ClusterResult(total_candidates=len(candidates))

        for cat, group in cat_groups.items():
            sub_result = self._cluster_group(group, category_prefix=cat)
            merged_result.cluster_records.extend(sub_result.cluster_records)
            merged_result.surface_to_canonical.update(sub_result.surface_to_canonical)
            merged_result.surface_to_cluster_id.update(sub_result.surface_to_cluster_id)
            merged_result.outlier_count += sub_result.outlier_count

        return merged_result

    def _cluster_global(
        self,
        candidates: List[CandidateAspect],
    ) -> ClusterResult:
        """Cluster all candidates globally."""
        return self._cluster_group(candidates, category_prefix="global")

    def _cluster_group(
        self,
        candidates: List[CandidateAspect],
        category_prefix: str,
    ) -> ClusterResult:
        """Execute clustering on a specific subset of candidate aspects."""
        if not candidates:
            return ClusterResult()

        # Deduplicate unique surface terms to avoid redundant embeddings
        unique_surfaces = sorted(list({c.surface.strip().lower() for c in candidates}))
        surface_counts = Counter(c.surface.strip().lower() for c in candidates)

        # Get embeddings
        embeddings = self.embedder.encode(unique_surfaces, normalize_embeddings=True)

        n_samples = len(unique_surfaces)
        min_cluster_size = min(self.min_cluster_size, max(2, n_samples // 3)) if n_samples > 3 else 2
        min_samples = min(self.min_samples, min_cluster_size)

        # Run clustering algorithm
        labels, probabilities = self._run_hdbscan(
            embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        cluster_records: List[AspectClusterRecord] = []
        surface_to_canonical: Dict[str, str] = {}
        surface_to_cluster_id: Dict[str, str] = {}
        outlier_count = 0

        # Group indices by cluster label
        clusters: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            clusters.setdefault(lbl, []).append(idx)

        # Process valid clusters (label >= 0)
        for lbl, member_indices in clusters.items():
            if lbl == -1:
                continue

            member_surfaces = [unique_surfaces[i] for i in member_indices]
            member_embeddings = embeddings[member_indices]
            member_probs = [probabilities[i] for i in member_indices]

            # Enforce safety guard: split cluster if disallowed merge terms are grouped together
            valid_groups = self._partition_disallowed_merges(member_surfaces, member_indices)

            for group_idx, (grp_surfaces, grp_indices) in enumerate(valid_groups):
                # Derive canonical aspect:
                # 1. Check if any member has an explicit alias
                # 2. Or choose the medoid / most frequent member
                canonical = self._select_canonical_aspect(
                    grp_surfaces,
                    surface_counts,
                    embeddings[grp_indices],
                    category_prefix,
                )

                cluster_hash = hashlib.sha256(
                    f"{category_prefix}_{lbl}_{group_idx}_{canonical}".encode("utf-8")
                ).hexdigest()[:16]
                cluster_id = f"cls_{category_prefix}_{cluster_hash}"

                mean_conf = float(np.mean([probabilities[i] for i in grp_indices])) if grp_indices else 0.8
                # Representative terms (sorted by member frequency)
                sorted_rep = sorted(
                    list(set(grp_surfaces)),
                    key=lambda s: surface_counts[s],
                    reverse=True,
                )[:5]
                total_grp_members = sum(surface_counts[s] for s in grp_surfaces)

                rec = AspectClusterRecord(
                    cluster_id=cluster_id,
                    canonical_aspect=canonical,
                    cluster_confidence=round(mean_conf, 4),
                    representative_terms=sorted_rep,
                    member_count=total_grp_members,
                )
                cluster_records.append(rec)

                for s in grp_surfaces:
                    surface_to_canonical[s] = canonical
                    surface_to_cluster_id[s] = cluster_id

        # Process outliers (label == -1): outliers remain represented per Spec §14
        if -1 in clusters:
            outlier_indices = clusters[-1]
            outlier_count = len(outlier_indices)
            for idx in outlier_indices:
                surface = unique_surfaces[idx]
                canonical = self.alias_resolver.resolve(surface, category=category_prefix)
                cluster_hash = hashlib.sha256(f"outlier_{category_prefix}_{surface}".encode("utf-8")).hexdigest()[:16]
                cluster_id = f"cls_outlier_{cluster_hash}"

                rec = AspectClusterRecord(
                    cluster_id=cluster_id,
                    canonical_aspect=canonical,
                    cluster_confidence=0.5,
                    representative_terms=[surface],
                    member_count=surface_counts[surface],
                )
                cluster_records.append(rec)
                surface_to_canonical[surface] = canonical
                surface_to_cluster_id[surface] = cluster_id

        return ClusterResult(
            cluster_records=cluster_records,
            surface_to_canonical=surface_to_canonical,
            surface_to_cluster_id=surface_to_cluster_id,
            total_candidates=len(candidates),
            outlier_count=outlier_count,
        )

    def _partition_disallowed_merges(
        self,
        surfaces: List[str],
        indices: List[int],
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Check if any pairs in the cluster violate alias merge rules
        (e.g., 'screen' vs 'screen protector', or shared generic word only).
        Partitions the cluster into mutually compatible sub-groups.
        """
        groups: List[Tuple[List[str], List[int]]] = []
        for s, idx in zip(surfaces, indices):
            placed = False
            for grp_surfaces, grp_indices in groups:
                # Check if compatible with all existing members of this group
                if all(self.alias_resolver.can_merge(s, existing) for existing in grp_surfaces):
                    grp_surfaces.append(s)
                    grp_indices.append(idx)
                    placed = True
                    break
            if not placed:
                groups.append(([s], [idx]))
        return groups

    def _select_canonical_aspect(
        self,
        surfaces: List[str],
        surface_counts: Counter,
        embeddings: np.ndarray,
        category: str,
    ) -> str:
        """
        Select the most representative, clean canonical aspect name from cluster members.
        """
        # First check if any surface maps directly to an explicit alias
        for s in surfaces:
            alias = self.alias_resolver.resolve(s, category=category)
            if alias in DEFAULT_ALIASES.values():
                return alias

        # Medoid: calculate member closest to cluster centroid
        if len(surfaces) > 1 and len(embeddings) == len(surfaces):
            centroid = np.mean(embeddings, axis=0, keepdims=True)
            centroid /= (np.linalg.norm(centroid) + 1e-9)
            sims = np.dot(embeddings, centroid.T).flatten()
            best_idx = int(np.argmax(sims))
            medoid_surface = surfaces[best_idx]
            return self.alias_resolver.resolve(medoid_surface, category=category)

        # Fallback to most frequent term
        most_freq = max(surfaces, key=lambda s: (surface_counts[s], -len(s)))
        return self.alias_resolver.resolve(most_freq, category=category)

    def _run_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run HDBSCAN clustering with fallback to scikit-learn DBSCAN / distance matching.
        """
        n_samples = len(embeddings)
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=float)

        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=self.metric,
                core_dist_n_jobs=1,
            )
            labels = clusterer.fit_predict(embeddings)
            probs = getattr(clusterer, "probabilities_", np.ones(n_samples, dtype=float))
            return labels, probs
        except Exception:
            pass

        # Fallback: scikit-learn DBSCAN with cosine distance
        try:
            from sklearn.cluster import DBSCAN
            # Convert cosine similarity to cosine distance: 1 - cos_sim
            db = DBSCAN(
                eps=0.35,
                min_samples=min_samples,
                metric="cosine",
            )
            labels = db.fit_predict(embeddings)
            probs = np.full(n_samples, 0.75, dtype=float)
            return labels, probs
        except Exception as exc:
            logger.warning("Clustering fallback to singletons: %s", exc)
            return np.arange(n_samples, dtype=int), np.ones(n_samples, dtype=float)
