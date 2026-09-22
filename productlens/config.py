"""
ProductLens configuration system.

Provides frozen dataclasses for all configuration sections, YAML loading,
profile support (smoke/dev/full), and runtime --set key=value overrides.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Project-level settings."""
    seed: int = 42
    artifact_root: str = "artifacts"


@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    categories: List[str] = field(default_factory=list)
    max_reviews: int = 10_000
    chunk_size: int = 1_000
    min_review_chars: int = 20
    language: str = "en"
    local_path: str = ""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    dedupe_near: bool = False
    near_dedupe_threshold: float = 0.9


@dataclass
class ModelsConfig:
    """Model identifiers."""
    aspect_extractor: str = "microsoft/deberta-v3-base"
    sentiment_classifier: str = "microsoft/deberta-v3-base"
    embedder: str = "BAAI/bge-small-en-v1.5"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    max_length: int = 256
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2.0e-5
    epochs: int = 2
    fp16: bool = True
    gradient_checkpointing: bool = True
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    save_total_limit: int = 2


@dataclass
class NormalizationConfig:
    """Aspect normalization / clustering settings."""
    mode: str = "category_aware"  # global | category_aware
    min_cluster_size: int = 5
    min_samples: int = 2
    metric: str = "cosine"
    alias_override_path: str = "artifacts/aspects/alias_overrides.yaml"


@dataclass
class EvidenceConfig:
    """Evidence selection settings."""
    top_n: int = 5
    mmr_lambda: float = 0.7


@dataclass
class AggregationConfig:
    """Aggregation settings."""
    min_mentions: int = 3
    quality_weighting: bool = True


@dataclass
class PipelineConfig:
    """Pipeline execution settings."""
    cache: bool = True
    deterministic: bool = True


@dataclass
class ProductLensConfig:
    """Root configuration combining all sections."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# ---------------------------------------------------------------------------
# Profile overlays
# ---------------------------------------------------------------------------

_PROFILE_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "data": {
            "max_reviews": 300,
            "chunk_size": 100,
            "categories": [],
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "fp16": False,
        },
        "normalization": {
            "min_cluster_size": 3,
            "min_samples": 1,
        },
        "pipeline": {
            "cache": False,
        },
    },
    "dev": {
        "data": {
            "max_reviews": 5_000,
            "chunk_size": 500,
        },
        "training": {
            "epochs": 2,
        },
    },
    "full": {
        "data": {
            "max_reviews": 100_000,
            "chunk_size": 5_000,
        },
        "training": {
            "epochs": 3,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataclass_from_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively construct a dataclass from a dictionary, ignoring unknown keys."""
    known_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return cls(**filtered)


def _apply_overlay(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *overlay* into *base* (mutates *base*)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _apply_overlay(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(data: Dict[str, Any], dotted_key: str, value: str) -> None:
    """Set a value in a nested dict using dotted notation, e.g. 'data.max_reviews=500'."""
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    # Attempt type coercion
    raw = value
    if raw.lower() in ("true", "false"):
        coerced: Any = raw.lower() == "true"
    else:
        try:
            coerced = int(raw)
        except ValueError:
            try:
                coerced = float(raw)
            except ValueError:
                coerced = raw
    current[parts[-1]] = coerced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    path: Optional[str] = None,
    profile: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> ProductLensConfig:
    """
    Load configuration from YAML, apply profile overlay, then runtime overrides.

    Parameters
    ----------
    path : str, optional
        Path to YAML config file. Defaults to ``configs/default.yaml``.
    profile : str, optional
        Profile name (``smoke``, ``dev``, ``full``).
    overrides : list of str, optional
        Runtime overrides in ``key=value`` dotted notation,
        e.g. ``["data.max_reviews=500", "training.epochs=1"]``.

    Returns
    -------
    ProductLensConfig
    """
    # --- locate config file ---
    if path is None:
        # Search relative to repo root
        candidates = [
            Path("configs/default.yaml"),
            Path(__file__).resolve().parent.parent / "configs" / "default.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = str(candidate)
                break

    # --- load YAML ---
    raw: Dict[str, Any] = {}
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded

    # --- apply profile ---
    if profile and profile in _PROFILE_OVERLAYS:
        raw = _apply_overlay(raw, copy.deepcopy(_PROFILE_OVERLAYS[profile]))

    # --- apply runtime overrides ---
    if overrides:
        for item in overrides:
            if "=" not in item:
                continue
            key, val = item.split("=", 1)
            _set_nested(raw, key.strip(), val.strip())

    # --- construct dataclasses ---
    cfg = ProductLensConfig(
        project=_dataclass_from_dict(ProjectConfig, raw.get("project", {})),
        data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        models=_dataclass_from_dict(ModelsConfig, raw.get("models", {})),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
        normalization=_dataclass_from_dict(NormalizationConfig, raw.get("normalization", {})),
        evidence=_dataclass_from_dict(EvidenceConfig, raw.get("evidence", {})),
        aggregation=_dataclass_from_dict(AggregationConfig, raw.get("aggregation", {})),
        pipeline=_dataclass_from_dict(PipelineConfig, raw.get("pipeline", {})),
    )
    return cfg


def config_hash(cfg: ProductLensConfig) -> str:
    """Return a deterministic SHA-256 hash of the configuration."""
    data = asdict(cfg)
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def config_to_dict(cfg: ProductLensConfig) -> Dict[str, Any]:
    """Serialize configuration to a plain dictionary."""
    return asdict(cfg)
