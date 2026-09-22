"""
Amazon Reviews 2023 data loader.

Supports HuggingFace dataset loading, local file discovery, schema
adaptation, configurable limits, chunked loading, and stable ID generation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from tqdm import tqdm

from productlens.config import DataConfig
from productlens.schemas import ReviewRecord
from productlens.utils import stable_id

logger = logging.getLogger("productlens.data.load_amazon")

# ---------------------------------------------------------------------------
# Field mapping — maps common Amazon Reviews 2023 fields to ReviewRecord
# ---------------------------------------------------------------------------

_FIELD_MAP: Dict[str, str] = {
    # review fields
    "rating": "rating",
    "overall": "rating",
    "star_rating": "rating",
    "title": "title",
    "reviewTitle": "title",
    "summary": "title",
    "text": "text",
    "reviewText": "text",
    "review_body": "text",
    "reviewerID": "user_id",
    "user_id": "user_id",
    "asin": "product_id",
    "product_id": "product_id",
    "parent_asin": "parent_product_id",
    "parent_product_id": "parent_product_id",
    "verified_purchase": "verified_purchase",
    "verified": "verified_purchase",
    "helpful_vote": "helpful_vote",
    "helpful": "helpful_vote",
    "vote": "helpful_vote",
    "timestamp": "timestamp",
    "unixReviewTime": "timestamp",
    "reviewTime": "timestamp",
    "date": "timestamp",
    # metadata fields
    "main_category": "category",
    "category": "category",
    "categories": "category",
}


def _safe_str(val: Any) -> str:
    """Convert a value to string safely, handling None and lists."""
    if val is None:
        return ""
    if isinstance(val, list):
        # categories can be nested lists
        flat = val
        while flat and isinstance(flat[0], list):
            flat = flat[0]
        return str(flat[0]) if flat else ""
    return str(val)


def _safe_float(val: Any, default: float = -1.0) -> float:
    """Convert to float safely."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert to int safely."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_bool(val: Any) -> bool:
    """Convert to bool safely."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1", "y")
    if isinstance(val, (int, float)):
        return bool(val)
    return False


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------

def discover_schema(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Discover how to map a raw record's fields to ReviewRecord fields.

    Parameters
    ----------
    sample : dict
        A single raw record from the dataset.

    Returns
    -------
    dict
        Mapping from raw field name → ReviewRecord field name.
    """
    mapping: Dict[str, str] = {}
    for raw_key in sample.keys():
        normalized_key = raw_key.strip().lower().replace(" ", "_")
        if raw_key in _FIELD_MAP:
            mapping[raw_key] = _FIELD_MAP[raw_key]
        elif normalized_key in _FIELD_MAP:
            mapping[raw_key] = _FIELD_MAP[normalized_key]
    return mapping


def normalize_record(
    raw: Dict[str, Any],
    schema_map: Dict[str, str],
    source: str = "",
    category_override: str = "",
) -> ReviewRecord:
    """
    Normalize a raw record dict into a ReviewRecord.

    Parameters
    ----------
    raw : dict
        Raw record from dataset.
    schema_map : dict
        Field mapping from ``discover_schema``.
    source : str
        Data source identifier.
    category_override : str
        If non-empty, override the category field.

    Returns
    -------
    ReviewRecord
    """
    mapped: Dict[str, Any] = {}
    for raw_key, target_key in schema_map.items():
        if raw_key in raw:
            mapped[target_key] = raw[raw_key]

    text = _safe_str(mapped.get("text", ""))
    product_id = _safe_str(mapped.get("product_id", ""))
    timestamp = _safe_str(mapped.get("timestamp", ""))
    category = category_override or _safe_str(mapped.get("category", ""))

    review_id_val = stable_id(source, product_id, text, timestamp)

    parent_product_id = _safe_str(mapped.get("parent_product_id", ""))
    if not parent_product_id:
        parent_product_id = product_id

    return ReviewRecord(
        review_id=review_id_val,
        product_id=product_id,
        parent_product_id=parent_product_id,
        category=category,
        title=_safe_str(mapped.get("title", "")),
        text=text,
        raw_text=text,
        clean_text="",
        rating=_safe_float(mapped.get("rating"), -1.0),
        verified_purchase=_safe_bool(mapped.get("verified_purchase", False)),
        helpful_vote=_safe_int(mapped.get("helpful_vote", 0)),
        timestamp=timestamp,
        source=source,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class AmazonReviewLoader:
    """
    Loads Amazon Reviews 2023 data from HuggingFace or local files.

    Parameters
    ----------
    config : DataConfig
        Data configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._schema_map: Optional[Dict[str, str]] = None

    def load(self) -> List[ReviewRecord]:
        """
        Load reviews according to configuration.

        Tries local path first, then HuggingFace datasets.

        Returns
        -------
        list of ReviewRecord
        """
        if self.config.local_path and Path(self.config.local_path).exists():
            return self._load_local(self.config.local_path)
        return self._load_huggingface()

    def load_chunked(self) -> Generator[List[ReviewRecord], None, None]:
        """
        Yield chunks of ReviewRecords for incremental processing.

        Yields
        ------
        list of ReviewRecord
            Each chunk of size at most ``config.chunk_size``.
        """
        reviews = self.load()
        chunk_size = max(1, self.config.chunk_size)
        for i in range(0, len(reviews), chunk_size):
            yield reviews[i: i + chunk_size]

    # ----- HuggingFace loading -----

    def _load_huggingface(self) -> List[ReviewRecord]:
        """Load from HuggingFace datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace loading. "
                "Install with: pip install datasets"
            )

        logger.info(
            "Loading from HuggingFace: %s (max_reviews=%d)",
            self.config.dataset_name,
            self.config.max_reviews,
        )

        categories = self.config.categories
        if not categories:
            # Load the default "full" subset or first available
            categories = ["All_Beauty"]  # small default for safety

        records: List[ReviewRecord] = []

        for category in categories:
            if len(records) >= self.config.max_reviews:
                break

            remaining = self.config.max_reviews - len(records)
            logger.info("Loading category '%s' (remaining budget: %d)", category, remaining)

            try:
                ds = load_dataset(
                    self.config.dataset_name,
                    category,
                    split="full",
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning("Failed to load category '%s': %s", category, e)
                continue

            # Schema discovery from first record
            if len(ds) > 0:
                first = ds[0]
                if self._schema_map is None:
                    self._schema_map = discover_schema(first)
                    logger.info("Discovered schema mapping: %s", self._schema_map)

            for i, row in enumerate(tqdm(ds, desc=f"Loading {category}", total=min(remaining, len(ds)))):
                if i >= remaining:
                    break
                record = normalize_record(
                    dict(row),
                    self._schema_map or {},
                    source=self.config.dataset_name,
                    category_override=category,
                )
                records.append(record)

        logger.info("Loaded %d reviews total from HuggingFace", len(records))
        return records

    # ----- Local file loading -----

    def _load_local(self, path: str) -> List[ReviewRecord]:
        """Load from local files (Parquet, JSON, JSONL, CSV)."""
        root = Path(path)
        files = self._discover_files(root)
        if not files:
            logger.warning("No supported files found in %s", path)
            return []

        logger.info("Found %d local files to process", len(files))
        records: List[ReviewRecord] = []

        for filepath in files:
            if len(records) >= self.config.max_reviews:
                break
            remaining = self.config.max_reviews - len(records)
            chunk = self._load_single_file(filepath, remaining)
            records.extend(chunk)

        logger.info("Loaded %d reviews total from local files", len(records))
        return records

    def _discover_files(self, root: Path) -> List[Path]:
        """Recursively discover supported data files."""
        extensions = {".parquet", ".json", ".jsonl", ".csv"}
        files: List[Path] = []
        if root.is_file():
            if root.suffix.lower() in extensions:
                files.append(root)
        else:
            for ext in extensions:
                files.extend(sorted(root.rglob(f"*{ext}")))
        return sorted(files)

    def _load_single_file(self, filepath: Path, max_records: int) -> List[ReviewRecord]:
        """Load records from a single file."""
        ext = filepath.suffix.lower()
        source = str(filepath)

        try:
            if ext == ".parquet":
                return self._load_parquet(filepath, source, max_records)
            elif ext == ".csv":
                return self._load_csv(filepath, source, max_records)
            elif ext in (".json", ".jsonl"):
                return self._load_json(filepath, source, max_records)
        except Exception as e:
            logger.error("Error loading %s: %s", filepath, e)

        return []

    def _load_parquet(self, filepath: Path, source: str, max_records: int) -> List[ReviewRecord]:
        """Load from Parquet file."""
        import pandas as pd

        df = pd.read_parquet(filepath)
        return self._df_to_records(df, source, max_records)

    def _load_csv(self, filepath: Path, source: str, max_records: int) -> List[ReviewRecord]:
        """Load from CSV file."""
        import pandas as pd

        df = pd.read_csv(filepath, nrows=max_records)
        return self._df_to_records(df, source, max_records)

    def _load_json(self, filepath: Path, source: str, max_records: int) -> List[ReviewRecord]:
        """Load from JSON or JSONL file."""
        import json as _json

        records: List[ReviewRecord] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            # Try JSONL first
            first_line = fh.readline().strip()
            fh.seek(0)

            if first_line.startswith("["):
                # Regular JSON array
                data = _json.load(fh)
                if isinstance(data, list):
                    for item in data[:max_records]:
                        if self._schema_map is None:
                            self._schema_map = discover_schema(item)
                        records.append(
                            normalize_record(item, self._schema_map or {}, source=source)
                        )
            else:
                # JSONL
                for line in fh:
                    if len(records) >= max_records:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = _json.loads(line)
                        if self._schema_map is None:
                            self._schema_map = discover_schema(item)
                        records.append(
                            normalize_record(item, self._schema_map or {}, source=source)
                        )
                    except _json.JSONDecodeError:
                        continue

        return records

    def _df_to_records(
        self, df: Any, source: str, max_records: int
    ) -> List[ReviewRecord]:
        """Convert a pandas DataFrame to ReviewRecords."""
        if self._schema_map is None and len(df) > 0:
            self._schema_map = discover_schema(dict(df.iloc[0]))
            logger.info("Discovered schema: %s", self._schema_map)

        records: List[ReviewRecord] = []
        for _, row in tqdm(
            df.head(max_records).iterrows(),
            desc="Converting",
            total=min(max_records, len(df)),
        ):
            records.append(
                normalize_record(dict(row), self._schema_map or {}, source=source)
            )
        return records
