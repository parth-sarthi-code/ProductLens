"""
Shared test fixtures for ProductLens.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Ensure productlens is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from productlens.config import load_config, ProductLensConfig
from productlens.data.synthetic import generate_synthetic_reviews
from productlens.data.clean import clean_reviews
from productlens.schemas import ReviewRecord


@pytest.fixture(scope="session")
def smoke_config() -> ProductLensConfig:
    """Load smoke profile configuration."""
    return load_config(profile="smoke")


@pytest.fixture(scope="session")
def default_config() -> ProductLensConfig:
    """Load default configuration."""
    return load_config()


@pytest.fixture(scope="session")
def synthetic_reviews() -> List[ReviewRecord]:
    """Generate synthetic reviews (deterministic, seed=42)."""
    return generate_synthetic_reviews(seed=42)


@pytest.fixture(scope="session")
def cleaned_reviews(synthetic_reviews: List[ReviewRecord]) -> List[ReviewRecord]:
    """Cleaned synthetic reviews (short/empty filtered out)."""
    return clean_reviews(synthetic_reviews, min_chars=10)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
