"""
ProductLens utility functions.

Provides logging setup, seed management, device detection, library version
printing, stable ID generation, GPU cleanup, timer, DONE.json management,
and safe I/O helpers.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("productlens")

# ---------------------------------------------------------------------------
# Library versions — printed at startup
# ---------------------------------------------------------------------------

_REQUIRED_LIBS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyyaml", "yaml"),
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
]

_OPTIONAL_LIBS = [
    ("hdbscan", "hdbscan"),
    ("sentence-transformers", "sentence_transformers"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pyarrow", "pyarrow"),
    ("spacy", "spacy"),
]


def get_library_versions() -> Dict[str, str]:
    """Return a dict of library name → version for installed packages."""
    versions: Dict[str, str] = {}
    versions["python"] = sys.version.split()[0]
    for name, module_name in _REQUIRED_LIBS + _OPTIONAL_LIBS:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "unknown")
            versions[name] = str(ver)
        except ImportError:
            versions[name] = "not installed"
    return versions


def print_library_versions() -> Dict[str, str]:
    """Print important library versions and return them."""
    versions = get_library_versions()
    logger.info("=== Library Versions ===")
    for name, ver in versions.items():
        logger.info("  %-25s %s", name, ver)
    return versions


def check_required_libraries() -> None:
    """Fail clearly if critical libraries are missing."""
    missing = []
    for name, module_name in _REQUIRED_LIBS:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            f"Missing required libraries: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    artifact_root: str = "artifacts",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging to console and optionally to a log file.

    Parameters
    ----------
    artifact_root : str
        Root directory for artifacts. Log files go under ``<artifact_root>/logs/``.
    level : int
        Logging level.
    log_file : str, optional
        Explicit log file name. If None, a timestamped name is generated.
    """
    root_logger = logging.getLogger("productlens")
    root_logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler
    log_dir = Path(artifact_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"productlens_{ts}.log"
    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(preferred: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect available compute device and return device info dict.

    Parameters
    ----------
    preferred : str, optional
        Force a specific device (``cpu``, ``cuda``, ``cuda:0``, etc.).

    Returns
    -------
    dict
        Keys: ``device``, ``gpu_name``, ``vram_mb``, ``cuda_version``,
        ``gpu_available``.
    """
    info: Dict[str, Any] = {
        "device": "cpu",
        "gpu_name": None,
        "vram_mb": 0,
        "cuda_version": None,
        "gpu_available": False,
    }

    try:
        import torch

        if preferred and preferred != "auto":
            info["device"] = preferred
        elif torch.cuda.is_available():
            info["device"] = "cuda"

        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    logger.info("Device: %s", info["device"])
    if info["gpu_available"]:
        logger.info(
            "GPU: %s | VRAM: %d MB | CUDA: %s",
            info["gpu_name"],
            info["vram_mb"],
            info["cuda_version"],
        )
    return info


# ---------------------------------------------------------------------------
# Stable ID generation
# ---------------------------------------------------------------------------

def stable_id(*parts: Any) -> str:
    """
    Generate a deterministic SHA-256 hex ID from the given parts.

    Parameters
    ----------
    *parts
        Values to hash. Each is converted to string and joined with ``|``.

    Returns
    -------
    str
        64-character hex digest.
    """
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Return SHA-256 hex digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GPU memory cleanup
# ---------------------------------------------------------------------------

def cleanup_gpu() -> None:
    """Release GPU memory and trigger garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = ""):
    """Context manager that logs elapsed time."""
    start = time.perf_counter()
    logger.info("⏱ START: %s", label or "timer")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("⏱ DONE: %s — %.2f seconds", label or "timer", elapsed)


# ---------------------------------------------------------------------------
# DONE.json management
# ---------------------------------------------------------------------------

def write_done_marker(
    stage: str,
    artifact_dir: str,
    config_hash_val: str = "",
    input_hash: str = "",
    seconds: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a ``DONE.json`` marker indicating a completed stage.

    Parameters
    ----------
    stage : str
        Stage name.
    artifact_dir : str
        Directory to write into.
    config_hash_val : str
        Hash of the configuration used.
    input_hash : str
        Hash of the input data used.
    seconds : float
        Runtime in seconds.
    extra : dict, optional
        Additional metadata to include.

    Returns
    -------
    Path
        Path to the created DONE.json file.
    """
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "stage": stage,
        "config_hash": config_hash_val,
        "library_versions": get_library_versions(),
        "input_hash": input_hash,
        "seconds": round(seconds, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }
    if extra:
        marker.update(extra)
    path = out_dir / "DONE.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2, default=str)
    return path


def read_done_marker(artifact_dir: str) -> Optional[Dict[str, Any]]:
    """Read a ``DONE.json`` marker if it exists."""
    path = Path(artifact_dir) / "DONE.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def stage_is_done(artifact_dir: str, config_hash_val: str = "") -> bool:
    """
    Check if a stage has completed successfully with matching config.

    If *config_hash_val* is empty, any successful DONE.json matches.
    """
    marker = read_done_marker(artifact_dir)
    if marker is None or marker.get("status") != "success":
        return False
    if config_hash_val and marker.get("config_hash") != config_hash_val:
        return False
    return True


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def log_processing_error(
    error_dir: str,
    record_id: str,
    stage: str,
    error_type: str,
    message: str,
) -> None:
    """Append a processing error to the error log for the given stage."""
    err_path = Path(error_dir)
    err_path.mkdir(parents=True, exist_ok=True)
    entry = {
        "record_id": record_id,
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file = err_path / f"{stage}_errors.jsonl"
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Safe I/O
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json_dump(data: Any, path: str | Path) -> None:
    """Write JSON to file, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def safe_json_load(path: str | Path) -> Any:
    """Read JSON from file, returning None if file doesn't exist."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)
