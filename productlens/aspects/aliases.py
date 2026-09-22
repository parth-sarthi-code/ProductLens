"""
Aspect alias resolution and typing.

Provides deterministic alias mapping, category-conditioned alias overrides,
prevention of invalid merges (e.g. 'screen' vs 'screen protector', shared generic words),
and aspect typing (product, service, delivery, seller, packaging, unknown).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


# ---------------------------------------------------------------------------
# Generic Words Guard
# ---------------------------------------------------------------------------

# Aspects must NEVER be merged solely because they share one of these generic words.
GENERIC_MODIFIERS: Set[str] = {
    "quality", "performance", "comfort", "durability", "design", "speed",
    "power", "level", "value", "feature", "feel", "look", "material",
    "size", "weight", "mode", "option", "rate", "efficiency", "sound",
}

# ---------------------------------------------------------------------------
# Built-in Canonical Aliases (Domain-Agnostic & Category-Aware)
# ---------------------------------------------------------------------------

# Deterministic alias dictionary: surface_term -> canonical_alias
DEFAULT_ALIASES: Dict[str, str] = {
    # Display / Screen synonyms
    "display": "screen",
    "screen": "screen",
    "monitor": "screen",
    "panel": "screen",
    "lcd": "screen",
    "oled": "screen",
    "touchscreen": "touchscreen",
    # Power
    "battery life": "battery",
    "battery backup": "battery",
    "runtime": "battery",
    "battery": "battery",
    "charging speed": "charging",
    "charger": "charger",
    # Audio
    "sound": "audio",
    "sound quality": "audio quality",
    "audio": "audio",
    "speakers": "speaker",
    "speaker": "speaker",
    "microphone": "microphone",
    "mic": "microphone",
    # Input
    "keyboard": "keyboard",
    "keys": "keyboard",
    "key travel": "keyboard",
    "trackpad": "trackpad",
    "touchpad": "trackpad",
    "mouse": "mouse",
    # Construction / Form
    "build": "build quality",
    "build quality": "build quality",
    "construction": "build quality",
    "craftsmanship": "build quality",
    # Comfort / Fit (wearables, clothing, furniture)
    "comfort": "comfort",
    "ergonomics": "ergonomics",
    "fit": "fit",
    "sizing": "fit",
    # Customer interactions
    "customer support": "customer service",
    "customer service": "customer service",
    "support": "customer service",
    "warranty": "warranty",
    "return policy": "returns",
    # Shipping
    "delivery speed": "shipping",
    "shipping": "shipping",
    "delivery": "shipping",
    # Packaging
    "box": "packaging",
    "packaging": "packaging",
    "unboxing": "packaging",
}

# Disallowed merges: pairs of terms that must never be merged even if embedding is close
DISALLOWED_MERGES: Set[Tuple[str, str]] = {
    ("screen", "screen protector"),
    ("screen protector", "screen"),
    ("phone", "phone case"),
    ("phone case", "phone"),
    ("sound quality", "build quality"),
    ("build quality", "sound quality"),
    ("battery performance", "gaming performance"),
    ("camera", "camera lens protector"),
    ("laptop", "laptop sleeve"),
}


# ---------------------------------------------------------------------------
# Aspect Typing
# ---------------------------------------------------------------------------

ASPECT_TYPES: Set[str] = {
    "product",
    "service",
    "delivery",
    "seller",
    "packaging",
    "unknown",
}

_DELIVERY_KEYWORDS: Set[str] = {
    "delivery", "shipping", "shipped", "shipment", "carrier", "ups", "fedex",
    "usps", "dhl", "transit", "delivered", "arrived", "arrival", "courier",
    "tracking", "package transit",
}

_PACKAGING_KEYWORDS: Set[str] = {
    "packaging", "box", "package", "bubble wrap", "unboxing", "container",
    "wrapped", "wrapping", "seal", "packaged",
}

_SERVICE_KEYWORDS: Set[str] = {
    "customer service", "customer support", "support", "warranty", "refund",
    "return", "replacement", "representative", "helpdesk", "call center",
    "repair", "policy", "guarantee",
}

_SELLER_KEYWORDS: Set[str] = {
    "seller", "merchant", "vendor", "store", "dealer", "dealership",
    "third-party seller", "retailer", "fulfillment",
}


def classify_aspect_type(surface: str, sentence_context: str = "") -> str:
    """
    Classify an aspect mention into one of the supported aspect types:
    product, service, delivery, seller, packaging, unknown.

    Parameters
    ----------
    surface : str
        The extracted aspect surface string.
    sentence_context : str, optional
        The surrounding sentence context.

    Returns
    -------
    str : One of the 6 canonical aspect types.
    """
    s = surface.strip().lower()
    ctx = sentence_context.strip().lower()

    # Direct surface matches have highest priority
    if any(k in s for k in _DELIVERY_KEYWORDS):
        return "delivery"
    if any(k in s for k in _PACKAGING_KEYWORDS):
        return "packaging"
    if any(k in s for k in _SERVICE_KEYWORDS):
        return "service"
    if any(k in s for k in _SELLER_KEYWORDS):
        return "seller"

    # Contextual clues
    if ctx:
        if any(f"amazon {k}" in ctx for k in ["delivered", "shipped", "arrived", "delivery"]):
            return "delivery"
        if any(w in ctx for w in ["arrived damaged in box", "opened the packaging", "poor packaging"]):
            if any(w in s for w in ["box", "package", "wrapper", "container"]):
                return "packaging"
        if any(w in ctx for w in ["contacted support", "customer care", "warranty claim"]):
            return "service"
        if any(w in ctx for w in ["seller responded", "seller refunded", "shady seller"]):
            return "seller"

    # Default to product
    return "product"


# ---------------------------------------------------------------------------
# Alias Resolver
# ---------------------------------------------------------------------------

class AliasResolver:
    """
    Deterministic aspect alias resolver with override support.

    Maintains alias mappings separately from model-generated clusters,
    prevents spurious merges across semantically distinct entities,
    and supports YAML override files.
    """

    def __init__(
        self,
        override_path: Optional[str] = None,
        custom_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.aliases: Dict[str, str] = dict(DEFAULT_ALIASES)
        self.disallowed_pairs: Set[Tuple[str, str]] = set(DISALLOWED_MERGES)
        self.override_path = override_path

        if custom_aliases:
            self.aliases.update({k.lower().strip(): v.lower().strip() for k, v in custom_aliases.items()})

        if override_path:
            self.load_overrides(override_path)

    def load_overrides(self, path: str) -> None:
        """Load manual alias overrides from a YAML file."""
        p = Path(path)
        if not p.is_file():
            return
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    self.aliases[k.lower().strip()] = v.lower().strip()

    def save_overrides(self, path: str) -> None:
        """Write current alias mappings to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            yaml.dump(self.aliases, fh, default_flow_style=False)

    def can_merge(self, term_a: str, term_b: str) -> bool:
        """
        Check whether two aspect terms are legally allowed to merge.

        Guarantees:
        - Exact opposites / distinct modifiers (e.g. 'screen' vs 'screen protector') cannot merge.
        - Shared generic words (e.g. 'quality' in 'sound quality' vs 'build quality') cannot merge.
        """
        a = term_a.strip().lower()
        b = term_b.strip().lower()

        if a == b:
            return True

        if (a, b) in self.disallowed_pairs or (b, a) in self.disallowed_pairs:
            return False

        words_a = set(a.split())
        words_b = set(b.split())

        # If one term is a modifier of an accessory/protector, don't merge with the base
        accessories = {"protector", "cover", "case", "sleeve", "adapter", "cable", "stand"}
        has_acc_a = bool(words_a & accessories)
        has_acc_b = bool(words_b & accessories)
        if has_acc_a != has_acc_b:
            return False

        # If they only overlap on a generic modifier (e.g., 'quality', 'performance'), do not merge
        overlap = words_a & words_b
        if overlap and overlap.issubset(GENERIC_MODIFIERS):
            return False

        return True

    def resolve(
        self,
        surface: str,
        category: str = "",
        sentence_context: str = "",
    ) -> str:
        """
        Resolve an aspect surface mention to its canonical alias.

        Parameters
        ----------
        surface : str
            The raw or candidate aspect surface string.
        category : str, optional
            Product category context for domain disambiguation.
        sentence_context : str, optional
            Surrounding sentence text.

        Returns
        -------
        str : The canonical normalized aspect name.
        """
        s = surface.strip().lower()

        # Check explicit alias dictionary
        if s in self.aliases:
            return self.aliases[s]

        # Basic lemmatization / inflection normalization
        # e.g., 'keyboards' -> 'keyboard', 'speakers' -> 'speaker'
        if s.endswith("ies") and len(s) > 4:
            stemmed = s[:-3] + "y"
            if stemmed in self.aliases:
                return self.aliases[stemmed]
            return stemmed
        if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
            stemmed = s[:-1]
            if stemmed in self.aliases:
                return self.aliases[stemmed]
            return stemmed

        return s
