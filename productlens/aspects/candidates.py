"""
Candidate aspect extraction.

Extracts candidate aspect mentions from sentences using noun phrases,
compound nouns, and syntactic patterns, while filtering generic terms,
pronouns, and stopwords. Supports gold, weak, and pseudo-labeled candidate provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from productlens.schemas import SentenceRecord, ReviewRecord


# ---------------------------------------------------------------------------
# Filter Vocabularies (Domain-Agnostic)
# ---------------------------------------------------------------------------

_PRONOUNS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "nobody", "nothing", "none",
}

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "also", "really", "even",
}

_GENERIC_NOUNS: Set[str] = {
    "thing", "things", "stuff", "item", "items", "product", "products",
    "everything", "nothing", "something", "anything", "aspect", "aspects",
    "feature", "features", "lot", "bit", "way", "part", "parts", "side",
    "point", "points", "time", "times", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "one", "ones", "people", "person",
    "case", "cases", "use", "uses", "review", "reviews", "star", "stars",
    "purchase", "purchases", "buy", "order", "orders", "price", "money",
    "amount", "deal", "reason", "reasons", "problem", "problems", "issue",
    "issues", "experience", "factor", "piece", "pieces", "model", "models",
    "unit", "units", "version", "versions", "brand", "brands", "type", "types",
}

# Generic words that cannot stand alone as an aspect (e.g. "quality" alone is too vague)
_STANDALONE_GENERIC: Set[str] = {
    "quality", "performance", "comfort", "durability", "feel", "look", "looks",
    "size", "weight", "design", "style", "function", "value", "speed",
    "appearance", "fit", "work", "action", "behavior", "nature",
}

# Verbs, copulas, and conjunctions that break noun phrase boundaries
_COPULAS_AND_CONJUNCTIONS: Set[str] = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "but", "however", "although", "though", "while",
    "whereas", "because", "since", "so", "and", "or", "feel", "feels", "felt",
    "works", "worked", "working", "seems", "seemed", "looked", "looks", "sounded",
    "sounds", "left", "bought", "got", "get", "gets", "ordered", "delivered",
}

# Pure sentiment / opinion words that cannot be the head of an aspect
_OPINION_WORDS: Set[str] = {
    "good", "bad", "great", "terrible", "awful", "excellent", "poor", "amazing",
    "horrible", "wonderful", "fantastic", "decent", "superb", "fine", "mediocre",
    "subpar", "outstanding", "impressive", "unimpressive", "disappointing", "pleasing",
    "unacceptable", "satisfactory", "unsatisfactory", "exceptional", "abysmal",
    "flawless", "dreadful", "gorgeous", "lovely", "surprising", "surprisingly",
    "ordinary", "uneventful", "happier", "happy", "unhappy", "perfect",
}

# Common POS / syntactic noun phrase patterns
# e.g., "battery life", "noise cancellation", "customer support", "build quality"
_NP_PATTERN = re.compile(
    r"\b(?:[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?\s+){0,3}[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?\b"
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateAspect:
    """
    Extracted candidate aspect mention.

    Attributes
    ----------
    surface : str
        The exact surface mention text.
    start_char : int
        Start character offset within the source sentence.
    end_char : int
        End character offset within the source sentence.
    sentence_id : str
        Parent sentence stable ID.
    review_id : str
        Parent review stable ID.
    product_id : str
        Parent product ID (ASIN).
    category : str
        Product category.
    doc_start_char : int
        Start character offset within parent review clean_text.
    doc_end_char : int
        End character offset within parent review clean_text.
    origin : str
        Label provenance: 'gold', 'weak', or 'pseudo'.
    confidence : float
        Candidate extraction confidence score (0.0–1.0).
    aspect_type : str
        Type: 'product', 'service', 'delivery', 'seller', 'packaging', 'unknown'.
    """

    surface: str
    start_char: int
    end_char: int
    sentence_id: str = ""
    review_id: str = ""
    product_id: str = ""
    category: str = ""
    doc_start_char: int = 0
    doc_end_char: int = 0
    origin: str = "weak"
    confidence: float = 0.5
    aspect_type: str = "product"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateAspect":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Candidate Extractor
# ---------------------------------------------------------------------------

class CandidateExtractor:
    """
    Rule and syntactic candidate aspect extractor.

    Supports noun compound extraction, pattern matching, stopword/generic filtering,
    and provenance tagging ('gold', 'weak', 'pseudo').
    """

    def __init__(
        self,
        min_chars: int = 2,
        max_words: int = 4,
        allow_standalone_generics: bool = False,
    ) -> None:
        self.min_chars = min_chars
        self.max_words = max_words
        self.allow_standalone_generics = allow_standalone_generics

    def is_valid_candidate(self, text: str) -> bool:
        """
        Check whether a candidate surface string passes filtering rules.

        Rejects pronouns, stopwords, generic nouns, standalone generics,
        and single-character or punctuation tokens.
        """
        clean = text.strip().lower()
        if len(clean) < self.min_chars:
            return False

        # Reject purely non-alphanumeric
        if not any(c.isalnum() for c in clean):
            return False

        words = clean.split()
        if len(words) > self.max_words:
            return False

        # Reject pure pronouns or stopwords
        if len(words) == 1 and (words[0] in _PRONOUNS or words[0] in _STOPWORDS):
            return False

        # Reject pure opinion words (e.g. 'excellent', 'terrible')
        if len(words) == 1 and words[0] in _OPINION_WORDS:
            return False

        # Reject standalone generic nouns
        if len(words) == 1 and words[0] in _GENERIC_NOUNS:
            return False

        if not self.allow_standalone_generics and len(words) == 1 and words[0] in _STANDALONE_GENERIC:
            return False

        # Multi-word checks:
        # A valid aspect phrase should not contain conjunctions or copulas
        if any(w in _COPULAS_AND_CONJUNCTIONS for w in words):
            return False

        # A valid aspect phrase should not end with an opinion/predicate word
        if words[-1] in _OPINION_WORDS:
            return False

        # Reject if leading or trailing words are stopwords/pronouns
        if words[0] in _STOPWORDS or words[0] in _PRONOUNS:
            return False
        if words[-1] in _STOPWORDS or words[-1] in _PRONOUNS:
            return False

        return True

    def extract_from_sentence(
        self,
        sentence_text: str,
        sentence_id: str = "",
        review_id: str = "",
        product_id: str = "",
        category: str = "",
        doc_sentence_start: int = 0,
        origin: str = "weak",
    ) -> List[CandidateAspect]:
        """
        Extract candidate aspects from a single sentence string.

        Finds candidate noun phrases and compounds with exact character offsets.
        """
        candidates: List[CandidateAspect] = []
        seen_spans: Set[Tuple[int, int]] = set()

        # Regex patterns targeting noun phrases & compound mentions
        # Match sequences of capitalized or lowercase nouns/adjectives leading to a noun
        # Common syntactic patterns:
        # 1. Compound nouns: "battery life", "sound quality", "customer support"
        # 2. Adjective + noun: "bright display", "fast processor", "poor packaging"
        # 3. Single nouns (domain terms): "screen", "keyboard", "camera", "bass", "zipper"
        token_pattern = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?\b")
        tokens_with_spans = [
            (m.group(), m.start(), m.end())
            for m in token_pattern.finditer(sentence_text)
        ]

        n = len(tokens_with_spans)
        # Scan n-grams of lengths up to max_words
        for span_len in range(self.max_words, 0, -1):
            for i in range(n - span_len + 1):
                start_tok = tokens_with_spans[i]
                end_tok = tokens_with_spans[i + span_len - 1]

                start_idx = start_tok[1]
                end_idx = end_tok[2]

                # Check if this span is already covered by a longer valid candidate
                if any(s <= start_idx and e >= end_idx for (s, e) in seen_spans):
                    continue

                surface = sentence_text[start_idx:end_idx]

                if self.is_valid_candidate(surface):
                    seen_spans.add((start_idx, end_idx))
                    candidates.append(
                        CandidateAspect(
                            surface=surface,
                            start_char=start_idx,
                            end_char=end_idx,
                            sentence_id=sentence_id,
                            review_id=review_id,
                            product_id=product_id,
                            category=category,
                            doc_start_char=doc_sentence_start + start_idx,
                            doc_end_char=doc_sentence_start + end_idx,
                            origin=origin,
                            confidence=0.7 if span_len > 1 else 0.5,
                            aspect_type=self._heuristic_type(surface, sentence_text),
                        )
                    )

        # Sort by start_char
        candidates.sort(key=lambda c: c.start_char)
        return candidates

    def extract_from_sentence_record(
        self,
        sentence: SentenceRecord,
        review: Optional[ReviewRecord] = None,
        origin: str = "weak",
    ) -> List[CandidateAspect]:
        """Extract candidates from a SentenceRecord."""
        review_id = review.review_id if review else sentence.review_id
        product_id = review.product_id if review else ""
        category = review.category if review else ""

        return self.extract_from_sentence(
            sentence_text=sentence.text,
            sentence_id=sentence.sentence_id,
            review_id=review_id,
            product_id=product_id,
            category=category,
            doc_sentence_start=sentence.start_char,
            origin=origin,
        )

    def _heuristic_type(self, surface: str, context: str) -> str:
        """Classify aspect type into product, service, delivery, seller, packaging, unknown."""
        s = surface.lower()
        ctx = context.lower()

        delivery_terms = {"delivery", "shipping", "shipped", "arrived", "carrier", "ups", "fedex", "usps", "transit"}
        packaging_terms = {"packaging", "box", "package", "wrapped", "wrapping", "unboxing", "bubble wrap", "container"}
        service_terms = {"customer service", "support", "warranty", "refund", "return", "replacement", "representative"}
        seller_terms = {"seller", "merchant", "vendor", "store", "dealership"}

        if any(w in s for w in delivery_terms) or any(w in ctx for w in ["delivered", "arrived late", "shipping was"]):
            if any(w in s for w in delivery_terms):
                return "delivery"
        if any(w in s for w in packaging_terms):
            return "packaging"
        if any(w in s for w in service_terms):
            return "service"
        if any(w in s for w in seller_terms):
            return "seller"

        return "product"


def extract_candidates(
    sentences: List[SentenceRecord],
    reviews: Optional[List[ReviewRecord]] = None,
    origin: str = "weak",
) -> List[CandidateAspect]:
    """Convenience function to extract candidate aspects from a list of sentences."""
    extractor = CandidateExtractor()
    review_map = {r.review_id: r for r in reviews} if reviews else {}

    all_candidates: List[CandidateAspect] = []
    for sentence in sentences:
        rev = review_map.get(sentence.review_id)
        candidates = extractor.extract_from_sentence_record(sentence, rev, origin=origin)
        all_candidates.extend(candidates)
    return all_candidates
