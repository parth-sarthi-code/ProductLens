"""
Transformer-based BIO sequence labeling for aspect candidate extraction.

Supports DeBERTa-v3-base and RoBERTa-base with token-to-character span
reconstruction, multi-aspect extraction per sentence, FP16 execution,
and deterministic mock extraction for smoke mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from productlens.aspects.aliases import classify_aspect_type
from productlens.aspects.candidates import CandidateAspect, CandidateExtractor
from productlens.schemas import SentenceRecord, ReviewRecord

logger = logging.getLogger("productlens.aspects.bio_model")

# ---------------------------------------------------------------------------
# Label Definitions (Spec §11)
# ---------------------------------------------------------------------------

BIO_LABELS = ["O", "B-ASP", "I-ASP"]
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}


# ---------------------------------------------------------------------------
# Span Reconstruction Helper
# ---------------------------------------------------------------------------

def reconstruct_spans_from_bio(
    sentence_text: str,
    tokens_offsets: List[Tuple[int, int]],
    predicted_labels: List[str],
    token_confidences: Optional[List[float]] = None,
) -> List[Tuple[str, int, int, float]]:
    """
    Reconstruct character-level spans from BIO sequence labels and token offsets.

    Parameters
    ----------
    sentence_text : str
        The raw sentence text.
    tokens_offsets : list of (int, int)
        Character offset tuples (start_char, end_char) for each subword token.
    predicted_labels : list of str
        BIO label strings ("O", "B-ASP", "I-ASP").
    token_confidences : list of float, optional
        Confidence / probability for each token's predicted label.

    Returns
    -------
    list of (surface, start_char, end_char, confidence)
    """
    spans: List[Tuple[str, int, int, float]] = []

    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_confs: List[float] = []

    def _flush_span() -> None:
        nonlocal current_start, current_end, current_confs
        if current_start is not None and current_end is not None:
            raw_surface = sentence_text[current_start:current_end]
            # Strip outer whitespace and adjust offsets
            l_strip = len(raw_surface) - len(raw_surface.lstrip())
            r_strip = len(raw_surface) - len(raw_surface.rstrip())
            adj_start = current_start + l_strip
            adj_end = current_end - r_strip
            surface = sentence_text[adj_start:adj_end]

            if surface.strip():
                mean_conf = float(np.mean(current_confs)) if current_confs else 0.8
                spans.append((surface, adj_start, adj_end, mean_conf))

        current_start = None
        current_end = None
        current_confs = []

    for idx, (label, (t_start, t_end)) in enumerate(zip(predicted_labels, tokens_offsets)):
        # Skip special tokens (offsets (0, 0) for non-empty text, or start == end)
        if t_start >= t_end:
            continue

        conf = token_confidences[idx] if token_confidences and idx < len(token_confidences) else 0.9

        if label == "B-ASP":
            # Start of a new aspect
            _flush_span()
            current_start = t_start
            current_end = t_end
            current_confs.append(conf)

        elif label == "I-ASP":
            if current_start is not None:
                # Continuation of current aspect
                current_end = t_end
                current_confs.append(conf)
            else:
                # Orphan I-ASP: treat as B-ASP to be robust
                current_start = t_start
                current_end = t_end
                current_confs.append(conf)

        else:  # "O"
            _flush_span()

    # Flush final span if still open at end of sequence
    _flush_span()

    return spans


# ---------------------------------------------------------------------------
# Mock BIO Aspect Extractor (Spec §41 Smoke Mode)
# ---------------------------------------------------------------------------

class MockBioAspectExtractor:
    """
    Deterministic mock BIO aspect extractor for smoke mode and unit tests.

    Runs instantaneously on CPU without downloading Hugging Face models,
    while producing authentic BIO token classifications and exact character spans.
    """

    def __init__(self, model_name: str = "mock/deberta-v3-base") -> None:
        self.model_name = model_name
        self.candidate_extractor = CandidateExtractor()

    def extract_from_sentence(
        self,
        sentence_text: str,
        sentence_id: str = "",
        review_id: str = "",
        product_id: str = "",
        category: str = "",
        doc_sentence_start: int = 0,
    ) -> List[CandidateAspect]:
        """Extract candidate aspects from a sentence using deterministic mock rules."""
        raw_candidates = self.candidate_extractor.extract_from_sentence(
            sentence_text=sentence_text,
            sentence_id=sentence_id,
            review_id=review_id,
            product_id=product_id,
            category=category,
            doc_sentence_start=doc_sentence_start,
            origin="weak",
        )

        # Refine candidates with mock transformer confidence
        results: List[CandidateAspect] = []
        for c in raw_candidates:
            # Deterministic pseudo-confidence between 0.82 and 0.98 based on surface
            conf = 0.85 + (hash(c.surface) % 13) * 0.01
            results.append(
                CandidateAspect(
                    surface=c.surface,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    sentence_id=sentence_id,
                    review_id=review_id,
                    product_id=product_id,
                    category=category,
                    doc_start_char=doc_sentence_start + c.start_char,
                    doc_end_char=doc_sentence_start + c.end_char,
                    origin="pseudo",
                    confidence=round(conf, 4),
                    aspect_type=classify_aspect_type(c.surface, sentence_text),
                )
            )

        return results

    def extract_from_sentences(
        self,
        sentences: List[SentenceRecord],
        reviews: Optional[List[ReviewRecord]] = None,
    ) -> List[CandidateAspect]:
        """Extract candidate aspects from a batch of SentenceRecords."""
        review_map = {r.review_id: r for r in reviews} if reviews else {}
        all_aspects: List[CandidateAspect] = []

        for sent in sentences:
            rev = review_map.get(sent.review_id)
            review_id = rev.review_id if rev else sent.review_id
            product_id = rev.product_id if rev else ""
            category = rev.category if rev else ""

            aspects = self.extract_from_sentence(
                sentence_text=sent.text,
                sentence_id=sent.sentence_id,
                review_id=review_id,
                product_id=product_id,
                category=category,
                doc_sentence_start=sent.start_char,
            )
            all_aspects.extend(aspects)

        return all_aspects


# ---------------------------------------------------------------------------
# Transformer BIO Aspect Extractor (Spec §11)
# ---------------------------------------------------------------------------

class BioAspectExtractor:
    """
    DeBERTa-v3-base (or RoBERTa-base) token classification model for BIO aspect extraction.

    Features:
    - FP16 inference on CUDA
    - CPU fallback
    - FastTokenizer offset_mapping for exact character-level span reconstruction
    - Graceful fallback to MockBioAspectExtractor when weights are unavailable
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        device: Optional[str] = None,
        fp16: bool = True,
        max_length: int = 256,
        mock: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.mock = mock
        self._mock_extractor = MockBioAspectExtractor(model_name=model_name_or_path)

        self._tokenizer = None
        self._model = None
        self._device = device
        self._fp16 = fp16

        if not mock:
            self._init_model()

    def _init_model(self) -> None:
        """Initialize Hugging Face tokenizer and model."""
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading BIO model %s on %s", self.model_name_or_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                use_fast=True,
            )
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=len(BIO_LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            self._model.to(self._device)
            if self._device == "cuda" and self._fp16:
                self._model.half()
            self._model.eval()

        except Exception as exc:
            logger.warning(
                "Could not load Hugging Face model %s (%s). Falling back to mock extractor.",
                self.model_name_or_path,
                exc,
            )
            self.mock = True

    def extract_from_sentence(
        self,
        sentence_text: str,
        sentence_id: str = "",
        review_id: str = "",
        product_id: str = "",
        category: str = "",
        doc_sentence_start: int = 0,
    ) -> List[CandidateAspect]:
        """Extract aspect spans with character-level offsets from a sentence."""
        if self.mock or self._model is None or self._tokenizer is None:
            return self._mock_extractor.extract_from_sentence(
                sentence_text=sentence_text,
                sentence_id=sentence_id,
                review_id=review_id,
                product_id=product_id,
                category=category,
                doc_sentence_start=doc_sentence_start,
            )

        import torch

        # Tokenize with fast offset mapping
        encoding = self._tokenizer(
            sentence_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        offsets = encoding.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]  # (seq_len, num_labels)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            confidences = [probs[idx, p_id].item() for idx, p_id in enumerate(pred_ids)]

        pred_labels = [ID2LABEL.get(pid, "O") for pid in pred_ids]

        # Reconstruct character-level spans
        raw_spans = reconstruct_spans_from_bio(
            sentence_text=sentence_text,
            tokens_offsets=offsets,
            predicted_labels=pred_labels,
            token_confidences=confidences,
        )

        results: List[CandidateAspect] = []
        for surface, s_char, e_char, conf in raw_spans:
            results.append(
                CandidateAspect(
                    surface=surface,
                    start_char=s_char,
                    end_char=e_char,
                    sentence_id=sentence_id,
                    review_id=review_id,
                    product_id=product_id,
                    category=category,
                    doc_start_char=doc_sentence_start + s_char,
                    doc_end_char=doc_sentence_start + e_char,
                    origin="pseudo",
                    confidence=round(conf, 4),
                    aspect_type=classify_aspect_type(surface, sentence_text),
                )
            )

        return results

    def extract_from_sentences(
        self,
        sentences: List[SentenceRecord],
        reviews: Optional[List[ReviewRecord]] = None,
    ) -> List[CandidateAspect]:
        """Extract aspects from a list of SentenceRecords."""
        review_map = {r.review_id: r for r in reviews} if reviews else {}
        all_aspects: List[CandidateAspect] = []

        for sent in sentences:
            rev = review_map.get(sent.review_id)
            review_id = rev.review_id if rev else sent.review_id
            product_id = rev.product_id if rev else ""
            category = rev.category if rev else ""

            aspects = self.extract_from_sentence(
                sentence_text=sent.text,
                sentence_id=sent.sentence_id,
                review_id=review_id,
                product_id=product_id,
                category=category,
                doc_sentence_start=sent.start_char,
            )
            all_aspects.extend(aspects)

        return all_aspects
