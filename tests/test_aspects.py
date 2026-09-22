"""
Unit tests for aspect candidate extraction and BIO sequence labeling.

Spec §42: Test BIO reconstruction, multiple aspects, spans, subwords.
"""

from pathlib import Path
import pytest

from productlens.aspects.aliases import classify_aspect_type
from productlens.aspects.bio_model import (
    BioAspectExtractor,
    MockBioAspectExtractor,
    reconstruct_spans_from_bio,
)
from productlens.aspects.candidates import (
    CandidateAspect,
    CandidateExtractor,
    extract_candidates,
)
from productlens.aspects.train_extractor import (
    AspectExtractorTrainer,
    compute_bio_metrics,
    compute_span_metrics,
)
from productlens.config import load_config
from productlens.schemas import ReviewRecord, SentenceRecord


# ---------------------------------------------------------------------------
# Span Reconstruction Tests
# ---------------------------------------------------------------------------

def test_reconstruct_spans_single_aspect():
    """Test span reconstruction for a single aspect."""
    text = "The display is vibrant."
    # Tokens: ["The", "display", "is", "vibrant", "."]
    offsets = [(0, 3), (4, 11), (12, 14), (15, 22), (22, 23)]
    labels = ["O", "B-ASP", "O", "O", "O"]

    spans = reconstruct_spans_from_bio(text, offsets, labels)
    assert len(spans) == 1
    surface, start, end, conf = spans[0]
    assert surface == "display"
    assert start == 4
    assert end == 11
    assert text[start:end] == "display"


def test_reconstruct_spans_multiple_aspects():
    """Test reconstructing multiple aspects from the same sentence."""
    text = "The sound quality is excellent but the microphone is terrible."
    # "sound quality" (4, 17) -> B-ASP, I-ASP
    # "microphone" (39, 49) -> B-ASP
    offsets = [
        (0, 3),    # The
        (4, 9),    # sound
        (10, 17),  # quality
        (18, 20),  # is
        (21, 30),  # excellent
        (31, 34),  # but
        (35, 38),  # the
        (39, 49),  # microphone
        (50, 52),  # is
        (53, 61),  # terrible
        (61, 62),  # .
    ]
    labels = ["O", "B-ASP", "I-ASP", "O", "O", "O", "O", "B-ASP", "O", "O", "O"]

    spans = reconstruct_spans_from_bio(text, offsets, labels)
    assert len(spans) == 2

    assert spans[0][0] == "sound quality"
    assert spans[0][1] == 4
    assert spans[0][2] == 17
    assert text[spans[0][1]:spans[0][2]] == "sound quality"

    assert spans[1][0] == "microphone"
    assert spans[1][1] == 39
    assert spans[1][2] == 49
    assert text[spans[1][1]:spans[1][2]] == "microphone"


def test_reconstruct_spans_orphan_iasp():
    """Test that orphan I-ASP tokens are treated as a span rather than discarded."""
    text = "Great battery life."
    offsets = [(0, 5), (6, 13), (14, 18), (18, 19)]
    # Orphan I-ASP starting without B-ASP
    labels = ["O", "I-ASP", "I-ASP", "O"]

    spans = reconstruct_spans_from_bio(text, offsets, labels)
    assert len(spans) == 1
    assert spans[0][0] == "battery life"
    assert text[spans[0][1]:spans[0][2]] == "battery life"


def test_reconstruct_spans_no_aspects():
    """Test sentence with no aspect tokens."""
    text = "It was completely ordinary and uneventful."
    offsets = [(0, 2), (3, 6), (7, 17), (18, 21), (22, 32), (32, 33)]
    labels = ["O", "O", "O", "O", "O", "O"]

    spans = reconstruct_spans_from_bio(text, offsets, labels)
    assert len(spans) == 0


# ---------------------------------------------------------------------------
# Candidate Extractor Filtering & Offset Tests
# ---------------------------------------------------------------------------

def test_candidate_filtering_pronouns_and_stopwords():
    """Verify that candidate extractor filters pronouns, stopwords, and generic terms."""
    extractor = CandidateExtractor()

    # Pronouns & stopwords must be rejected
    assert not extractor.is_valid_candidate("it")
    assert not extractor.is_valid_candidate("they")
    assert not extractor.is_valid_candidate("the")
    assert not extractor.is_valid_candidate("and")
    assert not extractor.is_valid_candidate("this")

    # Standalone generic nouns must be rejected
    assert not extractor.is_valid_candidate("thing")
    assert not extractor.is_valid_candidate("product")
    assert not extractor.is_valid_candidate("item")
    assert not extractor.is_valid_candidate("stuff")

    # Standalone generic modifiers rejected by default
    assert not extractor.is_valid_candidate("quality")
    assert not extractor.is_valid_candidate("performance")

    # Valid compounds and domain nouns must pass
    assert extractor.is_valid_candidate("battery life")
    assert extractor.is_valid_candidate("sound quality")
    assert extractor.is_valid_candidate("build quality")
    assert extractor.is_valid_candidate("screen")
    assert extractor.is_valid_candidate("camera")


def test_candidate_extraction_offsets():
    """Verify exact character offset tracking in sentences and documents."""
    extractor = CandidateExtractor()
    sentence_text = "The bright screen and battery life are amazing."
    doc_start = 50

    candidates = extractor.extract_from_sentence(
        sentence_text=sentence_text,
        sentence_id="sent_123",
        review_id="rev_456",
        product_id="PROD_789",
        category="Electronics",
        doc_sentence_start=doc_start,
    )

    assert len(candidates) >= 2
    surfaces = [c.surface for c in candidates]
    assert any("screen" in s for s in surfaces)
    assert any("battery life" in s for s in surfaces)

    for c in candidates:
        # Check sentence slice
        assert sentence_text[c.start_char:c.end_char] == c.surface
        # Check document offset math
        assert c.doc_start_char == doc_start + c.start_char
        assert c.doc_end_char == doc_start + c.end_char
        # Check provenance
        assert c.sentence_id == "sent_123"
        assert c.review_id == "rev_456"
        assert c.product_id == "PROD_789"
        assert c.category == "Electronics"


# ---------------------------------------------------------------------------
# Mock Extractor & Multi-Aspect Sentence Handling
# ---------------------------------------------------------------------------

def test_mock_bio_extractor_multi_aspect():
    """Test MockBioAspectExtractor on multi-aspect sentence."""
    extractor = MockBioAspectExtractor()
    sent = SentenceRecord(
        sentence_id="s1",
        review_id="r1",
        text="The sound quality is great but the microphone is terrible.",
        start_char=10,
        end_char=68,
    )
    rev = ReviewRecord(
        review_id="r1",
        product_id="B000TEST",
        category="Electronics",
        clean_text="Prefix... The sound quality is great but the microphone is terrible.",
    )

    aspects = extractor.extract_from_sentences([sent], [rev])
    assert len(aspects) >= 2
    surfaces = [a.surface.lower() for a in aspects]
    assert any("sound quality" in s or "sound" in s for s in surfaces)
    assert any("microphone" in s for s in surfaces)

    for a in aspects:
        assert a.product_id == "B000TEST"
        assert a.review_id == "r1"
        assert a.sentence_id == "s1"
        assert a.confidence > 0.0
        assert sent.text[a.start_char:a.end_char] == a.surface


# ---------------------------------------------------------------------------
# Aspect Typing Tests (Spec §16)
# ---------------------------------------------------------------------------

def test_aspect_typing():
    """Test classification into product, service, delivery, seller, packaging, unknown."""
    assert classify_aspect_type("delivery speed") == "delivery"
    assert classify_aspect_type("shipping") == "delivery"
    assert classify_aspect_type("carrier") == "delivery"
    assert classify_aspect_type("bubble wrap") == "packaging"
    assert classify_aspect_type("box") == "packaging"
    assert classify_aspect_type("customer service") == "service"
    assert classify_aspect_type("warranty") == "service"
    assert classify_aspect_type("third-party seller") == "seller"
    assert classify_aspect_type("battery life") == "product"
    assert classify_aspect_type("display") == "product"

    # Contextual disambiguation
    assert classify_aspect_type("delivery", sentence_context="Amazon delivered it yesterday.") == "delivery"


# ---------------------------------------------------------------------------
# Evaluation Metrics Tests (Spec §13)
# ---------------------------------------------------------------------------

def test_compute_bio_metrics():
    """Test token-level BIO metrics computation."""
    true_labels = [["O", "B-ASP", "I-ASP", "O"]]
    pred_labels = [["O", "B-ASP", "I-ASP", "O"]]

    metrics = compute_bio_metrics(true_labels, pred_labels)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

    # With a mistake
    pred_labels_err = [["O", "B-ASP", "O", "O"]]
    metrics_err = compute_bio_metrics(true_labels, pred_labels_err)
    assert metrics_err["recall"] < 1.0


def test_compute_span_metrics():
    """Test exact and partial span matching F1."""
    true_spans = [[(4, 17), (35, 45)]]  # 2 gold spans
    pred_spans = [[(4, 17), (36, 45)]]  # 1 exact match, 1 partial match (36..45 inside 35..45)

    metrics = compute_span_metrics(true_spans, pred_spans)
    # Exact: 1 match out of 2 gold / 2 pred
    assert metrics["exact_span_f1"] == 0.5
    # Partial: both match with overlap!
    assert metrics["partial_span_f1"] == 1.0


# ---------------------------------------------------------------------------
# Isolation Rule Test
# ---------------------------------------------------------------------------

def test_trainer_rejects_synthetic_smoke_data():
    """Enforce hard isolation: synthetic smoke data must never enter research training."""
    cfg = load_config(profile="smoke")
    trainer = AspectExtractorTrainer(cfg, output_dir="artifacts/test_models")

    synthetic_data = [
        {"text": "Synthetic test review", "source": "synthetic", "spans": [(0, 9)]}
    ]

    with pytest.raises(ValueError, match="Hard isolation violation"):
        trainer.train(synthetic_data)


# ---------------------------------------------------------------------------
# Architecture & Product Association Tests
# ---------------------------------------------------------------------------

def test_roberta_configured_alternative():
    """Verify roberta-base is supported as configured alternative."""
    extractor = MockBioAspectExtractor(model_name="roberta-base")
    assert extractor.model_name == "roberta-base"
    aspects = extractor.extract_from_sentence(
        "The screen is bright.",
        sentence_id="s1",
        review_id="r1",
        product_id="P1",
    )
    assert len(aspects) >= 1
    assert aspects[0].surface == "screen"


def test_product_association_integrity():
    """
    Verify every aspect remains associated with source product/review.
    Do not aggregate identical aspect names across unrelated products.
    """
    extractor = MockBioAspectExtractor()
    sent1 = SentenceRecord(sentence_id="s1", review_id="r1", text="The battery is great.", start_char=0, end_char=21)
    rev1 = ReviewRecord(review_id="r1", product_id="PROD_A", category="Electronics", clean_text="The battery is great.")

    sent2 = SentenceRecord(sentence_id="s2", review_id="r2", text="The battery is dead.", start_char=0, end_char=20)
    rev2 = ReviewRecord(review_id="r2", product_id="PROD_B", category="Electronics", clean_text="The battery is dead.")

    aspects1 = extractor.extract_from_sentences([sent1], [rev1])
    aspects2 = extractor.extract_from_sentences([sent2], [rev2])

    assert len(aspects1) >= 1
    assert len(aspects2) >= 1

    # Same aspect surface "battery", but distinct products & reviews
    assert aspects1[0].surface == "battery"
    assert aspects2[0].surface == "battery"
    assert aspects1[0].product_id == "PROD_A"
    assert aspects2[0].product_id == "PROD_B"
    assert aspects1[0].review_id == "r1"
    assert aspects2[0].review_id == "r2"
    assert aspects1[0].product_id != aspects2[0].product_id


def test_no_train_test_contamination():
    """Verify aspect extraction respects Stage 1 split assignments without contamination."""
    from productlens.data.synthetic import generate_synthetic_reviews
    from productlens.data.clean import clean_reviews
    from productlens.data.split import split_reviews
    from productlens.data.sentence_split import split_reviews_to_sentences

    reviews = clean_reviews(generate_synthetic_reviews(seed=42))
    splits = split_reviews(reviews, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    train_reviews = splits["train"]
    test_reviews = splits["test"]

    train_pids = set(r.product_id for r in train_reviews)
    test_pids = set(r.product_id for r in test_reviews)

    # Disjoint products across splits
    assert len(train_pids & test_pids) == 0

    test_sentences = split_reviews_to_sentences(test_reviews)
    extractor = MockBioAspectExtractor()
    test_aspects = extractor.extract_from_sentences(test_sentences, test_reviews)

    # All extracted aspects from test set must only have test product IDs
    for asp in test_aspects:
        assert asp.product_id in test_pids
        assert asp.product_id not in train_pids


def test_exact_sentence_offsets_traceable():
    """Verify that 100% of extracted aspects trace back to exact sentence and document offsets."""
    from productlens.data.synthetic import generate_synthetic_reviews
    from productlens.data.clean import clean_reviews
    from productlens.data.sentence_split import split_reviews_to_sentences

    reviews = clean_reviews(generate_synthetic_reviews(seed=42))[:20]
    sentences = split_reviews_to_sentences(reviews)
    extractor = MockBioAspectExtractor()
    aspects = extractor.extract_from_sentences(sentences, reviews)

    rev_map = {r.review_id: r for r in reviews}
    sent_map = {s.sentence_id: s for s in sentences}

    assert len(aspects) > 0
    for asp in aspects:
        sent = sent_map[asp.sentence_id]
        rev = rev_map[asp.review_id]

        # Sentence offset match
        assert sent.text[asp.start_char:asp.end_char] == asp.surface
        # Document offset match
        assert rev.clean_text[asp.doc_start_char:asp.doc_end_char] == asp.surface


def test_stage2_smoke_pipeline_end_to_end():
    """Verify that Stage 2 pipeline executes cleanly and writes DONE.json."""
    from productlens.aspects.run import run_aspect_pipeline
    exit_code = run_aspect_pipeline(profile="smoke", mock=True)
    assert exit_code == 0
    done_file = Path("artifacts/aspects/DONE.json")
    assert done_file.is_file()
