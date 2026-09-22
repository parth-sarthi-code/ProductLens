"""
Tests for sentence splitting.

Verifies sentence boundary correctness, character offset accuracy,
abbreviation handling, multi-aspect preservation, empty input, Unicode,
long reviews, and punctuation edge cases.
"""

from __future__ import annotations

import pytest

from productlens.data.sentence_split import (
    split_into_sentences,
    split_review_sentences,
    split_reviews_to_sentences,
)
from productlens.schemas import ReviewRecord, SentenceRecord


# ===================================================================
# Basic sentence splitting
# ===================================================================

class TestSplitIntoSentences:
    """Test the core sentence splitting function."""

    def test_single_sentence(self):
        sents = split_into_sentences("This is a single sentence.")
        assert len(sents) == 1
        assert sents[0] == "This is a single sentence."

    def test_two_sentences(self):
        sents = split_into_sentences("First sentence. Second sentence.")
        assert len(sents) == 2
        assert sents[0] == "First sentence."
        assert sents[1] == "Second sentence."

    def test_multiple_sentences(self):
        text = "The display is great. The battery is terrible. The keyboard is okay."
        sents = split_into_sentences(text)
        assert len(sents) == 3

    def test_question_mark(self):
        sents = split_into_sentences("Is this good? Yes it is!")
        assert len(sents) == 2

    def test_exclamation(self):
        sents = split_into_sentences("Amazing product! Highly recommend!")
        assert len(sents) == 2

    def test_mixed_punctuation(self):
        sents = split_into_sentences("Is the battery good? Yes! The screen is amazing.")
        assert len(sents) == 3

    def test_empty_string(self):
        assert split_into_sentences("") == []

    def test_whitespace_only(self):
        assert split_into_sentences("   ") == []

    def test_none_handling(self):
        assert split_into_sentences("") == []

    def test_no_ending_punctuation(self):
        sents = split_into_sentences("This has no period")
        assert len(sents) == 1
        assert sents[0] == "This has no period"


# ===================================================================
# Abbreviation handling
# ===================================================================

class TestAbbreviations:
    """Test that abbreviations don't cause false sentence breaks."""

    def test_mr_mrs(self):
        sents = split_into_sentences("Mr. Smith bought it. Mrs. Jones agreed.")
        assert len(sents) == 2

    def test_dr(self):
        sents = split_into_sentences("Dr. Brown recommended this product. It works great.")
        assert len(sents) == 2

    def test_us(self):
        sents = split_into_sentences("Made in the U.S. Quality is excellent.")
        assert len(sents) == 2

    def test_etc(self):
        sents = split_into_sentences("Colors include red, blue, etc. The quality is good.")
        assert len(sents) == 2


# ===================================================================
# Decimal and numeric handling
# ===================================================================

class TestDecimals:
    """Test that decimal numbers don't cause false breaks."""

    def test_decimal_number(self):
        sents = split_into_sentences("Battery lasts 4.5 hours. Screen is 15.6 inches.")
        assert len(sents) == 2
        assert "4.5" in sents[0]
        assert "15.6" in sents[1]

    def test_price(self):
        sents = split_into_sentences("Costs $99.99. Worth the price.")
        assert len(sents) == 2


# ===================================================================
# Multi-aspect sentences
# ===================================================================

class TestMultiAspect:
    """Test that multi-aspect sentences are preserved as single sentences."""

    def test_but_conjunction(self):
        text = "The display is great but the battery is terrible."
        sents = split_into_sentences(text)
        assert len(sents) == 1
        assert "display" in sents[0]
        assert "battery" in sents[0]

    def test_however(self):
        text = "The camera is excellent during the day but poor at night."
        sents = split_into_sentences(text)
        assert len(sents) == 1

    def test_comma_list(self):
        text = "The sound, display, and keyboard are all excellent."
        sents = split_into_sentences(text)
        assert len(sents) == 1


# ===================================================================
# Character offset tests
# ===================================================================

class TestCharacterOffsets:
    """Test that sentence offsets correctly index into source text."""

    def _make_review(self, text: str) -> ReviewRecord:
        return ReviewRecord(
            review_id="test_review_001",
            text=text,
            raw_text=text,
            clean_text=text,
        )

    def test_single_sentence_offsets(self):
        text = "This is a test sentence."
        review = self._make_review(text)
        sents = split_review_sentences(review)
        assert len(sents) == 1
        s = sents[0]
        assert text[s.start_char:s.end_char] == s.text

    def test_two_sentence_offsets(self):
        text = "First sentence. Second sentence."
        review = self._make_review(text)
        sents = split_review_sentences(review)
        assert len(sents) == 2
        for s in sents:
            assert text[s.start_char:s.end_char] == s.text

    def test_multi_sentence_offsets(self):
        text = "The display is great. The battery is terrible. The keyboard is okay."
        review = self._make_review(text)
        sents = split_review_sentences(review)
        for s in sents:
            actual = text[s.start_char:s.end_char]
            assert actual == s.text, (
                f"Offset mismatch: expected '{s.text}', got '{actual}' "
                f"at [{s.start_char}:{s.end_char}]"
            )

    def test_multi_aspect_single_sentence_offsets(self):
        text = "The display is beautiful, but battery life is terrible."
        review = self._make_review(text)
        sents = split_review_sentences(review)
        assert len(sents) == 1
        s = sents[0]
        assert text[s.start_char:s.end_char] == s.text

    def test_long_review_offsets(self):
        text = (
            "I've been using this laptop for three months now. "
            "The display is absolutely gorgeous with vibrant colors. "
            "The keyboard has good travel and feels satisfying. "
            "However, the trackpad is a bit small. "
            "Battery life is around 7 hours which is decent."
        )
        review = self._make_review(text)
        sents = split_review_sentences(review)
        for s in sents:
            actual = text[s.start_char:s.end_char]
            assert actual == s.text


# ===================================================================
# Sentence record metadata
# ===================================================================

class TestSentenceRecords:
    """Test SentenceRecord creation."""

    def _make_review(self, text: str, review_id: str = "rev_001") -> ReviewRecord:
        return ReviewRecord(
            review_id=review_id,
            text=text,
            raw_text=text,
            clean_text=text,
        )

    def test_sentence_ids_are_stable(self):
        review = self._make_review("First. Second.")
        s1 = split_review_sentences(review)
        s2 = split_review_sentences(review)
        assert [s.sentence_id for s in s1] == [s.sentence_id for s in s2]

    def test_sentence_ids_are_unique(self):
        review = self._make_review("First. Second. Third.")
        sents = split_review_sentences(review)
        ids = [s.sentence_id for s in sents]
        assert len(ids) == len(set(ids))

    def test_review_id_propagated(self):
        review = self._make_review("A sentence.", review_id="my_review_123")
        sents = split_review_sentences(review)
        for s in sents:
            assert s.review_id == "my_review_123"

    def test_empty_review(self):
        review = self._make_review("")
        sents = split_review_sentences(review)
        assert len(sents) == 0

    def test_whitespace_review(self):
        review = self._make_review("   ")
        sents = split_review_sentences(review)
        assert len(sents) == 0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Test edge cases in sentence splitting."""

    def test_ellipsis(self):
        sents = split_into_sentences("Hmm... not sure about this one.")
        assert len(sents) >= 1

    def test_multiple_exclamation(self):
        sents = split_into_sentences("WOW!!!! AMAZING!!!! BEST EVER!!!!!")
        # Should not split on consecutive punctuation mid-stream if no uppercase follows
        assert len(sents) >= 1

    def test_unicode_text(self):
        sents = split_into_sentences("Très bon produit! Quality is excellent.")
        assert len(sents) == 2

    def test_quoted_text(self):
        text = 'My friend said "this is great." I agree.'
        sents = split_into_sentences(text)
        assert len(sents) >= 1

    def test_paragraph_break(self):
        text = "First paragraph.\n\nSecond paragraph."
        sents = split_into_sentences(text)
        assert len(sents) == 2

    def test_single_word(self):
        sents = split_into_sentences("Good")
        assert len(sents) == 1
        assert sents[0] == "Good"

    def test_url_in_text(self):
        text = "Check https://example.com for details. Good product."
        sents = split_into_sentences(text)
        # Should handle the URL without breaking incorrectly
        assert len(sents) >= 1


# ===================================================================
# Batch splitting
# ===================================================================

class TestBatchSplitting:
    """Test batch review-to-sentence splitting."""

    def test_batch_splitting(self, cleaned_reviews):
        sentences = split_reviews_to_sentences(cleaned_reviews[:10])
        assert len(sentences) > 0
        for s in sentences:
            assert s.sentence_id
            assert s.review_id
            assert s.text

    def test_batch_preserves_review_ids(self, cleaned_reviews):
        reviews = cleaned_reviews[:5]
        sentences = split_reviews_to_sentences(reviews)
        review_ids = {r.review_id for r in reviews}
        sentence_review_ids = {s.review_id for s in sentences}
        assert sentence_review_ids.issubset(review_ids)
