"""
Tests for span selection module.
"""

import pytest
import numpy as np
from framing_audit.spans import SpanSelector, Span


class TestSpanSelector:
    """Test cases for SpanSelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = SpanSelector(
            min_span_length=2,
            max_span_length=5,
            rarity_weight=0.5,
            pmi_weight=0.3,
            ablation_weight=0.2,
            frequency_threshold=0.1
        )
        
    def test_tokenize(self):
        """Test tokenization."""
        text = "Hello, world! This is a test."
        tokens = self.selector.tokenize(text)
        
        assert tokens == ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
        
    def test_build_background_frequencies(self):
        """Test background frequency building."""
        corpus = [
            "The cat sat on the mat",
            "The dog sat on the floor",
            "The cat and dog played"
        ]
        
        self.selector.build_background_frequencies(corpus)
        
        # Check unigram counts
        assert self.selector.unigram_counts['the'] == 6
        assert self.selector.unigram_counts['cat'] == 2
        assert self.selector.unigram_counts['dog'] == 2
        assert self.selector.unigram_counts['sat'] == 2
        
        # Check bigram counts
        assert self.selector.bigram_counts[('the', 'cat')] == 2
        assert self.selector.bigram_counts[('sat', 'on')] == 2
        
    def test_get_rarity_score(self):
        """Test rarity score calculation."""
        # Build simple background
        self.selector.unigram_counts = {'the': 10, 'cat': 2, 'rare': 1}
        self.selector.total_tokens = 20
        
        # Common tokens should have low rarity
        common_score = self.selector.get_rarity_score(['the', 'cat'])
        
        # Rare tokens should have high rarity
        rare_score = self.selector.get_rarity_score(['rare', 'word'])
        
        assert rare_score > common_score
        
    def test_get_pmi_score(self):
        """Test PMI score calculation."""
        # Set up counts for PMI calculation
        self.selector.unigram_counts = {'new': 10, 'york': 10, 'city': 10}
        self.selector.bigram_counts = {('new', 'york'): 8, ('york', 'city'): 2}
        self.selector.total_tokens = 100
        
        # High PMI for common bigram
        high_pmi = self.selector.get_pmi_score(['new', 'york'])
        
        # Lower PMI for less common bigram
        low_pmi = self.selector.get_pmi_score(['york', 'city'])
        
        assert high_pmi > low_pmi
        
    def test_get_ablation_score(self):
        """Test ablation score calculation."""
        original = "The policy has significant impact on the economy"
        ablated = "The measure has moderate effect on the economy"
        
        score = self.selector.get_ablation_score(original, ablated, method="rouge")
        
        # Should detect some drift
        assert 0 < score < 1
        
    def test_extract_candidate_spans(self):
        """Test candidate span extraction."""
        text = "The quick brown fox jumps"
        spans = self.selector.extract_candidate_spans(text)
        
        # Check we get spans of correct lengths
        span_texts = [s[0] for s in spans]
        
        # Should include 2-token spans
        assert "the quick" in span_texts
        assert "quick brown" in span_texts
        
        # Should include up to 5-token spans
        assert "the quick brown fox jumps" in span_texts
        
    def test_filter_overlapping_spans(self):
        """Test overlapping span filtering."""
        spans = [
            Span("the quick", 0, 9, combined_score=0.9),
            Span("quick brown", 4, 15, combined_score=0.8),
            Span("brown fox", 10, 19, combined_score=0.7),
            Span("fox jumps", 16, 25, combined_score=0.6)
        ]
        
        filtered = self.selector.filter_overlapping_spans(spans, top_k=3)
        
        # Should get non-overlapping spans in score order
        assert len(filtered) == 3
        assert filtered[0].text == "the quick"
        assert filtered[1].text == "brown fox"  # Non-overlapping with first
        assert filtered[2].text == "fox jumps"  # Non-overlapping with previous
        
    def test_select_spans_integration(self):
        """Test full span selection pipeline."""
        # Build background corpus
        corpus = [
            "The climate change affects global temperatures",
            "Climate policy requires immediate action",
            "Global warming threatens ecosystems"
        ]
        self.selector.build_background_frequencies(corpus)
        
        # Test text with some rare phrases
        text = "The climate hoax narrative undermines scientific consensus on global warming"
        
        spans = self.selector.select_spans(text, top_k=3)
        
        # Should return requested number of spans
        assert len(spans) <= 3
        
        # All spans should have scores
        for span in spans:
            assert span.rarity_score >= 0
            assert span.pmi_score >= 0
            assert span.combined_score >= 0
            
        # Spans should be non-overlapping
        positions = set()
        for span in spans:
            span_positions = set(range(span.start, span.end))
            assert not (span_positions & positions)
            positions.update(span_positions)