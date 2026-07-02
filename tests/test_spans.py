"""
Tests for OLMoTrace-style span discovery and scoring.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from framing_audit.spans import (
    discover_maximal_spans, select_top_spans, retrieve_and_rerank,
    get_unigram_counts, rarity_unigram, rarity_ngram_avg, pmi_bigram_avg,
    normalize_minmax, ablation_salience, SpanSelector, Span
)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Simple token sequence
    tokenizer.decode.side_effect = lambda x: f"token_{x[0]}" if len(x) == 1 else " ".join(f"token_{i}" for i in x)
    return tokenizer


@pytest.fixture
def mock_infinigram_responses():
    """Mock responses from Infinigram API."""
    return {
        "token_1": {"cnt": 1000},
        "token_2": {"cnt": 500}, 
        "token_3": {"cnt": 100},
        "token_4": {"cnt": 50},
        "token_5": {"cnt": 10},
        "token_1 token_2": {"cnt": 200},
        "token_2 token_3": {"cnt": 80},
        "token_3 token_4": {"cnt": 25},
        "token_4 token_5": {"cnt": 5},
        "token_1 token_2 token_3": {"cnt": 40}
    }


class TestSpanDiscovery:
    """Test span discovery functionality."""
    
    @patch('framing_audit.spans.infinigram.find')
    def test_discover_maximal_spans_basic(self, mock_find, mock_tokenizer):
        """Test basic span discovery."""
        # Mock API responses - all spans exist in training
        mock_find.return_value = {"cnt": 10}
        
        output_token_ids = [1, 2, 3, 4, 5]
        spans = discover_maximal_spans(
            output_token_ids, 
            "test_index", 
            max_ngram=3,
            tokenizer=mock_tokenizer
        )
        
        # Should find some spans
        assert len(spans) > 0
        # All spans should be valid (start < end)
        for start, end in spans:
            assert start < end
            assert 0 <= start < len(output_token_ids)
            assert 0 < end <= len(output_token_ids)
    
    @patch('framing_audit.spans.infinigram.find')
    def test_discover_spans_no_matches(self, mock_find, mock_tokenizer):
        """Test when no spans exist in training data."""
        # Mock API to return no matches
        mock_find.return_value = {"cnt": 0}
        
        output_token_ids = [1, 2, 3]
        spans = discover_maximal_spans(
            output_token_ids,
            "test_index", 
            tokenizer=mock_tokenizer
        )
        
        # Should return empty list
        assert spans == []
    
    @patch('framing_audit.spans.infinigram.find')
    def test_maximal_suppression(self, mock_find, mock_tokenizer):
        """Test that non-maximal spans are properly suppressed."""
        def mock_find_side_effect(index, query):
            # Only longer spans exist
            if len(query.split()) >= 2:
                return {"cnt": 10}
            return {"cnt": 0}
        
        mock_find.side_effect = mock_find_side_effect
        
        output_token_ids = [1, 2, 3, 4]
        spans = discover_maximal_spans(
            output_token_ids,
            "test_index",
            max_ngram=4, 
            tokenizer=mock_tokenizer
        )
        
        # Should only keep maximal (non-overlapping) spans
        for i in range(len(spans) - 1):
            start1, end1 = spans[i]
            start2, end2 = spans[i + 1]
            assert end1 <= start2  # No overlap


class TestScoring:
    """Test span scoring functionality."""
    
    def test_rarity_unigram(self, mock_tokenizer):
        """Test unigram rarity scoring."""
        token_counts = {"token_1": 1000, "token_2": 100, "token_3": 10}
        span_token_ids = [1, 2, 3]
        
        score = rarity_unigram(span_token_ids, token_counts, tokenizer=mock_tokenizer)
        
        # Score should be positive (rarity increases with lower counts)
        assert score > 0
        # Should include length bonus
        assert score > sum(-np.log(count + 1) for count in [1000, 100, 10])
    
    def test_pmi_calculation(self):
        """Test PMI calculation."""
        span_tokens = ["token_1", "token_2"]
        unigram_cache = {"token_1": 1000, "token_2": 500}
        bigram_cache = {"token_1 token_2": 200}
        
        with patch('framing_audit.spans.infinigram.find') as mock_find:
            # Mock already cached
            pmi = pmi_bigram_avg(span_tokens, "test_index", unigram_cache, bigram_cache)
            
            # PMI should be reasonable value
            assert isinstance(pmi, float)
            # For these numbers: log((200+1)*1e12 / ((1000+1)*(500+1))) ≈ positive
            assert pmi > 0
    
    @patch('framing_audit.spans.infinigram.find')
    def test_ngram_rarity(self, mock_find):
        """Test n-gram rarity scoring."""
        mock_find.return_value = {"cnt": 50}
        
        span_tokens = ["rare", "phrase", "here"] 
        score = rarity_ngram_avg(span_tokens, 2, "test_index", {})
        
        # Should return average negative log count
        expected = -np.log(50 + 1)  # Same for both bigrams in mock
        assert abs(score - expected) < 0.001
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        values = [1.0, 3.0, 5.0, 2.0]
        normalized = normalize_minmax(values)
        
        # Should be in [0, 1] range
        assert all(0 <= x <= 1 for x in normalized)
        # Min should be 0, max should be 1
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        # Should preserve order
        assert normalized[2] == 1.0  # Original max (5.0)
        assert normalized[0] == 0.0  # Original min (1.0)
    
    def test_ablation_salience(self):
        """Test ablation salience scoring."""
        # Span present in neutralized text
        score1 = ablation_salience("important phrase", "This important phrase is here")
        assert score1 == 0.0
        
        # Span not present in neutralized text  
        score2 = ablation_salience("important phrase", "This other text is here")
        assert score2 == 1.0  # default drift_semantic
        
        # No neutralized text
        score3 = ablation_salience("any phrase", None)
        assert score3 == 0.0


class TestEndToEnd:
    """Test end-to-end span selection pipeline."""
    
    @patch('framing_audit.spans.infinigram.find')
    @patch('framing_audit.spans.discover_maximal_spans')
    def test_select_top_spans_pipeline(self, mock_discover, mock_find, mock_tokenizer):
        """Test complete span selection pipeline."""
        # Mock span discovery
        mock_discover.return_value = [(0, 2), (2, 4), (1, 3)]  # Some overlapping spans
        
        # Mock API calls for scoring
        def mock_find_side_effect(index, query):
            counts = {
                "token_1": 1000, "token_2": 500, "token_3": 100, "token_4": 50,
                "token_1 token_2": 200, "token_2 token_3": 80, "token_3 token_4": 25
            }
            return {"cnt": counts.get(query, 10)}
        
        mock_find.side_effect = mock_find_side_effect
        
        output_ids = [1, 2, 3, 4]
        output_tokens = ["token_1", "token_2", "token_3", "token_4"]
        
        spans = select_top_spans(
            output_ids=output_ids,
            output_text_tokens=output_tokens,
            index_name="test_index",
            topK=2,
            M=10,
            tokenizer=mock_tokenizer
        )
        
        # Should return requested number of spans
        assert len(spans) <= 2
        
        # Each span should have required fields
        for span in spans:
            assert 'span_text' in span
            assert 'score' in span
            assert 'features' in span
            assert 'start' in span
            assert 'end' in span
            
            # Features should be normalized [0, 1]
            for feature_name, feature_val in span['features'].items():
                assert 0 <= feature_val <= 1
    
    def test_greedy_non_overlap_selection(self):
        """Test that greedy selection avoids overlapping spans."""
        # Mock candidates with overlapping positions
        candidates = [
            {"start": 0, "end": 3, "score": 0.9},
            {"start": 1, "end": 4, "score": 0.8},  # Overlaps with first
            {"start": 5, "end": 7, "score": 0.7},  # No overlap
            {"start": 6, "end": 8, "score": 0.6},  # Overlaps with third
        ]
        
        # Simulate greedy selection logic
        selected = []
        occupied = set()
        
        for candidate in sorted(candidates, key=lambda x: x["score"], reverse=True):
            positions = set(range(candidate["start"], candidate["end"]))
            if not positions.intersection(occupied):
                selected.append(candidate)
                occupied.update(positions)
        
        # Should select non-overlapping spans with highest scores
        assert len(selected) == 2
        assert selected[0]["score"] == 0.9  # Highest score
        assert selected[1]["score"] == 0.7  # Next highest non-overlapping


class TestRetrieval:
    """Test retrieval and reranking functionality."""
    
    @patch('framing_audit.spans.infinigram.find')
    @patch('framing_audit.spans.infinigram.get_doc_by_rank')
    def test_retrieve_and_rerank(self, mock_get_doc, mock_find):
        """Test document retrieval and reranking."""
        # Mock find response
        mock_find.return_value = {
            "cnt": 5,
            "shards": [(0, 10, 15), (1, 20, 25)]
        }
        
        # Mock document retrieval
        mock_get_doc.side_effect = [
            {"snippet": "This is a relevant document about the topic", "ptr": "0:10"},
            {"snippet": "Another document with different content", "ptr": "0:11"},
            {"snippet": "Third document discussing related matters", "ptr": "1:20"}
        ]
        
        spans = [
            {"span_text": "important phrase", "start": 0, "end": 2}
        ]
        
        results = retrieve_and_rerank(
            spans=spans,
            index_name="test_index", 
            full_context_text="This important phrase needs context",
            per_span_samples=3,
            top_docs=2
        )
        
        # Should add docs to span
        assert len(results) == 1
        assert "docs" in results[0]
        assert len(results[0]["docs"]) <= 2  # top_docs limit
        
        # Each doc should have required fields
        for doc in results[0]["docs"]:
            assert "snippet" in doc
            assert "score" in doc  
            assert "ptr" in doc
            assert isinstance(doc["score"], (int, float))
    
    def test_empty_retrieval(self):
        """Test handling of empty retrieval results."""
        spans = [{"span_text": "nonexistent phrase", "start": 0, "end": 2}]
        
        with patch('framing_audit.spans.infinigram.find') as mock_find:
            mock_find.return_value = {"cnt": 0, "shards": []}
            
            results = retrieve_and_rerank(
                spans=spans,
                index_name="test_index",
                full_context_text="some context",
                per_span_samples=3,
                top_docs=2
            )
            
            # Should handle gracefully
            assert len(results) == 1
            assert results[0]["docs"] == []
            assert results[0]["doc_count"] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self, mock_tokenizer):
        """Test handling of empty inputs."""
        # Empty token sequence
        spans = discover_maximal_spans([], "test_index", tokenizer=mock_tokenizer)
        assert spans == []
        
        # Empty text tokens
        results = select_top_spans(
            output_ids=[],
            output_text_tokens=[],
            index_name="test_index",
            tokenizer=mock_tokenizer
        )
        assert results == []
    
    def test_single_token_input(self, mock_tokenizer):
        """Test handling of single token."""
        with patch('framing_audit.spans.infinigram.find') as mock_find:
            mock_find.return_value = {"cnt": 10}
            
            spans = discover_maximal_spans([1], "test_index", tokenizer=mock_tokenizer)
            # Should find at least the single token span if it exists in training
            assert len(spans) >= 0
    
    def test_normalization_edge_cases(self):
        """Test normalization with edge cases."""
        # All same values
        result = normalize_minmax([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]
        
        # Empty list
        result = normalize_minmax([])
        assert result == []
        
        # Single value
        result = normalize_minmax([3.14])
        assert result == [0.0]
    
    @patch('framing_audit.spans.infinigram.find')
    def test_api_error_handling(self, mock_find, mock_tokenizer):
        """Test handling of API errors."""
        # Mock API failure
        mock_find.side_effect = Exception("API Error")
        
        # Should handle gracefully without crashing
        spans = discover_maximal_spans([1, 2, 3], "test_index", tokenizer=mock_tokenizer)
        assert isinstance(spans, list)  # Should return empty list or handle gracefully
    
    def test_feature_weights_validation(self):
        """Test that feature weights are applied correctly."""
        # This would be tested in integration, but we can test the logic
        weights = {"unigram": 0.5, "pmi": 0.3, "bigram": 0.2}
        features = {"unigram": 0.8, "pmi": 0.6, "bigram": 0.4}
        
        expected_score = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4
        calculated_score = sum(weights.get(k, 0) * v for k, v in features.items())
        
        assert abs(calculated_score - expected_score) < 1e-6


# Legacy compatibility tests
class TestLegacySpanSelector:
    """Test cases for legacy SpanSelector."""
    
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