"""
Tests for Infini-gram API client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from framing_audit.infinigram import InfinigramClient, InfinigramResult


class TestInfinigramClient:
    """Test cases for InfinigramClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = InfinigramClient(
            api_url="https://api.test.com",
            timeout=30,
            retry_attempts=3
        )
        
    @patch('requests.Session.request')
    def test_make_request_success(self, mock_request):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "data": []}
        mock_request.return_value = mock_response
        
        result = self.client._make_request("test_endpoint", method="POST", json={"test": "data"})
        
        assert result == {"status": "success", "data": []}
        mock_request.assert_called_once()
        
    @patch('requests.Session.request')
    def test_make_request_retry(self, mock_request):
        """Test request retry on failure."""
        # Mock timeout then success
        mock_request.side_effect = [
            requests.exceptions.Timeout(),
            Mock(status_code=200, json=lambda: {"status": "success"})
        ]
        
        result = self.client._make_request("test_endpoint")
        
        assert result == {"status": "success"}
        assert mock_request.call_count == 2
        
    @patch.object(InfinigramClient, '_make_request')
    def test_find(self, mock_make_request):
        """Test find method."""
        mock_make_request.return_value = {
            "results": [
                {"doc_id": "doc1", "shard": 0, "rank": 1},
                {"doc_id": "doc2", "shard": 0, "rank": 2}
            ],
            "total_count": 100
        }
        
        result = self.client.find("test_index", "test query", max_results=10)
        
        assert "results" in result
        assert len(result["results"]) == 2
        mock_make_request.assert_called_once_with(
            "find",
            json={
                "index": "test_index",
                "query": "test query",
                "max_results": 10
            }
        )
        
    @patch.object(InfinigramClient, '_make_request')
    def test_find_cnf(self, mock_make_request):
        """Test find_cnf method."""
        mock_make_request.return_value = {"results": [], "count": 0}
        
        result = self.client.find_cnf(
            "test_index",
            [["word1", "word2"], ["word3"]],
            [1, 0],
            clause_freq_cap=1000
        )
        
        assert "results" in result
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args[1]["json"]
        assert call_args["clauses"] == [["word1", "word2"], ["word3"]]
        assert call_args["max_gaps"] == [1, 0]
        assert call_args["clause_freq_cap"] == 1000
        
    @patch.object(InfinigramClient, '_make_request')
    def test_get_doc_by_rank(self, mock_make_request):
        """Test get_doc_by_rank method."""
        mock_make_request.return_value = {
            "text": "This is a test document snippet",
            "offset_left": 100,
            "offset_right": 150
        }
        
        result = self.client.get_doc_by_rank("test_index", shard=0, rank=42)
        
        assert result["text"] == "This is a test document snippet"
        mock_make_request.assert_called_once_with(
            "get_doc_by_rank",
            json={"index": "test_index", "shard": 0, "rank": 42}
        )
        
    @patch.object(InfinigramClient, 'find')
    @patch.object(InfinigramClient, 'get_doc_by_rank')
    def test_retrieve_snippets_for_span(self, mock_get_doc, mock_find):
        """Test snippet retrieval for a span."""
        # Mock find response
        mock_find.return_value = {
            "results": [
                {"doc_id": "doc1", "shard": 0, "rank": 1},
                {"doc_id": "doc2", "shard": 1, "rank": 5}
            ]
        }
        
        # Mock document responses
        mock_get_doc.side_effect = [
            {
                "text": "First document with span text here",
                "offset_left": 0,
                "offset_right": 35
            },
            {
                "text": "Second document also has span text",
                "offset_left": 10,
                "offset_right": 44
            }
        ]
        
        results = self.client.retrieve_snippets_for_span(
            "span text",
            ["index1", "index2"],
            max_results_per_index=10,
            samples_per_index=2
        )
        
        # Should get results from both indexes
        assert len(results) == 4  # 2 indexes * 2 samples each
        
        # Check result structure
        assert all(isinstance(r, InfinigramResult) for r in results)
        assert results[0].snippet_text == "First document with span text here"
        
    def test_rerank_snippets(self):
        """Test snippet reranking."""
        snippets = [
            InfinigramResult(
                index_name="idx1",
                doc_ptr="doc1",
                shard=0,
                rank=1,
                snippet_text="The climate crisis requires immediate action",
                offset_left=0,
                offset_right=50
            ),
            InfinigramResult(
                index_name="idx1",
                doc_ptr="doc2",
                shard=0,
                rank=2,
                snippet_text="Unrelated content about something else entirely",
                offset_left=0,
                offset_right=50
            ),
            InfinigramResult(
                index_name="idx1",
                doc_ptr="doc3",
                shard=0,
                rank=3,
                snippet_text="Climate change affects global temperatures significantly",
                offset_left=0,
                offset_right=60
            )
        ]
        
        reranked = self.client.rerank_snippets(
            snippets,
            "climate change impacts",
            top_k=2
        )
        
        # Should return top 2 most relevant
        assert len(reranked) == 2
        
        # Most relevant snippets should be about climate
        assert "climate" in reranked[0].snippet_text.lower()
        
    @patch.object(InfinigramClient, 'retrieve_snippets_for_span')
    @patch.object(InfinigramClient, 'rerank_snippets')
    def test_batch_retrieve(self, mock_rerank, mock_retrieve):
        """Test batch retrieval for multiple spans."""
        # Mock retrieve responses
        mock_retrieve.side_effect = [
            [InfinigramResult("idx1", "doc1", 0, 1, "snippet 1", 0, 10)],
            [InfinigramResult("idx1", "doc2", 0, 2, "snippet 2", 0, 10)]
        ]
        
        # Mock rerank (return as-is)
        mock_rerank.side_effect = lambda x, y, z: x[:z]
        
        spans = ["span one", "span two"]
        results = self.client.batch_retrieve(
            spans,
            ["index1"],
            max_results_per_span=10,
            samples_per_span=1
        )
        
        assert len(results) == 2
        assert "span one" in results
        assert "span two" in results
        assert len(results["span one"]) == 1
        assert len(results["span two"]) == 1