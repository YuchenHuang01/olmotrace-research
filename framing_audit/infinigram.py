"""
Infini-gram API client for retrieving training data snippets.
"""

import requests
import backoff
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class InfinigramResult:
    """Result from Infini-gram API query."""
    index_name: str
    doc_ptr: str
    shard: int
    rank: int
    snippet_text: str
    offset_left: int
    offset_right: int
    

class InfinigramClient:
    """Client for interacting with the Infini-gram API."""
    
    def __init__(
        self,
        api_url: str = "https://api.infini-gram.io",
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Infini-gram client.
        
        Args:
            api_url: Base URL for Infini-gram API
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=3,
        max_time=60
    )
    def _make_request(self, endpoint: str, method: str = "POST", **kwargs) -> Dict[str, Any]:
        """
        Make a request to the API with retry logic.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response as dict
        """
        url = f"{self.api_url}/{endpoint}"
        logger.debug(f"Making {method} request to {url}")
        
        response = self.session.request(
            method,
            url,
            timeout=self.timeout,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def find(self, index: str, query: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Find occurrences of a query in an index.
        
        Args:
            index: Index name to search
            query: Query string
            max_results: Maximum number of results
            
        Returns:
            API response with counts and document pointers
        """
        payload = {
            "index": index,
            "query": query,
            "max_results": max_results
        }
        
        logger.info(f"Searching for '{query}' in index {index}")
        
        # Example request format (adjust based on actual API)
        response = self._make_request("find", json=payload)
        
        return response
    
    def find_cnf(
        self,
        index: str,
        clauses: List[List[str]],
        max_gaps: List[int],
        clause_freq_cap: Optional[int] = None,
        max_results: int = 100
    ) -> Dict[str, Any]:
        """
        Find documents matching CNF query with gaps.
        
        Args:
            index: Index name to search
            clauses: List of clause lists (CNF format)
            max_gaps: Maximum gaps between clauses
            clause_freq_cap: Frequency cap for clauses
            max_results: Maximum number of results
            
        Returns:
            API response
        """
        payload = {
            "index": index,
            "clauses": clauses,
            "max_gaps": max_gaps,
            "max_results": max_results
        }
        
        if clause_freq_cap is not None:
            payload["clause_freq_cap"] = clause_freq_cap
            
        logger.info(f"CNF search in index {index} with {len(clauses)} clauses")
        
        response = self._make_request("find_cnf", json=payload)
        
        return response
    
    def get_doc_by_rank(self, index: str, shard: int, rank: int) -> Dict[str, Any]:
        """
        Get document by shard and rank.
        
        Args:
            index: Index name
            shard: Shard number
            rank: Document rank in shard
            
        Returns:
            Document content and metadata
        """
        payload = {
            "index": index,
            "shard": shard,
            "rank": rank
        }
        
        logger.debug(f"Fetching doc from {index} shard {shard} rank {rank}")
        
        response = self._make_request("get_doc_by_rank", json=payload)
        
        return response
    
    def retrieve_snippets_for_span(
        self,
        span_text: str,
        indexes: List[str],
        max_results_per_index: int = 100,
        samples_per_index: int = 5,
    ) -> List[InfinigramResult]:
        """
        Retrieve training snippets for a text span across indexes.
        
        Args:
            span_text: Text span to search for
            indexes: List of index names to search
            max_results_per_index: Max results per index
            samples_per_index: Number of snippets to sample per index
            
        Returns:
            List of InfinigramResult objects
        """
        all_results = []
        
        for index in indexes:
            try:
                # Search for span in index
                find_response = self.find(index, span_text, max_results_per_index)
                
                # Extract document pointers (format depends on actual API response)
                # This is a mock structure - adjust based on real API
                if "results" in find_response:
                    doc_pointers = find_response["results"][:samples_per_index]
                    
                    for doc_ptr in doc_pointers:
                        # Fetch document snippet
                        doc_response = self.get_doc_by_rank(
                            index,
                            doc_ptr.get("shard", 0),
                            doc_ptr.get("rank", 0)
                        )
                        
                        # Create result object
                        result = InfinigramResult(
                            index_name=index,
                            doc_ptr=doc_ptr.get("doc_id", ""),
                            shard=doc_ptr.get("shard", 0),
                            rank=doc_ptr.get("rank", 0),
                            snippet_text=doc_response.get("text", ""),
                            offset_left=doc_response.get("offset_left", 0),
                            offset_right=doc_response.get("offset_right", 0)
                        )
                        all_results.append(result)
                        
            except Exception as e:
                logger.error(f"Error retrieving snippets from {index}: {e}")
                
        return all_results
    
    def rerank_snippets(
        self,
        snippets: List[InfinigramResult],
        query_text: str,
        top_k: int = 5
    ) -> List[InfinigramResult]:
        """
        Rerank snippets by relevance to query.
        
        Args:
            snippets: List of snippets to rerank
            query_text: Query text for relevance scoring
            top_k: Number of top snippets to return
            
        Returns:
            Reranked list of snippets
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not snippets:
            return []
            
        # Extract snippet texts
        texts = [s.snippet_text for s in snippets]
        texts.append(query_text)  # Add query
        
        try:
            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(texts)
            
            # Compute cosine similarity with query
            query_vector = vectors[-1]
            snippet_vectors = vectors[:-1]
            similarities = cosine_similarity(query_vector, snippet_vectors).flatten()
            
            # Sort by similarity
            ranked_indices = similarities.argsort()[::-1][:top_k]
            
            return [snippets[i] for i in ranked_indices]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original order")
            return snippets[:top_k]
    
    def batch_retrieve(
        self,
        spans: List[str],
        indexes: List[str],
        max_results_per_span: int = 50,
        samples_per_span: int = 3,
    ) -> Dict[str, List[InfinigramResult]]:
        """
        Retrieve snippets for multiple spans in batch.
        
        Args:
            spans: List of text spans
            indexes: List of indexes to search
            max_results_per_span: Max results per span
            samples_per_span: Samples per span
            
        Returns:
            Dict mapping spans to results
        """
        results = {}
        
        for span in spans:
            logger.info(f"Retrieving snippets for span: {span}")
            
            snippets = self.retrieve_snippets_for_span(
                span,
                indexes,
                max_results_per_span,
                samples_per_span
            )
            
            # Rerank by relevance
            snippets = self.rerank_snippets(snippets, span, samples_per_span)
            
            results[span] = snippets
            
            # Rate limiting
            time.sleep(self.retry_delay)
            
        return results