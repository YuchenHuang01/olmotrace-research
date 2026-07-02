"""
Infini-gram API client for retrieving training data snippets.
"""

import requests
import backoff
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_client: Optional["InfinigramClient"] = None


def get_client() -> "InfinigramClient":
    """Get global client instance."""
    global _client
    if _client is None:
        _client = InfinigramClient()
    return _client


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
        api_url: str = "https://api.infini-gram.io/",
        default_index: str = "v4_dolma-v1_7_llama",
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Infini-gram client.

        API spec: https://infini-gram.readthedocs.io/en/latest/api.html
        """
        self.api_url = api_url.rstrip("/")
        self.default_index = default_index
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session = requests.Session()

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=3,
        max_time=60,
    )
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the API with retry logic."""
        logger.debug(f"POST {self.api_url} payload={payload}")
        resp = self.session.post(self.api_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Infini-gram API error: {data['error']}")
        return data

    # ------------------------------------------------------------------
    # Low-level primitives
    # ------------------------------------------------------------------

    def find(self, query: str, index: Optional[str] = None) -> Dict[str, Any]:
        """
        Count occurrences of a query in an index.

        Returns: {"count": int, "approx": bool, "token_ids": [...], "tokens": [...], ...}
        """
        if index is None:
            index = self.default_index
        payload = {
            "index": index,
            "query_type": "count",
            "query": query,
        }
        logger.debug(f"Counting occurrences of {query!r} in {index}")
        return self._make_request(payload)

    def find_positions(self, query: str, index: Optional[str] = None) -> Dict[str, Any]:
        """
        Find matching positions for a query (document search step 1).

        This wraps query_type="find" and returns at least:
            - "cnt": total occurrences
            - "segment_by_shard": list of [start, end] rank ranges per shard
        """
        if index is None:
            index = self.default_index
        payload = {
            "index": index,
            "query_type": "find",
            "query": query,
        }
        logger.debug(f"Finding positions for {query!r} in {index}")
        return self._make_request(payload)

    def get_doc_by_rank(
        self,
        query: str,
        shard: int,
        rank: int,
        index: Optional[str] = None,
        max_disp_len: int = 1000,
    ) -> Dict[str, Any]:
        """
        Get document snippet by shard and rank, including any available metadata.

        According to the API spec, get_doc_by_rank must receive the same
        `query` or `query_ids` used in the preceding `find` call.

        Expected response includes:
            - doc_ix, doc_len, disp_len, token_ids, spans
            - For some indexes (e.g., Dolma via AllenAI wrapper), may also
              include metadata with URLs.
        """
        if index is None:
            index = self.default_index

        payload = {
            "index": index,
            "query_type": "get_doc_by_rank",
            "query": query,        # IMPORTANT: must include query
            "s": shard,
            "rank": rank,
            "max_disp_len": max_disp_len,
        }
        logger.debug(f"Fetching doc {index} query={query!r} s={shard} rank={rank}")
        return self._make_request(payload)
