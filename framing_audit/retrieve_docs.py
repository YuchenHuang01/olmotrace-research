#!/usr/bin/env python3
"""
Script to retrieve documents for spans using the public Infini-gram API.
Reads spans from maximal_spans.json and searches the v4_dolma-v1_7_llama corpus.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from framing_audit.infinigram import InfinigramClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_spans(spans_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load spans from JSON file."""
    logger.info(f"Loading spans from {spans_file}")
    with open(spans_file, "r") as f:
        data = json.load(f)

    spans = data.get("raw_spans", {})
    logger.info(f"Loaded {len(spans)} categories with spans")
    return spans


def search_span_documents(
    client: InfinigramClient,
    span_text: str,
    max_docs: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search for documents containing the span text and retrieve snippets + metadata.

    This uses:
      - client.find() to check if there are any occurrences.
      - client.search_docs() to get shard/rank.
      - client.get_doc_by_rank() to fetch snippet/metadata.
    """
    try:
        count_result = client.find(span_text)
        count = count_result.get("count", 0)

        if count == 0:
            logger.warning(f"No documents found for span: {span_text!r}")
            return []

        logger.info(f"Found {count} occurrences of span: {span_text!r}")

        search_result = client.search_docs(span_text, k=max_docs)
        results = search_result.get("results", [])
        documents_with_metadata: List[Dict[str, Any]] = []

        for doc_result in results:
            shard = doc_result.get("shard")
            rank = doc_result.get("rank")
            doc_ix = doc_result.get("doc_ix")

            if shard is None or rank is None:
                logger.warning(f"Search result missing shard/rank: {doc_result}")
                continue

            try:
                doc_metadata = client.get_doc_by_rank(shard=shard, rank=rank)

                combined = {
                    "search_info": doc_result,
                    "document_metadata": doc_metadata,
                    "shard": shard,
                    "rank": rank,
                    "doc_ix": doc_ix,
                }
                documents_with_metadata.append(combined)
            except Exception as e:
                logger.error(
                    f"Error fetching metadata for doc shard={shard} rank={rank}: {e}"
                )
                continue

        return documents_with_metadata

    except Exception as e:
        logger.error(f"Error searching for span {span_text!r}: {e}")
        return []


def process_spans(
    spans_data: Dict[str, List[Dict[str, Any]]],
    client: InfinigramClient,
    max_spans_per_category: int = 5,
    max_docs_per_span: int = 3,
) -> Dict[str, Any]:
    """Process all spans and retrieve documents."""
    results: Dict[str, Any] = {}

    for category, spans in spans_data.items():
        logger.info(f"\nProcessing category: {category}")
        results[category] = []

        for i, span in enumerate(spans[:max_spans_per_category]):
            span_text = span.get("text", "")
            if not span_text:
                continue

            logger.info(
                f"  Processing span {i+1}/{min(len(spans), max_spans_per_category)}: {span_text!r}"
            )

            search_results = search_span_documents(
                client, span_text, max_docs=max_docs_per_span
            )

            span_result = {
                "span": span,
                "search_results": search_results,
            }
            results[category].append(span_result)

    return results


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    logger.info(f"Saving results to {output_file}")

    output_data = {
        "metadata": {
            "index": "v4_dolma-v1_7_llama",
            "api_url": "https://api.infini-gram.io/",
            "total_categories": len(results),
            "total_spans_processed": sum(len(spans) for spans in results.values()),
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved successfully")


def main():
    spans_file = "maximal_spans.json"
    output_file = "span_document_retrieval_results.json"
    max_spans_per_category = 3
    max_docs_per_span = 3

    logger.info("Initializing Infinigram client...")
    client = InfinigramClient()

    spans_data = load_spans(spans_file)

    logger.info("Starting span processing...")
    results = process_spans(
        spans_data,
        client,
        max_spans_per_category=max_spans_per_category,
        max_docs_per_span=max_docs_per_span,
    )

    save_results(results, output_file)

    logger.info("Document retrieval completed!")

    total_spans = sum(len(spans) for spans in results.values())
    print("\n=== SUMMARY ===")
    print(f"Categories processed: {len(results)}")
    print(f"Total spans processed: {total_spans}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
