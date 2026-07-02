#!/usr/bin/env python3
"""
Retrieve documents for framing spans using the public Infini-gram API.

We use a curated list of conceptually important spans for each framing
(business_ownership vs monopolistic_power), query them via the public API,
and retrieve snippets + whatever metadata is available (including URLs if
exposed for v4_dolma-v1_7_llama).
"""

import json
import logging
import re
import random
from typing import List, Dict, Any

from framing_audit.infinigram import InfinigramClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. Curated query spans (slightly longer phrases)
# ----------------------------------------------------------------------

QUERY_SPANS: Dict[str, List[str]] = {
    "business_ownership": [
        "Business ownership refers to the legal right to operate",
        "profit from a business entity",
        "business ownership plays a critical role",
        "shaping the financial landscape",
        "Ownership can manifest in various forms",
        "from sole proprietorships to partnerships",
        "Small family firms and large corporations",
        "principles of business ownership",
        "business ownership is increasingly complex",
        "globalization and the rise of multinational corporations",
        "multitude of shareholders",
        "stake in the company's direction and financial performance",
        "business ownership is not just limited to private entities",
        "governments also own and operate various businesses",
        "sectors considered essential to the public interest",
        "state-owned enterprises in mixed economies",
        "discourse on business ownership and economic theories",
        "corporate social responsibility",
        "who benefits from a company's profits",
        "how decisions are made",
        "Ownership is a crucial factor in shaping business strategies",
        "corporate social responsibility initiatives",
        "distribution of wealth and power within an economy",
        "business ownership in modern economies is a multifaceted concept",
    ],
    "monopolistic_power": [
        "Monopolistic power in modern economies",
        "ability of a single business entity to exert significant control over the market",
        "power often stems from business ownership concentration",
        "single firm or a small group of firms dominate the industry",
        "Monopolistic power can be achieved through various means",
        "barriers to entry for potential competitors",
        "significant economies of scale that make it difficult for smaller firms to compete",
        "monopolistic power can set prices that maximize profits",
        "higher prices and lower output levels than in a competitive market",
        "market dominance can limit innovation and stifle potential competitors",
        "harming consumers and reducing economic efficiency",
        "own and control a significant share of the business landscape within an industry",
        "manipulate market dynamics to their advantage",
        "regulatory bodies scrutinize and regulate monopolistic power",
        "Antitrust laws and regulations promote fair competition",
        "prevent harmful monopolistic practices",
        "balancing the benefits of efficiency and innovation",
        "downsides of excessive market concentration is an ongoing policy challenge",
    ],
}

# ----------------------------------------------------------------------
# 2. Normalization + n-gram backoff helpers
# ----------------------------------------------------------------------


def normalize_span(span_text: str) -> str:
    """Normalize span text to improve exact n-gram matching."""
    s = span_text.strip()

    # Normalize fancy quotes/dashes
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("–", "-").replace("—", "-")

    # Remove outer quotes if the whole string is quoted
    if len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"')
        or (s[0] == "'" and s[-1] == "'")
    ):
        s = s[1:-1].strip()

    # Remove trailing punctuation like . , ? ! ; : ) ]
    s = re.sub(r"[.,?!;:\)\]]+$", "", s).strip()

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)

    return s


def generate_ngram_candidates(
    normalized_span: str,
    max_n: int = 7,
    min_n: int = 4,
) -> List[str]:
    """
    Generate n-gram candidates from a normalized span.
    We start from longer n (up to 7) and back off to shorter n (down to 4).
    """
    tokens = normalized_span.split()
    if not tokens:
        return []

    candidates: List[str] = []

    for n in range(min(max_n, len(tokens)), min_n - 1, -1):
        for i in range(len(tokens) - n + 1):
            cand = " ".join(tokens[i : i + n])
            candidates.append(cand)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped


# ----------------------------------------------------------------------
# 3. Metadata URL extraction helper
# ----------------------------------------------------------------------


def extract_url_from_raw_metadata(raw_meta: Any) -> str | None:
    """
    raw_meta is document_metadata["metadata"], which may be:
      - a dict
      - a JSON string
      - something else

    We parse it and then recursively search for a field named "url".
    """

    # 1) Parse string to dict if needed
    if isinstance(raw_meta, str):
        try:
            meta = json.loads(raw_meta)
        except Exception:
            return None
    elif isinstance(raw_meta, dict):
        meta = raw_meta
    else:
        return None

    # 2) DFS to find the first "url" field
    def dfs(obj):
        if isinstance(obj, dict):
            if "url" in obj and isinstance(obj["url"], str):
                return obj["url"]
            for v in obj.values():
                res = dfs(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for v in obj:
                res = dfs(v)
                if res is not None:
                    return res
        return None

    return dfs(meta)


# ----------------------------------------------------------------------
# 4. Main retrieval logic for a single span
# ----------------------------------------------------------------------


def search_span_documents(
    client: InfinigramClient,
    span_text: str,
    max_docs: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search for documents containing the span text using a robust strategy:
      1. Normalize the span.
      2. Break into overlapping n-gram windows (7-grams down to 4-grams).
      3. For each candidate n-gram:
           - query_type="count"; if count == 0, skip.
           - query_type="find" to get segment_by_shard.
           - randomly sample (shard, rank) pairs (like OLMoTrace's
             "up to 10 documents per span") and call get_doc_by_rank()
             with the same query to fetch snippet + metadata.
      4. Stop when we have max_docs unique documents (by URL or (shard, rank)).
    """
    # 1) Normalize
    normalized = normalize_span(span_text)
    if not normalized:
        logger.warning(f"Span {span_text!r} normalized to empty; skipping")
        return []

    logger.info(f"Normalized span: {normalized!r}")

    # 2) Generate candidate n-grams
    candidates = generate_ngram_candidates(
        normalized_span=normalized,
        max_n=7,
        min_n=4,
    )
    if not candidates:
        logger.warning(f"No n-gram candidates for span: {span_text!r}")
        return []

    documents_with_metadata: List[Dict[str, Any]] = []
    seen_keys = set()  # dedupe docs by URL or (shard, rank)

    for cand in candidates:
        if len(documents_with_metadata) >= max_docs:
            break

        # 3a) Quick count filter
        try:
            count_result = client.find(cand)
        except Exception as e:
            logger.error(f"Error in count query for candidate {cand!r}: {e}")
            continue

        cnt = count_result.get("count", 0)
        if cnt == 0:
            continue

        logger.info(f"Candidate {cand!r} has {cnt} occurrences")

        # 3b) Use query_type="find" to get positions across shards
        try:
            find_result = client.find_positions(cand)
        except Exception as e:
            logger.error(f"Error in find_positions for candidate {cand!r}: {e}")
            continue

        if isinstance(find_result, dict) and "error" in find_result:
            logger.error(f"find_positions error for {cand!r}: {find_result['error']}")
            continue

        segs = find_result.get("segment_by_shard", [])
        if not segs:
            continue

        remaining = max_docs - len(documents_with_metadata)
        if remaining <= 0:
            break

        # 3c) Randomly sample ranks across shards (OLMoTrace-style, up to remaining)
        for shard_idx, seg in enumerate(segs):
            if remaining <= 0:
                break

            if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                continue

            start, end = seg
            if start >= end:
                continue

            span_len = end - start
            k = min(span_len, remaining)
            # Sample k distinct ranks in this shard
            try:
                ranks = random.sample(range(start, end), k=k)
            except ValueError:
                # If span_len < k for some weird reason, fall back to simple range
                ranks = list(range(start, min(end, start + k)))

            for rank in ranks:
                if remaining <= 0:
                    break

                try:
                    doc_metadata = client.get_doc_by_rank(
                        query=cand,
                        shard=shard_idx,
                        rank=rank,
                        max_disp_len=1000,
                    )
                except Exception as e:
                    logger.error(
                        f"Error get_doc_by_rank for cand={cand!r} s={shard_idx} rank={rank}: {e}"
                    )
                    continue

                raw_meta = doc_metadata.get("metadata", {})
                url = extract_url_from_raw_metadata(raw_meta)

                key = url or (shard_idx, rank)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                combined = {
                    "original_span": span_text,
                    "normalized_span": normalized,
                    "span_candidate": cand,
                    "shard": shard_idx,
                    "rank": rank,
                    "search_info": {
                        "shard": shard_idx,
                        "rank": rank,
                    },
                    "document_metadata": doc_metadata,
                    "url": url,
                }
                documents_with_metadata.append(combined)
                remaining -= 1

    if not documents_with_metadata:
        logger.warning(f"No documents found for span (after backoff): {span_text!r}")

    return documents_with_metadata


# ----------------------------------------------------------------------
# 5. Process all spans + save results
# ----------------------------------------------------------------------


def process_spans(
    client: InfinigramClient,
    max_docs_per_span: int = 3,
) -> Dict[str, Any]:
    """Process all query spans and retrieve documents."""
    results: Dict[str, Any] = {}

    for category, spans in QUERY_SPANS.items():
        logger.info(f"\nProcessing category: {category}")
        cat_results = []
        for i, span_text in enumerate(spans, start=1):
            logger.info(
                f"  Processing span {i}/{len(spans)} in category {category}: {span_text!r}"
            )
            search_results = search_span_documents(
                client, span_text, max_docs=max_docs_per_span
            )
            cat_results.append(
                {
                    "span_text": span_text,
                    "search_results": search_results,
                }
            )
        results[category] = cat_results

    return results


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    logger.info(f"Saving results to {output_file}")

    total_spans = sum(len(spans) for spans in results.values())
    output_data = {
        "metadata": {
            "index": "v4_dolma-v1_7_llama",
            "api_url": "https://api.infini-gram.io/",
            "total_categories": len(results),
            "total_spans_processed": total_spans,
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved successfully")


def main():
    output_file = "span_document_retrieval_results.json"
    max_docs_per_span = 3

    logger.info("Initializing Infinigram client...")
    client = InfinigramClient()

    logger.info("Starting span processing...")
    results = process_spans(client, max_docs_per_span=max_docs_per_span)

    save_results(results, output_file)

    logger.info("Document retrieval completed!")

    total_spans = sum(len(spans) for spans in results.values())
    print("\n=== SUMMARY ===")
    print(f"Categories processed: {len(results)}")
    print(f"Total spans processed: {total_spans}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
