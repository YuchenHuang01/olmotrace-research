#!/usr/bin/env python3
"""
DJ Search (exact-match) span finder over the Infini-gram API.

Reimplements the exact-match span-decomposition algorithm ("DJ Search") from:
  "AI as Humanity's Salieri: Quantifying Linguistic Creativity of Language
   Models via Systematic Attribution of Machine Text against Web Text"
  (Lu et al., 2024) -- https://arxiv.org/abs/2410.04265
  original repo: https://github.com/GXimingLu/creativity_index

The algorithm is reimplemented here (not copied) to fit this codebase: it
reuses this repo's InfinigramClient and index, and returns per-span occurrence
counts as a rarity signal for downstream filtering / document retrieval.

Given a text, it greedily segments it into the longest consecutive token spans
(>= min_ngram) that occur verbatim in the corpus, using only the Infini-gram
`count` query -- no local index required.
"""

import argparse
import json
import logging
import time
from typing import Callable, List, Dict, Any

import nltk
from unidecode import unidecode

from .infinigram import InfinigramClient

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tokenization helpers
# ----------------------------------------------------------------------
def tokenize_words(text: str) -> List[str]:
    """Word-level tokenization (Twitter/casual tokenizer, no punkt download)."""
    return nltk.tokenize.casual.casual_tokenize(text)


def _make_detokenizer() -> Callable[[List[str]], str]:
    """Return a token-list -> string detokenizer (Moses if available)."""
    try:
        from sacremoses import MosesDetokenizer
        md = MosesDetokenizer(lang="en")
        return lambda toks: md.detokenize(toks)
    except Exception:
        return _simple_detokenize


_ATTACH_LEFT = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%",
                "'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "'"}


def _simple_detokenize(tokens: List[str]) -> str:
    """Lightweight fallback: join with spaces, but glue punctuation/clitics."""
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out = t
        elif t in _ATTACH_LEFT:
            out += t
        elif out.endswith(("(", "[", "{", "$", "#", "@")):
            out += t
        else:
            out += " " + t
    return out


# ----------------------------------------------------------------------
# Core algorithm (pure: takes a count_fn, so it is unit-testable)
# ----------------------------------------------------------------------
def find_exact_match(
    tokens: List[str],
    detokenize: Callable[[List[str]], str],
    count_fn: Callable[[str], int],
    min_ngram: int = 5,
) -> Dict[str, Any]:
    """
    Two-pointer greedy longest-match segmentation (DJ Search, exact).

    For each position, grow the right edge while the span still occurs in the
    corpus; when it stops occurring, slide the left edge forward. Returns the
    matched spans, coverage, and average span length.
    """
    n = len(tokens)
    spans: List[Dict[str, Any]] = []

    def add_or_replace(new: Dict[str, Any]) -> None:
        # If the new span strictly contains the previous one (we just extended
        # the right edge), replace it; otherwise it is a distinct span.
        if spans:
            last = spans[-1]
            if (new["start_index"] <= last["start_index"]
                    and last["end_index"] <= new["end_index"]):
                spans[-1] = new
                return
        spans.append(new)

    first, second = 0, min_ngram
    while second <= n:
        span_text = detokenize(tokens[first:second])
        occurrence = count_fn(span_text)

        if occurrence:
            add_or_replace({
                "start_index": first,
                "end_index": second,
                "span_text": span_text,
                "occurrence": occurrence,
            })
            second += 1                       # try to extend the match
        else:
            if second - first > min_ngram:
                first += 1                    # shrink from the left, retry
            else:                             # window == min_ngram: no match here
                first += 1
                second += 1

    # coverage over token indices (overlaps counted once)
    covered = [False] * n
    for s in spans:
        for j in range(s["start_index"], s["end_index"]):
            covered[j] = True
    coverage = (sum(covered) / n) if n else 0.0
    if spans:
        avg_span_len = sum(s["end_index"] - s["start_index"] for s in spans) / len(spans)
    else:
        avg_span_len = 0.0

    return {
        "matched_spans": spans,
        "coverage": coverage,
        "avg_span_len": avg_span_len,
    }


def select_nonoverlapping(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reduce DJ Search's (deliberately overlapping) spans to a clean,
    non-overlapping set for attribution.

    Greedy longest-first: keep a span only if it does not overlap an already-kept
    span. Each kept span is one of the ORIGINAL spans (never a fused union), so it
    still individually occurs in the corpus (count > 0) -- unioning overlapping
    spans could create a string that does not exist verbatim.
    """
    ordered = sorted(
        spans, key=lambda s: (-(s["end_index"] - s["start_index"]), s["start_index"]))
    kept: List[Dict[str, Any]] = []
    for s in ordered:
        if all(s["end_index"] <= k["start_index"] or s["start_index"] >= k["end_index"]
               for k in kept):
            kept.append(s)
    kept.sort(key=lambda s: s["start_index"])
    return kept


# ----------------------------------------------------------------------
# Client wrapper (count via Infini-gram API, with an in-memory cache)
# ----------------------------------------------------------------------
class DJSpanFinder:
    def __init__(self, client: InfinigramClient, index: str, min_ngram: int = 5,
                 request_sleep: float = 0.1):
        self.client = client
        self.index = index
        self.min_ngram = min_ngram
        self.request_sleep = request_sleep      # throttle to avoid API 403 bursts
        self.detokenize = _make_detokenizer()
        self._cache: Dict[str, int] = {}        # span_text -> count

    def _count(self, span_text: str, max_retries: int = 6) -> int:
        if span_text in self._cache:
            return self._cache[span_text]
        for attempt in range(max_retries):
            if self.request_sleep:
                time.sleep(self.request_sleep)
            try:
                result = self.client.find(span_text, index=self.index)
                count = int(result.get("count", 0)) if isinstance(result, dict) else 0
                self._cache[span_text] = count
                return count
            except Exception as e:
                # Usually a 403 rate-limit burst; cool down longer and retry.
                cooldown = 2.0 * (attempt + 1)
                logger.warning(f"count retry {attempt+1}/{max_retries} "
                               f"({str(e)[:60]}); sleeping {cooldown:.0f}s")
                time.sleep(cooldown)
        raise RuntimeError(f"count failed after {max_retries} retries: {span_text!r}")

    def find_spans(self, text: str) -> Dict[str, Any]:
        tokens = tokenize_words(unidecode(text))
        if len(tokens) <= self.min_ngram:
            return {"matched_spans": [], "coverage": 0.0, "avg_span_len": 0.0}
        return find_exact_match(tokens, self.detokenize, self._count, self.min_ngram)


# ----------------------------------------------------------------------
# Input loading (accepts a few common shapes)
# ----------------------------------------------------------------------
def load_texts(path: str, text_key: str) -> List[Dict[str, Any]]:
    """Load a JSON list of strings or dicts; return list of {id, text, extra}."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list (of strings or objects)")
    out = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            out.append({"doc_id": f"doc_{i}", "text": item})
        elif isinstance(item, dict) and item.get(text_key):
            rec = {"doc_id": item.get("id", f"doc_{i}"), "text": item[text_key]}
            out.append(rec)
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="DJ Search exact-match span finder (Infini-gram API)")
    ap.add_argument("--input", required=True,
                    help="JSON list of strings or objects with a text field")
    ap.add_argument("--text-key", default="text",
                    help="Field holding the text when input items are objects")
    ap.add_argument("--out", default="dj_spans.json", help="Output JSON path")
    ap.add_argument("--index", default="v4_olmo-mix-1124_llama",
                    help="Infini-gram index to query. Use OLMo's OWN training "
                         "corpus so 'span present' means 'in OLMo's training "
                         "data': v4_olmo-mix-1124_llama (pretrain mix) or "
                         "v4_olmo-2-1124-13b-instruct_llama (post-training). "
                         "Dolma (v4_dolma-v1_7_llama) exposes source URLs but is "
                         "only an approximation of OLMo's data.")
    ap.add_argument("--min-ngram", type=int, default=5,
                    help="Minimum span length in (word) tokens")
    ap.add_argument("--max-occurrence", type=int, default=None,
                    help="If set, also emit spans-for-retrieval keeping only "
                         "spans with occurrence <= this (rarity filter)")
    ap.add_argument("--spans-out", default=None,
                    help="Optional path to write {doc_id: [span_text,...]} for retrieval")
    ap.add_argument("--subset", type=int, default=None,
                    help="Only process the first N documents")
    ap.add_argument("--request-sleep", type=float, default=0.1,
                    help="Throttle between count calls to avoid rate-limit 403s")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = InfinigramClient()
    finder = DJSpanFinder(client, index=args.index, min_ngram=args.min_ngram,
                          request_sleep=args.request_sleep)

    docs = load_texts(args.input, args.text_key)
    if args.subset:
        docs = docs[:args.subset]
    logger.info(f"Loaded {len(docs)} documents | index={args.index} "
                f"min_ngram={args.min_ngram}")

    results = []
    spans_for_retrieval: Dict[str, List[str]] = {}
    for k, doc in enumerate(docs, 1):
        try:
            found = finder.find_spans(doc["text"])
        except Exception as e:
            logger.error(f"[{k}/{len(docs)}] {doc['doc_id']} failed: {e}")
            continue
        # Keep raw matched_spans (+coverage) faithful to DJ Search; also emit a
        # clean non-overlapping set for attribution / retrieval.
        merged = select_nonoverlapping(found["matched_spans"])
        rec = {**doc, **found, "merged_spans": merged}
        results.append(rec)

        keep = merged
        if args.max_occurrence is not None:
            keep = [s for s in keep if s["occurrence"] <= args.max_occurrence]
        spans_for_retrieval[doc["doc_id"]] = [s["span_text"] for s in keep]

        logger.info(f"[{k}/{len(docs)}] {doc['doc_id']}: "
                    f"{len(found['matched_spans'])} raw -> {len(merged)} merged spans, "
                    f"coverage={found['coverage']:.3f}, "
                    f"avg_len={found['avg_span_len']:.1f}")

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if args.spans_out is not None:
        with open(args.spans_out, "w") as f:
            json.dump(spans_for_retrieval, f, indent=2)
        logger.info(f"Wrote spans-for-retrieval -> {args.spans_out}")

    logger.info(f"Done. {len(results)} docs -> {args.out}")


if __name__ == "__main__":
    main()
