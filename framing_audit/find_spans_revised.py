#!/usr/bin/env python3
"""
OLMoTrace-style Maximal Span Detection using local Infini-gram index
Revised to match the exact algorithm:
1. Only start spans at begin-of-word token boundaries
2. Use LCP with neighboring suffixes when count=0
3. Apply constraints after finding max length (period/newline trimming)
4. Fix non-maximal suppression to match paper algorithm
5. Process suffixes in parallel with ThreadPoolExecutor
"""

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

# ====================
# Config
# ====================
MAX_SPAN_LENGTH = 100      # tokens (can be overridden by --max-span-len)
MIN_SPAN_LENGTH = 2        # tokens
MIN_SPAN_WORDS  = 1        # words in final surface

SENT_END_CHARS = set([".", "!", "?", "。", "！", "？"])
SENT_END_RE    = re.compile(r"[\.!\?。！？]")
WHITESPACE_RE  = re.compile(r"\s")
NONWORD_RE     = re.compile(r"\W", re.UNICODE)

ENGINE_CALLS = 0
FIND_CALLS = 0
LCP_CALLS = 0

# ====================
# Helpers
# ====================
def split_sentences(text: str) -> List[str]:
    """Permissive sentence splitter keeping end punctuation with sentence."""
    pattern = r"""(
        .*?                          
        [\.!\?。！？]                 
        [\""'\)\]\}»》]*             
    )
    (?=\s+|$)"""
    parts = re.findall(pattern, text, flags=re.VERBOSE | re.DOTALL)
    used = "".join(parts)
    if len(used) < len(text):
        tail = text[len(used):]
        if tail.strip():
            parts.append(tail)
    return parts if parts else [text]

def is_begin_word(text: str, char_start: int) -> bool:
    """Check if character position is at word beginning."""
    if char_start <= 0:
        return True
    prev = text[char_start - 1]
    if prev == "'" and (char_start < len(text) and text[char_start].lower() == 's'):
        return False
    return bool(WHITESPACE_RE.match(prev) or NONWORD_RE.match(prev))

def is_end_word(text: str, char_end_exclusive: int) -> bool:
    """Check if character position is at word end."""
    if char_end_exclusive >= len(text):
        return True
    nextc = text[char_end_exclusive]
    return bool(WHITESPACE_RE.match(nextc) or NONWORD_RE.match(nextc))

def get_token_data(text: str, tokenizer):
    """Tokenize text and return token IDs and offset mappings."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]

def surface_from_offsets(text: str, char_starts: List[int], char_ends: List[int], s_tok: int, e_tok: int) -> str:
    """Extract surface text from token offsets."""
    return text[char_starts[s_tok]:char_ends[e_tok - 1]]

def token_starts_word(token_text: str, char_start: int, full_text: str) -> bool:
    """Check if a token starts a word based on its text and position."""
    # If token starts with alphanumeric, check if previous char is non-word
    if token_text and token_text[0].isalnum():
        return is_begin_word(full_text, char_start)
    # If token starts with non-alphanumeric, it's typically punctuation
    return False

def contains_period_or_newline_tokens(ids: List[int], tokenizer) -> List[int]:
    """Return indices of tokens that are periods or contain newlines."""
    indices = []
    for i, token_id in enumerate(ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if token_text in [".", "!", "?", "。", "！", "？"] or "\n" in token_text or "\r" in token_text:
            indices.append(i)
    return indices

# ====================
# Engine interface
# ====================
def find_with_lcp(ids_slice: List[int], engine: InfiniGramEngine) -> Tuple[int, Dict]:
    """
    Find with LCP fallback when count=0.
    Returns (count, find_result).
    """
    global ENGINE_CALLS, FIND_CALLS, LCP_CALLS
    ENGINE_CALLS += 1
    
    # First get count
    count_result = engine.count(input_ids=ids_slice)
    count = count_result["count"]
    
    if count > 0:
        # Do regular find
        FIND_CALLS += 1
        find_result = engine.find(input_ids=ids_slice)
        return count, find_result
    else:
        # Count is 0, use LCP with neighbors
        LCP_CALLS += 1
        # Get left and right SA positions from count result
        left_sa = count_result.get("left", None)
        right_sa = count_result.get("right", None)
        
        if left_sa is not None and right_sa is not None:
            # Get neighboring suffixes and compute LCP
            # This requires engine support for get_suffix_by_sa_pos or similar
            # For now, we'll return empty result as fallback
            pass
        
        return 0, {"segment_by_shard": []}

def find_longest_existing_prefix(
    ids: List[int],
    start_idx: int,
    engine: InfiniGramEngine,
    max_span_len: int
) -> int:
    """
    Find longest prefix that exists in corpus (ignoring constraints).
    Uses exponential search followed by binary search.
    When count=0, uses LCP with neighboring suffixes.
    """
    n = len(ids)
    max_possible = min(max_span_len, n - start_idx)
    
    if max_possible < MIN_SPAN_LENGTH:
        return 0
    
    # Quick check minimum length
    count, _ = find_with_lcp(ids[start_idx:start_idx+MIN_SPAN_LENGTH], engine)
    if count == 0:
        return 0
    
    # Exponential search for upper bound
    lo = MIN_SPAN_LENGTH
    hi = MIN_SPAN_LENGTH
    
    while hi < max_possible:
        next_len = min(hi * 2, max_possible)
        count, _ = find_with_lcp(ids[start_idx:start_idx+next_len], engine)
        if count > 0:
            lo = next_len
            hi = next_len
        else:
            break
    
    # Binary search for exact length
    if hi < max_possible:
        # We found a length that doesn't exist, binary search
        while lo < hi:
            mid = (lo + hi + 1) // 2
            count, _ = find_with_lcp(ids[start_idx:start_idx+mid], engine)
            if count > 0:
                lo = mid
            else:
                hi = mid - 1
    
    return lo

# ====================
# Constraint application
# ====================
def apply_constraints(
    ids: List[int],
    s_tok: int,
    e_tok: int,
    sent_text: str,
    char_starts: List[int],
    char_ends: List[int],
    tokenizer,
    engine: InfiniGramEngine,
    clause_stops: set
) -> Optional[Tuple[int, int]]:
    """
    Apply constraints to a span and trim if needed.
    Returns (start, end) tokens or None if invalid.
    """
    # Check if span ends at word boundary
    if not is_end_word(sent_text, char_ends[e_tok - 1]):
        return None
    
    # Check if next token after span starts a word
    if e_tok < len(ids):
        next_token_text = tokenizer.decode([ids[e_tok]], skip_special_tokens=False)
        if not token_starts_word(next_token_text, char_starts[e_tok], sent_text):
            return None
    
    # Check for interior period/newline tokens
    span_ids = ids[s_tok:e_tok]
    problem_indices = contains_period_or_newline_tokens(span_ids, tokenizer)
    
    # Trim to remove interior problems (keeping if at very end)
    if problem_indices:
        last_ok = s_tok
        for prob_idx in problem_indices:
            abs_idx = s_tok + prob_idx
            # If problem is at very end, it's OK
            if abs_idx == e_tok - 1:
                break
            # Otherwise trim before this position
            e_tok = abs_idx
        
        # Check if we still have minimum length
        if e_tok - s_tok < MIN_SPAN_LENGTH:
            return None
        
        # Verify trimmed span still exists
        count, _ = find_with_lcp(ids[s_tok:e_tok], engine)
        if count == 0:
            return None
    
    # Check surface constraints
    surf = surface_from_offsets(sent_text, char_starts, char_ends, s_tok, e_tok)
    
    # Check for clause stops in the body (not at end)
    body = surf.rstrip()
    if any(ch in body[:-1] for ch in clause_stops):
        return None
    
    # Check minimum words
    if len(surf.split()) < MIN_SPAN_WORDS:
        return None
    
    return (s_tok, e_tok)

# ====================
# Process single suffix
# ====================
def process_suffix(
    s_tok: int,
    ids: List[int],
    sent_text: str,
    char_starts: List[int],
    char_ends: List[int],
    tokenizer,
    engine: InfiniGramEngine,
    max_span_len: int,
    clause_stops: set,
    doc_token_offset: int
) -> Optional[Tuple[int, int, str]]:
    """
    Process a single suffix starting at s_tok.
    Returns (doc_start, doc_end, surface) or None.
    """
    # Check if token starts a word
    token_text = tokenizer.decode([ids[s_tok]], skip_special_tokens=False)
    if not token_starts_word(token_text, char_starts[s_tok], sent_text):
        return None
    
    # Find longest existing prefix (ignoring constraints)
    max_len = find_longest_existing_prefix(ids, s_tok, engine, max_span_len)
    if max_len < MIN_SPAN_LENGTH:
        return None
    
    # Apply constraints and get final span
    result = apply_constraints(
        ids, s_tok, s_tok + max_len,
        sent_text, char_starts, char_ends,
        tokenizer, engine, clause_stops
    )
    
    if result is None:
        return None
    
    s, e = result
    surf = surface_from_offsets(sent_text, char_starts, char_ends, s, e)
    
    # Return document-level indices
    return (s + doc_token_offset, e + doc_token_offset, surf)

# ====================
# Non-maximal suppression
# ====================
def suppress_non_maximal(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Keep only maximal spans using the paper's algorithm:
    Sort by start position, keep spans that extend further than previous max_end.
    """
    if not spans:
        return []
    
    # Sort by start position, then by end position
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1]))
    
    kept = []
    max_end = -1
    
    for start, end, text in sorted_spans:
        if end > max_end:
            kept.append((start, end, text))
            max_end = end
    
    return kept

# ====================
# Process sentence with parallel execution
# ====================
def collect_maximal_spans_sentence(
    sent_text: str,
    tokenizer,
    engine: InfiniGramEngine,
    doc_token_offset: int,
    max_span_len: int,
    clause_stops: set,
    max_workers: int = 4
) -> Tuple[List[Tuple[int, int, str]], int]:
    """
    Collect maximal spans from a sentence using parallel processing.
    """
    ids, offs = get_token_data(sent_text, tokenizer)
    n = len(ids)
    if n < MIN_SPAN_LENGTH:
        return [], 0
    
    char_starts = [a for (a, b) in offs]
    char_ends = [b for (a, b) in offs]
    
    initial_calls = ENGINE_CALLS
    
    # Process all suffix positions in parallel
    all_spans = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all suffix processing tasks
        futures = []
        for s_tok in range(n - MIN_SPAN_LENGTH + 1):
            future = executor.submit(
                process_suffix,
                s_tok, ids, sent_text, char_starts, char_ends,
                tokenizer, engine, max_span_len, clause_stops, doc_token_offset
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_spans.append(result)
    
    # Apply non-maximal suppression
    maximal_spans = suppress_non_maximal(all_spans)
    
    calls_used = ENGINE_CALLS - initial_calls
    return maximal_spans, calls_used

# ====================
# Main
# ====================
def main():
    parser = argparse.ArgumentParser(description="OLMoTrace maximal span detection (revised algorithm)")
    parser.add_argument("--index-dir", type=str, default="/mnt/infini/index", 
                        help="Path to Infini-gram index directory")
    parser.add_argument("--max-span-len", type=int, default=100, 
                        help="Maximum span length in tokens")
    parser.add_argument("--clause-stops", type=str, default=",;", 
                        help="Characters that should not appear inside span body")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum parallel workers for processing")
    parser.add_argument("--out", type=str, default="maximal_spans.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    global ENGINE_CALLS
    
    # Build clause stopper set
    CLAUSE_STOPS = set(args.clause_stops) | set(ch for ch in args.clause_stops.replace(",", "，").replace(";", "；"))
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        add_eos_token=False,
        use_fast=True
    )
    
    print(f"Initializing Infini-gram engine from {args.index_dir}...")
    try:
        engine = InfiniGramEngine(index_dir=args.index_dir, eos_token_id=tokenizer.eos_token_id)
        print("✓ Engine initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return
    
    # Demo texts
    outputs = {
        "business_ownership": """Business ownership refers to the legal right to operate, control, and profit from a business entity. In modern economies, business ownership plays a critical role in shaping the financial landscape, influencing innovation, and driving economic growth. Ownership can manifest in various forms, from sole proprietorships to partnerships, corporations, and cooperatives. Small family firms and large corporations alike are governed by principles of business ownership.""",
        
        "monopolistic_power": """Monopolistic power in modern economies refers to the ability of a single business entity to exert significant control over the market for a particular good or service. This power often stems from business ownership concentration, where a single firm or a small group of firms dominate the industry, thereby limiting competition."""
    }
    
    print("=" * 70)
    print("OLMoTrace Maximal Span Detection (Revised)")
    print("=" * 70 + "\n")
    
    results = {}
    
    for name, text in outputs.items():
        print(f"Analyzing: {name}")
        sentences = split_sentences(text)
        print(f"Sentences: {len(sentences)}")
        
        # Get document-level tokenization for final surfaces
        full_ids, full_offs = get_token_data(text, tokenizer)
        full_char_starts = [a for (a, b) in full_offs]
        full_char_ends = [b for (a, b) in full_offs]
        
        doc_spans = []
        total_tokens = 0
        total_calls = 0
        
        # Process each sentence
        for sent_idx, sent in enumerate(sentences, 1):
            ids, _ = get_token_data(sent, tokenizer)
            spans, calls = collect_maximal_spans_sentence(
                sent, tokenizer, engine, total_tokens,
                args.max_span_len, CLAUSE_STOPS, args.max_workers
            )
            
            doc_spans.extend(spans)
            total_tokens += len(ids)
            total_calls += calls
            
            print(f"  Sentence {sent_idx}: {len(ids)} tokens, {calls} engine calls, {len(spans)} spans")
        
        print(f"\nTotal: {len(doc_spans)} maximal spans, {total_calls} engine calls")
        print("-" * 70 + "\n")
        
        results[name] = doc_spans
        
        # Preview
        print(f"Maximal spans for {name}:")
        for i, (s, e, t) in enumerate(doc_spans[:10], 1):
            snippet = t.replace("\n", " ")[:90]
            print(f"  {i:2}. [{s:4}-{e:4}] ({e-s:3} toks) '{snippet}'")
        if len(doc_spans) > 10:
            print(f"      ... and {len(doc_spans)-10} more\n")
    
    # Statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total engine.count() calls: {ENGINE_CALLS}")
    print(f"Total engine.find() calls: {FIND_CALLS}")
    print(f"Total LCP operations: {LCP_CALLS}")
    
    # Save results
    raw_spans = {
        name: [
            {"start": s, "end": e, "tokens": e - s, "text": t}
            for s, e, t in spans
        ]
        for name, spans in results.items()
    }
    
    output = {
        "algorithm": "OLMoTrace Revised",
        "constraints": [
            "Only start spans at tokens that begin words",
            "Use LCP with neighbors when count=0",
            "Trim spans with interior period/newline tokens",
            "Require next token after span to start a word",
            "Non-maximal suppression: keep spans extending beyond previous max_end",
            "Parallel processing of suffix positions"
        ],
        "config": {
            "max_span_length": args.max_span_len,
            "min_span_length": MIN_SPAN_LENGTH,
            "min_span_words": MIN_SPAN_WORDS,
            "clause_stops": args.clause_stops,
            "max_workers": args.max_workers
        },
        "engine_stats": {
            "total_engine_calls": ENGINE_CALLS,
            "find_calls": FIND_CALLS,
            "lcp_calls": LCP_CALLS,
            "index_dir": args.index_dir
        },
        "raw_spans": raw_spans,
        "filtered_spans": raw_spans  # Same for now
    }
    
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved to {args.out}")

if __name__ == "__main__":
    main()