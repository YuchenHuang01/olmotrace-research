#!/usr/bin/env python3
"""
OLMoTrace-style Maximal Span Detection using local Infini-gram index (token-based, strict, keep-all)
Shorter-clause mode:
- Forbid interior clause punctuation (commas/semicolons by default) so spans stop at clauses.
- Optional --max-span-len to cap span length (default 100).
"""

import argparse
import json
import math
import re
from typing import List, Tuple, Dict

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

# ============== Strict Config (paper-faithful base) ==============
MAX_SPAN_LENGTH = 100      # tokens (overridden by --max-span-len)
MIN_SPAN_LENGTH = 6        # tokens (minimum 6 tokens for filtering)
MIN_SPAN_WORDS  = 1        # words in final surface
RELAX_BOUNDARIES = False   # STRICT
RELAX_SELF_CONTAINED = False  # STRICT

SENT_END_CHARS = set([".", "!", "?", "。", "！", "？"])
SENT_END_RE    = re.compile(r"[\.!\?。！？]")
WHITESPACE_RE  = re.compile(r"\s")
NONWORD_RE     = re.compile(r"\W", re.UNICODE)

ENGINE_CALLS = 0

# ==================== Helpers ====================
def split_sentences(text: str) -> List[str]:
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
    if char_start <= 0:
        return True
    prev = text[char_start - 1]
    if prev == "'" and (char_start < len(text) and text[char_start].lower() == 's'):
        return False
    return bool(WHITESPACE_RE.match(prev) or NONWORD_RE.match(prev))

def is_end_word(text: str, char_end_exclusive: int) -> bool:
    if char_end_exclusive >= len(text):
        return True
    nextc = text[char_end_exclusive]
    return bool(WHITESPACE_RE.match(nextc) or NONWORD_RE.match(nextc))

def get_token_data(text: str, tokenizer):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]

def surface_from_offsets(text: str, char_starts: List[int], char_ends: List[int], s_tok: int, e_tok: int) -> str:
    return text[char_starts[s_tok]:char_ends[e_tok - 1]]

# ---------- engine helpers ----------
def exists_by_find(ids_slice: List[int], engine: InfiniGramEngine) -> bool:
    global ENGINE_CALLS
    ENGINE_CALLS += 1
    res = engine.find(input_ids=ids_slice)
    segs = res.get("segment_by_shard", [])
    return any(ranges for ranges in segs)

# ==================== Longest prefix w/ boundary snap ====================
def find_longest_prefix(
    ids: List[int],
    start_idx: int,
    sent_text: str,
    char_ends: List[int],
    engine: InfiniGramEngine,
    max_span_len: int
) -> int:
    n = len(ids)
    max_possible = min(max_span_len, n - start_idx)
    if max_possible < MIN_SPAN_LENGTH:
        return 0

    if not exists_by_find(ids[start_idx:start_idx+MIN_SPAN_LENGTH], engine):
        return 0

    lo = MIN_SPAN_LENGTH
    hi = MIN_SPAN_LENGTH
    while True:
        nxt = min(hi * 2, max_span_len, n - start_idx)
        if nxt == hi:
            break
        if exists_by_find(ids[start_idx:start_idx+nxt], engine):
            hi = nxt
        else:
            break

    L_exist_lo, L_exist_hi = lo, hi
    while L_exist_lo < L_exist_hi:
        mid = (L_exist_lo + L_exist_hi + 1) // 2
        if exists_by_find(ids[start_idx:start_idx+mid], engine):
            L_exist_lo = mid
        else:
            L_exist_hi = mid - 1
    L_exist = L_exist_lo

    L = L_exist
    while L >= MIN_SPAN_LENGTH:
        end_idx = start_idx + L
        if is_end_word(sent_text, char_ends[end_idx - 1]):
            return L
        L -= 1
    return 0

# ==================== Self-contained (CLAUSE MODE) ====================
def make_self_contained_checker(stop_chars: set):
    """
    Forbid interior . ! ? and also any of the provided clause stop characters (e.g., , ;)
    inside the span body; they are allowed only at the very end (paired with sentence enders).
    """
    def self_contained_ok(surface: str) -> bool:
        s = surface
        if "\n" in s or "\r" in s:
            return False
        trimmed = s.rstrip()
        trimmed = re.sub(r'["\'\)\]\}»》\s]+$', '', trimmed)
        # allow terminal sentence ender, but not in the body
        if len(trimmed) > 0 and trimmed[-1] in SENT_END_CHARS:
            body = trimmed[:-1]
        else:
            body = s
        if SENT_END_RE.search(body):
            return False
        # NEW: forbid clause-stopper punctuation in body
        if any(ch in body for ch in stop_chars):
            return False
        return True
    return self_contained_ok

# ==================== Sentence-level collection ====================
def collect_maximal_spans_sentence(
    sent_text: str,
    tokenizer,
    engine: InfiniGramEngine,
    doc_token_offset: int,
    self_contained_ok,       # injected checker (clause mode)
    max_span_len: int
) -> Tuple[List[Tuple[int, int, str]], int]:
    ids, offs = get_token_data(sent_text, tokenizer)
    n = len(ids)
    if n < MIN_SPAN_LENGTH:
        return [], 0

    char_starts = [a for (a, b) in offs]
    char_ends   = [b for (a, b) in offs]

    initial_calls = ENGINE_CALLS
    per_suffix: List[Tuple[int, int]] = []
    surfaces: Dict[Tuple[int, int], str] = {}

    for s_tok in range(n):
        s_char = char_starts[s_tok]
        if not is_begin_word(sent_text, s_char):
            continue

        best_len = find_longest_prefix(ids, s_tok, sent_text, char_ends, engine, max_span_len)
        if best_len < MIN_SPAN_LENGTH:
            continue
        s0, e0 = s_tok, s_tok + best_len

        if not is_end_word(sent_text, char_ends[e0 - 1]):
            continue

        surf = surface_from_offsets(sent_text, char_starts, char_ends, s0, e0)
        if not self_contained_ok(surf):
            ok = False
            e_tmp = e0
            while e_tmp > s0 + MIN_SPAN_LENGTH:
                e_tmp -= 1
                if not is_end_word(sent_text, char_ends[e_tmp - 1]):
                    continue
                if not exists_by_find(ids[s0:e_tmp], engine):
                    continue
                surf2 = surface_from_offsets(sent_text, char_starts, char_ends, s0, e_tmp)
                if self_contained_ok(surf2):
                    e0 = e_tmp
                    surf = surf2
                    ok = True
                    break
            if not ok:
                continue

        if surf.strip() and len(surf.split()) >= MIN_SPAN_WORDS:
            per_suffix.append((s0, e0))
            surfaces[(s0, e0)] = surf

    per_suffix.sort(key=lambda x: (x[0], x[1]))
    kept: List[Tuple[int, int, str]] = []
    max_end = -1
    for s0, e0 in per_suffix:
        if e0 > max_end:
            kept.append((s0 + doc_token_offset, e0 + doc_token_offset, surfaces[(s0, e0)]))
            max_end = e0

    calls_used = ENGINE_CALLS - initial_calls
    return kept, calls_used

# ==================== Remove overlapping spans ====================
def remove_overlapping_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Remove overlapping spans, keeping the longer one when overlaps occur.
    Two spans overlap if they share any tokens.
    """
    if not spans:
        return []
    
    # Sort by length (descending), then by start position
    sorted_spans = sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0]))
    
    kept = []
    for span in sorted_spans:
        s, e, text = span
        # Check if this span overlaps with any already kept span
        overlaps = False
        for kept_s, kept_e, _ in kept:
            # Check for overlap: spans overlap if they share any positions
            if not (e <= kept_s or s >= kept_e):
                overlaps = True
                break
        
        if not overlaps:
            kept.append(span)
    
    # Sort by start position for output
    return sorted(kept, key=lambda x: x[0])

# ==================== (Optional) rarity diagnostics ====================
def build_unigram_neglog_cache(engine: InfiniGramEngine, N_tokens: int):
    tok_count: Dict[int, int] = {}
    def unigram_neglog(tid: int) -> float:
        if tid not in tok_count:
            c = engine.count(input_ids=[tid])["count"]
            tok_count[tid] = c
        c = tok_count[tid]
        p = (c / float(N_tokens)) if N_tokens else 0.0
        return -math.log(max(p, 0.5 / max(1.0, float(N_tokens))))
    return unigram_neglog

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="OLMoTrace-style maximal span detection (strict, keep-all, clause mode)")
    parser.add_argument("--index-dir", type=str, default="/mnt/infini/index", help="Path to Infini-gram index directory")
    parser.add_argument("--max-span-len", type=int, default=100, help="Cap on span length in tokens (default 100)")
    parser.add_argument("--clause-stops", type=str, default=",;", help="Characters that should not appear inside a span body (e.g., ',;')")
    args = parser.parse_args()

    # Build clause-stopper set (include common fullwidth variants)
    STOP_CHARS = set(args.clause_stops) | set(ch for ch in args.clause_stops.replace(",", "，").replace(";", "；"))
    self_contained_ok = make_self_contained_checker(STOP_CHARS)

    global ENGINE_CALLS, MAX_SPAN_LENGTH
    MAX_SPAN_LENGTH = args.max_span_len  # override global cap

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False, add_eos_token=False, use_fast=True
    )

    print(f"Initializing Infini-gram engine from {args.index_dir}...")
    try:
        engine = InfiniGramEngine(index_dir=args.index_dir, eos_token_id=tokenizer.eos_token_id)
        print("✓ Engine initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return

    # ===== Your hardcoded examples (keep these) =====
    outputs = {
        "business_ownership": """Business ownership refers to the legal right to operate, control, and profit from a business entity. In modern economies, business ownership plays a critical role in shaping the financial landscape, influencing innovation, and driving economic growth. Ownership can manifest in various forms, from sole proprietorships to partnerships, corporations, and cooperatives. Small family firms and large corporations alike are governed by principles of business ownership. In contemporary settings, business ownership is increasingly complex due to globalization and the rise of multinational corporations. These entities are often owned by a multitude of shareholders, each having a stake in the company's direction and financial performance. Further, the concept of business ownership has been extended through various investment vehicles and financial instruments, allowing individuals and institutions to own parts of businesses they may never directly interact with. Moreover, business ownership is not just limited to private entities; governments also own and operate various businesses in sectors considered essential to the public interest, such as utilities and transportation. This ownership model is referred to as state-owned enterprises and is prevalent in mixed economies. The discourse on business ownership often touches upon economic theories, corporate governance, and social responsibilities. It frames the discussion around who benefits from a company's profits and how decisions are made. Ownership is a crucial factor in shaping business strategies, corporate social responsibility initiatives, and the distribution of wealth and power within an economy. Thus, business ownership in modern economies is a multifaceted concept that influences virtually every aspect of economic life.""",
        "monopolistic_power": """Monopolistic power in modern economies refers to the ability of a single business entity to exert significant control over the market for a particular good or service. This power often stems from business ownership concentration, where a single firm or a small group of firms dominate the industry, thereby limiting competition. Monopolistic power can be achieved through various means, including barriers to entry for potential competitors, exclusive ownership of key resources, or significant economies of scale that make it difficult for smaller firms to compete effectively. Firms with monopolistic power can set prices that maximize their profits, potentially leading to higher prices and lower output levels than in a competitive market. Such firms may also use their market dominance to limit innovation and stifle potential competitors, ultimately harming consumers and reducing economic efficiency. When a few entities own and control a significant share of the business landscape within an industry, it becomes easier for them to manipulate market dynamics to their advantage. Policymakers and regulatory bodies often scrutinize and regulate monopolistic power to prevent unfair practices and maintain competitive markets. Antitrust laws and regulations are designed to promote fair competition and prevent harmful monopolistic practices. Balancing the benefits of efficiency and innovation with the potential downsides of excessive market concentration is an ongoing challenge in modern economies."""
    }

    print("=" * 70)
    print("OLMoTrace Maximal Span Detection — STRICT (KEEP ALL, CLAUSE MODE)")
    print("=" * 70 + "\n")

    results: Dict[str, List[Tuple[int, int, str]]] = {}
    per_doc_token_ids: Dict[str, List[int]] = {}

    for name, text in outputs.items():
        print(f"Analyzing: {name}")
        sentences = split_sentences(text)
        print(f"  Sentences: {len(sentences)}")

        full_ids, full_offs = get_token_data(text, tokenizer)
        full_char_starts = [a for (a, b) in full_offs]
        full_char_ends   = [b for (a, b) in full_offs]
        per_doc_token_ids[name] = full_ids

        doc_spans: List[Tuple[int, int, str]] = []
        total_tokens = 0
        total_calls  = 0

        for sent_idx, sent in enumerate(sentences, 1):
            ids, _ = get_token_data(sent, tokenizer)
            spans, calls = collect_maximal_spans_sentence(
                sent, tokenizer, engine, total_tokens, self_contained_ok, args.max_span_len
            )
            doc_spans.extend(spans)
            total_tokens += len(ids)
            total_calls  += calls
            print(f"    Sentence {sent_idx:2}: {len(ids)} toks, ~{calls} probes, {len(spans)} spans")

        print(f"  Maximal spans found: {len(doc_spans)} (clause stops: '{args.clause_stops}', max_len={args.max_span_len})")
        
        # Remove overlapping spans, keeping longer ones
        doc_spans_before = len(doc_spans)
        doc_spans = remove_overlapping_spans(doc_spans)
        if doc_spans_before != len(doc_spans):
            print(f"  Removed {doc_spans_before - len(doc_spans)} overlapping spans, kept {len(doc_spans)}")

        # Optional rarity diagnostics
        N_tokens = getattr(engine, "total_tokens", None) or getattr(engine, "num_tokens", None) or 10**12
        _neglog = build_unigram_neglog_cache(engine, N_tokens)  # kept for future use

        results[name] = doc_spans
        print(f"  Engine probes in doc: ~{total_calls}\n")

    raw_spans = {
        name: [
            {"start": s, "end": e, "tokens": e - s, "ids": per_doc_token_ids[name][s:e], "text": t}
            for (s, e, t) in results[name]
        ]
        for name in results
    }
    filtered_spans = raw_spans  # keep-all mode

    output = {
        "step": "OLMoTrace Maximal Span Detection (strict, keep-all, clause mode)",
        "constraints": [
            "Existence-first via FIND, boundary-snap later",
            "Self-contained: no interior sentence enders/newlines (allowed only at end)",
            "Clause mode: also forbid interior clause punctuation (configurable)",
            "Word boundaries: no mid-word; possessive 's guarded",
            "Maximality: keep spans whose end strictly increases (single-pass)",
            "Surfaces from original text offsets (no decode())",
            "Keep all maximal spans (no 5% filter)"
        ],
        "engine_stats": {
            "total_find_calls": ENGINE_CALLS,
            "index_dir": args.index_dir
        },
        "params": {
            "max_span_length": args.max_span_len,
            "min_span_length": MIN_SPAN_LENGTH,
            "min_span_words": MIN_SPAN_WORDS,
            "clause_stops": args.clause_stops
        },
        "raw_spans": raw_spans,
        "filtered_spans": filtered_spans
    }

    with open("maximal_spans.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✓ Saved to maximal_spans.json (shorter-clause spans; each includes token IDs)")

if __name__ == "__main__":
    main()
