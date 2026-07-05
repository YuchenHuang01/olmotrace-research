#!/usr/bin/env python3
"""
End-to-end demo: OLMo output -> DJ span detection -> document retrieval.

Uses a real OLMo-7B-Instruct generation and the Dolma index (OLMo-7B's training
corpus, which also exposes source URLs).
"""
import json
import logging
import os
import sys
import textwrap

# This script lives in outputs/; add the repo root so imports resolve when run
# from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

logging.basicConfig(level=logging.WARNING)

from framing_audit.infinigram import InfinigramClient
from framing_audit.find_spans_dj import DJSpanFinder
from retrieve_span_documents import search_span_documents

INDEX = "v4_olmo-mix-1124_llama"
MIN_NGRAM = 5

# --- 1. OLMo output paragraph (local sample, no external dependency) ---
olmo = json.load(open(os.path.join(_HERE, "olmo_full_input.json")))
text = olmo[0]["text"]

print("=" * 90)
print("1) OLMO-7B-INSTRUCT OUTPUT PARAGRAPH")
print("=" * 90)
print(textwrap.fill(text, 90))

# --- 2. Span detection (DJ Search exact, over Infini-gram count) ---
client = InfinigramClient()
finder = DJSpanFinder(client, index=INDEX, min_ngram=MIN_NGRAM, request_sleep=0.25)
result = finder.find_spans(text)
spans = result["matched_spans"]

print("\n" + "=" * 90)
print(f"2) SPANS DETECTED  (index={INDEX}, min_ngram={MIN_NGRAM})")
print(f"   coverage={result['coverage']:.3f}  avg_span_len={result['avg_span_len']:.1f}  "
      f"n_spans={len(spans)}")
print("=" * 90)
for s in spans:
    print(f"  occ={s['occurrence']:>8}  [{s['start_index']:>3}:{s['end_index']:<3}]  "
          f"\"{s['span_text']}\"")

# --- 3. Document retrieval for a couple of example spans ---
# pick the 2 longest spans (most distinctive) that are not ultra-frequent
ranked = sorted(spans, key=lambda s: (s["end_index"] - s["start_index"]), reverse=True)
examples = ranked[:2]

print("\n" + "=" * 90)
print("3) RETRIEVED TRAINING DOCUMENTS  (find -> get_doc_by_rank -> URL)")
print("=" * 90)
for s in examples:
    print(f"\n>>> span: \"{s['span_text']}\"  (occurrence={s['occurrence']})")
    docs = search_span_documents(client, s["span_text"], max_docs=2)
    if not docs:
        print("    (no documents retrieved)")
        continue
    for d in docs:
        snippet = ""
        dm = d.get("document_metadata", {})
        # snippet text if present
        for key in ("spans", "text", "snippet"):
            if isinstance(dm, dict) and dm.get(key):
                snippet = str(dm[key]); break
        print(f"    - matched n-gram : \"{d.get('span_candidate')}\"")
        print(f"      shard/rank     : {d.get('shard')}/{d.get('rank')}")
        print(f"      URL            : {d.get('url')}")
