#!/usr/bin/env python3
"""
Flatten framing_pairs_generated.jsonl (generation + screening metrics) into a CSV.

Usage:
    python jsonl_to_csv.py --input framing_pairs_generated.jsonl \
                           --out framing_pairs_generated.csv
"""

import argparse
import csv
import json

COLUMNS = [
    ("Topic", lambda r, m: r.get("domain", "")),
    ("Referent", lambda r, m: r.get("referent", "")),
    ("Phrase 1 (neutral framing)", lambda r, m: r.get("positive", "")),
    ("Phrase 2 (charged framing)", lambda r, m: r.get("negative", "")),
    ("Sentiment P1", lambda r, m: m.get("sentiment_pos", "")),
    ("Sentiment P2", lambda r, m: m.get("sentiment_neg", "")),
    ("Polarity Delta (P1-P2)", lambda r, m: m.get("polarity_delta", "")),
    ("Ref Fidelity P1", lambda r, m: m.get("ref_fidelity_pos", "")),
    ("Ref Fidelity P2", lambda r, m: m.get("ref_fidelity_neg", "")),
    ("Same Referent", lambda r, m: m.get("same_referent", "")),
    ("Passed", lambda r, m: r.get("passed", "")),
    ("Attempts", lambda r, m: r.get("attempts", "")),
    ("Judge Reason", lambda r, m: m.get("reason", "")),
]


def main():
    ap = argparse.ArgumentParser(description="framing pairs JSONL -> CSV")
    ap.add_argument("--input", default="framing_pairs_generated.jsonl")
    ap.add_argument("--out", default="framing_pairs_generated.csv")
    args = ap.parse_args()

    rows = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([name for name, _ in COLUMNS])
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            m = r.get("metrics", {})
            writer.writerow([fn(r, m) for _, fn in COLUMNS])
            rows += 1

    print(f"Wrote {rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
