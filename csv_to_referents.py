#!/usr/bin/env python3
"""
Extract referents from a source CSV (Topic / Referent / ...) into the standard
referents-input format used by generate_framing_phrases.py:

    domain:
      - <referent>
      - <referent>

Only the Topic and Referent columns are used. The framings (positive/negative)
are intentionally dropped -- those are what the generation step produces.

Usage:
    python csv_to_referents.py --input path/to.csv --out config/referents_from_csv.yaml
"""

import argparse
import csv
from collections import OrderedDict

import yaml

COL_TOPIC = "Topic"
COL_REFERENT = "Referent"


def slugify(topic: str) -> str:
    """Turn a human topic label into a compact domain key."""
    s = topic.strip().lower()
    keep = [c if (c.isalnum() or c == " ") else " " for c in s]
    return "_".join("".join(keep).split())


def convert(input_path: str):
    result = OrderedDict()
    skipped = 0
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = (row.get(COL_TOPIC) or "").strip()
            referent = (row.get(COL_REFERENT) or "").strip()
            if not (topic and referent):
                skipped += 1
                continue
            domain = slugify(topic)
            result.setdefault(domain, [])
            if referent not in result[domain]:      # de-dup within a domain
                result[domain].append(referent)
    return result, skipped


def main():
    ap = argparse.ArgumentParser(description="CSV -> referents-only YAML")
    ap.add_argument("--input", required=True, help="Source CSV file")
    ap.add_argument("--out", default="config/referents_from_csv.yaml",
                    help="Output YAML path")
    args = ap.parse_args()

    result, skipped = convert(args.input)

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(result), f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False)

    total = sum(len(v) for v in result.values())
    print(f"Wrote {total} referents across {len(result)} domains -> {args.out}")
    if skipped:
        print(f"Skipped {skipped} rows missing Topic/Referent.")
    for domain, items in result.items():
        print(f"  {domain}: {len(items)}")


if __name__ == "__main__":
    main()
