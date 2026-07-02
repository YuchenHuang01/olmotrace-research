#!/usr/bin/env python3
"""
Generate neutral underlying concepts for referents with opposing frames.
Uses Claude API to create academic, value-neutral descriptions.
"""

import os
import json
import yaml
import time
import re
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """
You are an academic writing assistant studying linguistic framing effects.

TASK
Given a REFERENT with opposing positive/negative frames, write a concise, strictly neutral
"Underlying concept" that defines what the referent is or involves—independent of either frame.

GUIDELINES
- Academic tone; avoid value-laden adjectives.
- Explain scope, typical contexts, and relevant actors/institutions if helpful.
- 1–2 sentences, ≤60 words total.
- Do NOT copy or echo the given frames; generalize beyond them.
- Output JSON only in this exact schema:
{
  "domain": "...",
  "referent": "...",
  "underlying_concept": "...",
  "notes": "optional (<=40 words)"
}
"""

USER_TMPL = """DOMAIN: {domain}
REFERENT: {referent}
POSITIVE_FRAME: {positive}
NEGATIVE_FRAME: {negative}

Return JSON only per the schema.
"""

BAD_WORDS = re.compile(
    r"\b(reckless|heroic|corrupt|tyrannical|elitist|magnate|crony|propaganda|visionary)\b",
    flags=re.I
)

def validate(obj):
    if not isinstance(obj, dict):
        return False, "not a dict"
    uc = obj.get("underlying_concept", "").strip()
    if not uc:
        return False, "missing underlying_concept"
    if len(uc.split()) > 60:
        return False, "too long"
    if BAD_WORDS.search(uc):
        return False, "judgmental language"
    return True, ""

def call_claude(client, domain, item, retries=3, backoff=1.5):
    user = USER_TMPL.format(domain=domain, **item)
    
    for k in range(retries):
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        
        # try to extract JSON if wrapped
        print(f"    Raw response: {text[:200]}...")  # Debug output
        m = re.search(r"\{[\s\S]*\}", text)
        try:
            obj = json.loads(m.group(0) if m else text)
        except Exception as e:
            print(f"    JSON parse error: {e}")
            print(f"    Trying to parse: {(m.group(0) if m else text)[:100]}...")
            if k < retries - 1:
                time.sleep(backoff * (k + 1))
                continue
            raise

        # fill domain/referent if omitted
        obj.setdefault("domain", domain)
        obj.setdefault("referent", item["referent"])

        ok, why = validate(obj)
        if ok:
            return obj
        
        if k < retries - 1:
            # nudge + retry
            nudged = user + f"\n\nNOTE: Your last answer violated the rules ({why}). Rewrite neutrally, 1–2 sentences, <=60 words."
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                temperature=0.1,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": nudged}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content).strip()
            m = re.search(r"\{[\s\S]*\}$", text)
            obj = json.loads(m.group(0) if m else text)
            obj.setdefault("domain", domain)
            obj.setdefault("referent", item["referent"])
            ok, _ = validate(obj)
            if ok:
                return obj
            time.sleep(backoff * (k + 1))
        else:
            raise RuntimeError(f"Validation failed: {why}")

def main(yaml_path="config/framing_pairs_generated.yaml",
         out_jsonl="underlying_concepts.jsonl",
         out_txt="underlying_concepts.txt"):
    
    print(f"Loading framing pairs from {yaml_path}...")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Check for API key
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as fj, open(out_txt, "w", encoding="utf-8") as ft:
        for domain, items in data.items():
            print(f"\nProcessing domain: {domain}")
            for i, item in enumerate(items):
                print(f"  [{i+1}/{len(items)}] {item['referent']}", end="... ")
                try:
                    obj = call_claude(client, domain, item)
                    fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    ft.write(
                        f"{obj['domain']} — {obj['referent']}\n"
                        f"Underlying concept: {obj['underlying_concept']}\n"
                        + (f"Notes: {obj['notes']}\n" if obj.get("notes") else "")
                        + "\n"
                    )
                    n += 1
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    # Optional: continue processing other items
                    continue
                
                # Rate limiting
                time.sleep(0.5)
                
    print(f"\nComplete! Wrote {n} concepts → {out_jsonl}, {out_txt}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate neutral underlying concepts for framing pairs")
    ap.add_argument("--yaml", default="config/framing_pairs_generated.yaml",
                    help="Input YAML file with framing pairs")
    ap.add_argument("--out_jsonl", default="underlying_concepts.jsonl",
                    help="Output JSONL file")
    ap.add_argument("--out_txt", default="underlying_concepts.txt",
                    help="Output text file")
    args = ap.parse_args()
    
    main(args.yaml, args.out_jsonl, args.out_txt)