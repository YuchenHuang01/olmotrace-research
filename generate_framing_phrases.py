#!/usr/bin/env python3
"""
Generate paired framing phrases for a list of referents using a Gemini model.

For each REFERENT we ask the model for two competing labels that denote the
*same* referent but frame it differently:
  - positive_neutral : a positive or neutral framing
  - negative         : a negative or pejorative framing

Canonical example (referent = "people living in a country without legal
authorization"):
    positive_neutral -> "undocumented immigrants"
    negative         -> "illegal aliens"

Input  : a YAML mapping domain -> list of referents. Each list item may be a
         plain string, or a dict with a `referent` key (so you can point this
         at config/framing_pairs_five_domains.yaml and reuse its referents;
         any existing positive/negative fields are ignored).

Output : - a YAML file in the same shape as framing_pairs_*.yaml
           (domain -> [{referent, positive, negative}, ...])
         - a JSONL file with one record per referent (for logging/analysis)
"""

import argparse
import json
import os
import re
import time

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a linguistics researcher studying framing effects in political and media discourse.

TASK
Given a REFERENT (a person, group, policy, event, or thing), produce TWO alternative
labels that both denote the SAME referent but frame it differently:
  - positive_neutral: a positive or neutral framing
  - negative:         a negative or pejorative framing

REQUIREMENTS
- Both labels MUST refer to the same underlying referent. They are competing names
  for the exact same thing, NOT two different things.
- Keep each label SHORT: a noun phrase of roughly 2-5 words. No sentences, no
  explanations, no punctuation at the end.
- Use natural wording that actually appears in real-world discourse.

WORKED EXAMPLE (study this carefully)
Suppose the input is:
    REFERENT: people living in a country without legal authorization
Both of the following labels name that SAME group of people; they differ only in tone:
    - "undocumented immigrants" is the positive_neutral framing (sympathetic/neutral).
    - "illegal aliens"          is the negative framing (pejorative).
So the correct output for that input is EXACTLY:
    {"positive_neutral": "undocumented immigrants", "negative": "illegal aliens"}
Notice: both labels refer to the same people, each is a short noun phrase, and the
output is JSON only with no extra text.

OUTPUT
Now do the same for the REFERENT you are given. Return JSON only, exactly:
{"positive_neutral": "...", "negative": "..."}
"""

USER_TMPL = """REFERENT: {referent}
DOMAIN: {domain}

Return JSON only per the schema."""

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate(obj, max_words):
    if not isinstance(obj, dict):
        return False, "not a dict"
    pos = str(obj.get("positive_neutral", "")).strip()
    neg = str(obj.get("negative", "")).strip()
    if not pos or not neg:
        return False, "missing positive_neutral or negative"
    if len(pos.split()) > max_words or len(neg.split()) > max_words:
        return False, f"a label exceeds {max_words} words"
    if pos.lower() == neg.lower():
        return False, "the two labels are identical"
    return True, ""


def parse_json(text):
    """Extract the first JSON object from a model response."""
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group(0) if m else text)


# ----------------------------------------------------------------------
# Gemini call
# ----------------------------------------------------------------------
def call_gemini(client, model, domain, referent, max_words, temperature,
                retries=3, backoff=1.5):
    user = USER_TMPL.format(referent=referent, domain=domain)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
        response_mime_type="application/json",
    )

    last_why = ""
    for k in range(retries):
        contents = user if k == 0 else (
            user + f"\n\nNOTE: your previous answer was invalid ({last_why}). "
                   f"Return two short labels (<= {max_words} words each) for the "
                   f"SAME referent, JSON only."
        )
        resp = client.models.generate_content(
            model=model, contents=contents, config=config
        )
        text = (resp.text or "").strip()
        try:
            obj = parse_json(text)
        except Exception as e:
            last_why = f"JSON parse error: {e}"
            time.sleep(backoff * (k + 1))
            continue

        obj = {
            "domain": domain,
            "referent": referent,
            "positive": str(obj.get("positive_neutral", "")).strip(),
            "negative": str(obj.get("negative", "")).strip(),
        }
        ok, why = validate(
            {"positive_neutral": obj["positive"], "negative": obj["negative"]},
            max_words,
        )
        if ok:
            return obj
        last_why = why
        time.sleep(backoff * (k + 1))

    raise RuntimeError(f"Validation failed after {retries} tries: {last_why}")


# ----------------------------------------------------------------------
# Input loading
# ----------------------------------------------------------------------
def load_referents(path):
    """domain -> [referent, ...]; accepts items as strings or {referent: ...} dicts."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Input YAML must map domain -> list of referents")

    out = {}
    for domain, items in data.items():
        referents = []
        for item in items or []:
            if isinstance(item, str):
                referents.append(item)
            elif isinstance(item, dict) and item.get("referent"):
                referents.append(item["referent"])
        if referents:
            out[domain] = referents
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate framing phrase pairs with Gemini")
    ap.add_argument("--input", default="config/referents_from_csv.yaml",
                    help="YAML mapping domain -> list of referents (strings or {referent:...})")
    ap.add_argument("--out-yaml", default="config/framing_pairs_generated.yaml",
                    help="Output YAML (framing_pairs shape)")
    ap.add_argument("--out-jsonl", default="framing_pairs_generated.jsonl",
                    help="Output JSONL (one record per referent)")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model id")
    ap.add_argument("--max-words", type=int, default=6, help="Max words per label")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between calls")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env")
    client = genai.Client(api_key=api_key)

    referents_by_domain = load_referents(args.input)
    total = sum(len(v) for v in referents_by_domain.values())
    print(f"Loaded {total} referents across {len(referents_by_domain)} domains "
          f"from {args.input}\nModel: {args.model}\n")

    result = {}
    n_ok = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fj:
        for domain, referents in referents_by_domain.items():
            print(f"Domain: {domain}")
            result[domain] = []
            for i, referent in enumerate(referents, 1):
                print(f"  [{i}/{len(referents)}] {referent}", end=" ... ")
                try:
                    obj = call_gemini(
                        client, args.model, domain, referent,
                        args.max_words, args.temperature,
                    )
                    result[domain].append({
                        "referent": obj["referent"],
                        "positive": obj["positive"],
                        "negative": obj["negative"],
                    })
                    fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n_ok += 1
                    print(f"✓  +'{obj['positive']}'  -'{obj['negative']}'")
                except Exception as e:
                    print(f"✗ {e}")
                time.sleep(args.sleep)

    with open(args.out_yaml, "w", encoding="utf-8") as fy:
        yaml.safe_dump(result, fy, sort_keys=False, allow_unicode=True)

    print(f"\nDone. {n_ok}/{total} pairs generated.")
    print(f"  YAML : {args.out_yaml}")
    print(f"  JSONL: {args.out_jsonl}")


if __name__ == "__main__":
    main()
