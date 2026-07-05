#!/usr/bin/env python3
"""
Generate paired framing phrases for a list of referents using an OpenAI model.

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
from openai import OpenAI

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
# OpenAI call
# ----------------------------------------------------------------------
def call_openai(client, model, domain, referent, max_words, temperature,
                retries=3, backoff=1.5, feedback=""):
    user = USER_TMPL.format(referent=referent, domain=domain)
    if feedback:
        # Guidance from a previous screening attempt for this referent.
        user += ("\n\nThe previous pair was rejected by quality screening: "
                 f"{feedback}\nProduce a better pair that fixes this.")

    last_why = ""
    for k in range(retries):
        user_msg = user if k == 0 else (
            user + f"\n\nNOTE: your previous answer was invalid ({last_why}). "
                   f"Return two short labels (<= {max_words} words each) for the "
                   f"SAME referent, JSON only."
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
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
# Quality screening (LLM judge + thresholds)
# ----------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are a strict evaluator of framing pairs for a political-framing study.

You are given a REFERENT and two candidate labels for it:
  - POSITIVE: intended as a positive/neutral framing
  - NEGATIVE: intended as a negative/charged framing

Assess, critically and independently:
1. same_referent (bool): do BOTH labels denote the SAME underlying referent?
   (They must be competing names for the same thing, not different things.)
2. ref_fidelity_pos, ref_fidelity_neg (0.0-1.0): how faithfully each label
   captures the given referent.
3. sentiment_pos, sentiment_neg (-1.0 to 1.0): the political framing tone of
   each label, where -1 is very negative/pejorative, 0 is neutral, +1 is very
   positive/favorable.

Output JSON only, exactly:
{"same_referent": true, "ref_fidelity_pos": 0.0, "ref_fidelity_neg": 0.0,
 "sentiment_pos": 0.0, "sentiment_neg": 0.0, "reason": "<= 25 words"}
"""

JUDGE_USER_TMPL = """REFERENT: {referent}
DOMAIN: {domain}
POSITIVE: {positive}
NEGATIVE: {negative}

Return JSON only per the schema."""


def judge_pair(client, judge_model, domain, referent, positive, negative,
               retries=3, backoff=1.5):
    """Score a candidate pair. Returns a metrics dict (with polarity_delta)."""
    user = JUDGE_USER_TMPL.format(referent=referent, domain=domain,
                                  positive=positive, negative=negative)
    for k in range(retries):
        resp = client.chat.completions.create(
            model=judge_model,
            temperature=0,                      # stable, deterministic scoring
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        try:
            m = parse_json((resp.choices[0].message.content or "").strip())
        except Exception:
            time.sleep(backoff * (k + 1))
            continue
        metrics = {
            "same_referent": bool(m.get("same_referent", False)),
            "ref_fidelity_pos": float(m.get("ref_fidelity_pos", 0.0)),
            "ref_fidelity_neg": float(m.get("ref_fidelity_neg", 0.0)),
            "sentiment_pos": float(m.get("sentiment_pos", 0.0)),
            "sentiment_neg": float(m.get("sentiment_neg", 0.0)),
            "reason": str(m.get("reason", "")).strip(),
        }
        metrics["polarity_delta"] = round(
            metrics["sentiment_pos"] - metrics["sentiment_neg"], 3)
        return metrics
    raise RuntimeError("Judge failed to return valid JSON")


def passes(metrics, positive, negative, th):
    """Apply screening thresholds. Returns (ok: bool, reasons: list[str])."""
    reasons = []
    if not metrics["same_referent"]:
        reasons.append("labels do not refer to the same referent")
    if metrics["ref_fidelity_pos"] < th["min_ref_fidelity"]:
        reasons.append(f"positive label weakly matches referent "
                       f"({metrics['ref_fidelity_pos']:.2f})")
    if metrics["ref_fidelity_neg"] < th["min_ref_fidelity"]:
        reasons.append(f"negative label weakly matches referent "
                       f"({metrics['ref_fidelity_neg']:.2f})")
    if metrics["polarity_delta"] < th["min_polarity_delta"]:
        reasons.append(f"polarity gap too small "
                       f"({metrics['polarity_delta']:.2f}); make the negative "
                       f"more charged and the positive more neutral/favorable")
    if metrics["sentiment_neg"] > th["max_neg_sentiment"]:
        reasons.append(f"negative label not charged enough "
                       f"({metrics['sentiment_neg']:.2f})")
    if positive.strip().lower() == negative.strip().lower():
        reasons.append("labels are identical")
    return (len(reasons) == 0), reasons


def quality_score(metrics):
    """Composite score to pick the best attempt when none fully passes."""
    return (metrics["polarity_delta"]
            + metrics["ref_fidelity_pos"] + metrics["ref_fidelity_neg"]
            + (0.5 if metrics["same_referent"] else 0.0))


def generate_screened(client, args, th, domain, referent):
    """Generate -> judge -> regenerate until thresholds pass or attempts run out.

    Returns (record_dict, passed_bool).
    """
    best = None
    feedback = ""
    attempt = 0
    for attempt in range(1, args.max_attempts + 1):
        obj = call_openai(client, args.model, domain, referent,
                          args.max_words, args.temperature, feedback=feedback)
        metrics = judge_pair(client, args.judge_model, domain, referent,
                             obj["positive"], obj["negative"])
        ok, reasons = passes(metrics, obj["positive"], obj["negative"], th)
        record = {**obj, "attempts": attempt, "passed": ok, "metrics": metrics}
        if ok:
            return record, True
        # keep the best-so-far by composite score
        if best is None or quality_score(metrics) > quality_score(best["metrics"]):
            best = record
        feedback = "; ".join(reasons)
        time.sleep(args.sleep)
    best["attempts"] = attempt   # total attempts spent (not the best attempt's index)
    return best, False


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
    ap = argparse.ArgumentParser(description="Generate framing phrase pairs with OpenAI")
    ap.add_argument("--input", default="config/referents_from_csv.yaml",
                    help="YAML mapping domain -> list of referents (strings or {referent:...})")
    ap.add_argument("--out-yaml", default="config/framing_pairs_generated.yaml",
                    help="Output YAML (framing_pairs shape)")
    ap.add_argument("--out-jsonl", default="framing_pairs_generated.jsonl",
                    help="Output JSONL (one record per referent)")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model id (generator)")
    ap.add_argument("--judge-model", default="gpt-4o", help="OpenAI model id (screening judge)")
    ap.add_argument("--max-words", type=int, default=6, help="Max words per label")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between calls")
    # --- screening thresholds ---
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="Regeneration attempts per referent before keeping best")
    ap.add_argument("--min-ref-fidelity", type=float, default=0.6,
                    help="Min referent-fidelity (0-1) required for each label")
    ap.add_argument("--min-polarity-delta", type=float, default=0.4,
                    help="Min sentiment gap (pos - neg) required")
    ap.add_argument("--max-neg-sentiment", type=float, default=-0.15,
                    help="Negative label sentiment must be <= this")
    args = ap.parse_args()

    th = {
        "min_ref_fidelity": args.min_ref_fidelity,
        "min_polarity_delta": args.min_polarity_delta,
        "max_neg_sentiment": args.max_neg_sentiment,
    }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in your environment or .env")
    client = OpenAI(api_key=api_key)

    referents_by_domain = load_referents(args.input)
    total = sum(len(v) for v in referents_by_domain.values())
    print(f"Loaded {total} referents across {len(referents_by_domain)} domains "
          f"from {args.input}\nModel: {args.model}\n")

    result = {}
    n_ok = n_pass = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fj:
        for domain, referents in referents_by_domain.items():
            print(f"Domain: {domain}")
            result[domain] = []
            for i, referent in enumerate(referents, 1):
                print(f"  [{i}/{len(referents)}] {referent}", end=" ... ")
                try:
                    record, ok = generate_screened(client, args, th, domain, referent)
                except Exception as e:
                    print(f"✗ {e}")
                    time.sleep(args.sleep)
                    continue

                m = record["metrics"]
                result[domain].append({
                    "referent": record["referent"],
                    "positive": record["positive"],
                    "negative": record["negative"],
                    "passed": record["passed"],
                    "polarity_delta": m["polarity_delta"],
                })
                fj.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1
                n_pass += int(ok)
                flag = "✓" if ok else "⚠ best"
                print(f"{flag} [{record['attempts']}x] "
                      f"+'{record['positive']}' -'{record['negative']}' "
                      f"Δ={m['polarity_delta']:.2f}"
                      + ("" if ok else f"  ({m['reason']})"))
                time.sleep(args.sleep)

    with open(args.out_yaml, "w", encoding="utf-8") as fy:
        yaml.safe_dump(result, fy, sort_keys=False, allow_unicode=True)

    print(f"\nDone. {n_ok}/{total} referents processed; "
          f"{n_pass} passed screening, {n_ok - n_pass} kept as best-effort.")
    print(f"  YAML : {args.out_yaml}")
    print(f"  JSONL: {args.out_jsonl}  (full metrics per referent)")


if __name__ == "__main__":
    main()
