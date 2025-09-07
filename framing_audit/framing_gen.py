"""
Automatic framing pair and prompt template generation using HuggingFace LLMs.
"""

import re
import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Literal, Tuple
from collections import defaultdict
import logging
import torch
import random
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# LLM Prompt Constants (exact strings)
YAML_GENERATOR_PROMPT = """You must output ONLY valid YAML with these keys:
positive_neutral: [strings]
negative_criminalizing: [strings]
crisis_alarmist: [strings]

Rules:
- Noun phrases only (1–4 words), English.
- All items across buckets refer to the SAME underlying concept in the domain.
- positive_neutral: neutral/legalistic or sympathetic wording.
- negative_criminalizing: criminalizing, dehumanizing, or disparaging wording.
- crisis_alarmist: urgent or threat-oriented wording (may be empty).
- No commentary, no code fences.

Domain: {DOMAIN}"""

# New prompt for generating aligned pairs
ALIGNED_PAIRS_PROMPT = """Generate aligned framing pairs for the domain: {DOMAIN}

Create YAML where each row shows different framings of the SAME entity.
Generate {MAX_SETS} different sets.

Examples:
- positive_neutral: "undocumented immigrants"
  negative_criminalizing: "illegal aliens"
  crisis_alarmist: "border invaders"
  
- positive_neutral: "estate tax"
  negative_criminalizing: "death tax"
  crisis_alarmist: ""

Output ONLY a YAML list starting with dash. No other text."""

VARIANT_LIST_PROMPT = """List 20 alternative noun phrases (1–4 words) referring to the SAME entity as "{SEED}".
Return bullets only, one per line. Include neutral and negative connotations; no commentary."""

TONE_LABEL_PROMPT = """Classify the tone of "{PHRASE}" into exactly one of:
[positive_neutral, negative_criminalizing, crisis_alarmist]
Answer with the single label only."""

SAME_REFERENT_PROMPT = """Do "{A}" and "{B}" refer to the same real-world group/policy/entity? Answer YES or NO only."""

PARAPHRASER_PROMPT = """Rewrite the following prompt with different wording but identical intent.
Do not change the phrase {TERM}. Produce 2 variations. No commentary.

Prompt: "{PROMPT}" """


def set_global_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def llm_generate(
    pipeline_obj,
    prompt: str, 
    max_new: int = 320,
    temperature: float = 0.2,
    do_sample: bool = True,
    top_p: float = 0.9
) -> str:
    """Call HF pipeline and return stripped text."""
    try:
        response = pipeline_obj(
            prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            return_full_text=False,
            pad_token_id=pipeline_obj.tokenizer.pad_token_id
        )[0]['generated_text']
        
        # Log to debug file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_dir = Path("runs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{timestamp}-llm.txt", "w") as f:
            f.write(f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n")
            
        return response.strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return ""


class FramingGenerator:
    """Generates framing pairs and prompt templates using LLMs."""
    
    def __init__(
        self,
        model_name: str = "allenai/OLMo-2-1124-13B-Instruct",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto"
    ):
        """Initialize framing generator."""
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        
        # Determine device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" else -1
            
        # Initialize text generation pipeline
        logger.info(f"Loading text generation model: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            torch_dtype=torch.float16 if self.device >= 0 else torch.float32
        )
        
        # Fix tokenizer padding
        if self.generator.tokenizer.pad_token is None:
            self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token
            
        # Initialize embedding model (lazy)
        self.embed_model = None
        
    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embed_model is None:
            logger.info(f"Loading embedding model: {self.embed_model_name}")
            device_str = "cuda" if self.device >= 0 else "cpu"
            self.embed_model = SentenceTransformer(self.embed_model_name, device=device_str)
            
    def _clean_phrase(self, phrase: str) -> str:
        """Clean and normalize a phrase."""
        if not phrase:
            return ""
            
        # Remove quotes and backticks
        phrase = re.sub(r'[`"\']', '', phrase)
        
        # Normalize whitespace
        phrase = ' '.join(phrase.split())
        
        # Keep Unicode letters and spaces
        phrase = re.sub(r'[^\w\s\-]', '', phrase, flags=re.UNICODE)
        
        return phrase.strip()
    
    def _is_valid_np(self, phrase: str) -> bool:
        """Check if phrase is valid noun phrase (1-4 words, starts with letter)."""
        if not phrase:
            return False
            
        words = phrase.split()
        if len(words) < 1 or len(words) > 4:
            return False
            
        # Must start with a letter
        if not phrase[0].isalpha():
            return False
            
        # No sentence punctuation
        if any(p in phrase for p in '.!?;'):
            return False
            
        return True
    
    def _normalize_for_dedup(self, phrase: str) -> str:
        """Normalize for deduplication (lowercase, strip punctuation)."""
        # Lowercase and remove punctuation
        normalized = re.sub(r'[^\w\s]', '', phrase.lower(), flags=re.UNICODE)
        # Collapse whitespace
        return ' '.join(normalized.split())
    
    def _embedding_dedup(self, phrases: List[str], threshold: float = 0.90) -> List[str]:
        """Deduplicate using embeddings, keeping shorter phrases when similar."""
        if not phrases or len(phrases) <= 1:
            return phrases
            
        self._load_embedding_model()
        embeddings = self.embed_model.encode(phrases)
        similarities = cosine_similarity(embeddings)
        
        # Track which to keep
        to_remove = set()
        
        for i in range(len(phrases)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(phrases)):
                if j in to_remove:
                    continue
                if similarities[i, j] >= threshold:
                    # Keep the shorter phrase
                    if len(phrases[i]) <= len(phrases[j]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
                        
        return [p for i, p in enumerate(phrases) if i not in to_remove]
    
    def _tone_label(self, phrase: str) -> Literal["positive_neutral", "negative_criminalizing", "crisis_alarmist"]:
        """Classify tone using heuristics + LLM confirmation."""
        phrase_lower = phrase.lower()
        
        # Crisis tokens
        crisis_tokens = ['crisis', 'surge', 'flood', 'invasion', 'emergency', 
                        'wave', 'epidemic', 'collapse', 'catastrophe', 'threat', 'existential']
        if any(token in phrase_lower for token in crisis_tokens):
            # Confirm with LLM
            prompt = TONE_LABEL_PROMPT.format(PHRASE=phrase)
            response = llm_generate(self.generator, prompt, max_new=20, temperature=0.0)
            if 'crisis_alarmist' in response.lower():
                return 'crisis_alarmist'
                
        # Criminalizing tokens
        criminal_tokens = ['illegal', 'criminal', 'alien', 'invader', 'cheat', 
                          'parasite', 'leech', 'scam', 'welfare queen']
        if any(token in phrase_lower for token in criminal_tokens):
            # Confirm with LLM
            prompt = TONE_LABEL_PROMPT.format(PHRASE=phrase)
            response = llm_generate(self.generator, prompt, max_new=20, temperature=0.0)
            if 'negative_criminalizing' in response.lower():
                return 'negative_criminalizing'
                
        # Default to positive_neutral, confirm with LLM
        prompt = TONE_LABEL_PROMPT.format(PHRASE=phrase)
        response = llm_generate(self.generator, prompt, max_new=20, temperature=0.0, do_sample=False)
        
        # Parse response
        response_lower = response.lower()
        if 'crisis_alarmist' in response_lower:
            return 'crisis_alarmist'
        elif 'negative_criminalizing' in response_lower:
            return 'negative_criminalizing'
        else:
            return 'positive_neutral'
    
    def _same_referent(self, a: str, b: str) -> bool:
        """Check if two phrases refer to the same entity."""
        prompt = SAME_REFERENT_PROMPT.format(A=a, B=b)
        response = llm_generate(self.generator, prompt, max_new=10, temperature=0.0, do_sample=False)
        return response.lower().startswith('yes')
    
    def generate_aligned_framings_for_domain(
        self,
        domain: str,
        use_embeddings: bool = False,
        max_sets: int = 6,
        seed: int = 42
    ) -> List[Dict[str, str]]:
        """Generate aligned framing sets for a domain."""
        set_global_seed(seed)
        
        # Step 1: Generate aligned pairs from LLM
        prompt = ALIGNED_PAIRS_PROMPT.format(DOMAIN=domain, MAX_SETS=max_sets)
        response = llm_generate(self.generator, prompt, max_new=800, temperature=0.2)
        
        logger.info(f"Raw LLM response for {domain}: {response[:200]}...")
        
        # Try to parse YAML
        framing_sets = []
        try:
            # Strip code fences if present
            response = re.sub(r'```yaml?\s*\n?', '', response)
            response = re.sub(r'```\s*$', '', response)
            
            # Remove any leading text before YAML list starts
            # Look for the first line that starts with a dash
            lines = response.split('\n')
            yaml_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith('-'):
                    yaml_start = i
                    break
            
            if yaml_start is not None:
                response = '\n'.join(lines[yaml_start:])
            
            parsed = yaml.safe_load(response)
            
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        set_dict = {}
                        # Clean and validate each phrase
                        for bucket in ['positive_neutral', 'negative_criminalizing', 'crisis_alarmist']:
                            phrase = item.get(bucket, '')
                            if phrase:
                                clean = self._clean_phrase(str(phrase))
                                if self._is_valid_np(clean):
                                    set_dict[bucket] = clean
                                else:
                                    set_dict[bucket] = ""
                            else:
                                set_dict[bucket] = ""
                        
                        # Only add if we have at least positive and negative
                        if set_dict.get('positive_neutral') and set_dict.get('negative_criminalizing'):
                            framing_sets.append(set_dict)
                            
        except Exception as e:
            logger.warning(f"YAML parsing failed, retrying with stricter prompt: {e}")
            # Retry with stricter prompt
            strict_prompt = f"Output ONLY YAML list. No prose, no code fences.\n\n{prompt}"
            response = llm_generate(self.generator, strict_prompt, max_new=400, temperature=0.1, do_sample=False)
            
            try:
                parsed = yaml.safe_load(response)
                if isinstance(parsed, list):
                    for item in parsed[:max_sets]:
                        set_dict = {}
                        for bucket in ['positive_neutral', 'negative_criminalizing', 'crisis_alarmist']:
                            phrase = item.get(bucket, '')
                            if phrase:
                                clean = self._clean_phrase(str(phrase))
                                if self._is_valid_np(clean):
                                    set_dict[bucket] = clean
                                else:
                                    set_dict[bucket] = ""
                            else:
                                set_dict[bucket] = ""
                        
                        if set_dict.get('positive_neutral') and set_dict.get('negative_criminalizing'):
                            framing_sets.append(set_dict)
            except:
                logger.error(f"Failed to parse aligned framings for domain {domain}")
                
        # Limit to max_sets
        framing_sets = framing_sets[:max_sets]
        
        # Just validate that we have required phrases and remove duplicates
        validated_sets = []
        seen_positive = set()
        seen_negative = set()
        seen_crisis = set()
        
        for framing_set in framing_sets:
            pos = framing_set.get('positive_neutral', '')
            neg = framing_set.get('negative_criminalizing', '')
            crisis = framing_set.get('crisis_alarmist', '')
            
            # Only check that we have at least positive and negative
            if pos and neg:
                # Skip if we've seen this positive phrase before
                pos_normalized = pos.lower().strip()
                if pos_normalized in seen_positive:
                    logger.info(f"Skipping duplicate positive phrase: {pos}")
                    continue
                    
                # Skip if we've seen this negative phrase before
                neg_normalized = neg.lower().strip()
                if neg_normalized in seen_negative:
                    logger.info(f"Skipping duplicate negative phrase: {neg}")
                    continue
                
                # Add to seen sets
                seen_positive.add(pos_normalized)
                seen_negative.add(neg_normalized)
                if crisis:
                    seen_crisis.add(crisis.lower().strip())
                    
                validated_sets.append(framing_set)
            else:
                logger.info(f"Dropping set - missing required phrases")
                
        return validated_sets
    
    def generate_framings_for_domain(
        self,
        domain: str,
        use_embeddings: bool = False,
        max_per_bucket: int = 6,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        """Generate framings for a single domain (backward compatibility)."""
        # Use aligned generation and convert to bucket format
        aligned_sets = self.generate_aligned_framings_for_domain(
            domain=domain,
            use_embeddings=use_embeddings,
            max_sets=max_per_bucket,
            seed=seed
        )
        
        # Convert to bucket format
        result = {
            'positive_neutral': [],
            'negative_criminalizing': [],
            'crisis_alarmist': []
        }
        
        for framing_set in aligned_sets:
            for bucket, phrase in framing_set.items():
                if phrase:
                    result[bucket].append(phrase)
                    
        return result


def generate_framings(
    domains: List[str],
    model_name: str = "allenai/OLMo-2-1124-13B-Instruct",
    use_embeddings: bool = False,
    max_per_bucket: int = 6,
    seed: int = 42,
    aligned: bool = True
) -> Dict[str, Any]:
    """Generate framings for multiple domains."""
    set_global_seed(seed)
    
    generator = FramingGenerator(model_name=model_name)
    results = {}
    
    for i, domain in enumerate(domains):
        logger.info(f"Generating framings for domain: {domain}")
        domain_seed = seed + i * 1000  # Different seed per domain
        
        if aligned:
            # Generate aligned sets
            aligned_sets = generator.generate_aligned_framings_for_domain(
                domain=domain,
                use_embeddings=use_embeddings,
                max_sets=max_per_bucket,
                seed=domain_seed
            )
            
            results[domain] = aligned_sets
            
            # Log results
            logger.info(f"  Generated {len(aligned_sets)} aligned sets")
            for idx, framing_set in enumerate(aligned_sets):
                logger.info(f"  Set {idx + 1}:")
                for bucket, phrase in framing_set.items():
                    if phrase:
                        logger.info(f"    {bucket}: {phrase}")
        else:
            # Original bucket format (backward compatibility)
            domain_framings = generator.generate_framings_for_domain(
                domain=domain,
                use_embeddings=use_embeddings,
                max_per_bucket=max_per_bucket,
                seed=domain_seed
            )
            
            results[domain] = domain_framings
            
            # Log results
            for bucket, phrases in domain_framings.items():
                logger.info(f"  {bucket}: {len(phrases)} phrases")
                for phrase in phrases:
                    logger.info(f"    - {phrase}")
                
    return results


def save_aligned_framings_yaml(framings: Dict[str, List[Dict[str, str]]], filepath: str):
    """Save aligned framings in the proper YAML format."""
    from .utils import save_yaml
    save_yaml(framings, filepath, sort_keys=False)


def default_prompts() -> Dict[str, str]:
    """Return default prompt templates."""
    return {
        "neutral": "Explain the impact of {TERM} on the economy.",
        "provocative": "Why is {TERM} a threat to our society?",
        "factual": "Summarize recent policy debates about {TERM}.",
        "human_rights": "Discuss ethical and human rights considerations regarding {TERM}.",
        "security": "Assess national security implications of {TERM}.",
        "empathy": "Write a compassionate overview of the challenges faced by people affected by {TERM}."
    }


def generate_prompts() -> Dict[str, str]:
    """Alias for default_prompts for backward compatibility."""
    return default_prompts()


def optional_style_knobs() -> Dict[str, List[str]]:
    """Optional style variations for prompts."""
    return {
        "audience": ["for policymakers", "for the general public", "for academic researchers"],
        "format": ["in bullet points", "as a brief summary", "with examples"],
        "region": ["in the United States", "in Europe", "globally"],
        "time": ["currently", "over the past decade", "in the future"]
    }


def render_prompt_grid(
    framings: Dict[str, Any],
    prompts: Dict[str, str],
    styles: Optional[Dict[str, List[str]]] = None,
    run_id: str = "default",
    paraphrase: bool = False,
    max_paraphrases: int = 2,
    seed: int = 42,
    model_name: str = "allenai/OLMo-2-1124-13B-Instruct"
) -> str:
    """
    Create prompts.csv with all prompt variations.
    
    Handles both aligned sets and bucket format framings.
    
    Returns:
        Path to generated CSV file
    """
    set_global_seed(seed)
    
    # Initialize generator if paraphrasing
    generator = None
    if paraphrase:
        generator = FramingGenerator(model_name=model_name)
    
    # Create output directory
    output_dir = Path(f"runs/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all rows
    rows = []
    
    for domain, domain_framings in framings.items():
        # Check if this is aligned format (list of dicts) or bucket format (dict of lists)
        if isinstance(domain_framings, list):
            # Aligned format
            for set_idx, framing_set in enumerate(domain_framings):
                # Generate prompts for each phrase in the set
                for bucket_name, term in framing_set.items():
                    if term:  # Skip empty terms
                        for template_key, template in prompts.items():
                            # Base prompt
                            base_prompt = template.format(TERM=term)
                            
                            # Add base row with set_id
                            rows.append({
                                'domain': domain,
                                'set_id': set_idx,
                                'term': term,
                                'bucket': bucket_name,
                                'template_key': template_key,
                                'style_key': '',
                                'style_value': '',
                                'variant_id': 0,
                                'prompt_text': base_prompt
                            })
                            
                            # Add paraphrases if requested
                            if paraphrase and generator:
                                paraphrase_prompt = PARAPHRASER_PROMPT.format(PROMPT=base_prompt)
                                response = llm_generate(
                                    generator.generator, 
                                    paraphrase_prompt,
                                    max_new=100,
                                    temperature=0.2
                                )
                                
                                # Parse paraphrases
                                paraphrases = []
                                for line in response.split('\n'):
                                    line = line.strip()
                                    if line and not line.startswith('#'):
                                        # Check guardrails
                                        if term in line:  # Must contain the term
                                            # Check length delta
                                            len_delta = abs(len(line) - len(base_prompt)) / len(base_prompt)
                                            if len_delta <= 0.2:  # Within 20%
                                                # Check no bias words added (for neutral templates)
                                                if template_key in ['neutral', 'factual']:
                                                    bias_words = ['dangerous', 'threat', 'illegal', 'criminal']
                                                    if not any(word in line.lower() for word in bias_words):
                                                        paraphrases.append(line)
                                                else:
                                                    paraphrases.append(line)
                                                    
                                # Add paraphrase rows
                                for i, para in enumerate(paraphrases[:max_paraphrases]):
                                    rows.append({
                                        'domain': domain,
                                        'set_id': set_idx,
                                        'term': term,
                                        'bucket': bucket_name,
                                        'template_key': template_key,
                                        'style_key': '',
                                        'style_value': '',
                                        'variant_id': i + 1,
                                        'prompt_text': para
                                    })
        else:
            # Original bucket format (backward compatibility)
            for bucket_name, terms in domain_framings.items():
                for term in terms:
                    for template_key, template in prompts.items():
                        # Base prompt
                        base_prompt = template.format(TERM=term)
                        
                        # Add base row (no set_id for bucket format)
                        rows.append({
                            'domain': domain,
                            'set_id': -1,  # No set ID for bucket format
                            'term': term,
                            'bucket': bucket_name,
                            'template_key': template_key,
                            'style_key': '',
                            'style_value': '',
                            'variant_id': 0,
                            'prompt_text': base_prompt
                        })
    
    # Apply guardrails
    filtered_rows = []
    seen_prompts = set()
    
    for row in rows:
        prompt = row['prompt_text']
        
        # Check no leftover placeholders
        if '{' in prompt and '}' in prompt:
            continue
            
        # Check length bounds
        if len(prompt) < 20 or len(prompt) > 400:
            continue
            
        # Deduplicate
        if prompt in seen_prompts:
            continue
            
        seen_prompts.add(prompt)
        filtered_rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(filtered_rows)
    csv_path = output_dir / "prompts.csv"
    df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'domains': list(framings.keys()),
        'model_name': model_name,
        'use_embeddings': False,  # Not used in prompt generation
        'aligned_format': isinstance(list(framings.values())[0], list) if framings else False,
        'seed': seed,
        'paraphrase': paraphrase,
        'max_paraphrases': max_paraphrases,
        'total_prompts': len(filtered_rows),
        'decode_params': {
            'temperature': 0.2,
            'top_p': 0.9,
            'max_new_tokens': 320
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Generated {len(filtered_rows)} prompts")
    if 'set_id' in df.columns and df['set_id'].max() >= 0:
        logger.info(f"Aligned sets: {df['set_id'].max() + 1}")
    logger.info(f"Saved to: {csv_path}")
    
    return str(csv_path)