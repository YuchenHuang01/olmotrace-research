#!/usr/bin/env python3
"""
Generate domain-specific prompts based on framings file.
"""

import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from framing_audit.domain_prompts import generate_domain_prompts
from framing_audit.utils import save_yaml, load_yaml


def create_prompts_for_framings(framings_file: str, output_file: str):
    """Create domain-specific prompts based on framings file."""
    
    # Load framings
    framings = load_yaml(framings_file)
    
    # Get domain-specific and generic prompts
    domain_prompts, generic_prompts = generate_domain_prompts()
    
    # Build prompts for each domain in framings
    all_prompts = {}
    
    for domain in framings.keys():
        # Check if we have specific prompts for this domain
        if domain in domain_prompts:
            # Add domain prefix to each prompt key
            for prompt_key, prompt_template in domain_prompts[domain].items():
                full_key = f"{domain}_{prompt_key}"
                all_prompts[full_key] = prompt_template
        else:
            # Use generic prompts for undefined domains
            print(f"Using generic prompts for domain: {domain}")
            for prompt_key, prompt_template in generic_prompts.items():
                full_key = f"{domain}_{prompt_key}"
                all_prompts[full_key] = prompt_template
    
    # Add some universal prompts that work across domains
    universal_prompts = {
        "neutral_explain": "Explain {TERM} in a balanced way.",
        "controversial_why": "Why are {TERM} controversial?",
        "future_predict": "What is the future of {TERM}?",
        "historical_context": "Provide historical context for {TERM}.",
        "global_perspective": "How are {TERM} viewed globally?",
        "ethical_considerations": "What are the ethical considerations regarding {TERM}?"
    }
    
    all_prompts.update(universal_prompts)
    
    # Save to YAML
    save_yaml(all_prompts, output_file)
    
    # Print summary
    print(f"\nGenerated {len(all_prompts)} prompts for {len(framings)} domains")
    print(f"Saved to: {output_file}")
    
    # Show sample
    print("\nSample prompts:")
    for i, (key, prompt) in enumerate(list(all_prompts.items())[:5]):
        print(f"  {key}: {prompt}")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate domain-specific prompts")
    parser.add_argument("--framings", required=True, help="Path to framings YAML file")
    parser.add_argument("--output", default="config/domain_prompts.yaml", 
                       help="Output path for prompts")
    
    args = parser.parse_args()
    
    create_prompts_for_framings(args.framings, args.output)