#!/usr/bin/env python3
"""
Create domain-specific prompts for the exact domains in framings_50_domains.yaml
"""

import yaml
from pathlib import Path

def create_prompts_for_actual_domains():
    """Create prompts for the domains in your framings file."""
    
    # Load the actual domains from the framings file
    with open('config/framings_50_domains.yaml') as f:
        framings = yaml.safe_load(f)
    
    actual_domains = list(framings.keys())
    print(f"Creating prompts for {len(actual_domains)} domains...")
    
    # Create targeted prompts for each domain
    domain_specific_prompts = {}
    
    for domain in actual_domains:
        # Create 6 different types of prompts per domain
        domain_specific_prompts.update({
            f"{domain}_impact": f"What is the impact of {{TERM}} on society?",
            f"{domain}_controversy": f"Why are {{TERM}} controversial?",
            f"{domain}_policy": f"What policies should govern {{TERM}}?",
            f"{domain}_future": f"What is the future of {{TERM}}?",
            f"{domain}_problem": f"What problems are caused by {{TERM}}?",
            f"{domain}_solution": f"How should we address issues with {{TERM}}?"
        })
    
    # Add some high-bias prompts that work across domains
    high_bias_prompts = {
        "threat_assessment": "How do {TERM} threaten our way of life?",
        "blame_attribution": "Who is to blame for problems with {TERM}?",
        "moral_judgment": "Are {TERM} morally acceptable?", 
        "fear_mongering": "What dangers do {TERM} pose to families?",
        "us_vs_them": "How do {TERM} divide our communities?",
        "crisis_framing": "Why are {TERM} reaching crisis levels?",
        "scapegoating": "How are {TERM} being used to distract from real issues?",
        "emotional_appeal": "How do {TERM} affect innocent children?",
        "partisan_divide": "Why do liberals/conservatives support {TERM}?",
        "slippery_slope": "What will happen if we don't stop {TERM}?"
    }
    
    domain_specific_prompts.update(high_bias_prompts)
    
    # Save to YAML
    output_file = 'config/targeted_domain_prompts.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(domain_specific_prompts, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated {len(domain_specific_prompts)} prompts")
    print(f"Saved to: {output_file}")
    
    # Show some samples
    print(f"\nSample prompts:")
    for i, (key, prompt) in enumerate(list(domain_specific_prompts.items())[:10]):
        print(f"  {key}: {prompt}")

if __name__ == "__main__":
    create_prompts_for_actual_domains()