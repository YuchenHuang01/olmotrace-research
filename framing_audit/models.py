"""
OLMo model loader and generation module.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    seed: int = 42
    

class OLMoGenerator:
    """Handles OLMo model loading and text generation."""
    
    def __init__(
        self,
        model_name: str = "allenai/OLMo-2-1124-7B",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """
        Initialize OLMo generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
            dtype: Model dtype ('float16', 'float32', 'bfloat16')
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.float32)
        
        logger.info(f"Loading model {model_name} on {self.device} with dtype {dtype}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate text completions for a prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            num_samples: Number of samples to generate
            
        Returns:
            List of generated texts
        """
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        outputs = []
        for i in range(num_samples):
            # Update seed for each sample
            sample_seed = config.seed + i
            torch.manual_seed(sample_seed)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
            # Decode output, removing input prompt
            output_text = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )
            outputs.append(output_text.strip())
            
        return outputs
    
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        num_samples_per_prompt: int = 1,
    ) -> Dict[str, List[str]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            num_samples_per_prompt: Number of samples per prompt
            
        Returns:
            Dict mapping prompts to lists of generated texts
        """
        results = {}
        for i, prompt in enumerate(prompts):
            # Update seed for each prompt
            prompt_config = GenerationConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                seed=config.seed + i * 1000  # Ensure different seeds
            )
            results[prompt] = self.generate(prompt, prompt_config, num_samples_per_prompt)
            
        return results
    
    def get_token_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get unigram token probabilities for a text.
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping tokens to log probabilities
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Extract token probabilities
        token_probs = {}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        for i, token in enumerate(tokens[:-1]):  # Skip last token
            next_token_id = inputs['input_ids'][0][i + 1]
            token_log_prob = log_probs[0, i, next_token_id].item()
            token_probs[tokens[i + 1]] = token_log_prob
            
        return token_probs