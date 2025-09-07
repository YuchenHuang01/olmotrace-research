"""
Utility functions for logging, seeding, and I/O operations.
"""

import os
import json
import yaml
import logging
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set up file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded data as dict
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str, sort_keys: bool = False):
    """
    Save data as YAML file.
    
    Args:
        data: Data to save
        file_path: Output file path
        sort_keys: Whether to sort keys
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, indent=2, sort_keys=sort_keys, default_flow_style=False)


def save_json(data: Dict[str, Any], file_path: str):
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(df: pd.DataFrame, file_path: str):
    """
    Save DataFrame as CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def create_run_directory(base_dir: str, run_id: Optional[str] = None) -> str:
    """
    Create a directory for the current run.
    
    Args:
        base_dir: Base directory for runs
        run_id: Optional run ID
        
    Returns:
        Path to created run directory
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = os.path.join(base_dir, run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_metadata(
    run_dir: str,
    config: Dict[str, Any],
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save run metadata.
    
    Args:
        run_dir: Run directory
        config: Configuration dict
        additional_info: Additional metadata
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    save_json(metadata, os.path.join(run_dir, 'metadata.json'))


def format_prompt(template: str, term: str) -> str:
    """
    Format a prompt template with a term.
    
    Args:
        template: Prompt template with {TERM} placeholder
        term: Term to insert
        
    Returns:
        Formatted prompt
    """
    return template.replace('{TERM}', term)


def batch_data(data: list, batch_size: int) -> list:
    """
    Batch a list into chunks.
    
    Args:
        data: List to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = datetime.now()
        
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            print(f"\r{self.desc}: {self.current}/{self.total} "
                  f"({self.current/self.total*100:.1f}%) "
                  f"ETA: {remaining:.0f}s", end="", flush=True)
            
    def finish(self):
        """Mark as complete."""
        print()  # New line