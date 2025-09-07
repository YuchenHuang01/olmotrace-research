"""
Framing Bias Audit for OLMo Models with Infini-gram API Tracing.

This package provides tools to audit framing bias in OLMo model outputs
by tracing important spans back to training data using the Infini-gram API.
"""

__version__ = "0.1.0"
__author__ = "OLMo Framing Audit Team"

from .models import OLMoGenerator
from .spans import SpanSelector
from .infinigram import InfinigramClient
from .analysis import BiasAnalyzer
from .viz import Visualizer
from .report import ReportGenerator
from .framing_gen import (
    FramingGenerator, generate_framings, generate_prompts,
    render_prompt_grid, optional_style_knobs, set_global_seed
)

__all__ = [
    "OLMoGenerator",
    "SpanSelector", 
    "InfinigramClient",
    "BiasAnalyzer",
    "Visualizer",
    "ReportGenerator",
    "FramingGenerator",
    "generate_framings", 
    "generate_prompts",
    "render_prompt_grid",
    "optional_style_knobs",
    "set_global_seed"
]