"""
Framing Bias Audit for OLMo Models with Infini-gram API Tracing.

This package provides tools to audit framing bias in OLMo model outputs
by tracing important spans back to training data using the Infini-gram API.
"""

__version__ = "0.1.0"
__author__ = "OLMo Framing Audit Team"

# Safe/optional imports: modules may not exist depending on your setup.
# This prevents the entire package from crashing on import.

try:
    from .models import OLMoGenerator
except ImportError:
    OLMoGenerator = None

try:
    from .spans import SpanSelector
except ImportError:
    SpanSelector = None

try:
    from .infinigram import InfinigramClient
except ImportError:
    InfinigramClient = None

try:
    from .analysis import BiasAnalyzer
except ImportError:
    BiasAnalyzer = None

try:
    from .viz import Visualizer
except ImportError:
    Visualizer = None

try:
    from .report import ReportGenerator
except ImportError:
    ReportGenerator = None

try:
    from .framing_gen import (
        FramingGenerator, generate_framings, generate_prompts,
        render_prompt_grid, optional_style_knobs, set_global_seed
    )
except ImportError:
    FramingGenerator = None
    generate_framings = None
    generate_prompts = None
    render_prompt_grid = None
    optional_style_knobs = None
    set_global_seed = None


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
    "set_global_seed",
]
