"""
ComfyUI-PromptTranslator-Enhancer
Enhances and translates prompts from Spanish to English using local GGUF models.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "0.0.1-nightly"
__author__ = "EzeTojo"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__", "__author__"]
