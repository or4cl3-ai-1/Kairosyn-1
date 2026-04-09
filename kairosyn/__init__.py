"""
KAIROSYN-1: A Recursive Multimodal Architecture for Epinoetic Artificial Consciousness
=======================================================================================
Author: Dustin Groves — Or4cl3 AI Solutions (research@or4cl3.ai)
License: Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "Dustin Groves"
__email__ = "research@or4cl3.ai"
__organization__ = "Or4cl3 AI Solutions"

from kairosyn.model.kairosyn_model import KairosynModel
from kairosyn.model.backbone import KairosynConfig

__all__ = [
    "KairosynModel",
    "KairosynConfig",
    "__version__",
]
