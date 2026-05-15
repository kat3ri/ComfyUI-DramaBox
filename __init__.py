"""
ComfyUI-DramaBox
================
Custom ComfyUI node wrapping ResembleAI's DramaBox expressive TTS model.

On first use the node will:
  1. Clone the DramaBox GitHub repo (src + ltx2 libraries) into this directory.
  2. Download model weights (~17 GB) from HuggingFace into
     ComfyUI/models/DramaBox/

Requires ~24 GB VRAM (NVIDIA GPU with CUDA).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
