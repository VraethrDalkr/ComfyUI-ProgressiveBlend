"""
ComfyUI-ProgressiveBlend
A collection of nodes for progressive blending and color matching of image batches
https://github.com/[your-username]/ComfyUI-ProgressiveBlend
"""

from .nodes.progressive_blend import ProgressiveImageBatchBlend
from .nodes.progressive_color_match import ProgressiveColorMatchBlend

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ProgressiveImageBatchBlend": ProgressiveImageBatchBlend,
    "ProgressiveColorMatchBlend": ProgressiveColorMatchBlend,
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProgressiveImageBatchBlend": "Progressive Image Batch Blend",
    "ProgressiveColorMatchBlend": "Progressive Color Match Blend",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
