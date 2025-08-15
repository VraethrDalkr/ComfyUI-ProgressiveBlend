"""
Progressive Image Batch Blend Node for ComfyUI
Blends two image batches with linearly increasing blend factor from 0 to 1
"""

import torch
from typing import Tuple, Dict, Any

# Use relative import from parent package
from ..utils.blend_utils import calculate_blend_factor, get_blend_curve_options


class ProgressiveImageBatchBlend:
    """
    A ComfyUI node that progressively blends two image batches.

    The blend factor increases linearly from 0 (first frame) to 1 (last frame),
    creating a smooth transition from the first batch appearance to the second.
    """

    def __init__(self):
        """Initialize the Progressive Image Batch Blend node."""
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define the input types for the node.

        Returns:
            Dict containing the required input specifications
        """
        curve_options, curve_default = get_blend_curve_options()

        return {
            "required": {
                "images1": ("IMAGE",),  # First image batch
                "images2": ("IMAGE",),  # Second image batch
            },
            "optional": {
                "blend_curve": (curve_options, curve_default),  # Blend curve type
                "reverse": ("BOOLEAN", {"default": False}),  # Reverse blend direction
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "blend_progressive"

    CATEGORY = "image/blend"

    def blend_progressive(self, images1: torch.Tensor, images2: torch.Tensor,
                         blend_curve: str = "linear", reverse: bool = False) -> Tuple[torch.Tensor]:
        """
        Progressively blend two image batches with linearly increasing blend factor.

        Args:
            images1: First image batch tensor [batch, height, width, channels]
            images2: Second image batch tensor [batch, height, width, channels]
            blend_curve: Type of blending curve to apply
            reverse: Whether to reverse the blend direction

        Returns:
            Tuple containing the blended image batch tensor

        Raises:
            ValueError: If image batches have different dimensions
        """
        # Validate input dimensions
        if images1.shape != images2.shape:
            raise ValueError(
                f"Image batches must have the same dimensions. "
                f"Got images1: {images1.shape}, images2: {images2.shape}"
            )

        batch_size = images1.shape[0]

        # Handle single image case
        if batch_size == 1:
            # For single image, use 0.5 blend factor (middle blend)
            blended = images1 * 0.5 + images2 * 0.5
            return (blended,)

        # Create list to store blended frames
        blended_frames = []

        # Process each frame pair
        for i in range(batch_size):
            # Calculate blend factor for current frame using shared utility
            blend_factor = calculate_blend_factor(i, batch_size, blend_curve, reverse)

            # Extract current frames
            frame1 = images1[i]
            frame2 = images2[i]

            # Perform weighted blend
            # blend_factor = 0: 100% frame1, 0% frame2
            # blend_factor = 1: 0% frame1, 100% frame2
            blended_frame = frame1 * (1.0 - blend_factor) + frame2 * blend_factor

            # Ensure values stay in valid range [0, 1]
            blended_frame = torch.clamp(blended_frame, 0.0, 1.0)

            blended_frames.append(blended_frame)

        # Stack all blended frames back into a batch
        output_batch = torch.stack(blended_frames, dim=0)

        return (output_batch,)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Indicate that the node should always re-execute.

        Returns:
            Float value that changes to force re-execution
        """
        return float("NaN")