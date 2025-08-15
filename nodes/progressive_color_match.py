"""
Progressive Color Match Blend Node for ComfyUI
Performs progressive color matching and blending on image batches

Based on color matching implementation from ComfyUI-KJNodes by Kijai
https://github.com/kijai/ComfyUI-KJNodes
Original ColorMatch node licensed under GPL-3.0
"""

import torch
import os
from typing import Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Use relative import from parent package
from ..utils.blend_utils import calculate_blend_factor, get_blend_curve_options


class ProgressiveColorMatchBlend:
    """
    A ComfyUI node that progressively blends color matching effects between two reference images.

    This node applies color matching from two reference images to a target batch,
    with the influence progressively transitioning from the start reference to the end reference.
    """

    def __init__(self):
        """Initialize the Progressive Color Match Blend node."""
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define the input types for the node.

        Returns:
            Dict containing the required and optional input specifications
        """
        curve_options, curve_default = get_blend_curve_options()

        return {
            "required": {
                "start_reference": ("IMAGE",),  # Start reference image for color matching
                "end_reference": ("IMAGE",),    # End reference image for color matching
                "target_images": ("IMAGE",),    # Target image batch to process
                "method": (
                    [
                        'mkl',
                        'hm',
                        'reinhard',
                        'mvgd',
                        'hm-mvgd-hm',
                        'hm-mkl-hm',
                    ],
                    {"default": 'mkl'}
                ),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "multithread": ("BOOLEAN", {"default": True}),
                "blend_curve": (curve_options, curve_default),
                "reverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "progressive_color_match"

    CATEGORY = "image/blend"

    DESCRIPTION = """
Progressively applies color matching from two reference images across a batch.
The color matching effect transitions from the start reference to the end reference.

Based on color-matcher library which enables color transfer across images.
Useful for automatic color-grading of photographs, paintings and film sequences.

Methods available:
- mkl: Monge-Kantorovich Linearization
- hm: Histogram Matching
- reinhard: Reinhard et al. approach
- mvgd: Multi-Variate Gaussian Distribution
- hm-mvgd-hm: HM-MVGD-HM compound
- hm-mkl-hm: HM-MKL-HM compound

Original color matching implementation by Kijai (ComfyUI-KJNodes)
"""

    def apply_color_match(self, target_image: torch.Tensor, ref_image: torch.Tensor,
                         method: str, strength: float) -> torch.Tensor:
        """
        Apply color matching from a reference image to a target image.

        Args:
            target_image: Target image tensor to be color matched
            ref_image: Reference image tensor for color matching
            method: Color matching method to use
            strength: Strength of the color matching effect (0.0 to 10.0)

        Returns:
            Color matched image tensor
        """
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise Exception(
                "Can't import color-matcher. Please install it:\n"
                "pip install color-matcher"
            )

        # Convert tensors to numpy arrays
        target_np = target_image.cpu().numpy()
        ref_np = ref_image.cpu().numpy()

        # Perform color matching
        cm = ColorMatcher()
        try:
            result = cm.transfer(src=target_np, ref=ref_np, method=method)
            # Apply strength factor
            result = target_np + strength * (result - target_np)
            return torch.from_numpy(result).to(torch.float32)
        except Exception as e:
            print(f"Color matching error: {e}")
            # Return original image on error
            return target_image

    def progressive_color_match(self, start_reference: torch.Tensor,
                               end_reference: torch.Tensor,
                               target_images: torch.Tensor,
                               method: str,
                               strength: float,
                               multithread: bool = True,
                               blend_curve: str = "linear",
                               reverse: bool = False) -> Tuple[torch.Tensor]:
        """
        Progressively apply color matching and blending to an image batch.

        Args:
            start_reference: Reference image for start of sequence color matching
            end_reference: Reference image for end of sequence color matching
            target_images: Target image batch to process
            method: Color matching method to use
            strength: Strength of color matching effect
            multithread: Whether to use multithreading for processing
            blend_curve: Type of blending curve to apply
            reverse: Whether to reverse the blend direction

        Returns:
            Tuple containing the processed image batch tensor
        """
        # Move tensors to CPU for processing
        start_ref = start_reference.cpu().squeeze()
        end_ref = end_reference.cpu().squeeze()
        target_batch = target_images.cpu()

        batch_size = target_batch.shape[0]

        # Ensure reference images are 3D (H, W, C)
        if start_ref.dim() == 4:
            start_ref = start_ref[0]
        if end_ref.dim() == 4:
            end_ref = end_ref[0]

        def process_frame(i: int) -> torch.Tensor:
            """
            Process a single frame with progressive color matching.

            Args:
                i: Frame index

            Returns:
                Processed frame tensor
            """
            # Get current target frame
            target_frame = target_batch[i]

            # Calculate blend factor for this frame
            blend_factor = calculate_blend_factor(i, batch_size, blend_curve, reverse)

            # Apply color matching with start reference
            matched_start = self.apply_color_match(
                target_frame, start_ref, method, strength
            )

            # Apply color matching with end reference
            matched_end = self.apply_color_match(
                target_frame, end_ref, method, strength
            )

            # Blend between the two color matched results based on position
            # blend_factor = 0: 100% start_ref matching
            # blend_factor = 1: 100% end_ref matching
            blended = matched_start * (1.0 - blend_factor) + matched_end * blend_factor

            # Ensure values stay in valid range
            blended = torch.clamp(blended, 0.0, 1.0)

            return blended

        # Process frames with optional multithreading
        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                processed_frames = list(executor.map(process_frame, range(batch_size)))
        else:
            processed_frames = [process_frame(i) for i in range(batch_size)]

        # Stack processed frames back into a batch
        output_batch = torch.stack(processed_frames, dim=0).to(torch.float32)

        # Ensure final clamping
        output_batch.clamp_(0.0, 1.0)

        return (output_batch,)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Indicate that the node should always re-execute.

        Returns:
            Float value that changes to force re-execution
        """
        return float("NaN")