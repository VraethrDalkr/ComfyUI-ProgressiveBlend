"""
Progressive Color Match Blend Node for ComfyUI
Performs progressive color matching and blending on image batches

Based on color matching implementation from ComfyUI-KJNodes by Kijai
https://github.com/kijai/ComfyUI-KJNodes
Original ColorMatch node licensed under GPL-3.0
"""

import torch
import os
from typing import Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Use relative import from parent package
from ..utils.blend_utils import calculate_blend_factor, get_blend_curve_options


class ProgressiveColorMatchBlend:
    """
    A ComfyUI node that progressively blends color matching effects between reference images.

    This node can work with:
    - Two references: Progressive transition from start to end reference
    - Start reference only: Fade from color matched to original
    - End reference only: Fade from original to color matched
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
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable color matching. When disabled, passes through original images"
                }),
                "target_images": ("IMAGE", {
                    "tooltip": "Image batch/video frames to apply color matching to"
                }),
                "method": (
                    [
                        'mkl',
                        'hm',
                        'reinhard',
                        'mvgd',
                        'hm-mvgd-hm',
                        'hm-mkl-hm',
                    ],
                    {
                        "default": 'mkl',
                        "tooltip": "Color matching algorithm. mkl=general purpose, hm=fast, reinhard=classic, mvgd=statistical, compounds=best quality"
                    }
                ),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Intensity of color matching effect. 0=no effect, 1=normal, >1=amplified"
                }),
            },
            "optional": {
                "start_reference": ("IMAGE", {
                    "tooltip": "Reference image for start colors. Leave empty to fade from original"
                }),
                "end_reference": ("IMAGE", {
                    "tooltip": "Reference image for end colors. Leave empty to fade to original"
                }),
                "multithread": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use multiple CPU threads for faster processing of large batches"
                }),
                "blend_curve": (curve_options, {
                    **curve_default,
                    "tooltip": "Transition curve shape. linear=constant, ease_in=accelerate, ease_out=decelerate, ease_in_out=S-curve, ease_out_in=inverse-S"
                }),
                "reverse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the blend direction (end to start instead of start to end)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "progressive_color_match"

    CATEGORY = "image/blend"

    DESCRIPTION = """
Progressively applies color matching from reference images across a batch.

Modes:
- Both references: Transitions from start to end reference colors
- Start reference only: Fades from color matched to original
- End reference only: Fades from original to color matched
- No references: Returns original images

Based on color-matcher library which enables color transfer across images.
Useful for automatic color-grading of photographs, paintings and film sequences.

Methods available:
- mkl: Monge-Kantorovich Linearization
- hm: Histogram Matching
- reinhard: Reinhard et al. approach
- mvgd: Multi-Variate Gaussian Distribution
- hm-mvgd-hm: HM-MVGD-HM compound
- hm-mkl-hm: HM-MKL-HM compound

Blend curves:
- linear: Constant rate of change
- ease_in: Slow start, accelerating
- ease_out: Fast start, decelerating
- ease_in_out: Slow at both ends, fast middle
- ease_out_in: Fast at both ends, slow middle

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

    def progressive_color_match(self, enabled: bool,
                               target_images: torch.Tensor,
                               method: str,
                               strength: float,
                               start_reference: Optional[torch.Tensor] = None,
                               end_reference: Optional[torch.Tensor] = None,
                               multithread: bool = True,
                               blend_curve: str = "linear",
                               reverse: bool = False) -> Tuple[torch.Tensor]:
        """
        Progressively apply color matching and blending to an image batch.

        Args:
            enabled: Whether to apply color matching or pass through
            target_images: Target image batch to process
            method: Color matching method to use
            strength: Strength of color matching effect
            start_reference: Optional reference image for start of sequence
            end_reference: Optional reference image for end of sequence
            multithread: Whether to use multithreading for processing
            blend_curve: Type of blending curve to apply
            reverse: Whether to reverse the blend direction

        Returns:
            Tuple containing the processed image batch tensor
        """
        # If disabled, pass through original images
        if not enabled:
            return (target_images,)

        # Check which references are provided
        has_start = start_reference is not None
        has_end = end_reference is not None

        # If no references provided, return original images
        if not has_start and not has_end:
            print("No reference images provided, returning original images")
            return (target_images,)

        # Move tensors to CPU for processing
        target_batch = target_images.cpu()
        batch_size = target_batch.shape[0]

        # Prepare reference images if provided
        start_ref = None
        end_ref = None

        if has_start:
            start_ref = start_reference.cpu().squeeze()
            if start_ref.dim() == 4:
                start_ref = start_ref[0]

        if has_end:
            end_ref = end_reference.cpu().squeeze()
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

            if has_start and has_end:
                # Both references: blend between two color matched versions
                matched_start = self.apply_color_match(
                    target_frame, start_ref, method, strength
                )
                matched_end = self.apply_color_match(
                    target_frame, end_ref, method, strength
                )
                # blend_factor = 0: 100% start_ref matching
                # blend_factor = 1: 100% end_ref matching
                blended = matched_start * (1.0 - blend_factor) + matched_end * blend_factor

            elif has_start:
                # Start reference only: fade from color matched to original
                matched = self.apply_color_match(
                    target_frame, start_ref, method, strength
                )
                # blend_factor = 0: 100% color matched
                # blend_factor = 1: 100% original
                blended = matched * (1.0 - blend_factor) + target_frame * blend_factor

            else:  # has_end only
                # End reference only: fade from original to color matched
                matched = self.apply_color_match(
                    target_frame, end_ref, method, strength
                )
                # blend_factor = 0: 100% original
                # blend_factor = 1: 100% color matched
                blended = target_frame * (1.0 - blend_factor) + matched * blend_factor

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