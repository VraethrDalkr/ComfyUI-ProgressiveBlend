"""
Shared utility functions for progressive blending nodes
"""

from typing import Tuple


def calculate_blend_factor(index: int, total: int, curve: str = "linear", 
                          reverse: bool = False) -> float:
    """
    Calculate the blend factor for a specific frame index.
    
    This function determines how much weight to give to the second element
    in a blend operation based on the position in a sequence.
    
    Args:
        index: Current frame index (0-based)
        total: Total number of frames
        curve: Type of blending curve to use ('linear', 'ease_in', 'ease_out', 'ease_in_out', 'ease_out_in')
        reverse: Whether to reverse the blend direction

    Returns:
        Float blend factor between 0 and 1

    Examples:
        >>> calculate_blend_factor(0, 10, 'linear', False)
        0.0
        >>> calculate_blend_factor(9, 10, 'linear', False)
        1.0
        >>> calculate_blend_factor(4, 10, 'linear', False)
        0.4444...
    """
    # Calculate normalized position (0 to 1)
    if total <= 1:
        t = 0.0
    else:
        t = index / (total - 1)

    # Apply reverse if needed
    if reverse:
        t = 1.0 - t

    # Apply curve transformation
    if curve == "ease_in":
        # Quadratic ease-in: slow start, accelerating
        blend_factor = t * t
    elif curve == "ease_out":
        # Quadratic ease-out: fast start, decelerating
        blend_factor = 1.0 - (1.0 - t) * (1.0 - t)
    elif curve == "ease_in_out":
        # Cubic ease-in-out: slow start and end, fast middle
        if t < 0.5:
            blend_factor = 2 * t * t
        else:
            blend_factor = 1.0 - pow(-2 * t + 2, 2) / 2
    elif curve == "ease_out_in":
        # Inverse S-curve: fast at both ends, slow in middle
        if t < 0.5:
            # First half: ease-out (fast start, slow approach to middle)
            t_half = t * 2  # Scale to 0-1 for first half
            blend_factor = (1.0 - (1.0 - t_half) * (1.0 - t_half)) * 0.5
        else:
            # Second half: ease-in (slow departure from middle, fast end)
            t_half = (t - 0.5) * 2  # Scale to 0-1 for second half
            blend_factor = 0.5 + (t_half * t_half) * 0.5
    else:  # linear
        # Linear: constant rate of change
        blend_factor = t

    return blend_factor


def get_blend_curve_options() -> Tuple[list, dict]:
    """
    Get the available blend curve options for ComfyUI nodes.

    Returns:
        Tuple of (options_list, default_dict) for ComfyUI INPUT_TYPES
    """
    options = ["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in"]
    default = {"default": "linear"}
    return options, default