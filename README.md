# ComfyUI-ProgressiveBlend

A collection of custom nodes for ComfyUI that enable progressive blending and color matching effects across image batches/video frames.

## Features

This node pack includes two powerful nodes for creating smooth transitions and color grading effects:

### 1. Progressive Image Batch Blend
Blends two image batches with a progressively changing blend factor from 0.0 to 1.0 across the sequence.

**Use Cases:**
- Creating smooth transitions between two videos
- Morphing effects between different image sequences
- Artistic blend effects with customizable curves

**Inputs:**
- `images1`: First image batch
- `images2`: Second image batch
- `blend_curve`: Blending curve type (linear, ease_in, ease_out, ease_in_out, ease_out_in)
- `reverse`: Reverse the blend direction

**Output:**
- `images`: Blended image batch

### 2. Progressive Color Match Blend
Applies progressive color matching from reference images across a target batch, creating smooth color transitions.

**Modes:**
- **Both references**: Transitions from start to end reference colors
- **Start reference only**: Fades from color matched to original colors
- **End reference only**: Fades from original to color matched colors
- **No references**: Returns original images unchanged

**Use Cases:**
- Color grading transitions in videos
- Matching footage shot under different lighting conditions
- Creating artistic color progression effects
- Automatic color correction with smooth transitions
- Creative pulse effects with single reference and ease_in_out curves

**Required Inputs:**
- `target_images`: Target image batch to process
- `method`: Color matching algorithm (mkl, hm, reinhard, mvgd, hm-mvgd-hm, hm-mkl-hm)
- `strength`: Strength of color matching effect (0.0-10.0)

**Optional Inputs:**
- `start_reference`: Reference image for start color palette
- `end_reference`: Reference image for end color palette
- `multithread`: Enable multithreading for faster processing
- `blend_curve`: Blending curve type
- `reverse`: Reverse the blend direction

**Output:**
- `images`: Processed image batch with progressive color matching

## Installation

### Method 1: ComfyUI Manager (Recommended when available)
Once this node pack is registered with ComfyUI Manager, you'll be able to install it directly from the manager interface.

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ~/ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/[your-username]/ComfyUI-ProgressiveBlend.git
```

3. Install required dependencies:
```bash
cd ComfyUI-ProgressiveBlend
pip install -r requirements.txt
```

4. Restart ComfyUI

## How It Works

### Progressive Blending
Both nodes use a progressive blending system where each frame in a batch gets a unique blend factor based on its position:

- **Frame 0**: blend_factor = 0.0 (100% first input)
- **Frame N**: blend_factor = 1.0 (100% second input)
- **Frames 1 to N-1**: Linear or curved interpolation

### Blend Curves
- **linear**: Constant rate of change
- **ease_in**: Starts slow, accelerates (quadratic)
- **ease_out**: Starts fast, decelerates (quadratic)  
- **ease_in_out**: Slow at both ends, fast in middle (cubic)
- **ease_out_in**: Fast at both ends, slow in middle (inverse S-curve)

### Color Matching Methods
The Progressive Color Match Blend node uses the `color-matcher` library with these methods:

- **mkl**: Monge-Kantorovich Linearization - Good general purpose
- **hm**: Histogram Matching - Fast and simple
- **reinhard**: Reinhard et al. approach - Classic method
- **mvgd**: Multi-Variate Gaussian Distribution - Statistical approach
- **hm-mvgd-hm**: Compound method, often best results
- **hm-mkl-hm**: Alternative compound method

## Examples

### Example 1: Video Transition
Create a smooth transition between two video clips:
1. Load two video batches
2. Connect to Progressive Image Batch Blend
3. Set blend_curve to "ease_in_out" for smooth transition

### Example 2: Color Grading Transition
Apply a day-to-night color grade transition:
1. Load a day reference image as `start_reference`
2. Load a night reference image as `end_reference`
3. Load your video as `target_images`
4. Use Progressive Color Match Blend with method "hm-mvgd-hm"

## Technical Details

- **Thread-safe**: Supports multithreading for batch processing
- **Memory efficient**: Processes frames individually to minimize memory usage
- **GPU compatible**: Works with CUDA tensors when available
- **Error handling**: Gracefully handles edge cases and dimension mismatches

## Credits & Attribution

### Color Matching Implementation
The color matching functionality in the Progressive Color Match Blend node is based on the ColorMatch node from:
- **ComfyUI-KJNodes** by Kijai
- Repository: https://github.com/kijai/ComfyUI-KJNodes
- License: GPL-3.0

### Color-Matcher Library
This project uses the color-matcher library:
- Repository: https://github.com/hahnec/color-matcher
- Provides the underlying color transfer algorithms

## License

This project is licensed under GPL-3.0 License, in compliance with the inherited license from ComfyUI-KJNodes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues or questions:
1. Check the ComfyUI documentation
2. Review the examples in this README
3. Submit an issue on the repository

## Version History

- **v1.1.0**: Added Progressive Color Match Blend with multi-threading support
- **v1.0.0**: Initial release with Progressive Image Batch Blend

---

<p align="center">
Made with ❤️ for the ComfyUI Community
</p>