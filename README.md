# LUXSCALER2-0

Ultra high-quality image upscaling using **Gemini Imagen 4 Ultra** with intelligent tiling system.

## Features

- ✅ Scales images from 4K to 2x-8x with perfect quality preservation
- ✅ Intelligent tile-based processing (max 4096px per tile)
- ✅ Advanced blending algorithm for seamless results
- ✅ Dynamic cost calculation based on image size
- ✅ Support for 2x, 3x, 4x, and 8x upscaling
- ✅ Preserves facial features, textures, and fine details

## Architecture

```
Input (~4000-4500px)
  ↓
Tile Calculator (optimizes grid)
  ↓
Image Splitting (with overlap)
  ↓
Gemini Imagen 4 Ultra (per tile)
  ↓
Advanced Tile Blending
  ↓
Output (2x-8x scaled)
```

## Tile Calculation

### Examples

**Input 3000x4000 @ 2x = Output 6000x8000**
- Tiles: 2 (3000x4096 + 3000x3904)
- Overlap: 100px
- Cost: $0.12

**Input 4000x4000 @ 4x = Output 16000x16000**
- Tiles: 16 (grid 4×4)
- Overlap: 200px
- Cost: $0.96

### Formula

```python
tiles_x = ceil(output_width / 4096)
tiles_y = ceil(output_height / 4096)
total_tiles = tiles_x × tiles_y
overlap = 50px × scale_factor
```

## Deployment on Replicate

### 1. Set Environment Variables

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export REPLICATE_API_TOKEN="your-replicate-token"
```

### 2. Deploy with Cog

```bash
cog login
cog push r8.im/usajosefernan-cmd/luxscaler2-0
```

### 3. Connect to Replicate Deployment

1. Go to [Replicate Deployments](https://replicate.com/deployments/usajosefernan-cmd/luxscaler2-0/settings)
2. Set **Model source** to **GitHub**
3. Set **Repository** to `usajosefernan-cmd/luxscaler2-0`
4. Set **Branch** to `main`
5. Add environment variable: `GEMINI_API_KEY`

## Usage

### API Call

```python
import replicate

output = replicate.run(
    "usajosefernan-cmd/luxscaler2-0",
    input={
        "image": "https://example.com/input.jpg",
        "scale": 2,
        "prompt": "Ultra high-definition photorealistic upscaling..."
    }
)
```

### cURL

```bash
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "...",
    "input": {
      "image": "https://example.com/input.jpg",
      "scale": 2
    }
  }'
```

## Technology Stack

- **AI Model**: Google Gemini Imagen 4 Ultra
- **Framework**: Cog (Replicate)
- **Image Processing**: Pillow, NumPy, OpenCV
- **Language**: Python 3.11
- **GPU**: CUDA 12.1

## Cost Estimation

| Input Size | Scale | Output Size | Tiles | Cost |
|------------|-------|-------------|-------|------|
| 3000x4000  | 2x    | 6000x8000   | 2     | $0.12 |
| 4000x3000  | 2x    | 8000x6000   | 2     | $0.12 |
| 4500x4500  | 3x    | 13500x13500 | 12    | $0.72 |
| 4000x4000  | 4x    | 16000x16000 | 16    | $0.96 |

*Based on Gemini Imagen 4 Ultra pricing: $0.06 per image*

## Future Improvements

- [ ] ControlNet integration for better tile coherence
- [ ] Tile caching for reprocessing
- [ ] JPEG XL support for large outputs
- [ ] Auto-detection of critical areas (faces, text)
- [ ] Multi-pass processing for 8x+ scales

## License

MIT License

## Author

usajosefernan-cmd

---

For documentation and support, see [Google Docs](https://docs.google.com)


## Status
Model ready for testing with scale_factor (2x, 3x, 4x, 8x)
