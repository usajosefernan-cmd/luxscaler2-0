import os
import math
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from google import genai
from google.genai import types
from cog import BasePredictor, Input, Path
from typing import Optional

class Predictor(BasePredictor):
    def setup(self):
        """Initialize Gemini API clients"""
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
    
    def calculate_tiles(self, width: int, height: int, scale: int):
        """Calculate optimal tile distribution"""
        output_w = width * scale
        output_h = height * scale
        
        max_tile_size = 4096
        overlap = 50 * scale
        
        tiles_x = math.ceil(output_w / max_tile_size)
        tiles_y = math.ceil(output_h / max_tile_size)
        
        tile_width = (output_w + overlap * (tiles_x - 1)) // tiles_x
        tile_height = (output_h + overlap * (tiles_y - 1)) // tiles_y
        
        return {
            'tiles_x': tiles_x,
            'tiles_y': tiles_y,
            'total': tiles_x * tiles_y,
            'tile_width': tile_width,
            'tile_height': tile_height,
            'overlap': overlap
        }
    
    def split_image(self, image: Image.Image, tile_config: dict):
        """Split image into tiles with overlap"""
        tiles = []
        img_array = np.array(image)
        
        for y in range(tile_config['tiles_y']):
            for x in range(tile_config['tiles_x']):
                start_x = x * (tile_config['tile_width'] - tile_config['overlap'])
                start_y = y * (tile_config['tile_height'] - tile_config['overlap'])
                
                end_x = min(start_x + tile_config['tile_width'], img_array.shape[1])
                end_y = min(start_y + tile_config['tile_height'], img_array.shape[0])
                
                tile = img_array[start_y:end_y, start_x:end_x]
                tiles.append({
                    'image': Image.fromarray(tile),
                    'position': (x, y),
                    'coords': (start_x, start_y, end_x, end_y)
                })
        
        return tiles
    
    def analyze_tile_with_gemini_vision(self, tile_image: Image.Image, base_prompt: str):
        """Use Gemini 2.5 Pro to analyze tile and generate detailed description"""
        try:
            # Convert image to bytes
            buffered = BytesIO()
            tile_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            
            # Create prompt for analysis
            analysis_prompt = f"""{base_prompt}
            
Analyze this image section in extreme detail. Describe:
- Main subjects, objects, and their positions
- Colors, textures, materials, and lighting
- Facial features, skin details, hair texture (if person)
- Background elements and atmosphere
- Style and mood

Provide a comprehensive description that can be used to regenerate this image at higher resolution while preserving all details."""
            
            # Call Gemini 2.5 Pro for analysis
            response = self.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
                    analysis_prompt
                ]
            )
            
            description = response.text
            print(f"Generated description: {description[:100]}...")
            return description
            
        except Exception as e:
            print(f"Error analyzing tile: {e}")
            return base_prompt
    
    def regenerate_tile_with_imagen(self, description: str, target_size: tuple):
        """Use Gemini 3 Pro Imagen to regenerate tile at higher resolution"""
        try:
            # Use Gemini 3 Pro Image model for generation
            response = self.client.models.generate_content(
                model='gemini-3-pro-image-preview',
                contents=[description],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio='1:1' if target_size[0] == target_size[1] else '16:9'
                    )
                )
            )
            
            # Extract image from response
            for part in response.parts:
                if part.inline_data is not None:
                    # Get the image
                    img = Image.open(BytesIO(part.inline_data.data))
                    # Resize to exact target size
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    return img
            
            # Fallback: simple upscale if generation fails
            print("Warning: Image generation failed, using fallback")
            return None
            
        except Exception as e:
            print(f"Error regenerating tile: {e}")
            return None
    
    def upscale_tile(self, tile_image: Image.Image, scale: int, prompt: str):
        """Upscale tile using Gemini Vision + Imagen pipeline"""
        target_size = (tile_image.width * scale, tile_image.height * scale)
        
        print(f"  → Analyzing tile with Gemini 2.5 Pro...")
        description = self.analyze_tile_with_gemini_vision(tile_image, prompt)
        
        print(f"  → Regenerating at {target_size[0]}x{target_size[1]} with Imagen...")
        regenerated = self.regenerate_tile_with_imagen(description, target_size)
        
        if regenerated:
            return regenerated
        else:
            # Fallback to high-quality interpolation
            print(f"  → Using fallback interpolation")
            return tile_image.resize(target_size, Image.Resampling.LANCZOS)
    
    def blend_tiles(self, tiles: list, tile_config: dict, output_size: tuple):
        """Reconstruct image from upscaled tiles with blending"""
        output = np.zeros((output_size[1], output_size[0], 3), dtype=np.float32)
        weight_map = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
        
        overlap = tile_config['overlap']
        
        for tile_data in tiles:
            tile_img = np.array(tile_data['upscaled']).astype(np.float32)
            x, y = tile_data['position']
            coords = tile_data['coords']
            
            # Create weight mask for blending
            weight = np.ones((tile_img.shape[0], tile_img.shape[1]), dtype=np.float32)
            
            if overlap > 0:
                # Feather edges
                for i in range(min(overlap, tile_img.shape[1])):
                    alpha = i / overlap
                    if x > 0 and i < tile_img.shape[1]:
                        weight[:, i] *= alpha
                
                for i in range(min(overlap, tile_img.shape[0])):
                    alpha = i / overlap
                    if y > 0 and i < tile_img.shape[0]:
                        weight[i, :] *= alpha
            
            # Calculate actual tile size in output
            out_h = min(tile_img.shape[0], output_size[1] - coords[1])
            out_w = min(tile_img.shape[1], output_size[0] - coords[0])
            
            # Add tile to output with weights
            output[coords[1]:coords[1]+out_h, coords[0]:coords[0]+out_w] += \
                tile_img[:out_h, :out_w] * weight[:out_h, :out_w, np.newaxis]
            weight_map[coords[1]:coords[1]+out_h, coords[0]:coords[0]+out_w] += \
                weight[:out_h, :out_w]
        
        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-5)
        output = (output / weight_map[:, :, np.newaxis]).astype(np.uint8)
        
        return Image.fromarray(output)
    
    def predict(
        self,
        image: Path = Input(description="Input image (~4000-4500px)"),
        scale: int = Input(description="Scale factor", default=2, choices=[2, 3, 4, 8]),
        prompt: str = Input(
            description="Enhancement prompt",
            default="""Ultra high-definition photorealistic upscaling at 4096px resolution. 
Preserve with extreme fidelity: facial features, skin texture (pores, fine lines), 
eyes (iris detail, catchlights), hair strands, fabric weave, material surfaces. 
Enhance sharpness and clarity while maintaining natural appearance. 
Perfect color accuracy, no artifacts, seamless quality."""
        ),
        gemini_api_key: str = Input(description="Gemini API Key (GEMINI_API_KEY)", default="")
    ) -> Path:
        """AI-powered upscaling using Gemini 2.5 Pro Vision + Gemini 3 Pro Imagen"""
        
        # Set API key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            self.client = genai.Client(api_key=gemini_api_key)
        
        # Load image
        input_image = Image.open(image).convert('RGB')
        width, height = input_image.size
        
        print(f"Input: {width}x{height}")
        print(f"Scale: {scale}x")
        print(f"Output: {width*scale}x{height*scale}")
        print(f"Method: Gemini 2.5 Pro Vision → Gemini 3 Pro Imagen")
        
        # Calculate tiles
        tile_config = self.calculate_tiles(width, height, scale)
        print(f"Tiles: {tile_config['total']} ({tile_config['tiles_x']}x{tile_config['tiles_y']})")
        print(f"Cost estimate: ${tile_config['total'] * 0.10:.2f} (Gemini calls)")
        
        # Split image
        tiles = self.split_image(input_image, tile_config)
        
        # Process each tile with AI pipeline
        for i, tile_data in enumerate(tiles):
            print(f"\nProcessing tile {i+1}/{len(tiles)}...")
            upscaled = self.upscale_tile(tile_data['image'], scale, prompt)
            tile_data['upscaled'] = upscaled
        
        # Blend tiles
        print("\nBlending tiles...")
        output_image = self.blend_tiles(tiles, tile_config, (width*scale, height*scale))
        
        # Save output
        output_path = "/tmp/output.png"
        output_image.save(output_path, quality=95, optimize=True)
        
        print(f"\nDone! Output: {output_image.size[0]}x{output_image.size[1]}")
        return Path(output_path)
