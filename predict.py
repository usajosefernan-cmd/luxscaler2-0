import os
import math
from google import genai
from google.genai import types
from PIL import Image
import numpy as np
import cv2
from cog import BasePredictor, Input, Path
from typing import Optional
from io import BytesIO

class Predictor(BasePredictor):
    def setup(self):
        """Initialize Gemini API"""
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
    
    def upscale_tile(self, tile_image: Image.Image, prompt: str):
        """Upscale single tile using Gemini Imagen 4 Ultra"""
        try:
            response = self.client.models.generate_images(
                model='imagen-4.0-ultra-generate-001',
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1"
                )
            )
            
            return response.generated_images[0].image
        except Exception as e:
            print(f"Error upscaling tile: {e}")
            # Fallback: return original tile if API fails
            return tile_image
    
    def blend_tiles(self, tiles: list, tile_config: dict, output_size: tuple):
        """Reconstruct image from upscaled tiles with blending"""
        output = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        weight_map = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
        
        overlap = tile_config['overlap']
        
        for tile_data in tiles:
            tile_img = np.array(tile_data['upscaled'])
            x, y = tile_data['position']
            coords = tile_data['coords']
            
            # Create weight mask for blending
            weight = np.ones((tile_img.shape[0], tile_img.shape[1]), dtype=np.float32)
            
            if overlap > 0:
                # Feather edges
                for i in range(overlap):
                    alpha = i / overlap
                    if x > 0:  # Left edge
                        weight[:, i] *= alpha
                    if y > 0:  # Top edge
                        weight[i, :] *= alpha
            
            # Add tile to output with weights
            output[coords[1]:coords[3], coords[0]:coords[2]] += (tile_img * weight[:, :, np.newaxis]).astype(np.uint8)
            weight_map[coords[1]:coords[3], coords[0]:coords[2]] += weight
        
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
        """Main prediction function"""
        
        # Set API key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            self.client = genai.Client(api_key=gemini_api_key)
        
        # Load image
        input_image = Image.open(image)
        width, height = input_image.size
        
        print(f"Input: {width}x{height}")
        print(f"Scale: {scale}x")
        print(f"Output: {width*scale}x{height*scale}")
        
        # Calculate tiles
        tile_config = self.calculate_tiles(width, height, scale)
        print(f"Tiles: {tile_config['total']} ({tile_config['tiles_x']}x{tile_config['tiles_y']})")
        print(f"Cost estimate: ${tile_config['total'] * 0.06:.2f}")
        
        # Split image
        tiles = self.split_image(input_image, tile_config)
        
        # Upscale each tile
        for i, tile_data in enumerate(tiles):
            print(f"Processing tile {i+1}/{len(tiles)}...")
            upscaled = self.upscale_tile(tile_data['image'], prompt)
            tile_data['upscaled'] = upscaled
        
        # Blend tiles
        print("Blending tiles...")
        output_image = self.blend_tiles(tiles, tile_config, (width*scale, height*scale))
        
        # Save output
        output_path = "/tmp/output.png"
        output_image.save(output_path, quality=95)
        
        print(f"Done! Output: {output_image.size[0]}x{output_image.size[1]}")
        return Path(output_path)
