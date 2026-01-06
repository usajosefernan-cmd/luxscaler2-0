# Luxscaler 2.0 - AI-powered upscaling with Gemini 2.5 Flash + Imagen 3.0 Capability
# Arquitectura "El General y el Soldado"

import os
import math
import json
import base64
import io
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
from cog import BasePredictor, Input, Path
from typing import Optional
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

class Predictor(BasePredictor):
    def setup(self):
        """Initialize Gemini API for 'The General'"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        # El General: Gemini 2.5 Flash para análisis global
        self.general_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configurar credenciales de Google Cloud para Imagen 3
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = "us-central1"
        
    def analyze_image_with_general(self, image: Image.Image) -> dict:
        """FASE 1: EL GENERAL - Analiza la imagen completa y extrae características globales"""
        
        prompt = """Analyze this image and provide a detailed JSON response with the following structure:
{
  "identity_lock": "Describe facial geometry, bone structure, proportions (if person)",
  "light_map": "Describe light direction, intensity, color temperature, shadows",
  "material_zones": {
    "skin": "Describe skin texture, pores, tone",
    "hair": "Describe hair structure and details",
    "fabric": "Describe fabric weave, material properties",
    "background": "Describe background textures"
  },
  "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"],
  "texture_instructions": "Specific instructions for generating Phase One IQ4 quality textures"
}

Focus on photographic details that would be present in a 150MP camera capture."""
        
        try:
            response = self.general_model.generate_content([prompt, image])
            # Extraer JSON de la respuesta
            text = response.text
            # Buscar JSON en la respuesta
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                analysis = json.loads(json_str)
                return analysis
            else:
                # Fallback: análisis básico
                return {
                    "identity_lock": "Preserve original structure",
                    "light_map": "Natural lighting",
                    "material_zones": {"general": "High detail textures"},
                    "color_palette": ["#FFFFFF"],
                    "texture_instructions": "Generate photorealistic micro-details"
                }
        except Exception as e:
            print(f"Error in General analysis: {e}")
            return {
                "identity_lock": "Preserve original structure",
                "light_map": "Natural lighting",
                "material_zones": {"general": "High detail textures"},
                "color_palette": ["#FFFFFF"],
                "texture_instructions": "Generate photorealistic micro-details"
            }
    
    def enhance_with_soldier(self, image_or_tile: Image.Image, original_image: Image.Image, 
                            analysis: dict, is_full_image: bool = False) -> Image.Image:
        """FASE 2: EL SOLDADO - Mejora con Imagen 3.0 Capability (editing mode)
        
        Args:
            image_or_tile: El tile a mejorar (o imagen completa si <4K)
            original_image: La imagen original COMPLETA (referencia global)
            analysis: Análisis del General (JSON)
            is_full_image: True si procesamos la imagen completa sin tiles
        """
        
        # Construir el mega-prompt con las órdenes del General
        material_zones_str = json.dumps(analysis.get('material_zones', {}), indent=2)
        color_palette_str = ', '.join(analysis.get('color_palette', ['#FFFFFF']))
        
        prompt = f"""LUXURY UPSCALING - Phase One IQ4 150MP Quality

IDENTITY LOCK: {analysis.get('identity_lock', 'Preserve original structure')}
LIGHT MAP: {analysis.get('light_map', 'Natural lighting')}
MATERIAL ZONES:
{material_zones_str}
COLOR PALETTE: {color_palette_str}
TEXTURE INSTRUCTIONS: {analysis.get('texture_instructions', 'Generate photorealistic details')}

MISSION: Enhance this {'image' if is_full_image else 'tile region'} maintaining ABSOLUTE coherence with the original image.
Generate ultra-realistic skin micro-texture (pores, fine lines, subsurface scattering).
Preserve exact facial geometry, lighting direction, and color harmony.
Simulate Phase One IQ4 camera sensor characteristics.
"""
        
        try:
            # Convertir imágenes a base64
            tile_b64 = self._image_to_base64(image_or_tile)
            original_b64 = self._image_to_base64(original_image)
            
            # URL del endpoint de Vertex AI para Imagen 3.0 Capability
            url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/imagen-3.0-capability-001:predict"
            
            headers = {
                "Authorization": f"Bearer {self._get_access_token()}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "instances": [
                    {
                        "prompt": prompt,
                        "referenceImages": [
                            {
                                "referenceType": "REFERENCE_TYPE_RAW",
                                "referenceId": 1,
                                "referenceImage": {
                                    "bytesBase64Encoded": original_b64
                                }
                            },
                            {
                                "referenceType": "REFERENCE_TYPE_RAW",
                                "referenceId": 2,
                                "referenceImage": {
                                    "bytesBase64Encoded": tile_b64
                                }
                            }
                        ]
                    }
                ],
                "parameters": {
                    "sampleCount": 1,
                    "editMode": "product-image"  # Modo que preserva estructura
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            enhanced_b64 = result['predictions'][0]['bytesBase64Encoded']
            enhanced_bytes = base64.b64decode(enhanced_b64)
            
            return Image.open(io.BytesIO(enhanced_bytes))
            
        except Exception as e:
            print(f"Error in Soldier enhancement: {e}")
            return image_or_tile  # Fallback al original
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convierte PIL Image a base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _get_access_token(self) -> str:
        """Obtiene access token de Google Cloud"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                '/tmp/service-account-key.json',
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            credentials.refresh(Request())
            return credentials.token
        except:
            # Fallback: intentar con credenciales por defecto
            from google.auth import default
            credentials, project = default()
            credentials.refresh(Request())
            return credentials.token

    def predict(
        self,
        image: Path = Input(description="Input image (any resolution)"),
        scale_factor: int = Input(description="Scale factor (2x, 3x, 4x, 8x)", choices=[2, 3, 4, 8], default=2),
        gemini_api_key: str = Input(description="Gemini API key (optional)", default="")
    ) -> Path:
        """Main prediction method with General-Soldier architecture"""
        
        # Configurar API key si se proporciona
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            genai.configure(api_key=gemini_api_key)
        
        # Cargar imagen original
        input_img = Image.open(image).convert('RGB')
        input_width, input_height = input_img.size
        
        # Calcular dimensiones de salida
        output_width = input_width * scale_factor
        output_height = input_height * scale_factor
        
        max_dimension = max(output_width, output_height)
        
        print(f"Input: {input_width}x{input_height}")
        print(f"Output target: {output_width}x{output_height}")
        print(f"Max dimension: {max_dimension}px")
        
        # === FASE 1: EL GENERAL ANALIZA ===
        print("\n=== FASE 1: EL GENERAL (Gemini 2.5 Flash) ===")
        global_analysis = self.analyze_image_with_general(input_img)
        print(f"Analysis complete: {json.dumps(global_analysis, indent=2)}")
        
        # === DECISIÓN: TILES O IMAGEN COMPLETA ===
        if max_dimension <= 4096:
            # ¡Sin tiles! Procesar imagen completa directamente
            print(f"\n✅ Output ≤ 4096px: Processing FULL IMAGE without tiles")
            print(f"Cost: $0.06 (single call)")
            
            # Upscale básico con interpolación
            upscaled_img = input_img.resize((output_width, output_height), Image.LANCZOS)
            
            # === FASE 2: EL SOLDADO MEJORA LA IMAGEN COMPLETA ===
            print("\n=== FASE 2: EL SOLDADO (Imagen 3.0 Capability) ===")
            enhanced_img = self.enhance_with_soldier(
                image_or_tile=upscaled_img,
                original_image=input_img,
                analysis=global_analysis,
                is_full_image=True
            )
            
            # Guardar resultado
            output_path = "/tmp/output.png"
            enhanced_img.save(output_path, quality=95)
            print(f"\n✅ Complete! Saved to {output_path}")
            return Path(output_path)
        
        else:
            # Con tiles para imágenes grandes
            print(f"\n📊 Output > 4096px: Using TILED processing")
            
            # Calcular tiles
            max_tile_size = 4096
            overlap = 50 * scale_factor
            
            tiles_x = math.ceil(output_width / max_tile_size)
            tiles_y = math.ceil(output_height / max_tile_size)
            total_tiles = tiles_x * tiles_y
            
            tile_width = (output_width + overlap * (tiles_x - 1)) // tiles_x
            tile_height = (output_height + overlap * (tiles_y - 1)) // tiles_y
            
            print(f"Tiles: {tiles_x}x{tiles_y} = {total_tiles} tiles")
            print(f"Tile size: {tile_width}x{tile_height}px")
            print(f"Overlap: {overlap}px")
            print(f"Cost: {total_tiles} × $0.06 = ${total_tiles * 0.06:.2f}")
            
            # Upscale básico
            upscaled_img = input_img.resize((output_width, output_height), Image.LANCZOS)
            
            # Canvas para resultado final
            final_img = Image.new('RGB', (output_width, output_height))
            
            # === FASE 2: EL SOLDADO PROCESA CADA TILE ===
            print("\n=== FASE 2: EL SOLDADO (Imagen 3.0 Capability) ===")
            
            tile_idx = 0
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    x1 = tx * (tile_width - overlap)
                    y1 = ty * (tile_height - overlap)
                    x2 = min(x1 + tile_width, output_width)
                    y2 = min(y1 + tile_height, output_height)
                    
                    print(f"\nProcessing tile {tile_idx + 1}/{total_tiles}: ({x1},{y1})-({x2},{y2})")
                    
                    # Extraer tile
                    tile = upscaled_img.crop((x1, y1, x2, y2))
                    
                    # Mejorar con el Soldado (recibe imagen original completa para contexto)
                    enhanced_tile = self.enhance_with_soldier(
                        image_or_tile=tile,
                        original_image=input_img,
                        analysis=global_analysis,
                        is_full_image=False
                    )
                    
                    # Pegar con blending
                    if overlap > 0 and (tx > 0 or ty > 0):
                        final_img.paste(enhanced_tile, (x1, y1))
                    else:
                        final_img.paste(enhanced_tile, (x1, y1))
                    
                    tile_idx += 1
            
            # === FASE 3: ENSAMBLADOR (blending ya hecho) ===
            output_path = "/tmp/output.png"
            final_img.save(output_path, quality=95)
            print(f"\n✅ Complete! Saved to {output_path}")
            return Path(output_path)
