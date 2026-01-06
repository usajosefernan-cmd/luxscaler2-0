# LuxScaler 2.0 - AI-powered upscaling with Gemini 2.5 Flash + Imagen 3.0 Capability
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
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from typing import Optional

class Predictor(BasePredictor):
    def setup(self):
        """Initialize - project configured for luxnode01"""
        self.project_id = "luxifier-node-3362-1"
        self.location = "us-central1"
        print(f"Setup completado - Proyecto: {self.project_id}")
    
    def _setup_gcp_credentials(self, gcp_json: str):
        """Escribe credenciales GCP en archivo temporal"""
        creds_path = "/tmp/gcp-key.json"
        with open(creds_path, 'w') as f:
            f.write(gcp_json)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
        return creds_path
    
    def _get_access_token(self, creds_path: str):
        """Obtiene token de acceso para Imagen 3.0 Capability API"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            print(f"Error obteniendo token: {e}")
            return None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convierte imagen PIL a base64 PNG"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def analyze_image_with_general(self, image: Image.Image, api_key: str) -> dict:
        """FASE 1: El General (Gemini) analiza la imagen global"""
        print("[GENERAL] Analizando imagen completa con Gemini 2.0 Flash...")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = """Analiza esta imagen en detalle y extrae:
1. identity_lock: Geometría facial, proporciones, rasgos únicos
2. light_map: Dirección de luz, intensidad, calidad de sombras
3. material_zones: Zonas de piel, cabello, ropa, metales
4. color_palette: Paleta cromática dominante (hex codes)
5. texture_instructions: Texturas de piel, cabello, telas

Responde SOLO con un objeto JSON válido, sin markdown ni explicaciones."""
        
        try:
            response = model.generate_content([prompt, image])
            text = response.text.strip()
            
            # Extraer JSON del texto
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                analysis = json.loads(json_str)
                print(f"[GENERAL] Análisis completado: {list(analysis.keys())}")
                return analysis
        except Exception as e:
            print(f"[GENERAL] Error: {e}")
        
        # Fallback
        return {
            "identity_lock": "facial geometry preserved",
            "light_map": "natural lighting",
            "material_zones": "skin, hair, clothing detected",
            "color_palette": ["#F5E6D3", "#8B7355"],
            "texture_instructions": "enhance skin texture, preserve hair detail"
        }
    
    def enhance_with_soldier(self, tile: Image.Image, original_img: Image.Image, 
                           analysis: dict, scale_factor: int, creds_path: str,
                           is_full_image: bool = False) -> Image.Image:
        """FASE 2: El Soldado (Imagen 3.0) mejora el tile con contexto global"""
        tile_type = "imagen completa" if is_full_image else "tile"
        print(f"[SOLDADO] Mejorando {tile_type} con Imagen 3.0 Capability...")
        
        # Construir mega-prompt con análisis del General
        mega_prompt = f"""Ultra high-resolution {scale_factor}x upscale. 
Identity: {analysis.get('identity_lock', '')}
Lighting: {analysis.get('light_map', '')}
Materials: {analysis.get('material_zones', '')}
Colors: {analysis.get('color_palette', [])}
Textures: {analysis.get('texture_instructions', '')}

Generate photorealistic detail with Phase One IQ4 camera quality:
- Organic skin texture with visible pores and microdetail
- Natural fiber structure in fabrics
- Realistic hair strands with fine detail
- Controlled hallucination for ultra-sharp 100MP equivalent
- Preserve facial geometry and identity
- Magnific-style HDR micro-contrast
- RAW-like depth and dynamic range"""
        
        try:
            # Convertir imágenes a base64
            tile_b64 = self._image_to_base64(tile)
            original_b64 = self._image_to_base64(original_img)
            
            # Obtener token
            token = self._get_access_token(creds_path)
            if not token:
                print("[SOLDADO] No hay token de GCP, retornando tile sin mejorar")
                return tile
            
            # Llamada a Imagen 3.0 Capability API
            url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/imagen-3.0-capability-001:predict"
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "instances": [{
                    "prompt": mega_prompt,
                    "referenceImages": [
                        {"referenceId": 1, "referenceType": "REFERENCE_TYPE_RAW", "referenceImage": {"bytesBase64Encoded": tile_b64}},
                        {"referenceId": 2, "referenceType": "REFERENCE_TYPE_RAW", "referenceImage": {"bytesBase64Encoded": original_b64}}
                    ]
                }],
                "parameters": {
                    "sampleCount": 1,
                    "editMode": "product-image"
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                if "predictions" in result and len(result["predictions"]) > 0:
                    img_b64 = result["predictions"][0].get("bytesBase64Encoded")
                    if img_b64:
                        img_data = base64.b64decode(img_b64)
                        enhanced = Image.open(io.BytesIO(img_data))
                        print(f"[SOLDADO] {tile_type.capitalize()} mejorado exitosamente")
                        return enhanced
            else:
                print(f"[SOLDADO] Error API: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"[SOLDADO] Error: {e}")
        
        # Si falla, retornar tile original
        return tile
    
    def predict(
        self,
        image: Path = Input(description="Imagen a escalar"),
        scale_factor: int = Input(description="Factor de escala", default=2, choices=[2, 3, 4, 8]),
        gemini_api_key: str = Input(description="Gemini API Key", default=""),
        gcp_credentials_json: str = Input(description="GCP Service Account JSON (completo)", default="")
    ) -> Path:
        """Escala imagen usando arquitectura General-Soldier"""
        
        # Validar API keys
        if not gemini_api_key:
            gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            raise ValueError("gemini_api_key es requerido")
            
        if not gcp_credentials_json:
            gcp_credentials_json = os.environ.get("GCP_CREDENTIALS_JSON", "")
        if not gcp_credentials_json:
            raise ValueError("gcp_credentials_json es requerido (JSON completo de Service Account)")
        
        # Setup credenciales GCP
        creds_path = self._setup_gcp_credentials(gcp_credentials_json)
        
        print(f"\n{'='*60}")  
        print(f"LUXSCALER 2.0 - Arquitectura 'El General y el Soldado'")
        print(f"Proyecto GCP: {self.project_id}")
        print(f"Factor de escala: {scale_factor}x")
        print(f"{'='*60}\n")
        
        # Cargar imagen
        img = Image.open(str(image)).convert("RGB")
        width, height = img.size
        output_width = width * scale_factor
        output_height = height * scale_factor
        max_dimension = max(output_width, output_height)
        
        print(f"Resolución entrada: {width}x{height}")
        print(f"Resolución salida: {output_width}x{output_height}")
        
        # FASE 1: Análisis del General
        analysis = self.analyze_image_with_general(img, gemini_api_key)
        
        # LÓGICA ADAPTATIVA: Sin tiles hasta 4K, con tiles más allá
        if max_dimension <= 4096:
            print("\n[MODO SIN TILES] Imagen ≤4K - procesamiento en 1 llamada")
            print(f"Costo estimado: $0.06\n")
            
            # Upscale con interpolación básica
            upscaled = img.resize((output_width, output_height), Image.Resampling.LANCZOS)
            
            # FASE 2: Mejora con Soldado (imagen completa)
            final_img = self.enhance_with_soldier(upscaled, img, analysis, scale_factor, creds_path, is_full_image=True)
            
        else:
            print("\n[MODO CON TILES] Imagen >4K - procesamiento por tiles")
            
            # Configuración de tiles
            max_tile_size = 4096
            overlap = 50 * scale_factor
            
            tiles_x = math.ceil(output_width / max_tile_size)
            tiles_y = math.ceil(output_height / max_tile_size)
            total_tiles = tiles_x * tiles_y
            
            tile_width = (output_width + (tiles_x - 1) * overlap) // tiles_x
            tile_height = (output_height + (tiles_y - 1) * overlap) // tiles_y
            
            print(f"Grid: {tiles_x}x{tiles_y} = {total_tiles} tiles")
            print(f"Tamaño tile: {tile_width}x{tile_height} (overlap: {overlap}px)")
            print(f"Costo estimado: ${total_tiles * 0.06:.2f}\n")
            
            # Upscale completo
            upscaled = img.resize((output_width, output_height), Image.Resampling.LANCZOS)
            final_img = Image.new("RGB", (output_width, output_height))
            
            # FASE 2: Procesar cada tile
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    tile_num = ty * tiles_x + tx + 1
                    print(f"Procesando tile {tile_num}/{total_tiles} [{tx},{ty}]")
                    
                    # Extraer tile con overlap
                    x1 = max(0, tx * tile_width - overlap//2)
                    y1 = max(0, ty * tile_height - overlap//2)
                    x2 = min(output_width, x1 + tile_width + overlap)
                    y2 = min(output_height, y1 + tile_height + overlap)
                    
                    tile = upscaled.crop((x1, y1, x2, y2))
                    
                    # FASE 2: Mejorar tile con Soldado (contexto global)
                    enhanced_tile = self.enhance_with_soldier(tile, img, analysis, scale_factor, creds_path, is_full_image=False)
                    
                    # FASE 3: Pegar tile (el overlap ayuda con el blending)
                    final_img.paste(enhanced_tile, (x1, y1))
        
        # Guardar resultado
        output_path = "/tmp/output.png"
        final_img.save(output_path, "PNG", quality=100)
        print(f"\n[COMPLETADO] Imagen guardada: {output_width}x{output_height}")
        
        return Path(output_path)
