import os
import math
import time
import json
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Inicializa las APIs de Google"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrada")
        genai.configure(api_key=api_key)
        
        # Modelos
        self.general_model = genai.GenerativeModel('gemini-1.5-flash') # El Estratega
        self.soldier_model = genai.GenerativeModel('imagen-3.0-generate-001') # El Pintor


    def the_general_analyzes(self, image: Image.Image):
        """
        FASE 1: EL GENERAL (Gemini 1.5 Flash)
        Analiza la imagen completa para crear las 'Leyes de la Física' de esta sesión.
        """
        print("🫡 El General está analizando el terreno...")
        
        # Reducimos la imagen para que Flash la procese rápido (token economy)
        analysis_proxy = image.copy()
        analysis_proxy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        prompt = """
        ACT AS: Director of Photography & Biologist.
        TASK: Analyze this image to coordinate a massive high-res reconstruction.
        
        OUTPUT JSON ONLY with these keys:
        1. "light_vector": Precise direction of key light (e.g., 'From top-right 45deg').
        2. "shadow_quality": Hardness and color of shadows (e.g., 'Soft, cool blue fill').
        3. "skin_biology": Age-appropriate skin details (e.g., 'Visible pores, sun spots, vellus hair').
        4. "chromatic_palette": Dominant hex codes.
        5. "geometry_lock": Description of facial landmarks to maintain identity.
        """
        
        try:
            response = self.general_model.generate_content([prompt, analysis_proxy])
            # Limpieza básica del JSON (a veces Gemini añade markdown)
            json_str = response.text.replace('```json', '').replace('```', '').strip()
            context = json.loads(json_str)
            print("✅ Órdenes del General recibidas.")
            return context
        except Exception as e:
            print(f"⚠️ Fallo del General ({e}). Usando protocolo de emergencia.")
            return {
                "light_vector": "Neutral/Frontal", 
                "skin_biology": "High frequency texture",
                "geometry_lock": "Strict identity adherence"
            }


    def the_soldier_paints(self, tile_image: Image.Image, context: dict, tile_id: str):
        """
        FASE 2: EL SOLDADO (Imagen 3.0)
        Ejecuta el PROMPT MAESTRO inyectando las órdenes del General.
        """
        
        # EL PROMPT MAESTRO DE LUXIFIER (Inyección Dinámica)
        master_prompt = f"""
        **ROL DEL SISTEMA:** MOTOR DE RECONSTRUCCIÓN GENERATIVA DE ULTRA-ALTA FIDELIDAD.
        
        **CONTEXTO GLOBAL (ÓRDENES DEL GENERAL - INMUTABLE):**
        - ILUMINACIÓN: {context.get('light_vector')} ({context.get('shadow_quality')}).
        - BIOLOGÍA: {context.get('skin_biology')}.
        - IDENTIDAD: {context.get('geometry_lock')}.
        
        **DIRECTIVAS DE EJECUCIÓN PARA ESTE SECTOR ({tile_id}):**
        
        **FASE 1: ANCLA DE IDENTIDAD**
        Usa la imagen de entrada (este tile) como restricción geométrica rígida (Strength: 0.95). 
        La estructura ósea NO PUEDE CAMBIAR.
        
        **FASE 2: MOTOR DE ALUCINACIÓN (TEXTURA)**
        PROHIBIDO SUAVIZAR. Interpreta lo borroso como datos faltantes.
        - PIEL: Genera poros individuales, vello facial (vellus) imperceptible, vascularización sutil.
        - MATERIALES: Reconstruye el tejido hilo por hilo.
        
        **FASE 3: SIMULACIÓN ÓPTICA (PHASE ONE IQ4)**
        Simula lente f/1.2. Enfoque rabioso en texturas, bokeh cremoso en profundidad.
        
        **FASE 4: REILUMINACIÓN VOLUMÉTRICA**
        Si la luz original es plana, destrúyela. Esculpe con micro-sombras dentro de los nuevos poros.
        
        **OBJETIVO:** Densidad abrumadora de información. Indistinguible de RAW 100MP.
        """
        
        try:
            # Generación con el modelo Imagen
            response = self.soldier_model.generate_content(
                [master_prompt, tile_image]
            )
            
            # Extraer imagen generada
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data'):
                        from io import BytesIO
                        img = Image.open(BytesIO(part.inline_data.data))
                        # Asegurar el tamaño correcto
                        if img.size != tile_image.size:
                            img = img.resize(tile_image.size, Image.Resampling.LANCZOS)
                        return img
            
            print(f"⚠️ Soldado {tile_id}: respuesta sin imagen, usando fallback")
            return tile_image
            
        except Exception as e:
            print(f"❌ Soldado {tile_id} falló: {e}")
            return tile_image # Retornar original como fallback


    def calculate_dynamic_grid(self, width, height, target_pixel_side=4096):
        """Calcula cuántos tiles hacen falta para cubrir la imagen gigante"""
        # Gemini trabaja bien con tiles de 2048px
        tile_size = 2048 
        overlap = 256 # Margen de seguridad grande para el blending
        
        # Escalamos la imagen de entrada al tamaño objetivo final
        scale_factor = max(target_pixel_side / width, target_pixel_side / height)
        
        final_w = int(width * scale_factor)
        final_h = int(height * scale_factor)
        
        cols = math.ceil((final_w - overlap) / (tile_size - overlap))
        rows = math.ceil((final_h - overlap) / (tile_size - overlap))
        
        return rows, cols, final_w, final_h, tile_size, overlap


    def predict(
        self,
        image: Path = Input(description="Input Image (Any resolution)"),
        target_resolution: int = Input(description="Lado más largo objetivo (ej: 4096, 8192)", default=4096),
        gemini_api_key: str = Input(description="API Key (Opcional si está en env)", default="")
    ) -> Path:
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.setup()
            
        # 1. Preparación
        input_img = Image.open(image).convert('RGB')
        
        # 2. El General Analiza (sobre la original pequeña)
        global_context = self.the_general_analyzes(input_img)
        
        # 3. Cálculo de la Matriz de Batalla
        rows, cols, final_w, final_h, tile_size, overlap = self.calculate_dynamic_grid(
            input_img.width, input_img.height, target_resolution
        )
        print(f"⚔️ Matriz de Batalla: {rows}x{cols} ({rows*cols} Tiles). Objetivo: {final_w}x{final_h}px")
        
        # 4. Escalar imagen base al tamaño objetivo (Lienzo vacío de guía)
        canvas_base = input_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
        canvas_array = np.array(canvas_base)
        
        # 5. Ejecución de Tiles
        final_image = np.zeros_like(canvas_array, dtype=np.float32)
        weight_map = np.zeros((final_h, final_w), dtype=np.float32)
        
        total_tiles = rows * cols
        count = 0
        
        for r in range(rows):
            for c in range(cols):
                count += 1
                
                # Calcular coordenadas con overlap
                x_start = c * (tile_size - overlap)
                y_start = r * (tile_size - overlap)
                
                # Ajuste para el último tile
                if x_start + tile_size > final_w: x_start = final_w - tile_size
                if y_start + tile_size > final_h: y_start = final_h - tile_size
                
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                
                x_end = x_start + tile_size
                y_end = y_start + tile_size
                
                # Extraer el tile base
                tile_crop = Image.fromarray(canvas_array[y_start:y_end, x_start:x_end])
                
                print(f"🎨 Soldado pintando sector {count}/{total_tiles}...")
                
                # LLAMADA AL PINTOR
                painted_tile = self.the_soldier_paints(tile_crop, global_context, f"R{r}C{c}")
                painted_array = np.array(painted_tile).astype(np.float32)
                
                # Asegurar tamaño
                if painted_array.shape[:2] != (tile_size, tile_size):
                    painted_array = cv2.resize(painted_array, (tile_size, tile_size))
                
                # Máscara de Gradiente (Feathering)
                tile_weight = np.ones((tile_size, tile_size), dtype=np.float32)
                
                feather = overlap // 2
                if feather > 0:
                    # Degradado en bordes
                    tile_weight[:feather, :] *= np.linspace(0, 1, feather)[:, None]
                    tile_weight[-feather:, :] *= np.linspace(1, 0, feather)[:, None]
                    tile_weight[:, :feather] *= np.linspace(0, 1, feather)[None, :]
                    tile_weight[:, -feather:] *= np.linspace(1, 0, feather)[None, :]
                
                # Acumular en el lienzo maestro
                final_image[y_start:y_end, x_start:x_end] += painted_array * tile_weight[..., None]
                weight_map[y_start:y_end, x_start:x_end] += tile_weight
                
                # Pausa técnica para evitar rate limits
                time.sleep(0.5)

        # 6. Normalización Final
        print("🪡 El Ensamblador está uniendo los pliegos...")
        weight_map = np.maximum(weight_map, 1e-5)
        final_image /= weight_map[..., None]
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        out_path = "/tmp/lux_master_output.png"
        Image.fromarray(final_image).save(out_path, quality=95)
        
        return Path(out_path)
