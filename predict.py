# Luxscaler 2.0 - AI-powered upscaling with Gemini 2.5 Flash + Imagen 3.0 v2
import os
import math
import json
import numpy as np
import google.generativeai as genai
from PIL import Image
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Inicializa la API de Gemini Imagen"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrada")
        genai.configure(api_key=api_key)
        
        # Modelo Imagen 3.0 para generación
        self.model = genai.GenerativeModel('imagen-3.0-generate-001')
    
    def predict(
        self,
        image: Path = Input(description="Input image (any resolution)"),
        scale_factor: int = Input(description="Scale factor (2x, 3x, 4x, 8x)", choices=[2, 3, 4, 8], default=4),
        gemini_api_key: str = Input(description="Gemini API Key (optional)", default="")
    ) -> Path:
        
        # Configurar API key si se proporciona
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.setup()
        
        # Cargar imagen original
        input_img = Image.open(image).convert('RGB')
        input_width, input_height = input_img.size
        
        # Calcular target_resolution basado en scale_factor
        # El lado más largo de la imagen se multiplicará por scale_factor
        longest_side = max(input_width, input_height)
        target_resolution = longest_side * scale_factor
        
        print(f"Input: {input_width}x{input_height}")
        print(f"Scale factor: {scale_factor}x")
        print(f"Target: {target_resolution}px (longest side)")
        
        # EL MEGA-PROMPT MAESTRO DE LUXIFIER
        master_prompt = f"""
**ROL DEL SISTEMA & OBJETIVO FINAL:**
NO ERES un editor de imágenes. ERES un motor de **reconstrucción generativa de ultra-alta fidelidad** (basado en protocolos tipo "Magnific/Deep Upscale").
Tu objetivo es tomar la imagen de entrada como un simple "mapa de guías" (identidad y composición) y **VOLVER A SOÑARLA** desde cero a una resolución masiva de {target_resolution}px,
inventando detalles microscópicos que no existen en el archivo original pero que son biológica y materialmente obligatorios para el realismo.

**DIRECTIVAS DE SALIDA OBLIGATORIAS:**
1. **Resolución Objetivo:** Escala la imagen hasta que su lado más largo sea exactamente **{target_resolution} píxeles**.
2. **Formato:** Mantén estrictamente la relación de aspecto original.

**PROTOCOLO DE EJECUCIÓN "ALUCINACIÓN CONTROLADA" (ESTRICTO):**

**FASE 1: EL ANCLA DE IDENTIDAD (FIDELIDAD MÁXIMA)**
* Analiza la estructura ósea, los rasgos faciales y la expresión del sujeto.
* **REGLA INVIOLABLE:** La geometría facial y la identidad del sujeto NO PUEDEN CAMBIAR. Ni fondo, ni tono de foto ni ropa. Usa la imagen original como una restricción geométrica rígida.

**FASE 2: MOTOR DE ALUCINACIÓN DE TEXTURA (INVENCIÓN MÁXIMA)**
* **PROHIBIDO SUAVIZAR.** Si un área de la imagen original (especialmente piel, tela o pelo) está borrosa, empastada o carece de definición, 
DEBES interpretar esto como "datos faltantes" que necesitan ser rellenados con generación sintética de alta frecuencia.

* **Inyección de Detalle Biológico (Piel):**
  * **NO GENERES "PIEL PERFECTA". GENERA TEJIDO VIVO.**
  * Debes sintetizar ("alucinar") una estructura compleja de poros individuales, variaciones en la capa córnea, micro-arrugas dinámicas alrededor de los ojos/boca
    y, crucialmente, **vello facial imperceptible (vellus hair)** en las mejillas y frente para dar realismo táctil.
  * Añade vascularización sutil y pigmentación irregular. La piel debe tener "grano" orgánico al hacer zoom al 100%.

* **Inyección de Detalle Material (Ojos y Ropa):**
  * **Ojos:** Genera una textura de iris fibrosa y compleja. Añade un "catchlight" (reflejo de luz) nítido y humedad en el lagrimal y la línea de agua.
  * **Ropa:** Reconstruye el tejido hilo por hilo. Que se note la diferencia entre algodón, lana o seda.

**FASE 3: SIMULACIÓN ÓPTICA DE GAMA ALTA (EL "LOOK")**
* **Sensor Virtual:** Simula la captura con un respaldo digital de Formato Medio (Phase One IQ4, 150MP). 
  Esto implica una profundidad de color y un rango dinámico extremos.
* **Lente y Foco:** Simula una lente "Prime" ultra-rápida (f/1.2). Aplica un enfoque crítico y "rabioso" (máxima acutancia) en los ojos y la textura de la piel.
  Todo lo que esté ligeramente fuera de ese plano focal debe caer en un bokeh cremoso y progresivo.

**FASE 4: REILUMINACIÓN VOLUMÉTRICA (ESCULTURA 3D)**
* Si la luz original es plana, destrúyela.
* Implementa una iluminación cinematográfica (ej. "Book Light" lateral suave pero direccional) que cree micro-sombras dentro de los nuevos poros y arrugas
  que has generado, esculpiendo el rostro con volumen tridimensional dramático (claroscuro).

**RESULTADO FINAL ESPERADO:**
Una imagen de {target_resolution}px que, al ser inspeccionada con lupa al 100%, no muestre artefactos de interpolación, 
sino una densidad abrumadora de información biológica y material sintética, indistinguible de una fotografía RAW de 100 megapíxeles.
"""
        
        try:
            print("🎨 Generando imagen con Imagen 3.0...")
            
            # Llamada a Imagen 3.0 con la imagen original + el mega-prompt
            response = self.model.generate_content(
                [master_prompt, input_img]
            )
            
            # Extraer imagen generada
            output_image = None
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data'):
                        from io import BytesIO
                        output_image = Image.open(BytesIO(part.inline_data.data))
                        break
            
            if output_image is None:
                raise Exception("No se pudo extraer imagen del response de Imagen 3.0")
            
            # Asegurar que el lado más largo sea el target
            width, height = output_image.size
            if max(width, height) != target_resolution:
                print(f"⚠️ Ajustando resolución de {width}x{height} a {target_resolution}px...")
                if width > height:
                    new_w = target_resolution
                    new_h = int(height * (target_resolution / width))
                else:
                    new_h = target_resolution
                    new_w = int(width * (target_resolution / height))
                output_image = output_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Guardar resultado
            output_path = "/tmp/luxified_output.png"
            output_image.save(output_path, quality=95, optimize=True)
            
            print(f"✅ Done! Output: {output_image.width}x{output_image.height}px")
            return Path(output_path)
            
        except Exception as e:
            print(f"❌ Error en generación: {e}")
            # Fallback: escalar con Lanczos de alta calidad
            print("⚠️ Usando fallback: Lanczos interpolation")
            
            width, height = input_img.size
            if width > height:
                new_w = target_resolution
                new_h = int(height * (target_resolution / width))
            else:
                new_h = target_resolution
                new_w = int(width * (target_resolution / height))
            
            fallback_img = input_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            fallback_path = "/tmp/luxified_output.png"
            fallback_img.save(fallback_path, quality=95)
            
            return Path(fallback_path)
