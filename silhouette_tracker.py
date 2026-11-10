import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class SilhouetteTracker:
    def __init__(self, model_path="selfie_segmenter.tflite", confidence=0.5):
        """
        Segmentación semántica de silueta humana basada en MediaPipe Tasks (v1.x+).
        Detecta la silueta estática o en movimiento sin foto previa de fondo.
        """
        self.confidence = confidence
        
        # Comprobar si el archivo del modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo en '{model_path}'. "
                "Descárgalo usando el comando curl facilitado."
            )

        # Configuración del detector con la API moderna de MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def reset(self):
        """No requiere reinicio de modelo."""
        pass

    def process_frame(self, frame):
        # 1. Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Convertir imagen de OpenCV a formato MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Realizar la inferencia con la red neuronal
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # 4. En el modelo selfie_segmenter, la categoría 1 (o valor distinto de 0) es la persona
        solid_mask = np.where(category_mask > 0, 255, 0).astype(np.uint8)

        # 5. Suavizado de bordes para la estela de color en visual_effects.py
        output = cv2.GaussianBlur(solid_mask, (5, 5), 0)

        return output