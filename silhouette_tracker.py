import cv2
import numpy as np


class SilhouetteTracker:
    def __init__(self, history=300, var_threshold=45):  # Subimos varThreshold (antes 32) para filtrar más ruido de fondo
        self.history = history
        self.var_threshold = var_threshold
        
        # 1. Aumentamos el tamaño del kernel morfológico de (5,5) a (7,7) para eliminar puntos gruesos de ruido
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.background = self._create_background_model()
        self.has_background_frame = False

    def _create_background_model(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )

    def reset(self):
        self.background = self._create_background_model()
        self.has_background_frame = False

    def process_frame(self, frame):
        foreground = self.background.apply(frame)

        if not self.has_background_frame:
            self.has_background_frame = True
            foreground = np.zeros_like(foreground)

        # 2. Umbralizado rígido (Binarización): Elimina sombras débiles o parpadeos de luz
        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)

        # 3. Operaciones morfológicas: 
        # MORPH_OPEN elimina píxeles sueltos en el fondo
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self.kernel, iterations=2)
        # MORPH_CLOSE rellena los huecos vacíos dentro de tu propia silueta
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 4. Suavizado Gaussian más potente de (5,5) a (9,9) para recortar los bordes limpios
        foreground = cv2.GaussianBlur(foreground, (9, 9), 0)

        return foreground