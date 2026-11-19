import cv2
import numpy as np
from collections import deque

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker

class SilhouetteStrobeFreezeEffect:
    def __init__(self, capture_interval=24, max_echoes=5):
        self.capture_interval = capture_interval 
        self.history = deque(maxlen=90)
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.history.clear()
        self.tracker.reset()

    def apply(self, frame):
        h, w = frame.shape[:2]
        output = np.zeros((h, w, 4), dtype=np.uint8)
        
        mask = self.tracker.process_frame(frame)
        if mask is None:
            return output
            
        # 1. Asegurarnos de que la máscara es plana y sólida
        if mask.ndim == 3:
            mask = mask[..., 0]
            
        if mask.dtype != np.uint8:
            mask_u8 = (mask > 0).astype(np.uint8) * 255
        else:
            _, mask_u8 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Rellenar contornos por dentro para que sea tu cuerpo macizo
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_maciza = np.zeros_like(mask_u8)
        cv2.drawContours(mask_maciza, contours, -1, 255, thickness=cv2.FILLED)

        # 2. Guardamos tu "yo" actual en la historia
        self.history.append((frame.copy(), mask_maciza.copy()))

        # 3. Configuramos la estela
        step = max(2, self.capture_interval // 3) 
        num_clones = 4 # Número de copias que te persiguen
        
        out_rgb = np.zeros((h, w, 3), dtype=np.float32)
        out_alpha = np.zeros((h, w), dtype=np.float32)
        
        # 4. Dibujamos los clones en el fondo (del más viejo al más nuevo)
        for i in range(num_clones, 0, -1):
            frames_atras = i * step
            
            if len(self.history) > frames_atras:
                past_frame, past_mask = self.history[-(frames_atras + 1)]
                
                # Opacidad escalonada: el más lejano es más transparente
                opacity_factor = 1.0 - (i / (num_clones + 1.5)) 
                alpha_mask = (past_mask.astype(np.float32) / 255.0) * opacity_factor
                
                # Apilamos el clon
                out_rgb = out_rgb * (1 - alpha_mask[..., None]) + past_frame.astype(np.float32) * alpha_mask[..., None]
                out_alpha = out_alpha * (1 - alpha_mask) + alpha_mask

        # 5. LA CLAVE: Dibujamos tu silueta ACTUAL al 100% de opacidad POR ENCIMA.
        # Esto hace que nunca te veas transparente y que los clones solo se vean "por fuera" de ti.
        current_alpha = mask_maciza.astype(np.float32) / 255.0
        out_rgb = out_rgb * (1 - current_alpha[..., None]) + frame.astype(np.float32) * current_alpha[..., None]
        out_alpha = out_alpha * (1 - current_alpha) + current_alpha

        # 6. Pasamos el resultado a la capa final
        output[..., 0:3] = out_rgb.astype(np.uint8)
        output[..., 3] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)
        
        return output

if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteStrobeFreezeEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                # Para la prueba, el fondo será negro para ver bien la transparencia
                comp = res[..., :3].copy()
                cv2.imshow("Efecto Estela Opaca", comp)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()