import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteBokehEffect:
    """
    Efecto Retrato / Bokeh Dinámico:
    - Mantiene al sujeto completamente nítido.
    - Aplica un desenfoque óptico intenso al fondo.
    """

    def __init__(self, blur_strength=35):
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()

    @staticmethod
    def _as_uint8_mask(mask):
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def apply(self, frame):
        h, w = frame.shape[:2]
        raw_mask = self.tracker.process_frame(frame)
        mask = self._as_uint8_mask(raw_mask)

        # Suavizar transición de bordes para evitar recorte artificial
        soft_mask = cv2.GaussianBlur(mask, (11, 11), 0)
        alpha_f = (soft_mask.astype(np.float32) / 255.0)[..., None]

        # Desenfocar fondo a resolución reducida para mantener altos FPS
        small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        k = max(3, (self.blur_strength // 2) | 1)
        blurred_small = cv2.GaussianBlur(small, (k, k), 0)
        blurred_bg = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # Combinar sujeto nítido y fondo desenfocado
        combined = frame.astype(np.float32) * alpha_f + blurred_bg * (1.0 - alpha_f)

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(combined, 0, 255).astype(np.uint8)
        out[..., 3] = 255
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteBokehEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                cv2.imshow("Modo Retrato Bokeh", res[:, :, :3])
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()