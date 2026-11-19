import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteVirtualBgEffect:
    """
    Efecto Fondo Virtual Dinámico:
    - Reemplaza el entorno real por un fondo degradado animado con rejilla.
    - Integra la silueta humana suavizando los bordes.
    """

    def __init__(self, line_thickness=1):
        self.line_thickness = line_thickness
        self.tick = 0
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()
        self.tick = 0

    def _generate_animated_bg(self, h, w):
        # Degradado animado entre azul profundo y magenta/púrpura
        y_grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        pulse = 0.5 + 0.5 * np.sin(self.tick * 0.05)

        bg = np.zeros((h, w, 3), dtype=np.float32)
        bg[..., 0] = 140 + 90 * y_grad                 # Azul
        bg[..., 1] = 20 + 40 * (1.0 - y_grad) * pulse  # Verde
        bg[..., 2] = 60 + 140 * y_grad * pulse         # Rojo

        bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)

        # Rejilla horizontal animada estilo synthwave
        offset = (self.tick * 2) % 40
        for y in range(offset, h, 40):
            cv2.line(bg_u8, (0, y), (w, y), (255, 180, 100), 1)
        for x in range(0, w, 60):
            cv2.line(bg_u8, (x, 0), (x, h), (255, 180, 100), 1)

        return bg_u8.astype(np.float32)

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

        soft_mask = cv2.GaussianBlur(mask, (7, 7), 0)
        alpha_f = (soft_mask.astype(np.float32) / 255.0)[..., None]

        virtual_bg = self._generate_animated_bg(h, w)
        combined = frame.astype(np.float32) * alpha_f + virtual_bg * (1.0 - alpha_f)
        output_bgr = np.clip(combined, 0, 255).astype(np.uint8)

        self.tick += 1
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = output_bgr
        out[..., 3] = 255
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteVirtualBgEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                cv2.imshow("Fondo Virtual", res[:, :, :3])
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()