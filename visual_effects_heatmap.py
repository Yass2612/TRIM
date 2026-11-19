import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteHeatmapEffect:
    """
    Efecto Cámara Térmica / Mapa de Calor:
    - Silueta humana renderizada con paleta térmica (frío a caliente).
    - Fondo transformado en visión nocturna oscura para máximo contraste.
    """

    def __init__(self, colormap=cv2.COLORMAP_JET, thermal_blur=7, line_thickness=2):
        self.colormap = colormap
        self.thermal_blur = thermal_blur if thermal_blur % 2 == 1 else thermal_blur + 1
        self.line_thickness = line_thickness
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
        alpha_f = (mask.astype(np.float32) / 255.0)[..., None]

        # 1. Convertir frame a escala de grises y suavizar para simular radiación térmica
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.thermal_blur > 1:
            gray_smooth = cv2.GaussianBlur(gray, (self.thermal_blur, self.thermal_blur), 0)
        else:
            gray_smooth = gray

        # Aumentar contraste térmico dentro de la silueta
        gray_norm = cv2.normalize(gray_smooth, None, 0, 255, cv2.NORM_MINMAX)
        thermal_body = cv2.applyColorMap(gray_norm, self.colormap).astype(np.float32)

        # 2. Fondo en modo visión nocturna fría (azulado oscuro)
        night_bg = np.zeros_like(frame, dtype=np.float32)
        night_bg[..., 0] = gray * 0.45  # Canal Azul
        night_bg[..., 1] = gray * 0.20  # Canal Verde
        night_bg[..., 2] = gray * 0.10  # Canal Rojo

        # 3. Combinar silueta térmica y fondo frío
        combined = thermal_body * alpha_f + night_bg * (1.0 - alpha_f)
        output_bgr = np.clip(combined, 0, 255).astype(np.uint8)

        # 4. Contorno térmico brillante
        if self.line_thickness > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(
                    output_bgr, contours, -1, (255, 255, 255),
                    thickness=self.line_thickness, lineType=cv2.LINE_AA
                )

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = output_bgr
        out[..., 3] = 255
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteHeatmapEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                cv2.imshow("Mapa de Calor", res[:, :, :3])
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()