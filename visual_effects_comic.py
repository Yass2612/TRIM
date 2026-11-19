import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteComicEffect:
    """
    Efecto Estilo Cómic / Cel-Shading:
    - Suaviza texturas y reduce los niveles de color dentro de la silueta.
    - Dibuja trazos de tinta negra en los bordes internos y el contorno exterior.
    """

    def __init__(self, color_levels=6, edge_block_size=9, line_thickness=3, dim_bg=0.75):
        self.color_levels = max(2, color_levels)
        self.edge_block_size = edge_block_size if edge_block_size % 2 == 1 else edge_block_size + 1
        self.line_thickness = line_thickness
        self.dim_bg = dim_bg
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

        # 1. Suavizado preservando bordes + cuantización de colores (Cel-shading)
        small = cv2.pyrDown(frame)
        filtered = cv2.bilateralFilter(small, d=7, sigmaColor=75, sigmaSpace=75)
        smooth = cv2.pyrUp(filtered, dstsize=(w, h))

        step = 256.0 / self.color_levels
        quantized = (np.floor(smooth.astype(np.float32) / step) * step + step * 0.5)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        # 2. Extraer líneas de tinta internas (ojos, ropa, pliegues)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        ink_edges = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=self.edge_block_size,
            C=5,
        )

        # Combinar tinta con el relleno cuantizado
        comic_body = cv2.bitwise_and(quantized, quantized, mask=ink_edges).astype(np.float32)

        # 3. Fondo real ligeramente atenuado para resaltar el estilo ilustración
        bg = frame.astype(np.float32) * self.dim_bg
        combined = comic_body * alpha_f + bg * (1.0 - alpha_f)
        output_bgr = np.clip(combined, 0, 255).astype(np.uint8)

        # 4. Contorno exterior grueso tipo viñeta
        if self.line_thickness > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(
                    output_bgr, contours, -1, (15, 15, 15),
                    thickness=self.line_thickness, lineType=cv2.LINE_AA
                )

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = output_bgr
        out[..., 3] = 255
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteComicEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                cv2.imshow("Estilo Comic", res[:, :, :3])
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()